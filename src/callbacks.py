"""Trainer callbacks: FLOPs, eval loss over held-out roots, keep-checkpoints,
volume commit.

Eval design (see plan rev 3): ShardedSingleStepDataset is a shard-ABC
(get_shard(idx) -> list of samples; no __iter__/__getitem__), built with
processor=None. We attach the TRAIN pipeline's processor instance (its
statistics were set from the merged train mixture — identity reuse is the
normalization guarantee), pull samples once via get_shard, collate once with
the pipeline collator, and cache batches on CPU. Every eval reuses identical
tensors -> deterministic metric (single augmentation draw, noted in wandb
config) with zero per-eval video-decode cost.
"""

from __future__ import annotations

import os
import shutil
import time

import torch
from transformers.trainer_callback import TrainerCallback

H100_BF16_PEAK = 989e12  # dense BF16 FLOP/s per H100


def _model_call(model, batch):
    import inspect

    try:
        params = list(inspect.signature(model.forward).parameters)
        if params[:1] == ["inputs"]:
            return model(batch)
    except (ValueError, TypeError):
        pass
    return model(**batch)


def _to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    return batch


def calibrate_flops(model, batch) -> float:
    """Measure fwd FLOPs of one (cached eval) batch with FlopCounterMode; x3 ~ fwd+bwd.

    Caveats (logged to wandb config as flops_caveat): FlopCounterMode has no
    formula for flash-attn custom ops (attention undercounted) and x3
    overcounts since the frozen backbone runs no weight-grad backward. Treat
    as a monitoring index, not exact MFU.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    flop_counter = FlopCounterMode(display=False)
    device = next(model.parameters()).device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), flop_counter:
        _model_call(model, _to_device(batch, device))
    if was_training:
        model.train()
    fwd = flop_counter.get_total_flops()
    per_step = float(fwd) * 3.0
    print(f"[flops] calibrated on eval batch: fwd={fwd:.3e}, per_step~{per_step:.3e}")
    return per_step


class FlopsCallback(TrainerCallback):
    def __init__(self, flops_per_batch: float, batches_per_step: float = 1.0, n_gpus: int = 1):
        # flops_per_batch is calibrated on an eval-sized batch; scale to the
        # train batch via batches_per_step = train_batch/eval_batch ratio.
        self.flops_per_step = flops_per_batch * batches_per_step
        self.n_gpus = n_gpus
        self._t0 = None
        self._f0 = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        total = self.flops_per_step * state.global_step
        now = time.time()
        logs["flops/total"] = total
        if self._t0 is not None and now > self._t0:
            rate = (total - self._f0) / (now - self._t0)
            logs["flops/per_sec"] = rate
            logs["flops/mfu_approx"] = rate / (H100_BF16_PEAK * self.n_gpus)
        self._t0, self._f0 = now, total


class EvalLossCallback(TrainerCallback):
    """Deterministic eval loss on cached, train-processor-processed val batches."""

    def __init__(
        self,
        trainer,
        batch_factory,
        every_steps: int,
        first_batch_hook=None,
    ):
        """batch_factory: () -> list[batch] (called once, lazily, on first eval).
        first_batch_hook: called with (model, first_batch) after caching —
        used to wire FLOPs calibration without a second decode pass."""
        self.trainer = trainer
        self.factory = batch_factory
        self.every_steps = max(1, every_steps)
        self.first_batch_hook = first_batch_hook
        self._batches = None

    def ensure_cached(self):
        if self._batches is None:
            t0 = time.time()
            self._batches = self.factory()
            assert self._batches, "eval batch factory produced no batches"
            print(f"[eval] cached {len(self._batches)} batches in {time.time() - t0:.1f}s")
            if self.first_batch_hook is not None:
                self.first_batch_hook(self.trainer.model, self._batches[0])

    def _eval(self, state):
        self.ensure_cached()
        model = self.trainer.model
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in self._batches:
                out = _model_call(model, _to_device(batch, device))
                loss = out["loss"] if isinstance(out, dict) else out.loss
                losses.append(float(loss.detach().float().cpu()))
        if was_training:
            model.train()
        mean = sum(losses) / len(losses)
        import wandb

        if wandb.run is not None:
            wandb.log({"eval/loss": mean, "eval/n_batches": len(losses)}, commit=False)
        print(f"[eval] step {state.global_step}: eval/loss={mean:.4f} ({len(losses)} batches)")
        return mean

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1 or state.global_step % self.every_steps == 0:
            self._eval(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._eval(state)


class KeepCheckpointCallback(TrainerCallback):
    """Copy rolling checkpoints to a durable keep/ dir every keep_steps."""

    def __init__(self, keep_steps: int, keep_dir: str):
        self.keep_steps = max(1, keep_steps)
        self.keep_dir = keep_dir

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.keep_steps != 0:
            return
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(self.keep_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(src) and not os.path.exists(dst):
            os.makedirs(self.keep_dir, exist_ok=True)
            shutil.copytree(src, dst)
            print(f"[keep] {src} -> {dst}")


class VolumeCommitCallback(TrainerCallback):
    """Commit the Modal checkpoint volume after each save (durability under preemption)."""

    def __init__(self, volume=None):
        self.volume = volume
        if self.volume is None:
            try:
                import modal

                if os.environ.get("MODAL_TASK_ID"):
                    self.volume = modal.Volume.from_name("gr00t-so101-ckpt")
            except Exception:  # noqa: BLE001
                self.volume = None

    def on_save(self, args, state, control, **kwargs):
        if self.volume is None:
            return
        try:
            t0 = time.time()
            self.volume.commit()
            print(f"[volume] commit ok ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[volume] commit failed (will retry next save): {e}")
