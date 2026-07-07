"""Trainable-only checkpointing for HF Trainer.

Full GR00T checkpoints are ~6 GB; at save_steps=5 that would dominate
wall-clock. With LoRA, only a small fraction of params train — so rolling
checkpoints store ONLY trainable weights (plus HF's own optimizer/scheduler/
RNG/trainer_state, which are already trainable-sized). Frozen base weights are
reconstructed from base_model_path at every (re)start, then the trainable
subset is loaded on top with strict=False.

Constraints (asserted): single process, no deepspeed, save_only_model=False.
"""

from __future__ import annotations

import json
import os
import shutil

import torch

TRAINABLE_WEIGHTS = "trainable.safetensors"
TRAINABLE_KEYS = "trainable_keys.json"


def patch_trainable_only_checkpointing(trainer_cls):
    """Patch a Trainer subclass in place. Call once, before instantiation."""

    def _trainable_keys(model) -> set[str]:
        return {n for n, p in model.named_parameters() if p.requires_grad}

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        from safetensors.torch import save_file

        model = self.model
        keys = _trainable_keys(model)
        sd = model.state_dict()
        # named_parameters uses '.' paths identical to state_dict keys
        trainable_sd = {k: v.detach().to("cpu", copy=True).contiguous() for k, v in sd.items() if k in keys}
        missing = keys - set(trainable_sd)
        assert not missing, f"trainable params missing from state_dict: {sorted(missing)[:5]}"
        save_file(trainable_sd, os.path.join(output_dir, TRAINABLE_WEIGHTS))
        with open(os.path.join(output_dir, TRAINABLE_KEYS), "w") as f:
            json.dump(sorted(keys), f)
        # config for provenance (tiny)
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            try:
                model.config.save_pretrained(output_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[ckpt] config save skipped: {e}")
        print(f"[ckpt] saved {len(trainable_sd)} trainable tensors -> {output_dir}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from safetensors.torch import load_file

        model = model if model is not None else self.model
        path = os.path.join(resume_from_checkpoint, TRAINABLE_WEIGHTS)
        assert os.path.exists(path), (
            f"{path} not found — checkpoint {resume_from_checkpoint} is not a "
            "trainable-only checkpoint from this launcher"
        )
        sd = load_file(path)
        with open(os.path.join(resume_from_checkpoint, TRAINABLE_KEYS)) as f:
            saved_keys = set(json.load(f))
        current_keys = _trainable_keys(model)
        assert saved_keys == current_keys, (
            "trainable key sets differ between checkpoint and model "
            f"(saved-only: {sorted(saved_keys - current_keys)[:5]}, "
            f"model-only: {sorted(current_keys - saved_keys)[:5]}). "
            "LoRA config must match the original run."
        )
        result = model.load_state_dict(sd, strict=False)
        assert not result.unexpected_keys, f"unexpected keys: {result.unexpected_keys[:5]}"
        loaded = set(sd.keys())
        assert loaded == saved_keys
        print(f"[ckpt] restored {len(loaded)} trainable tensors from {resume_from_checkpoint}")

    trainer_cls._save = _save
    trainer_cls._load_from_checkpoint = _load_from_checkpoint
    return trainer_cls


def assert_compat(training_args):
    assert not getattr(training_args, "deepspeed", None), "trainable-only ckpt: no deepspeed"
    assert not training_args.save_only_model, "save_only_model breaks resume"
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "trainable-only ckpt: single process only"


def wipe_or_resume(exp_dir: str, fresh: bool, volume=None) -> bool:
    """Decide resume INSIDE the container at every (re)start.

    fresh=True wipes exactly once per operator launch: the wipe is recorded in
    a `.fresh_done` marker so retries of the same invocation never re-wipe.
    The volume is committed immediately after the wipe so a preemption in the
    window can't resurrect the old experiment dir.
    Returns True iff a resumable checkpoint exists after any wipe.
    """
    from transformers.trainer_utils import get_last_checkpoint

    os.makedirs(exp_dir, exist_ok=True)
    marker = os.path.join(exp_dir, ".fresh_done")
    if fresh and not os.path.exists(marker):
        for entry in os.listdir(exp_dir):
            p = os.path.join(exp_dir, entry)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        with open(marker, "w") as f:
            f.write("wiped once; retries must not re-wipe\n")
        print(f"[resume] fresh launch: wiped {exp_dir}")
        if volume is not None:
            try:
                volume.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[resume] volume commit after wipe failed: {e}")
    _discard_nonfinite_checkpoints(exp_dir)
    last = get_last_checkpoint(exp_dir)
    if last is None:
        last = _promote_keep_checkpoint(exp_dir)
    print(f"[resume] last checkpoint in {exp_dir}: {last}")
    return last is not None


def _ckpt_is_finite(ckpt_dir: str) -> bool:
    """Complete AND finite. A container killed mid-save (e.g. `modal app stop`
    during checkpoint write) leaves partial dirs — weights present but
    trainer_state.json/optimizer.pt missing — which crash-loop HF resume."""
    from safetensors.torch import load_file

    for required in (TRAINABLE_WEIGHTS, TRAINABLE_KEYS, "trainer_state.json",
                     "optimizer.pt", "scheduler.pt"):
        if not os.path.exists(os.path.join(ckpt_dir, required)):
            return False
    try:
        sd = load_file(os.path.join(ckpt_dir, TRAINABLE_WEIGHTS))
    except Exception:  # noqa: BLE001
        return False
    return all(torch.isfinite(t).all() for t in sd.values())


def _discard_nonfinite_checkpoints(exp_dir: str):
    """Delete rolling checkpoints with NaN/inf trainable weights so resume
    never restores a poisoned state (a NaN'd run saves poisoned checkpoints
    every 5 steps before the NanGuard can fire)."""
    for entry in sorted(os.listdir(exp_dir)):
        if not entry.startswith("checkpoint-"):
            continue
        p = os.path.join(exp_dir, entry)
        if os.path.isdir(p) and not _ckpt_is_finite(p):
            print(f"[resume] discarding non-finite checkpoint {p}")
            shutil.rmtree(p)


def _promote_keep_checkpoint(exp_dir: str) -> str | None:
    """If no valid rolling checkpoint remains, copy the newest finite keep/
    checkpoint back into the exp dir so training resumes from the last good
    durable state instead of starting over."""
    keep = os.path.join(exp_dir, "keep")
    if not os.path.isdir(keep):
        return None
    candidates = sorted(
        (e for e in os.listdir(keep) if e.startswith("checkpoint-")),
        key=lambda e: int(e.split("-")[1]),
        reverse=True,
    )
    for entry in candidates:
        src = os.path.join(keep, entry)
        if _ckpt_is_finite(src):
            dst = os.path.join(exp_dir, entry)
            shutil.copytree(src, dst)
            print(f"[resume] promoted keep checkpoint {src} -> {dst}")
            return dst
    return None


def torch_rng_sanity():
    """Tiny helper used by tests."""
    return torch.initial_seed()
