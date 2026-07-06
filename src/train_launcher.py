"""Custom GR00T N1.7 fine-tune launcher: LoRA + eval loss + trainable-only
5-step rolling checkpoints + GPU metrics + FLOPs, all on top of the stock
`gr00t.experiment.experiment.run` (config built exactly like
launch_finetune.py, Apache-2.0).

Run inside the Isaac-GR00T uv env (Modal container):
  uv run python /root/proj/src/train_launcher.py --exp-name smoke \
      --dataset-roots /data/v2/megamix --val-roots /data/v2/megamix_val ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GR00T_DIR = os.environ.get("GR00T_DIR", "/root/Isaac-GR00T")
MODALITY_CONFIG = f"{GR00T_DIR}/examples/SO100/so100_config.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True)
    p.add_argument("--dataset-roots", required=True, help="os.pathsep-separated train roots")
    p.add_argument("--val-roots", required=True, help="os.pathsep-separated val roots")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--save-steps", type=int, default=5)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--keep-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-batches", type=int, default=24)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--ds-weights-alpha", type=float, default=None)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--output-dir", default="/ckpt/runs")
    p.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    return p.parse_args()


def resolve_auto_roots(args):
    """--dataset-roots auto[:/base] globs prepared stage-2 roots inside the
    container: every /base/<slug>/ with meta/info.json, excluding *_val and
    megamix (megamix is stage-1, not part of the MolmoAct2 manifest). Val
    roots are the matching <root>_val dirs."""
    import glob

    base = "/data/v2"
    if ":" in args.dataset_roots:
        base = args.dataset_roots.split(":", 1)[1]
    roots = []
    for info in sorted(glob.glob(f"{base}/*/meta/info.json")):
        root = os.path.dirname(os.path.dirname(info))
        name = os.path.basename(root)
        if name.endswith("_val") or name.startswith("megamix"):
            continue
        if os.path.exists(f"{root}_val/meta/info.json"):
            roots.append(root)
        else:
            print(f"[launcher] skipping {root}: no _val sibling")
    assert roots, f"no prepared roots found under {base}"
    args.dataset_roots = os.pathsep.join(roots)
    args.val_roots = os.pathsep.join(f"{r}_val" for r in roots)
    print(f"[launcher] auto roots: {len(roots)} datasets")


def load_modality_config(path: str):
    import importlib

    p = Path(path)
    assert p.exists(), f"modality config not found: {path}"
    sys.path.append(str(p.parent))
    importlib.import_module(p.stem)
    print(f"[launcher] loaded modality config: {path}")


def build_config(args, resume: bool):
    """Mirror launch_finetune.py's config construction (Apache-2.0)."""
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    embodiment_tag = EmbodimentTag.resolve("NEW_EMBODIMENT").value
    dataset_paths = [p for p in args.dataset_roots.split(os.pathsep) if p]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    m = config.model
    m.tune_llm = False
    m.tune_visual = False
    # LoRA replaces full projector/DiT tuning; keep flags True so upstream code
    # marks those subtrees trainable, then apply_lora() re-freezes and adapts.
    m.tune_projector = True
    m.tune_diffusion_model = True
    m.state_dropout_prob = 0.2
    m.random_rotation_angle = None
    m.color_jitter_params = None
    m.use_percentiles = True
    m.extra_augmentation_config = None
    m.load_bf16 = False
    m.reproject_vision = False
    m.model_name = "nvidia/Cosmos-Reason2-2B"
    m.backbone_trainable_params_fp32 = True
    m.use_relative_action = True

    t = config.training
    t.experiment_name = args.exp_name
    t.start_from_checkpoint = args.base_model
    t.optim = "adamw_torch"
    t.global_batch_size = args.global_batch_size
    t.dataloader_num_workers = args.dataloader_num_workers
    t.learning_rate = args.lr
    t.gradient_accumulation_steps = 1
    t.output_dir = args.output_dir
    t.save_steps = args.save_steps
    t.save_total_limit = args.save_total_limit
    t.num_gpus = 1
    t.use_wandb = True
    t.wandb_project = os.environ.get("WANDB_PROJECT", "gr00t-n17-so101")
    t.max_steps = args.max_steps
    t.weight_decay = args.weight_decay
    t.warmup_ratio = args.warmup_ratio
    t.save_only_model = False
    t.resume_from_checkpoint = resume
    t.skip_weight_loading = False
    # eval stays "no": upstream factory asserts it; we do our own eval callback.
    t.eval_strategy = "no"

    d = config.data
    d.shard_size = 1024
    d.episode_sampling_rate = 1.0  # use ALL timesteps (default 0.1 would use 10%)
    d.num_shards_per_epoch = int(1e5)
    d.ds_weights_alpha = args.ds_weights_alpha

    return config, embodiment_tag


def make_eval_batch_factory(args, pipeline_ref, config, embodiment_tag):
    """Returns () -> list[cpu_batches]; called lazily at first eval (after the
    train mixture has set statistics on the pipeline processor)."""

    def factory():
        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.stats import generate_rel_stats, generate_stats

        pipeline = pipeline_ref["pipeline"]
        assert pipeline is not None, "pipeline not captured"
        processor = pipeline.processor
        # Identity guarantee: the SAME processor object the train mixture
        # configured with merged train statistics (normalization parity).
        stats = getattr(processor, "statistics", None)
        assert stats is not None and len(stats) > 0, "train processor has no statistics set"
        print(f"[eval] reusing train processor id={id(processor)} (statistics set)")

        val_roots = [p for p in args.val_roots.split(os.pathsep) if p]
        n_needed = args.eval_batch_size * args.eval_batches
        datasets = []
        for root in val_roots:
            generate_stats(root)
            generate_rel_stats(root, EmbodimentTag(embodiment_tag))
            ds = ShardedSingleStepDataset(
                dataset_path=root,
                embodiment_tag=EmbodimentTag(embodiment_tag),
                modality_configs=config.data.modality_configs[embodiment_tag],
                shard_size=64,
                episode_sampling_rate=1.0,
                seed=config.data.seed,
                allow_padding=config.data.allow_padding,
            )
            ds.set_processor(processor)
            datasets.append(ds)
        # Round-robin one shard at a time across roots so the cached eval set
        # is representative of the whole mixture, not just the first root.
        samples = []
        cursors = [0] * len(datasets)
        while len(samples) < n_needed and any(
            cursors[i] < len(datasets[i]) for i in range(len(datasets))
        ):
            for i, ds in enumerate(datasets):
                if len(samples) >= n_needed or cursors[i] >= len(ds):
                    continue
                samples.extend(ds.get_shard(cursors[i]))
                cursors[i] += 1
        samples = samples[:n_needed]
        assert samples, f"no eval samples collected from {val_roots}"
        collator = processor.collator
        batches = [
            collator(samples[i : i + args.eval_batch_size])
            for i in range(0, len(samples), args.eval_batch_size)
        ]
        print(f"[eval] cached {len(batches)} batches x {args.eval_batch_size} from {len(val_roots)} roots")
        return batches

    return factory


def main():
    args = parse_args()

    import torch  # noqa: F401  (fail fast if env broken)

    from callbacks import (
        EvalLossCallback,
        FlopsCallback,
        KeepCheckpointCallback,
        VolumeCommitCallback,
        calibrate_flops,
    )
    from ckpt_patches import (
        assert_compat,
        patch_trainable_only_checkpointing,
        wipe_or_resume,
    )
    from gpu_metrics import start_gpu_metrics_thread
    from lora import apply_lora

    load_modality_config(MODALITY_CONFIG)

    if args.dataset_roots.startswith("auto"):
        resolve_auto_roots(args)

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    volume = None
    if os.environ.get("MODAL_TASK_ID"):
        try:
            import modal

            volume = modal.Volume.from_name("gr00t-so101-ckpt")
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] no modal volume handle: {e}")
    resume = wipe_or_resume(exp_dir, args.fresh, volume)

    config, embodiment_tag = build_config(args, resume)

    import gr00t.experiment.experiment as exp
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model import MODEL_REGISTRY

    pipeline_ref = {"pipeline": None}

    # --- capture pipeline + apply LoRA on model creation ---
    PipelineCls = MODEL_REGISTRY[Gr00tN1d7Config]
    _orig_return_model = PipelineCls.return_model

    def patched_return_model(self):
        pipeline_ref["pipeline"] = self
        model = _orig_return_model(self)
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        try:
            import wandb

            if wandb.run is not None:
                wandb.config.update(
                    {
                        "lora": model._lora_summary,
                        "flops_caveat": "FlopCounterMode misses flash-attn ops; x3 overcounts frozen backbone",
                        "eval_note": "deterministic cached eval batches (single augmentation draw), train processor reused",
                        "launcher_args": vars(args),
                    },
                    allow_val_change=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] wandb config update skipped: {e}")
        return model

    PipelineCls.return_model = patched_return_model

    batch_factory = make_eval_batch_factory(args, pipeline_ref, config, embodiment_tag)

    # --- patched trainer: callbacks + trainable-only checkpointing ---
    class PatchedTrainer(exp.Gr00tTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            assert_compat(self.args)

            flops_cb = FlopsCallback(
                flops_per_batch=0.0,
                batches_per_step=args.global_batch_size / args.eval_batch_size,
            )

            def first_batch_hook(model, batch):
                per_eval_batch = calibrate_flops(model, batch)
                flops_cb.flops_per_step = per_eval_batch * (
                    args.global_batch_size / args.eval_batch_size
                )

            eval_cb = EvalLossCallback(
                self, batch_factory, every_steps=args.eval_steps, first_batch_hook=first_batch_hook
            )
            self.add_callback(flops_cb)
            self.add_callback(eval_cb)
            self.add_callback(
                KeepCheckpointCallback(args.keep_steps, os.path.join(exp_dir, "keep"))
            )
            self.add_callback(VolumeCommitCallback(volume))

    patch_trainable_only_checkpointing(PatchedTrainer)
    exp.Gr00tTrainer = PatchedTrainer

    def is_rate_limited(e: BaseException) -> bool:
        s = f"{type(e).__name__}: {e}"
        return "429" in s or "rate limit" in s.lower() or "quota" in s.lower()

    sampler = start_gpu_metrics_thread(interval_s=10.0)
    try:
        import time

        for attempt in range(4):
            try:
                exp.run(config)
                break
            except Exception as e:  # noqa: BLE001
                # HF 429s strike during setup (model/processor load), before any
                # training step — safe to sleep out the 5-min quota window and
                # re-enter run() with a freshly recomputed resume decision.
                if attempt < 3 and is_rate_limited(e):
                    print(f"[launcher] rate limited during setup; retry {attempt + 1}/3 in 330s")
                    time.sleep(330)
                    config.training.resume_from_checkpoint = wipe_or_resume(exp_dir, False, volume)
                    continue
                raise
    finally:
        try:
            sampler.stop()
        except Exception:  # noqa: BLE001
            pass
        if volume is not None:
            try:
                volume.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[launcher] final volume commit failed: {e}")


if __name__ == "__main__":
    main()
