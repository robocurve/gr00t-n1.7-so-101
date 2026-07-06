# CLAUDE.md

Fine-tuning GR00T N1.7 (LoRA) on SO-101 data from the MolmoAct2 manifest, on Modal.
Read `plans/2026-07-05-gr00t-n17-so101-finetune.md` first — it is the source of truth
(subagent-critiqued, rev 3) and records verified upstream facts (e.g. GR00T's eval
path is dead code; we ship our own eval callback).

## Key context

- Modal profile `robocurve`; secrets `huggingface-token`, `wandb` already exist.
  Volumes: `gr00t-so101-data` (/data), `gr00t-so101-ckpt` (/ckpt).
- Isaac-GR00T pinned to `ab88b50c718f6528e1df9dcbaf75865d1b604760` in the image;
  our code is mounted at `/root/proj/src` (last image layer, cheap to iterate).
- Everything trains via `src/train_launcher.py`, which monkeypatches
  `gr00t.experiment.experiment` (LoRA via patched `return_model`, callbacks via a
  `Gr00tTrainer` subclass, trainable-only checkpointing via `_save`/`_load_from_checkpoint`).
- Checkpoints: rolling×3 trainable-only, save_steps=300 (Young-Daly optimum, see docs/checkpoint-interval.md; was 5 → ~50% wall-clock tax) + durable `keep/` copies every 500.
  Resume is auto-detected in-container (`wipe_or_resume`); NEVER pass static resume flags.
- wandb: project `gr00t-n17-so101`; run id == exp name (`WANDB_RUN_ID`) so preemption
  retries continue the same run.
- `.env` has HF_TOKEN/HF_USER_ID — public repo, never commit it.

## Commands

```bash
modal run src/modal_app.py::env_check          # env + DCGM probe + module dump
modal run src/modal_app.py::prepare_megamix    # stage-1 data
modal run src/modal_app.py::smoke_train        # 60-step smoke
modal run src/modal_app.py::sweep_one --lr 1e-4 --bs 32
modal run --detach src/modal_app.py::train --exp-name main-01 ...
uv run tests/test_ckpt_patches.py              # local ckpt/resume unit test
```

## Gotchas

- LeRobot v3 repos must go through GR00T's `convert_v3_to_v2.py` (separate uv project
  `scripts/lerobot_conversion`); output nests at `<root>/<user>/<repo>`.
- Camera keys must map to exactly `front`/`wrist` via `meta/modality.json` `original_key`.
- Eval normalization: the eval callback MUST reuse the train pipeline's processor object;
  never build a fresh processor or merge stats from val roots.
- `episode_sampling_rate` default is 0.1 (10% of timesteps) — we set 1.0 explicitly.
- Multi-GPU is out of scope: `experiment.py` hardcodes `ddp_find_unused_parameters=False`,
  which conflicts with category-lora; single H100 everywhere.
