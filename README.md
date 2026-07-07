# gr00t-n1.7-so-101

Fine-tuning [NVIDIA GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the SO-101 subset of the
[MolmoAct2-SO100_101 dataset](https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset)
(an index of 1,220 community LeRobot repos with annotated language instructions), with LoRA, on
[Modal](https://modal.com).

## Approach

Staged, per plan in [`plans/`](plans/):

1. **Pipeline prototype** — full Modal → GR00T loop on
   [`whosricky/so101-megamix-v1`](https://huggingface.co/datasets/whosricky/so101-megamix-v1)
   (400 episodes, 8 tasks, LeRobot v3 → GR00T-flavored v2 conversion).
2. **Hyperparameter sweep** — short runs over learning rate × batch size.
3. **Filtered subset** — SO-101-only repos from the MolmoAct2 manifest with identifiable
   front+wrist cameras, ≥20 episodes, and loadable data; main LoRA fine-tune on that mixture.

Key implementation points:

- **LoRA**: peft on the action DiT + [`category-lora`](https://github.com/jeqcho/category-lora)
  on GR00T's `CategorySpecificLinear` layers; backbone frozen.
- **Preemption-safe checkpointing**: trainable-only rolling checkpoints (3 slots) + durable
  keeps every 500 steps; auto-resume inside the Modal container. Interval started at 5 steps,
  re-derived from measured costs to 300 (Young-Daly; the 5-step cadence cost ~50% of
  wall-clock) — see [docs/checkpoint-interval.md](docs/checkpoint-interval.md).
- **Logging**: wandb train/test loss, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`,
  `DCGM_FI_PROF_DRAM_ACTIVE`, FLOPs/MFU.
- **Train/test split**: episode-level holdout (5%, min 1, seeded) per source repo.

## Layout

- `plans/` — implementation plan (subagent-critiqued)
- `data/repo_list.json` — snapshot of the MolmoAct2 manifest (1,220 repos)
- `src/` — Modal app + training/data code
- `logs/` — local monitoring logs

## Status (2026-07-06)

- ✅ Stage 1 gate passed: megamix converted (with metadata sanitization — the source repo has
  duplicate episode rows + stale file indices), 380/20 split, 60-step smoke train with LoRA
  (55.4M trainable / 3.20B = 1.73%), eval loss, FLOPs, rolling checkpoints (3×5-step,
  trainable-only ≈ seconds to save), keep-copies, and a live kill→resume drill (restore at
  step 60 → continue to 120 with loss continuity).
- ⚠️ DCGM `PIPE_TENSOR_ACTIVE`/`DRAM_ACTIVE` are **not collectable on Modal**: the DCP
  profiling data path is blocked under gVisor (dcgmi works but streams N/A; NVML GPM errors).
  The sampler transparently falls back to NVML utilization (`gpu/util`, `gpu/mem_util`,
  `dcgm/source_tier=3` in wandb) rather than mislabeling a different metric.
- ✅ LR × batch sweep (250 steps each, megamix, held-out eval loss on cached batches):

  | lr \ batch | 32 | 64 |
  |---|---|---|
  | 5e-5 | 1.04 | 0.97 |
  | 1e-4 | 0.93 | 0.79 |
  | 3e-4 | 0.75 | 0.59 |
  | 6e-4 | — | **0.44** |
  | 1.2e-3 | — | 0.79 |

  Optimum bracketed at **lr=6e-4, bs=64** (3e-4 → 0.59, 6e-4 → 0.44, 1.2e-3 → 0.79).
  (Eval-loss jitter of ±0.04 traced to the flow-matching head sampling noise in `forward`;
  now seeded via `fork_rng` for step-comparable curves.)
- ✅ Stage-2 prep: 39/39 repos converted (2,130 train / 112 val episodes, 1.8M frames).
- ✅ Main run (`main-04`, wandb project `gr00t-n17-so101`): lr=3e-4, bs=64, 22,000 steps
  (~0.8 epoch), 1× H100. Held-out eval loss **1.129 → 0.0273** (best at step 21,000;
  tail converged flat ≈0.0273 under LR decay). Train loss 0.019. Survived one real
  preemption (auto-resume, ≤5 steps lost). Earlier attempts documented in CLAUDE.md:
  lr=6e-4 NaN'd at step 2.5k; mixed-aspect-ratio camera crash fixed with a uniform
  256×256 letterbox; mid-run restarts re-derived the checkpoint cadence (Young–Daly,
  docs/checkpoint-interval.md) and fixed GPU starvation (cpu=16 + 8 workers → 2× throughput).

## Output

Final checkpoint published to HuggingFace: [`robocurve/gr00t-n1.7-so101-molmoact2`](https://huggingface.co/robocurve/gr00t-n1.7-so101-molmoact2) (upon completion).
