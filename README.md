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
- **Preemption-safe checkpointing**: trainable-only checkpoints every 5 steps (rolling window
  of 3) + durable keeps at an interval; auto-resume inside the Modal container.
- **Logging**: wandb train/test loss, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`,
  `DCGM_FI_PROF_DRAM_ACTIVE`, FLOPs/MFU.
- **Train/test split**: episode-level holdout (5%, min 1, seeded) per source repo.

## Layout

- `plans/` — implementation plan (subagent-critiqued)
- `data/repo_list.json` — snapshot of the MolmoAct2 manifest (1,220 repos)
- `src/` — Modal app + training/data code
- `logs/` — local monitoring logs

## Output

Final checkpoint published to HuggingFace: `jeqcho/gr00t-n1.7-so101-molmoact2` (upon completion).
