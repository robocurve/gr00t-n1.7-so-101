# Optimal preemption checkpoint interval (Young–Daly)

Decision record, 2026-07-06. Applied from main-04 step ~3,300 onward: `save_steps=250` — chosen over the raw optimum band
midpoint so saves align with `eval_steps=250` (every kept checkpoint carries an exact eval
number) and with `keep_steps=500` (keeps fire every 2nd save; with 300 they would only fire
at LCM(300,500)=1,500). Rolling window still 3 slots.

## Problem

The original spec saved a rolling checkpoint **every 5 steps** for Modal-preemption safety.
Measured on the live main run (653 saves, H100, trainable-only checkpoints of ~55M params
+ optimizer state to a Modal volume):

- save cost δ: safetensors write + `volume.commit()` ≈ **7.5 s median** (commit alone:
  median 6.0 s, p90 8.6 s, max 17.2 s)
- clean step time: ≈ **1.4 s/step** at batch 64 → 5 steps ≈ 7 s of compute

So the 5-step cadence spent **≈50% of wall-clock on checkpointing** (observed as
~2.2 s/it effective vs ~1.4 s/it clean).

## Model

Young–Daly optimal interval between checkpoints:

```
T_opt = sqrt(2 · δ · M)
```

- δ = 7.5 s (measured above)
- M = mean time between preemptions. Observed: 1 preemption in ≈11 GPU-hours across all
  runs of this project (wide error bars; Modal H100s appear to preempt on the order of
  once per day).

| MTBF M | T_opt | steps @1.4 s |
|---|---|---|
| 4 h | ≈ 8 min | ≈ 330 |
| 11 h (observed) | ≈ 13 min | ≈ 550 |
| 24 h | ≈ 19 min | ≈ 810 |

## Total overhead (save tax + expected rework T/2 per preemption, amortized at M = 11 h)

| save_steps | save tax | expected rework | total |
|---|---|---|---|
| 5 (original) | ~50% | ~0.005% | **~50%** |
| 50 | ~9.5% | ~0.09% | ~9.6% |
| 100 | ~4.8% | ~0.18% | ~5.0% |
| **300 (chosen)** | **~1.75%** | **~0.5%** | **~2.3%** |
| 500 | ~1.06% | ~0.9% | ~2.0% |
| 800 | ~0.66% | ~1.4% | ~2.1% |

The optimum is flat: anything in 200–800 steps is within ~0.3 pp of ideal. **300** sits at
the risk-averse end of the flat region (a preemption costs ≤7 min of rework) while
eliminating essentially all of the 2× wall-clock tax.

Applying mid-run was safe because checkpoints are format-compatible across `save_steps`
changes: stop app → relaunch without `--fresh` → auto-resume from the latest rolling
checkpoint (≤`save_steps` old) with the same LR schedule (`max_steps` unchanged).

## Caveats

- δ scales with trainable-parameter count: full-model checkpoints (~6 GB + optimizer)
  would have δ ≈ 30–60 s → T_opt ≈ 15–40 min. The trainable-only LoRA checkpoints
  (see `src/ckpt_patches.py`) are what make sub-10-minute intervals cheap at all.
- M is estimated from one observed preemption; the flatness of the optimum makes the
  choice robust to an order-of-magnitude error in M.
- `keep_steps` (durable history for best-checkpoint selection) is a separate concern and
  stays at 500.
