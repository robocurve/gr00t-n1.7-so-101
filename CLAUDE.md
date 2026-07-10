# CLAUDE.md

Fine-tuning GR00T N1.7 (LoRA) on SO-101 data from the MolmoAct2 manifest, on Modal.
Read `plans/2026-07-05-gr00t-n17-so101-finetune.md` first — the plan of record
(subagent-critiqued, rev 3). `docs/checkpoint-interval.md` holds the checkpoint-cadence
decision record. This file doubles as a playbook for sibling projects (e.g. N1.7 on the
MolmoAct YAM dataset) — the Lessons section is the distilled cost of ~$60 of debugging.

## Key context

- Modal profile `robocurve`; secrets `huggingface-token`, `wandb` already exist.
  Volumes: `gr00t-so101-data` (/data), `gr00t-so101-ckpt` (/ckpt).
- Isaac-GR00T pinned to `ab88b50c718f6528e1df9dcbaf75865d1b604760`; our code mounted at
  `/root/proj/src` (last image layer → cheap iteration; but a RUNNING detached app is
  frozen at launch-time code — retries do NOT pick up new commits, only relaunches do).
- Everything trains via `src/train_launcher.py`, which monkeypatches
  `gr00t.experiment.experiment`: LoRA via patched `return_model` (peft
  `inject_adapter_in_model` on action-head Linears + category-lora on
  `CategorySpecificLinear`), callbacks via a `Gr00tTrainer` subclass, trainable-only
  checkpointing via `_save`/`_load_from_checkpoint`, eval via our own callback.
- Cadence (all aligned): eval = rolling save = durable keep = **250 steps**; rolling
  window 3; resume auto-detected in-container (`wipe_or_resume`) — NEVER pass static
  resume flags into a retried Modal function.
- wandb: project `gr00t-n17-so101`; `WANDB_RUN_ID=<exp>` + `WANDB_RESUME=allow` so
  preemption retries continue one run.
- `.env` has HF_TOKEN/HF_USER_ID — public repo, never commit it.

## Commands

```bash
modal run src/modal_app.py::env_check          # env + DCGM probe + module dump
modal run src/modal_app.py::prepare_megamix    # stage-1 data
modal run src/modal_app.py::smoke_train        # 60-step smoke
modal run src/modal_app.py::sweep_all --configs "3e-4:64,6e-4:64"   # ONE app, many configs
modal run --detach src/modal_app.py::train --exp-name main-XX --dataset-roots auto --val-roots auto ...
modal run src/modal_app.py::publish --exp-name main-XX
uv run tests/test_ckpt_patches.py              # local ckpt/resume unit test
```

## Lessons learned (each cost real time/money — read before the YAM repo)

### Isaac-GR00T N1.7 internals
1. **The eval path is dead code.** `DatasetFactory.build` asserts `eval_strategy=="no"`;
   `val_dataset_path`/`eval_set_split_ratio` are declared but consumed nowhere; the model
   forward has no `labels` kwarg. Test loss requires a custom callback: build
   `ShardedSingleStepDataset` over val roots, `set_processor(<the train pipeline's
   processor object>)` (identity reuse = normalization parity — never construct a fresh
   processor or merge val stats), pull via `get_shard(i)` (it's a shard ABC, no
   `__getitem__`/`__iter__`), collate once with `processor.collator`, cache on CPU.
2. **The image pipeline is aspect-ratio-preserving BY DESIGN** (albumentations:
   SmallestMaxSize → FractionalRandomCrop → SmallestMaxSize). Any sample or batch mixing
   16:9 and 4:3 cameras crashes `torch.stack` — deterministically at the same step on
   every retry (seeded shard schedule) → crash loop. The `letter_box_transform` flag only
   exists in the UNUSED torchvision branch (setting it logs `True` and does nothing).
   Fix: monkeypatch `build_image_transformations_albumentations` to append
   `LongestMaxSize(256) + PadIfNeeded(256,256)` (true letterbox). Uniform 256×256 also
   made steps ~10% faster. **For YAM: check camera geometries across the dataset first.**
3. **Batches are `{"inputs": {...}}`** — call `model(**batch)` exactly like HF Trainer
   (a positional `model(batch)` buries the dict and dies with `KeyError: input_ids`).
   `BatchFeature` is a UserDict, not a dict — use `collections.abc.Mapping` checks.
4. **The flow-matching head samples noise inside `forward`** → eval loss on identical
   inputs jitters ±0.04. Seed evals with `torch.random.fork_rng` + `manual_seed` or your
   sweep decisions are made on noise (and duplicate evals look like a bug).
5. **`episode_sampling_rate` defaults to 0.1** (only 10% of timesteps per epoch). Set 1.0
   explicitly if you want honest epoch math.
6. **`nvidia/Cosmos-Reason2-2B` (the VLM backbone repo) is gated**, but nothing gated is
   ever downloaded — the tokenizer ships inside the GR00T repo. However the processor
   path calls `model_info()` on it → `HF_HUB_OFFLINE=1` hard-crashes. Don't use offline mode.
7. **megamix-style v3 repos can be internally corrupt**: duplicate episode rows and stale
   `data/file_index` pointing at consolidated files that no longer exist. Sanitize
   meta/episodes parquet (dedup by episode_index; repoint indices if global offsets fit)
   BEFORE running `convert_v3_to_v2.py`; the converter skips its own download when the
   local root already exists (that's how the sanitized copy survives).

### Hyperparameters (LoRA r=32/α=64 on action head, bs=64, 1×H100)
8. **Short-horizon sweeps lie about stability.** lr=6e-4 won the 250-step sweep decisively
   (0.44 vs 0.59 @3e-4) then NaN'd the real run at step ~2,500, right after warmup reached
   the plateau. lr=3e-4 trains clean. If the sweep winner sits at the edge of your grid,
   deduct stability points before betting a 20k-step run on it.
9. **Guard against NaN and poisoned checkpoints**: a NaN'd run keeps writing poisoned
   rolling checkpoints (rotating out good ones) while burning GPU. `NanGuardCallback`
   aborts on non-finite loss/grad; `wipe_or_resume` discards non-finite checkpoints and
   promotes the newest finite `keep/` copy.

### Modal
10. **Function `retries` caps at 10.** Auto-resume must be idempotent so a retry (or an
    external relaunch) is always safe.
11. **App-creation is rate-limited** — N concurrent `modal run` invocations die with
    "App creation failed". Fan out INSIDE one app (`Function.starmap`), see `sweep_all`.
12. **Preemption really happens and the design works**: observed once; retry auto-resumed
    from the rolling checkpoint losing <5 steps (then at 250-step cadence, ≤250).
13. **DCGM PROF metrics don't work under Modal's gVisor**: `dcgmi` runs but streams N/A
    (fresh watches also need warmup — use ONE persistent `dcgmi dmon` child, not repeated
    polls); NVML GPM claims support then errors on sampling. Tier-3 fallback
    (`nvmlDeviceGetUtilizationRates` → `gpu/util`, `gpu/mem_util`) is all you get; label
    it honestly, never under the DCGM key names.
14. **Volumes need explicit `.commit()`**; commit costs ~6s median — it's the dominant
    checkpoint cost (see docs/checkpoint-interval.md).
15. **`clang` + `build-essential` required** in the image: lerobot→pynput→evdev builds
    from source in the conversion env.
15b. **Set `cpu=` on GPU train functions — Modal's default CPU allocation starves the GPU
    on video datasets.** Measured on main-04: default alloc + 4 dataloader workers gave
    GPU util median 0% (bursts to ~80%) and ~2.5-3.6 s/step on frame-heavy shards;
    `cpu=16, memory=64G` + 8 workers → sustained 1.8 s/step incl. eval/save overhead
    (~2x throughput, confirmed over 30+ min). Video decode (torchcodec, long mp4 seeks)
    is the bottleneck, not the model — even after the CPU fix, GPU util stays bursty
    (mean ~34%, p90 ~89%, median ~1%): compute finishes fast, then waits on decode.
    ~40% more headroom exists. Ranked fixes for the YAM repo (try in order):
      (a) GPU-side decoding: torchcodec supports CUDA/NVDEC decoding (device="cuda") —
          uses the GPU's dedicated video engine, not SMs; likely a near-one-liner in the
          dataset's decode path and the highest-leverage option.
      (b) Prep-time re-encode: short GOP (keyframe every ~8-16 frames for cheap seeks),
          fixed 256px shortest edge, and/or fps subsampling — long-episode repos
          (140k-frame hanoi-style mp4s) are the worst offenders.
      (c) More CPU/workers (diminishing returns past cpu=16/8 workers at bs=64).
15c. **Never `modal app stop` during a checkpoint write** — it leaves a partial checkpoint
    dir (weights present, trainer_state.json missing) that crash-loops HF resume through
    all retries. Checkpoint validity checks must require ALL resume artifacts
    (trainer_state.json, optimizer.pt, scheduler.pt, weights), not just weight finiteness
    (`ckpt_patches._ckpt_is_finite`).

### HuggingFace Hub
16. **The org token has a 1000 req/5min quota shared by EVERYTHING.** 39 unthrottled prep
    containers + 6 sweeps = instant blackout. Mitigations that worked: serialize prep
    (`max_containers=1`), `snapshot_download(max_workers=4)`, retry-with-330s-backoff —
    and crucially the retry must ALSO match `LocalEntryNotFoundError` ("cannot find the
    appropriate snapshot"), which is how snapshot_download masks 429s. Training runs make
    only dozens of calls; keep them online (see lesson 6) plus a 429-aware setup retry.
17. A HF PRO/Team account would delete this entire lesson class.

### HF Trainer plumbing
18. **Callback order matters for `on_log` mutation**: integration callbacks (WandbCallback)
    are registered before user callbacks — anything you add to `logs` in your callback is
    dropped unless you move WandbCallback to the END of `callback_handler.callbacks`.
19. **Don't log placeholder zeros** (fake observability): emit nothing until calibrated;
    calibrate at `on_train_begin` from the cached eval batch (never pull a throwaway
    train dataloader — it spawns workers and mutates state).
20. **Checkpoint cadence**: Young–Daly `T_opt = sqrt(2δM)`. With δ≈7.5s (trainable-only
    save+commit) and MTBF≈11h, anything in 200–800 steps is near-optimal (~2% overhead);
    save_steps=5 cost ~50% of wall-clock. Align save=eval=keep so every durable
    checkpoint carries an exact eval number (publishing = argmin over keeps).

### Process / monitoring (for the operator agent)
21. **Never end a monitor pipeline with `| head -N`** — it buffers everything and the
    monitor stays silent through failures. A coarse 15-min heartbeat catches what event
    filters miss (it caught two).
22. Deterministic failures re-arm themselves: same seed → same shard schedule → same
    crash step on every retry. If a run dies twice at the same step, it's data/code, not
    infra.
23. Verify a fix changed the OBSERVABLE (tensor shapes, step-1 eval, log marker), not
    just the config echo — lesson 2's flag logged `True` while doing nothing.

## YAM-repo deltas to expect
- Bimanual YAM = different embodiment: new modality config (state/action dims ≈ 14+,
  different camera set) instead of `examples/SO100/so100_config.py`; still
  `NEW_EMBODIMENT` tag; new `meta/modality.json` index ranges.
- Multi-GPU (if used): `experiment.py` hardcodes `ddp_find_unused_parameters=False`,
  which conflicts with category-lora's per-category grads — patch it or stay single-GPU.
- MolmoAct YAM data may be first-party (not a 1220-repo community index) — the filter/
  annotation-rewrite machinery may shrink to a straight conversion, but keep the
  loadability validation and the episode-level split.
- The teammates' `gr00t-yam-data`/`gr00t-yam-ckpt` volumes already exist in this Modal
  workspace (created by aris) — check them before re-downloading anything.

## Writing style (public-facing text)

READMEs, docs pages, repo/collection descriptions, and HF model cards must
avoid AI-writing tells. The full rule with the gating checklist lives in
[worldevals docs/model-cards.md, "Writing style"](https://github.com/robocurve/worldevals/blob/main/docs/model-cards.md);
short version:

- No em dashes in prose. Use periods, colons, commas, or parentheses (`—` is
  fine as an empty table cell and inside code blocks).
- Bold only for definition-list lead-ins (`**term:**`) and at most one critical
  imperative per safety bullet. Never mid-sentence for emphasis.
- No decorative emoji (functional ✅/⚠️ marks and 🤗 for Hugging Face are fine),
  no slogans or chiasmus, no "not just X, but Y".
- Headers use colons, never em dashes or italics.

Style-only edits must never touch YAML frontmatter, code blocks, numbers,
links, or safety qualifiers.
