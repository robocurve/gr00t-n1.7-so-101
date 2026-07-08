"""One-off: publish aris's GR00T N1.7 YAM checkpoint (Modal volume ->
robocurve/gr00t-n1.7-yam-molmoact2). Full fine-tune checkpoint — no LoRA merge;
straight upload of model weights + inference artifacts + model card.

Usage: modal run src/publish_yam.py
"""

import modal

app = modal.App("publish-gr00t-yam")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")
yam_ckpt = modal.Volume.from_name("gr00t-yam-ckpt")

CKPT = "/ckpt/yam_gr00t/trial-pi05-sampling/milestones/checkpoint-10000"
REPO = "robocurve/gr00t-n1.7-yam-molmoact2"

# Everything Gr00tPolicy.from_pretrained needs; skip DeepSpeed leftovers
# (zero_to_fp32.py, latest, global_step*, training_args.bin, optimizer states).
INCLUDE = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "trainer_state.json",
]
INCLUDE_DIRS = ["experiment_cfg"]

CARD = """---
license: apache-2.0
base_model: nvidia/GR00T-N1.7-3B
tags:
  - robotics
  - gr00t
  - yam
  - bimanual
  - lerobot
  - vla
---

# GR00T N1.7 — bimanual YAM fine-tune on MolmoAct2 data

Full fine-tune (native Isaac-GR00T recipe: action head + projector trainable, VLM backbone
frozen) of [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the AllenAI
**MolmoAct2-BimanualYAM** dataset, for the I2RT YAM bimanual arm platform. The GR00T analog of
[robocurve/pi05-yam-molmoact2](https://huggingface.co/robocurve/pi05-yam-molmoact2).

## Training

| | |
|---|---|
| Data | AllenAI MolmoAct2-BimanualYAM: 124 repos (block / box / charging tasks), ~5,145 episodes, ~11.35M frames; LeRobot v3 → v2 converted; AV1 videos transcoded to H.264 (torchcodec cannot reliably decode AV1) |
| Embodiment | `NEW_EMBODIMENT` with a custom bimanual config: state/action keys `left_arm / left_gripper / right_arm / right_gripper` (absolute joints), 3 cameras |
| Recipe | native GR00T N1.7 fine-tune (HF Trainer + DeepSpeed ZeRO-2): lr 1e-4, global batch 256 x grad-accum 2 (effective 512), warmup 1%, episode_sampling_rate 1.0 |
| Steps | 10,000 (this checkpoint = final and best) |
| Held-out val | open-loop action **MSE 0.00279** on `allenai/19012026-block-13` (unseen episode repo); π0.5 reference on the identical protocol: 0.00206 |

## Provenance

| | |
|---|---|
| Trained by | aris @ [Robocurve](https://huggingface.co/robocurve), 2026-07 (uploaded from the training volume by the team) |
| Training code | `robocurve/gr00t-n17-yam-replication` (Modal pipeline: data prep, training, val sidecar) |
| Framework | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.7), PyTorch/HF Trainer/DeepSpeed ZeRO-2 |
| Compute provider | [Modal](https://modal.com), multi-GPU (B200-class) |

## Losses

- **Training:** GR00T N1.7's native flow-matching action objective (DiT predicts the velocity
  field for noised action chunks conditioned on frozen VLM features + robot state; MSE against
  the interpolation target).
- **Validation:** open-loop action MSE — predicted vs ground-truth action trajectories on a
  fully held-out episode repo, via the training pipeline's out-of-band val sidecar (upstream
  Isaac-GR00T's HF-Trainer eval path is non-functional for sharded datasets).

## Usage

```python
from gr00t.model.policy import Gr00tPolicy
policy = Gr00tPolicy.from_pretrained("robocurve/gr00t-n1.7-yam-molmoact2")
```

Serve with Isaac-GR00T's `run_gr00t_server.py`; drive the YAM via the
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam) adapters.

## Caveats

- Open-loop MSE is a proxy; no closed-loop/real-robot success rates reported yet.
- Trained on three task families (block, box, charging) — expect limited transfer beyond them.
"""


@app.function(
    image=image,
    volumes={"/ckpt": yam_ckpt},
    secrets=[modal.Secret.from_name("hf-robocurve-write")],
    timeout=2 * 3600,
    cpu=4,
    memory=16384,
)
def publish():
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_WRITE_TOKEN"])
    api.create_repo(REPO, repo_type="model", exist_ok=True)

    src = Path(CKPT)
    ops = []
    for f in INCLUDE:
        p = src / f
        assert p.exists(), f"missing {p}"
        ops.append((str(p), f))
    for d in INCLUDE_DIRS:
        for p in (src / d).rglob("*"):
            if p.is_file():
                ops.append((str(p), str(p.relative_to(src))))

    for local, remote in ops:
        print(f"uploading {remote} ({os.path.getsize(local)/1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO)

    api.upload_file(
        path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=REPO,
        commit_message="docs: model card",
    )
    print(f"published -> https://huggingface.co/{REPO}")
