"""Publish a trained checkpoint to HuggingFace.

Rebuilds the model (base + LoRA), loads a trainable-only checkpoint, merges
adapters into base weights, saves a full model folder (plus the processor /
experiment_cfg artifacts Gr00tPolicy.from_pretrained needs), and uploads it.
Raw adapters are uploaded under adapters/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GR00T_DIR = os.environ.get("GR00T_DIR", "/root/Isaac-GR00T")
BASE_MODEL = "nvidia/GR00T-N1.7-3B"


def pick_checkpoint(exp_dir: Path, step: int) -> Path:
    keep = exp_dir / "keep"
    candidates = []
    for base in (keep, exp_dir):
        if base.is_dir():
            candidates += [p for p in base.iterdir() if p.name.startswith("checkpoint-")]
    assert candidates, f"no checkpoints under {exp_dir}"
    if step:
        matches = [p for p in candidates if p.name == f"checkpoint-{step}"]
        assert matches, f"checkpoint-{step} not found; have {sorted(p.name for p in candidates)}"
        return matches[0]
    return max(candidates, key=lambda p: int(p.name.split("-")[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--repo-id", default="")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    args = p.parse_args()

    import torch
    from huggingface_hub import HfApi
    from safetensors.torch import load_file

    from lora import apply_lora, merge_lora, strip_lora_prefixes

    exp_dir = Path(args.exp_dir)
    ckpt = pick_checkpoint(exp_dir, args.step)
    print(f"publishing from {ckpt}")

    repo_id = args.repo_id or "robocurve/gr00t-n1.7-so101-molmoact2"

    # --- rebuild model with LoRA, load trainable weights, merge ---
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model = Gr00tN1d7.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, attn_implementation="eager")  # CPU container: flash-attn needs CUDA; no forward pass happens here
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    sd = load_file(str(ckpt / "trainable.safetensors"))
    result = model.load_state_dict(sd, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys[:5]
    print(f"loaded {len(sd)} trainable tensors")
    model = merge_lora(model)
    merged_sd = strip_lora_prefixes(model.state_dict())

    out = Path("/tmp/publish") / repo_id.replace("/", "__")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # save merged model with the original architecture's key layout
    base = Gr00tN1d7.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    missing, unexpected = base.load_state_dict(merged_sd, strict=False)
    assert not unexpected, f"unexpected keys after merge: {unexpected[:5]}"
    assert not missing, f"missing keys after merge: {missing[:5]}"
    base = base.to(torch.bfloat16)
    base.save_pretrained(str(out))

    # processor + experiment cfg artifacts for Gr00tPolicy.from_pretrained
    for artifact in ("processor", "experiment_cfg"):
        src = exp_dir / artifact
        if src.exists():
            shutil.copytree(src, out / artifact)
    # trainer provenance
    state_file = ckpt / "trainer_state.json"
    if state_file.exists():
        shutil.copyfile(state_file, out / "trainer_state.json")

    # raw adapters
    adapters_dir = out / "adapters"
    adapters_dir.mkdir()
    shutil.copyfile(ckpt / "trainable.safetensors", adapters_dir / "trainable.safetensors")
    if (ckpt / "trainable_keys.json").exists():
        shutil.copyfile(ckpt / "trainable_keys.json", adapters_dir / "trainable_keys.json")

    card = out / "README.md"
    step_str = ckpt.name.split("-")[1]
    card.write_text(f"""---
license: apache-2.0
base_model: nvidia/GR00T-N1.7-3B
tags:
  - robotics
  - gr00t
  - so101
  - lerobot
  - lora
---

# GR00T N1.7 — SO-101 (MolmoAct2 community data) LoRA fine-tune

LoRA fine-tune of [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)
for the SO-101 follower arm (front + wrist cameras), trained on the SO-101 subset of
[allenai/MolmoAct2-SO100_101-Dataset](https://huggingface.co/datasets/allenai/MolmoAct2-SO100_101-Dataset)
(community LeRobot repos with annotated language instructions). Adapters (peft on the
action DiT + [category-lora](https://github.com/jeqcho/category-lora) on
`CategorySpecificLinear`) are merged into base weights; raw adapters are in `adapters/`.

- Checkpoint step: {step_str}
- Training code: https://github.com/robocurve/gr00t-n1.7-so-101
- Embodiment: `NEW_EMBODIMENT` with the SO-100/101 modality config (`front`/`wrist` cameras,
  6-dof state/action)

## Usage

Load with [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.7):

```python
from gr00t.model.policy import Gr00tPolicy
policy = Gr00tPolicy.from_pretrained("{repo_id}")
```

See the training repo for eval and serving instructions.
""")

    # Upload with the org-write token (HF_WRITE_TOKEN); model LOADING above ran
    # under the default workspace token, whose account has accepted the gated
    # Cosmos-Reason2-2B license (ours hasn't -> 403 on processor_config.json).
    write_token = os.environ.get("HF_WRITE_TOKEN")
    api = HfApi(token=write_token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model")
    print(f"published -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
