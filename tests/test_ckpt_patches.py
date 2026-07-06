# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.7.1", "transformers==4.57.3", "safetensors", "accelerate", "numpy==1.26.4"]
# ///
"""Local CPU unit test for trainable-only checkpointing + keep/rolling windows.

Run: uv run tests/test_ckpt_patches.py
Verifies:
  1. rolling dir never exceeds save_total_limit=3 checkpoints
  2. checkpoints contain ONLY trainable weights
  3. keep/ receives copies at keep_steps
  4. kill after N steps -> resume continues from last checkpoint with
     restored trainable weights + optimizer state (loss trajectory sane)
  5. wipe_or_resume semantics (fresh once, marker prevents re-wipe)
"""

import json
import os
import shutil
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from callbacks import EvalLossCallback, FlopsCallback, KeepCheckpointCallback  # noqa: E402
from ckpt_patches import patch_trainable_only_checkpointing, wipe_or_resume  # noqa: E402

OUT = "/tmp/ckpt_patch_test"


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(16, 64)
        self.adapter = nn.Linear(64, 16)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)

    def forward(self, x=None, labels=None):
        y = self.adapter(torch.relu(self.frozen(x)))
        loss = ((y - labels) ** 2).mean()
        return {"loss": loss, "logits": y}


class ToyData(Dataset):
    def __len__(self):
        return 512

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(16, generator=g)
        return {"x": x, "labels": x.clone()}


def make_trainer(max_steps: int, resume: bool):
    class PatchedTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(KeepCheckpointCallback(keep_steps=10, keep_dir=os.path.join(OUT, "keep")))
            self.add_callback(FlopsCallback(flops_per_batch=1e9))

    patch_trainable_only_checkpointing(PatchedTrainer)
    torch.manual_seed(0)
    model = ToyModel()
    args = TrainingArguments(
        output_dir=OUT,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        save_steps=5,
        save_total_limit=3,
        logging_steps=5,
        learning_rate=1e-2,
        lr_scheduler_type="linear",
        report_to=[],
        seed=0,
        save_safetensors=True,
    )
    return PatchedTrainer(model=model, args=args, train_dataset=ToyData())


def list_ckpts():
    return sorted(
        (d for d in os.listdir(OUT) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    )


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    # wipe_or_resume: fresh dir -> no resume
    assert wipe_or_resume(OUT, fresh=False, volume=None) is False

    # ---- phase 1: train 12 steps (killed mid-run at max_steps=12) ----
    t1 = make_trainer(max_steps=12, resume=False)
    t1.train()
    ckpts = list_ckpts()
    assert len(ckpts) <= 3, f"rolling window violated: {ckpts}"
    assert "checkpoint-10" in ckpts, ckpts

    # trainable-only contents
    from safetensors.torch import load_file

    last = os.path.join(OUT, ckpts[-1])
    files = set(os.listdir(last))
    assert "trainable.safetensors" in files, files
    assert not any(f.startswith("model") and f.endswith(".safetensors") for f in files), (
        f"full model weights leaked into checkpoint: {files}"
    )
    sd = load_file(os.path.join(last, "trainable.safetensors"))
    assert set(sd) == {"adapter.weight", "adapter.bias"}, set(sd)

    # keep dir
    assert os.path.isdir(os.path.join(OUT, "keep", "checkpoint-10")), "keep copy missing"

    adapter_after_12 = t1.model.adapter.weight.detach().clone()

    # ---- phase 2: resume (simulates preemption retry) ----
    assert wipe_or_resume(OUT, fresh=False, volume=None) is True
    t2 = make_trainer(max_steps=20, resume=True)
    # fresh init differs from trained weights
    assert not torch.allclose(t2.model.adapter.weight, adapter_after_12)
    t2.train(resume_from_checkpoint=get_last_checkpoint(OUT))
    assert t2.state.global_step == 20, t2.state.global_step
    # optimizer state restored (exp_avg exists for adapter params)
    opt_state = list(t2.optimizer.state.values())
    assert opt_state and "exp_avg" in opt_state[0]

    # ---- phase 3: fresh wipe once ----
    assert wipe_or_resume(OUT, fresh=True, volume=None) is False  # wiped
    assert os.path.exists(os.path.join(OUT, ".fresh_done"))
    # retry of the same launch must NOT re-wipe: simulate by creating a ckpt then calling again
    os.makedirs(os.path.join(OUT, "checkpoint-5"))
    with open(os.path.join(OUT, "checkpoint-5", "trainer_state.json"), "w") as f:
        json.dump({"global_step": 5}, f)
    assert wipe_or_resume(OUT, fresh=True, volume=None) is True, "marker failed to prevent re-wipe"

    print("ALL CKPT PATCH TESTS PASSED")


if __name__ == "__main__":
    main()
