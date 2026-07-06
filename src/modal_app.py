"""Modal app for fine-tuning GR00T N1.7 on SO-101 data (MolmoAct2 manifest).

Usage:
  modal run src/modal_app.py::env_check                 # H100: env + DCGM/GPM + model download smoke
  modal run src/modal_app.py::prepare_megamix           # CPU: stage-1 dataset -> /data/v2/megamix{,_val}
  modal run src/modal_app.py::smoke_train --steps 60    # H100: pipeline smoke (loss, eval, ckpt, metrics)
  modal run src/modal_app.py::sweep_one --lr 1e-4 --bs 32 --steps 250
  modal run --detach src/modal_app.py::train --exp-name main-01 --lr 1e-4 --bs 32 --max-steps 20000
  modal run src/modal_app.py::filter_manifest           # CPU: manifest -> /data/repos_filtered.json
  modal run src/modal_app.py::prepare_subset            # CPU fanout: filtered repos -> /data/v2/*
  modal run src/modal_app.py::publish --exp-name main-01
"""

import os
import subprocess

import modal

GR00T_COMMIT = "ab88b50c718f6528e1df9dcbaf75865d1b604760"  # Isaac-GR00T main 2026-06-30 (N1.7)
CATEGORY_LORA_PIN = (
    "category-lora @ git+https://github.com/jeqcho/category-lora"
    "@0a02f3984836e77555a183c140ca51b5aac3376e"
)
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
GR00T_DIR = "/root/Isaac-GR00T"

app = modal.App("gr00t-n17-so101")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0", "wget", "gnupg", "curl",
        "clang", "build-essential",  # evdev (lerobot->pynput dep) builds from source
    )
    # DCGM 4 for PROF_PIPE_TENSOR_ACTIVE (1004) / PROF_DRAM_ACTIVE (1005)
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        " && dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update"
        " && apt-get install -y datacenter-gpu-manager-4-cuda12"
    )
    .pip_install("uv")
    .run_commands(
        f"git clone https://github.com/NVIDIA/Isaac-GR00T.git {GR00T_DIR}"
        f" && cd {GR00T_DIR} && git checkout {GR00T_COMMIT}"
        " && uv sync --python 3.10"
    )
    # separate uv project for LeRobot v3 -> v2 conversion
    .run_commands(
        f"cd {GR00T_DIR} && GIT_LFS_SKIP_SMUDGE=1 uv sync --project scripts/lerobot_conversion"
    )
    .run_commands(f"cd {GR00T_DIR} && uv pip install '{CATEGORY_LORA_PIN}' nvidia-ml-py")
    .env(
        {
            "HF_HOME": "/data/hf",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONPATH": "/root/proj/src",
            "WANDB_RESUME": "allow",
            "GROOT_COMMIT_HASH": GR00T_COMMIT,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # our code last so edits don't invalidate heavy layers
    .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__))), "/root/proj/src")
)

data_vol = modal.Volume.from_name("gr00t-so101-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gr00t-so101-ckpt", create_if_missing=True)
secrets = [
    modal.Secret.from_name("huggingface-token"),
    modal.Secret.from_name("wandb"),
]

VOLS = {"/data": data_vol, "/ckpt": ckpt_vol}


def _run(*cmd: str, cwd: str = GR00T_DIR, env: dict | None = None) -> None:
    """Run a command inside the gr00t uv env, streaming output."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    # HF token secret may expose HF_TOKEN under a different name; normalize.
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if k in full_env:
            full_env.setdefault("HF_TOKEN", full_env[k])
    subprocess.run(["uv", "run", "--no-sync", "python", *cmd], cwd=cwd, env=full_env, check=True)


@app.function(image=image, gpu="H100", volumes=VOLS, secrets=secrets, timeout=3600)
def env_check():
    """Verify the whole environment on the actual training GPU."""
    _run("/root/proj/src/env_check.py")
    data_vol.commit()


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=8 * 3600, cpu=8, memory=32768)
def prepare_megamix(val_ratio: float = 0.05):
    _run(
        "/root/proj/src/prepare_data.py",
        "--repo-id", "whosricky/so101-megamix-v1",
        "--out-root", "/data/v2/megamix",
        "--camera-map", '{"front": "observation.images.front", "wrist": "observation.images.gripper"}',
        "--val-ratio", str(val_ratio),
    )
    data_vol.commit()


# Training needs zero HF traffic (model + data cached on the volume). Offline
# mode makes runs immune to the org-wide HF API rate limit (1000 req/5min),
# which concurrent sweeps + data prep otherwise exhaust.
TRAIN_ENV = {"HF_HUB_OFFLINE": "1", "WANDB_PROJECT": "gr00t-n17-so101"}


def _train_cmd(
    exp_name: str,
    dataset_roots: str,
    val_roots: str,
    lr: float,
    bs: int,
    max_steps: int,
    keep_steps: int,
    eval_steps: int,
    fresh: bool,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "/root/proj/src/train_launcher.py",
        "--exp-name", exp_name,
        "--dataset-roots", dataset_roots,
        "--val-roots", val_roots,
        "--lr", str(lr),
        "--global-batch-size", str(bs),
        "--max-steps", str(max_steps),
        "--save-steps", "5",
        "--save-total-limit", "3",
        "--keep-steps", str(keep_steps),
        "--eval-steps", str(eval_steps),
        "--output-dir", "/ckpt/runs",
    ]
    if fresh:
        cmd.append("--fresh")
    if extra:
        cmd.extend(extra)
    return cmd


@app.function(image=image, gpu="H100", volumes=VOLS, secrets=secrets, timeout=4 * 3600)
def smoke_train(steps: int = 60, fresh: bool = True):
    try:
        _run(
            *_train_cmd(
                "smoke",
                "/data/v2/megamix",
                "/data/v2/megamix_val",
                1e-4,
                32,
                steps,
                keep_steps=25,
                eval_steps=25,
                fresh=fresh,
            ),
            env={**TRAIN_ENV, "WANDB_RUN_ID": "smoke"},
        )
    finally:
        ckpt_vol.commit()


@app.function(image=image, gpu="H100", volumes=VOLS, secrets=secrets, timeout=6 * 3600)
def sweep_one(lr: float, bs: int, steps: int = 250):
    exp = f"sweep-lr{lr:g}-bs{bs}"
    try:
        _run(
            *_train_cmd(
                exp,
                "/data/v2/megamix",
                "/data/v2/megamix_val",
                lr,
                bs,
                steps,
                keep_steps=250,
                eval_steps=50,
                fresh=True,
            ),
            env={**TRAIN_ENV, "WANDB_RUN_ID": exp},
        )
    finally:
        ckpt_vol.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes=VOLS,
    secrets=secrets,
    timeout=23 * 3600,
    retries=modal.Retries(max_retries=10, initial_delay=60.0, backoff_coefficient=1.0),  # Modal max
    # Resume is idempotent; if 10 retries ever exhaust, the local monitor relaunches.
)
def train(
    exp_name: str,
    dataset_roots: str,
    val_roots: str,
    lr: float = 1e-4,
    bs: int = 32,
    max_steps: int = 20000,
    keep_steps: int = 500,
    eval_steps: int = 250,
    fresh: bool = False,
):
    """Main training. Preemption-safe: auto-resumes from the latest rolling checkpoint.

    `fresh` wipes the exp dir exactly once (guarded by a marker file inside the
    launcher) so retries of the same invocation never re-wipe.
    """
    try:
        _run(
            *_train_cmd(
                exp_name, dataset_roots, val_roots, lr, bs, max_steps,
                keep_steps=keep_steps, eval_steps=eval_steps, fresh=fresh,
            ),
            env={**TRAIN_ENV, "WANDB_RUN_ID": exp_name},
        )
    finally:
        ckpt_vol.commit()


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=2 * 3600, cpu=4, memory=16384)
def filter_manifest():
    _run("/root/proj/src/filter_manifest.py", "--out", "/data/repos_filtered.json")
    data_vol.commit()


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=12 * 3600, cpu=8, memory=32768, max_containers=3)
def prepare_one_repo(spec_json: str):
    """Prepare a single filtered repo. spec_json: one entry of repos_filtered.json."""
    _run("/root/proj/src/prepare_data.py", "--spec-json", spec_json, "--out-base", "/data/v2")
    data_vol.commit()


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=24 * 3600, cpu=2, memory=8192)
def prepare_subset(limit: int = 0):
    """Read /data/repos_filtered.json and fan out prepare_one_repo.map over entries."""
    import json

    with open("/data/repos_filtered.json") as f:
        entries = json.load(f)
    if limit:
        entries = entries[:limit]
    specs = [json.dumps(e) for e in entries]
    results = list(prepare_one_repo.map(specs, return_exceptions=True))
    failures = [
        (e["repo_id"], repr(r)) for e, r in zip(entries, results) if isinstance(r, Exception)
    ]
    print(f"prepared {len(entries) - len(failures)}/{len(entries)}; failures: {failures}")
    with open("/data/prepare_failures.json", "w") as f:
        json.dump(failures, f, indent=2)
    data_vol.commit()


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=4 * 3600, cpu=8, memory=32768)
def publish(exp_name: str, step: int = 0, repo_id: str = ""):
    _run(
        "/root/proj/src/publish.py",
        "--exp-dir", f"/ckpt/runs/{exp_name}",
        "--step", str(step),
        "--repo-id", repo_id,
    )


@app.function(image=image, volumes=VOLS, secrets=secrets, timeout=3600, cpu=4, memory=16384)
def debug(code: str):
    """Run arbitrary python inside the gr00t env (ops escape hatch)."""
    _run("-c", code)
