"""Prepare one LeRobot repo as GR00T-flavored v2 train/val roots.

Steps:
  1. Detect LeRobot version from hub meta/info.json.
  2. v3   -> Isaac-GR00T's scripts/lerobot_conversion/convert_v3_to_v2.py
     v2.x -> snapshot_download.
  3. Write meta/modality.json (SO-100/101 schema; per-repo camera mapping).
  4. Optionally rewrite tasks with MolmoAct2 per-episode annotated instructions.
  5. Episode-level split: val_ratio (min 1 episode) -> sibling <out_root>_val,
     both sides rebuilt as fully valid v2 roots (contiguous renumbering of
     parquet/video filenames, episode_index/index columns, episodes.jsonl,
     episodes_stats.jsonl, info.json totals).
  6. Validate: GR00T loader smoke on both roots.

Runs inside the Isaac-GR00T uv env on Modal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GR00T_DIR = os.environ.get("GR00T_DIR", "/root/Isaac-GR00T")
MANIFEST_REPO = "allenai/MolmoAct2-SO100_101-Dataset"

MODALITY_TEMPLATE = {
    "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "video": {},  # filled per-repo: {"front": {"original_key": ...}, "wrist": {...}}
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def log(msg: str):
    print(f"[prepare] {msg}", flush=True)


def fetch_info(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


def download_v2(repo_id: str, dest: Path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(dest))


def convert_v3(repo_id: str, work: Path) -> Path:
    """Run the official v3->v2 converter; returns the (nested) output root."""
    cmd = [
        "uv", "run", "--no-sync", "--project", "scripts/lerobot_conversion",
        "python", "scripts/lerobot_conversion/convert_v3_to_v2.py",
        "--repo-id", repo_id,
        "--root", str(work),
    ]
    log(f"running converter: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=GR00T_DIR, check=True)
    nested = work / repo_id  # converter nests output at <root>/<user>/<repo>
    if nested.exists():
        return nested
    # fall back: find the dir containing meta/info.json
    for p in work.rglob("meta/info.json"):
        return p.parent.parent
    raise FileNotFoundError(f"converted dataset not found under {work}")


def load_annotations(repo_id: str):
    """Return {episode_index: task_str} from the MolmoAct2 manifest, or None."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            MANIFEST_REPO,
            f"language_annotations/{repo_id}/tasks_annotated.parquet",
            repo_type="dataset",
        )
    except Exception:  # noqa: BLE001
        return None
    df = pd.read_parquet(path)
    if "episode_index" in df.columns:
        df = df.set_index("episode_index")
    return {int(i): str(t) for i, t in df["task"].items()}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def episode_paths(root: Path, info: dict, ep: int) -> tuple[Path, list[Path]]:
    chunk = ep // info.get("chunks_size", 1000)
    parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    videos = []
    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            videos.append(
                root
                / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep)
            )
    return parquet, videos


def rewrite_split(
    src_root: Path,
    dst_root: Path,
    episodes: list[int],
    info: dict,
    episodes_meta: list[dict],
    episodes_stats: dict[int, dict] | None,
    annotations: dict[int, str] | None,
    tasks_rows: list[dict],
):
    """Copy `episodes` (original indices) from src to dst with contiguous
    renumbering and fully rewritten metadata."""
    import pandas as pd

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "meta").mkdir(exist_ok=True)

    # --- task table: original tasks + (optionally) annotated per-episode tasks
    task_to_index: dict[str, int] = {}
    new_tasks_rows: list[dict] = []

    def task_index_for(task: str) -> int:
        if task not in task_to_index:
            task_to_index[task] = len(new_tasks_rows)
            new_tasks_rows.append({"task_index": task_to_index[task], "task": task})
        return task_to_index[task]

    orig_task_by_index = {r["task_index"]: r["task"] for r in tasks_rows}
    ep_meta_by_index = {m["episode_index"]: m for m in episodes_meta}

    new_episodes_meta = []
    new_episodes_stats = []
    global_index = 0
    chunks_size = info.get("chunks_size", 1000)
    total_frames = 0
    n_videos = 0

    for new_ep, old_ep in enumerate(episodes):
        old_parquet, old_videos = episode_paths(src_root, info, old_ep)
        chunk = new_ep // chunks_size
        new_parquet = dst_root / info["data_path"].format(
            episode_chunk=chunk, episode_index=new_ep
        )
        new_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(old_parquet)
        length = len(df)
        df["episode_index"] = new_ep
        if "index" in df.columns:
            df["index"] = range(global_index, global_index + length)

        # task rewrite
        meta = ep_meta_by_index[old_ep]
        if annotations is not None and old_ep in annotations:
            task_str = annotations[old_ep]
        else:
            # keep original task (first task of the episode)
            orig_ti = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
            task_str = orig_task_by_index.get(orig_ti, (meta.get("tasks") or ["unknown"])[0])
        ti = task_index_for(task_str)
        if "task_index" in df.columns:
            df["task_index"] = ti

        df.to_parquet(new_parquet)
        global_index += length
        total_frames += length

        for old_v in old_videos:
            # video key = parent dir name in v2 layout .../<video_key>/episode_XXXXXX.mp4
            vkey = old_v.parent.name
            new_v = dst_root / info["video_path"].format(
                episode_chunk=chunk, video_key=vkey, episode_index=new_ep
            )
            new_v.parent.mkdir(parents=True, exist_ok=True)
            if not old_v.exists():
                raise FileNotFoundError(f"missing video {old_v}")
            shutil.copyfile(old_v, new_v)
            n_videos += 1

        new_episodes_meta.append(
            {**meta, "episode_index": new_ep, "tasks": [task_str], "length": length}
        )
        if episodes_stats is not None and old_ep in episodes_stats:
            new_episodes_stats.append(
                {"episode_index": new_ep, "stats": episodes_stats[old_ep]}
            )

    # --- metadata
    new_info = dict(info)
    new_info["total_episodes"] = len(episodes)
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = n_videos
    new_info["total_chunks"] = (len(episodes) + chunks_size - 1) // chunks_size
    new_info["total_tasks"] = len(new_tasks_rows)
    new_info["splits"] = {"train": f"0:{len(episodes)}"}
    with open(dst_root / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=2)
    write_jsonl(dst_root / "meta/episodes.jsonl", new_episodes_meta)
    write_jsonl(dst_root / "meta/tasks.jsonl", new_tasks_rows)
    if new_episodes_stats:
        write_jsonl(dst_root / "meta/episodes_stats.jsonl", new_episodes_stats)

    # copy any remaining meta files verbatim (stats.json etc.), except ones we rewrote
    for p in (src_root / "meta").iterdir():
        if p.name not in {"info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl",
                          "modality.json"} and p.is_file():
            shutil.copyfile(p, dst_root / "meta" / p.name)


def write_modality(root: Path, camera_map: dict[str, str]):
    modality = json.loads(json.dumps(MODALITY_TEMPLATE))
    for new_key, original_key in camera_map.items():
        modality["video"][new_key] = {"original_key": original_key}
    with open(root / "meta/modality.json", "w") as f:
        json.dump(modality, f, indent=2)


def validate_root(root: Path, expect_min_episodes: int = 1):
    """GR00T loader smoke: stats generation + sharded dataset construction."""
    sys.path.insert(0, GR00T_DIR)
    import importlib

    sys.path.append(f"{GR00T_DIR}/examples/SO100")
    importlib.import_module("so100_config")  # registers new_embodiment modality configs

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    with open(root / "meta/info.json") as f:
        info = json.load(f)
    assert info["total_episodes"] >= expect_min_episodes
    generate_stats(str(root))
    generate_rel_stats(str(root), EmbodimentTag("new_embodiment"))

    cfg = get_default_config()
    ds = ShardedSingleStepDataset(
        dataset_path=str(root),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        modality_configs=cfg.data.modality_configs["new_embodiment"],
        shard_size=64,
        episode_sampling_rate=1.0,
        seed=17,
        allow_padding=cfg.data.allow_padding,
    )
    assert len(ds) > 0, f"dataset has no shards: {root}"
    log(f"validated {root}: {info['total_episodes']} eps, {info['total_frames']} frames, {len(ds)} shards")


def prepare_repo(
    repo_id: str,
    out_root: Path,
    camera_map: dict[str, str],
    val_ratio: float,
    seed: int = 17,
    use_annotations: bool = True,
):
    if (out_root / "meta/info.json").exists() and (
        Path(str(out_root) + "_val") / "meta/info.json"
    ).exists():
        log(f"{out_root} already prepared; skipping")
        return

    info = fetch_info(repo_id)
    version = str(info.get("codebase_version", "")).lstrip("v")
    log(f"{repo_id}: codebase v{version}, {info.get('total_episodes')} episodes")

    with tempfile.TemporaryDirectory(dir=out_root.parent if out_root.parent.exists() else None) as td:
        work = Path(td)
        if version.startswith("3"):
            src = convert_v3(repo_id, work / "conv")
            with open(src / "meta/info.json") as f:
                info = json.load(f)  # refresh: converter rewrites layout to v2
        elif version.startswith("2"):
            src = work / "dl"
            download_v2(repo_id, src)
        else:
            raise ValueError(f"unsupported codebase_version {version} for {repo_id}")

        # sanity: mapped cameras exist
        video_feats = {k for k, v in info["features"].items() if v.get("dtype") == "video"}
        for orig in camera_map.values():
            assert orig in video_feats, f"camera {orig} not in {sorted(video_feats)}"

        episodes_meta = read_jsonl(src / "meta/episodes.jsonl")
        tasks_rows = read_jsonl(src / "meta/tasks.jsonl")
        stats_path = src / "meta/episodes_stats.jsonl"
        episodes_stats = None
        if stats_path.exists():
            episodes_stats = {
                r["episode_index"]: r["stats"] for r in read_jsonl(stats_path)
            }

        annotations = load_annotations(repo_id) if use_annotations else None
        if annotations:
            log(f"loaded {len(annotations)} annotated instructions")

        all_eps = sorted(m["episode_index"] for m in episodes_meta)
        rng = random.Random(seed)
        n_val = max(1, int(round(len(all_eps) * val_ratio)))
        val_eps = sorted(rng.sample(all_eps, n_val))
        train_eps = [e for e in all_eps if e not in set(val_eps)]
        log(f"split: {len(train_eps)} train / {len(val_eps)} val episodes")

        val_root = Path(str(out_root) + "_val")
        for dst in (out_root, val_root):
            if dst.exists():
                shutil.rmtree(dst)
        rewrite_split(src, out_root, train_eps, info, episodes_meta, episodes_stats,
                      annotations, tasks_rows)
        rewrite_split(src, val_root, val_eps, info, episodes_meta, episodes_stats,
                      annotations, tasks_rows)

    for root in (out_root, Path(str(out_root) + "_val")):
        write_modality(root, camera_map)
        validate_root(root)
    log(f"DONE {repo_id} -> {out_root} (+_val)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id")
    p.add_argument("--out-root")
    p.add_argument("--camera-map", help='JSON like {"front": "observation.images.front", "wrist": "..."}')
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--no-annotations", action="store_true")
    p.add_argument("--spec-json", help="stage-2 mode: one entry of repos_filtered.json")
    p.add_argument("--out-base", default="/data/v2")
    args = p.parse_args()

    if args.spec_json:
        spec = json.loads(args.spec_json)
        repo_id = spec["repo_id"]
        slug = repo_id.replace("/", "__")
        prepare_repo(
            repo_id,
            Path(args.out_base) / slug,
            spec["camera_map"],
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_annotations=spec.get("has_annotations", True),
        )
    else:
        assert args.repo_id and args.out_root and args.camera_map
        prepare_repo(
            args.repo_id,
            Path(args.out_root),
            json.loads(args.camera_map),
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_annotations=not args.no_annotations,
        )


if __name__ == "__main__":
    main()
