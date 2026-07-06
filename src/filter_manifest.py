"""Filter the MolmoAct2 manifest (1220 repos) down to trainable SO-101 repos.

For each repo: fetch meta/info.json (any failure -> drop, logged). Filters:
  - robot_type contains "so101"/"so-101"
  - identifiable wrist camera + a distinct front camera (synonym heuristics)
  - total_episodes >= 20
  - fps in [10, 60]
  - state AND action shape [6]

Output: JSON list of {repo_id, version, episodes, frames, fps, camera_map,
has_annotations, robot_type}.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import urllib.request

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "repo_list.json")

WRIST_SYNONYMS = ("wrist", "gripper", "hand", "robocam", "cam_wrist", "wristcam", "claw")
FRONT_PREFERRED = (
    "front", "top", "laptop", "base", "phone", "cam_high", "overhead", "main",
    "webcam", "side", "external", "table", "scene", "context", "camera", "cam",
)


def fetch_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "gr00t-so101-prep"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def pick_cameras(video_keys: list[str]) -> dict[str, str] | None:
    """Return {"front": orig_key, "wrist": orig_key} or None if ambiguous."""
    def short(k: str) -> str:
        return k.removeprefix("observation.images.").lower()

    wrist = [k for k in video_keys if any(s in short(k) for s in WRIST_SYNONYMS)]
    if len(wrist) != 1:
        return None
    rest = [k for k in video_keys if k != wrist[0]]
    if not rest:
        return None
    if len(rest) == 1:
        return {"front": rest[0], "wrist": wrist[0]}
    for pref in FRONT_PREFERRED:
        for k in rest:
            if pref in short(k):
                return {"front": k, "wrist": wrist[0]}
    return None


def check_repo(entry: str, annotated: set[str]) -> tuple[str, dict | None, str]:
    repo_id = entry.split(":", 1)[1] if ":" in entry else entry
    try:
        info = fetch_json(f"https://huggingface.co/datasets/{repo_id}/raw/main/meta/info.json")
    except Exception as e:  # noqa: BLE001
        return repo_id, None, f"info.json fetch failed: {type(e).__name__}"

    robot = str(info.get("robot_type", "")).lower()
    if "so101" not in robot.replace("-", "").replace("_", ""):
        return repo_id, None, f"robot_type={robot or 'missing'}"

    eps = info.get("total_episodes", 0)
    if eps < 20:
        return repo_id, None, f"episodes={eps}<20"

    fps = info.get("fps", 0)
    if not 10 <= fps <= 60:
        return repo_id, None, f"fps={fps}"

    feats = info.get("features", {})
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    cam = pick_cameras(video_keys)
    if cam is None:
        return repo_id, None, f"cameras ambiguous: {video_keys}"

    for key in ("observation.state", "action"):
        shape = feats.get(key, {}).get("shape")
        if shape != [6]:
            return repo_id, None, f"{key} shape={shape}"

    version = str(info.get("codebase_version", "")).lstrip("v")
    if not (version.startswith("2") or version.startswith("3")):
        return repo_id, None, f"codebase_version={version}"

    return repo_id, {
        "repo_id": repo_id,
        "version": version,
        "episodes": eps,
        "frames": info.get("total_frames"),
        "fps": fps,
        "robot_type": info.get("robot_type"),
        "camera_map": cam,
        "has_annotations": repo_id in annotated,
    }, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/repos_filtered.json")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    with open(args.manifest) as f:
        entries = json.load(f)

    # which repos have language annotations (all of them per manifest, but be exact)
    try:
        tree = fetch_json(
            "https://huggingface.co/api/datasets/allenai/MolmoAct2-SO100_101-Dataset/tree/main/language_annotations?recursive=true"
        )
        annotated = {
            "/".join(item["path"].split("/")[1:3])
            for item in tree
            if item["path"].endswith("tasks_annotated.parquet")
        }
    except Exception as e:  # noqa: BLE001
        print(f"annotation tree fetch failed ({e}); assuming all annotated")
        annotated = {e.split(":", 1)[1] for e in entries}

    kept, dropped = [], {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for repo_id, spec, reason in ex.map(lambda e: check_repo(e, annotated), entries):
            if spec:
                kept.append(spec)
            else:
                dropped[repo_id] = reason

    kept.sort(key=lambda s: -s["episodes"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(kept, f, indent=2)
    drop_path = args.out.replace(".json", "_dropped.json")
    with open(drop_path, "w") as f:
        json.dump(dropped, f, indent=2)

    from collections import Counter

    reasons = Counter(v.split(":")[0] for v in dropped.values())
    total_eps = sum(s["episodes"] for s in kept)
    total_frames = sum(s["frames"] or 0 for s in kept)
    print(f"kept {len(kept)}/{len(entries)} repos, {total_eps} episodes, {total_frames} frames")
    print(f"drop reasons: {dict(reasons)}")
    print(f"wrote {args.out} and {drop_path}")


if __name__ == "__main__":
    main()
