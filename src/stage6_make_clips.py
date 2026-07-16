#!/usr/bin/env python3
"""Stage6-b: Stage6-a(merge_bins)が出したマージ済みマニフェストからffmpegでクリップを切り出す。
make_clips.pyを無改修相当で移植。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="stage6_merge_bins.py出力")
    p.add_argument("--target", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        events = json.load(f)

    pad_sec = cfg["clip_pad_sec"]
    n_ok, n_fail = 0, 0
    for e in events:
        video = e["video_path"]
        start = max(0, e["start_sec"] - pad_sec)
        dur = (e["end_sec"] - e["start_sec"]) + 2 * pad_sec
        out_name = f"{e['event_id']}.mp4"
        out_path = out_dir / out_name
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", video,
            "-t", f"{dur:.2f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(out_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            n_fail += 1
            print(f"FAILED {out_name}: {r.stderr[-300:]}")
        else:
            n_ok += 1
            print(f"OK {out_name} ({dur:.1f}s)")

    write_run_manifest(out_dir, cfg, REPO_ROOT, extra={"stage": "stage6_make_clips"})
    print(f"clips: {n_ok} ok, {n_fail} failed -> {out_dir}")


if __name__ == "__main__":
    main()
