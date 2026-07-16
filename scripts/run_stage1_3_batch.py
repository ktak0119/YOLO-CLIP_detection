#!/usr/bin/env python3
"""指定ディレクトリ内の全mp4に対しStage1〜3を順に実行するバッチスクリプト。

各ステージスクリプトは出力が既に存在すればスキップする(resume)ため、
このバッチスクリプトを中断して再実行しても安全（未完了分だけ処理される）。
1動画の失敗が他の動画の処理を止めないよう、動画ごとにtry/exceptで独立させている。

動画単位の並列実行（xargs -P等）は行わない（1プロセスの逐次実行のみ）。
大量・長時間の動画セットを並列化したい場合は別途相談。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed (exit {r.returncode}): {cmd}")


def process_one(video: Path, target: str, out_dir: Path):
    stage1_json = out_dir / "stage1.json"
    run([sys.executable, SRC_DIR / "stage1_mog2_flag.py", str(video),
         "--target", target, "--out-json", stage1_json])

    stage2_csv = out_dir / "stage2_bins.csv"
    run([sys.executable, SRC_DIR / "stage2_yolo_bin.py", str(video),
         "--target", target, "--stage1-json", stage1_json,
         "--out-csv", stage2_csv, "--out-crops-dir", out_dir / "crops"])

    stage3_csv = out_dir / "stage3_scored.csv"
    run([sys.executable, SRC_DIR / "stage3_clip_score.py",
         "--in-csv", stage2_csv, "--target", target, "--out-csv", stage3_csv])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True, help="mp4が入っているディレクトリ（非再帰）")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True, help="動画ごとに <out-dir>/<動画名>/ を作る")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        print(f"ERROR: no mp4 files found in {videos_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(videos)} videos found in {videos_dir}\n")

    ok, failed = [], []
    t0 = time.time()
    for i, video in enumerate(videos, 1):
        video_out = out_dir / video.stem
        video_out.mkdir(parents=True, exist_ok=True)
        print(f"=== [{i}/{len(videos)}] {video.name} ===", flush=True)
        t_video0 = time.time()
        try:
            process_one(video, args.target, video_out)
            ok.append(video.name)
            print(f"--- done in {time.time()-t_video0:.1f}s ---\n", flush=True)
        except Exception as e:
            failed.append((video.name, str(e)))
            print(f"--- FAILED: {e} ---\n", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"\n=== batch done: {len(ok)} ok, {len(failed)} failed, {elapsed/60:.1f} min total ===")
    if failed:
        print("failed videos:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
