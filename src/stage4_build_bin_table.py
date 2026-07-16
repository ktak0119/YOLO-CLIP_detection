#!/usr/bin/env python3
"""Stage4-a: Stage3出力（複数動画分）を1つのビン単位テーブルに結合し、
data/labels/<target>/bin_labels.csv があれば (video, bin_start_sec, bin_end_sec) で
左結合してlabel/split列を付与する（一致しない行はlabel/split空欄のまま残る＝
スコアリング/ランキングには使えるが閾値決定の学習/評価には使えない行）。
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-glob", required=True, help="Stage3出力CSVのglobパターン（動画ごと1ファイル）")
    p.add_argument("--target", required=True)
    p.add_argument("--labels-csv", default=None,
                    help="data/labels/<target>/bin_labels.csv。未指定ならlabel/split空欄で出力")
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.target)

    files = sorted(glob.glob(args.stage3_glob))
    if not files:
        print(f"ERROR: no files matched {args.stage3_glob}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        with open(fp, newline="") as f:
            rows.extend(csv.DictReader(f))

    labels = {}
    if args.labels_csv and Path(args.labels_csv).exists():
        with open(args.labels_csv, newline="") as f:
            for lr in csv.DictReader(f):
                key = (lr["video"], lr["bin_start_sec"], lr["bin_end_sec"])
                labels[key] = (lr["label"], lr["split"])

    n_labeled = 0
    for r in rows:
        key = (r["video"], r["start_sec"], r["end_sec"])
        label, split = labels.get(key, ("", ""))
        r["label"] = label
        r["split"] = split
        if label != "":
            n_labeled += 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "video", "bin_id", "start_sec", "end_sec", "frame_idx", "t_sec",
        "yolo_conf", "crop_path", "clip_score", "label", "split",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_run_manifest(out_csv.parent, cfg, REPO_ROOT, extra={"stage": "stage4_build_bin_table"})
    print(f"combined {len(files)} videos -> {len(rows)} bin rows ({n_labeled} labeled) -> {out_csv}")


if __name__ == "__main__":
    main()
