#!/usr/bin/env python3
"""Stage4-a: Stage3出力（複数動画分）を1つのビン単位テーブルに結合し、
data/labels/<target>/bin_labels.csv があれば (video, bin_start_sec, bin_end_sec) で
左結合してlabel/split列を付与する（一致しない行はlabel/split空欄のまま残る＝
スコアリング/ランキングには使えるが閾値決定の学習/評価には使えない行）。

出力はCLIPスコア降順にソートする（仕様書§4「統合スコア上位から確認」に沿う。
Stage4のスコア統合モデルをまだ使っていない段階でも、そのままStage5に渡せるようにするため）。

--out-csv の隣に review_crops/ フォルダを作り、代表クロップ画像を順位・スコア付きの
ファイル名でコピーする（Finder/プレビュー等で名前順に並べれば上位から目視確認できる）。
"""
import argparse
import csv
import glob
import shutil
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
    p.add_argument("--skip-review-copy", action="store_true",
                    help="review_crops/へのクロップ画像コピーをスキップする（候補数が非常に多い場合）")
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

    rows.sort(key=lambda r: -(float(r["clip_score"]) if r.get("clip_score") not in (None, "") else -1))

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

    if not args.skip_review_copy and rows:
        review_dir = out_csv.parent / "review_crops"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows, 1):
            src = Path(r["crop_path"])
            if not src.exists():
                continue
            score = r.get("clip_score") or "NA"
            dst = review_dir / f"{i:04d}_score{score}_{src.name}"
            shutil.copyfile(src, dst)
        print(f"copied {len(rows)} crops -> {review_dir} (rank+score prefixed filenames)")

    write_run_manifest(out_csv.parent, cfg, REPO_ROOT, extra={"stage": "stage4_build_bin_table"})
    print(f"combined {len(files)} videos -> {len(rows)} bin rows ({n_labeled} labeled, "
          f"sorted by clip_score desc) -> {out_csv}")


if __name__ == "__main__":
    main()
