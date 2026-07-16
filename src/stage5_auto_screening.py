#!/usr/bin/env python3
"""Stage5の非対話版（スキップ用）。

stage5_screening.py（人間目視Tkinter GUI）と入出力の形を完全に同じにし
（video, bin_id, start_sec, end_sec, crop_path, screening）、Stage6以降が
どちらの経由か区別せず使えるようにする。人間の判定の代わりに、Stage4出力の
`above_threshold`列（既定）や任意のスコア列＋閾値で機械的にTP/FPを決める。

**運用上の位置づけ**: これは精度を担保する仕組みではない（Stage4の閾値をそのまま
素通しするだけ）。プロトタイピング・パイプライン疎通確認・大量データでの一次スクリーニング
（後で人間が`screening.py`で見直す前提）など、人間目視を意図的に省略してよい場面で使う。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage5_screening import OUTPUT_FIELDS, save_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-table", required=True,
                    help="video,bin_id,start_sec,end_sec,crop_pathを含むCSV（Stage4出力等）")
    p.add_argument("--output", required=True)
    p.add_argument("--score-col", default="above_threshold",
                    help="TP/FP判定に使う列名。既定はStage4 apply_score_model.py出力の"
                         "above_threshold(0/1)。他のスコア列を使う場合は--thresholdと併用する")
    p.add_argument("--threshold", type=float, default=None,
                    help="指定時: --score-colを数値スコアとして扱いthreshold以上をTPとする。"
                         "未指定時: --score-colを0/1(またはtrue/false)の真偽値として扱う")
    return p.parse_args()


def is_positive(value, threshold):
    if threshold is not None:
        return float(value) >= threshold
    return value in ("1", "true", "True", "TP", "TRUE")


def main():
    args = parse_args()

    with open(args.bin_table, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if rows and args.score_col not in rows[0]:
        print(f"ERROR: --score-col '{args.score_col}' not found in {args.bin_table} "
              f"(columns: {list(rows[0].keys())})", file=sys.stderr)
        sys.exit(1)

    results = []
    for r in rows:
        screening = "TP" if is_positive(r[args.score_col], args.threshold) else "FP"
        results.append({
            "video": r["video"],
            "bin_id": r["bin_id"],
            "start_sec": r["start_sec"],
            "end_sec": r["end_sec"],
            "crop_path": r["crop_path"],
            "screening": screening,
        })

    save_results(Path(args.output), results)
    n_tp = sum(1 for r in results if r["screening"] == "TP")
    print(f"auto-screened {len(results)} bins ({args.score_col}"
          + (f">={args.threshold}" if args.threshold is not None else " truthy")
          + f") -> {n_tp} TP, {len(results) - n_tp} FP -> {args.output}")


if __name__ == "__main__":
    main()
