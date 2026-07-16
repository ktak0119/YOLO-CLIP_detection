#!/usr/bin/env python3
"""1動画に対しStage1〜4を通しで実行するオーケストレーター（骨組み構築フェーズ版）。

Stage5は人間目視のTkinter GUI、Stage6はStage5の結果が要るため、このスクリプトは
Stage4（--models-json指定時）までで止まる。Stage5/6は個別に手動で実行する:
    python src/stage5_screening.py --bin-table <out_dir>/stage4_applied.csv --output <out_dir>/screening_result.csv
    python src/stage6_merge_bins.py --screening-csv <out_dir>/screening_result.csv --videos-dir <dir> --target <target> --out-json <out_dir>/merged.json
    python src/stage6_make_clips.py --manifest <out_dir>/merged.json --target <target> --out-dir <out_dir>/clips

動画単位の並列実行（xargs -P等）は将来の性能最適化課題であり、このスクリプトは
1動画=1プロセスの直列実行のみを提供する。
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models-json", default=None,
                     help="stage4_fit_score_model.py出力。指定時のみStage4 applyまで実行する")
    ap.add_argument("--method", default="logistic_regression",
                     choices=["yolo_only", "clip_only", "sum", "logistic_regression"])
    ap.add_argument("--limit-seconds", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = ["--limit-seconds", str(args.limit_seconds)] if args.limit_seconds else []

    stage1_json = out_dir / "stage1.json"
    run([sys.executable, SRC_DIR / "stage1_mog2_flag.py", args.video,
         "--target", args.target, "--out-json", stage1_json] + limit)

    stage2_csv = out_dir / "stage2_bins.csv"
    run([sys.executable, SRC_DIR / "stage2_yolo_bin.py", args.video,
         "--target", args.target, "--stage1-json", stage1_json,
         "--out-csv", stage2_csv, "--out-crops-dir", out_dir / "crops"] + limit)

    stage3_csv = out_dir / "stage3_scored.csv"
    run([sys.executable, SRC_DIR / "stage3_clip_score.py",
         "--in-csv", stage2_csv, "--target", args.target, "--out-csv", stage3_csv])

    if args.models_json:
        stage4_csv = out_dir / "stage4_applied.csv"
        run([sys.executable, SRC_DIR / "stage4_apply_score_model.py",
             "--bin-table", stage3_csv, "--models-json", args.models_json,
             "--method", args.method, "--target", args.target, "--out-csv", stage4_csv])
        print(f"\nStage1-4完了。次はStage5(人間目視)を手動で実行してください:\n"
              f"  python {SRC_DIR}/stage5_screening.py --bin-table {stage4_csv} "
              f"--output {out_dir}/screening_result.csv")
    else:
        print(f"\nStage1-3完了（--models-json未指定のためStage4はスキップ）。"
              f"Stage4は stage4_fit_score_model.py で先にモデルを用意してから実行してください。")


if __name__ == "__main__":
    main()
