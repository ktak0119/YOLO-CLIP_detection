#!/usr/bin/env python3
"""1動画に対しStage1〜6を通しで実行するオーケストレーター（骨組み構築フェーズ版）。

既定ではStage4（--models-json指定時）までで止まり、Stage5(人間目視)は個別に手動で
実行する想定:
    python src/stage5_screening.py --bin-table <out_dir>/stage4_applied.csv --output <out_dir>/screening_result.csv
    python src/stage6_merge_bins.py --screening-csv <out_dir>/screening_result.csv --videos-dir <dir> --target <target> --out-json <out_dir>/merged.json
    python src/stage6_make_clips.py --manifest <out_dir>/merged.json --target <target> --out-dir <out_dir>/clips

--auto-screening を付けると、Stage5の人間目視を stage5_auto_screening.py
（Stage4のabove_thresholdをそのままTP/FPとして使う非対話版、入出力の形はscreening.pyと同一）
に差し替え、Stage6まで完全自動で実行する。精度を担保する仕組みではない点に注意
（疎通確認・大量一次スクリーニング用途。本番の目視は別途screening.pyで行う）。

動画単位の並列実行（xargs -P等）は将来の性能最適化課題であり、このスクリプトは
1動画=1プロセスの直列実行のみを提供する。
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
from common.config import default_out_dir, load_config


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", default=None,
                     help="省略時は configs/targets/<target>.yaml の out_dir 配下に "
                          "<out_dir>/<動画名>/ を作る")
    ap.add_argument("--models-json", default=None,
                     help="stage4_fit_score_model.py出力。指定時のみStage4 applyまで実行する")
    ap.add_argument("--method", default="logistic_regression",
                     choices=["yolo_only", "clip_only", "sum", "logistic_regression"])
    ap.add_argument("--videos-dir", default=None,
                     help="--auto-screening使用時、Stage6のffmpeg入力を解決するために必要。"
                          "省略時はconfigのvideos_dirを使う")
    ap.add_argument("--auto-screening", action="store_true",
                     help="Stage5(人間目視)をstage5_auto_screening.pyに差し替え、Stage6まで自動実行する")
    ap.add_argument("--limit-seconds", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.target)
    videos_dir = args.videos_dir or cfg.get("videos_dir")

    if args.auto_screening and (not args.models_json or not videos_dir):
        print("ERROR: --auto-screening には models-json と videos_dir(CLIまたはconfig)の両方が必要",
              file=sys.stderr)
        sys.exit(1)

    base_out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.get("out_dir") or default_out_dir(args.target))
    out_dir = base_out_dir if args.out_dir else base_out_dir / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out_dir: {out_dir}")
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

    if not args.models_json:
        print(f"\nStage1-3完了（--models-json未指定のためStage4はスキップ）。"
              f"Stage4は stage4_fit_score_model.py で先にモデルを用意してから実行してください。")
        return

    stage4_csv = out_dir / "stage4_applied.csv"
    run([sys.executable, SRC_DIR / "stage4_apply_score_model.py",
         "--bin-table", stage3_csv, "--models-json", args.models_json,
         "--method", args.method, "--target", args.target, "--out-csv", stage4_csv])

    if not args.auto_screening:
        print(f"\nStage1-4完了。次はStage5(人間目視)を手動で実行してください:\n"
              f"  python {SRC_DIR}/stage5_screening.py --bin-table {stage4_csv} "
              f"--output {out_dir}/screening_result.csv")
        return

    screening_csv = out_dir / "screening_result.csv"
    run([sys.executable, SRC_DIR / "stage5_auto_screening.py",
         "--bin-table", stage4_csv, "--output", screening_csv])

    merged_json = out_dir / "merged.json"
    run([sys.executable, SRC_DIR / "stage6_merge_bins.py",
         "--screening-csv", screening_csv, "--videos-dir", videos_dir,
         "--target", args.target, "--out-json", merged_json])

    run([sys.executable, SRC_DIR / "stage6_make_clips.py",
         "--manifest", merged_json, "--target", args.target, "--out-dir", out_dir / "clips"])

    print(f"\nStage1-6完了（Stage5は自動判定・人間目視なし）。クリップ -> {out_dir}/clips")


if __name__ == "__main__":
    main()
