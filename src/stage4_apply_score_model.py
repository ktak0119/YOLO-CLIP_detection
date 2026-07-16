#!/usr/bin/env python3
"""Stage4-c: 固定済みモデル・閾値（stage4_fit_score_model.pyのmodels.json）を
新規（未知）データのビンテーブルに適用し、統合スコアでランク付け・閾値抽出する。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import should_skip, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-table", required=True)
    p.add_argument("--models-json", required=True, help="stage4_fit_score_model.py出力")
    p.add_argument("--method", required=True, choices=["yolo_only", "clip_only", "sum", "logistic_regression"])
    p.add_argument("--target", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def fuse(method, yolo_conf, clip_score, model):
    if method == "yolo_only":
        return yolo_conf
    if method == "clip_only":
        return clip_score
    if method == "sum":
        return yolo_conf + clip_score
    if method == "logistic_regression":
        z = model["intercept"] + model["coef_yolo"] * yolo_conf + model["coef_clip"] * clip_score
        return 1.0 / (1.0 + pow(2.718281828, -z))
    raise ValueError(method)


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_csv = Path(args.out_csv)

    if should_skip(out_csv, cfg.get("resume", True)):
        print(f"SKIP (resume): {out_csv} already exists")
        return

    with open(args.models_json) as f:
        models = json.load(f)
    if args.method not in models:
        print(f"ERROR: method {args.method} not in {args.models_json}", file=sys.stderr)
        sys.exit(1)
    model = models[args.method]
    threshold = model["threshold"]

    with open(args.bin_table, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        score = fuse(args.method, float(r["yolo_conf"]), float(r["clip_score"]), model)
        r["combined_score"] = round(score, 4)
        r["above_threshold"] = int(score >= threshold)

    rows.sort(key=lambda r: -r["combined_score"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_run_manifest(out_csv.parent, cfg, REPO_ROOT,
                        extra={"stage": "stage4_apply_score_model", "method": args.method, "threshold": threshold})
    n_above = sum(r["above_threshold"] for r in rows)
    print(f"applied {args.method} (threshold={threshold}) to {len(rows)} bins -> {n_above} above threshold -> {out_csv}")


if __name__ == "__main__":
    main()
