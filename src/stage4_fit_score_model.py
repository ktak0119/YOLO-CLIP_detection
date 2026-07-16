#!/usr/bin/env python3
"""Stage4-b: YOLOスコア＋CLIPスコアの統合方式を比較し、閾値を決定する（仕様書§6-7）。

比較する4方式: yolo_only / clip_only / sum / logistic_regression。
分割はビン単位ではなく、bin_labels.csvの`split`列（threshold_decision/validation）を
そのまま使う——分割そのもの（GroupKFold等）はこのスクリプトでは行わない。ユーザーが
動画・撮影日・地点単位であらかじめ分割した上でCSVに書き込んでおく前提（骨組み構築フェーズの
方針、plan参照）。

閾値決定手順（仕様書§7）: threshold_decision split を統合スコア降順に並べ、
Recallが目標値に初めて到達する閾値（＝その時点のスコア値）を採用する。
決定したモデル・閾値をvalidation splitに固定適用し、Recall/Precisionを確認する。

4方式それぞれの結果を比較レポートとして出力する（自動選択はしない。仕様書は
「比較する」と規定しているだけで、どれを採用するかは人間が判断する）。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin-table", required=True, help="stage4_build_bin_table.py出力（またはダミーデータ）")
    p.add_argument("--target", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def load_labeled_rows(bin_table_path):
    with open(bin_table_path, newline="") as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r.get("label", "") != "" and r.get("split", "") != ""]
    return labeled


def select_threshold(scored, target_recall):
    """scored: list of (score, label). 仕様書§7のRecallベース閾値決定。"""
    paired = sorted(scored, key=lambda x: -x[0])
    n_pos = sum(1 for _, label in paired if label == 1)
    if n_pos == 0:
        return None, 0.0
    n_hit = 0
    for score, label in paired:
        if label == 1:
            n_hit += 1
        recall = n_hit / n_pos
        if recall >= target_recall:
            return score, recall
    return paired[-1][0], n_hit / n_pos


def precision_recall_at(scored, threshold):
    n_pos = sum(1 for _, label in scored if label == 1)
    accepted = [(s, l) for s, l in scored if s >= threshold]
    tp = sum(1 for _, l in accepted if l == 1)
    recall = tp / n_pos if n_pos else 0.0
    precision = tp / len(accepted) if accepted else 0.0
    return {"recall": round(recall, 4), "precision": round(precision, 4),
            "n_accepted": len(accepted), "n_positives_total": n_pos, "n_true_positive": tp}


def fuse(method, yolo_conf, clip_score, logreg=None):
    if method == "yolo_only":
        return yolo_conf
    if method == "clip_only":
        return clip_score
    if method == "sum":
        return yolo_conf + clip_score
    if method == "logistic_regression":
        return logreg.predict_proba([[yolo_conf, clip_score]])[0][1]
    raise ValueError(method)


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_labeled_rows(args.bin_table)
    if not rows:
        print("ERROR: bin-table に label/split が付与された行が1件もない", file=sys.stderr)
        sys.exit(1)

    decision_rows = [r for r in rows if r["split"] == "threshold_decision"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if not decision_rows or not val_rows:
        print(f"ERROR: threshold_decision={len(decision_rows)}件, validation={len(val_rows)}件 "
              "-- 両方に最低1件ずつ必要", file=sys.stderr)
        sys.exit(1)

    def xy(rs):
        X = [[float(r["yolo_conf"]), float(r["clip_score"])] for r in rs]
        y = [int(float(r["label"])) for r in rs]
        return X, y

    X_dec, y_dec = xy(decision_rows)
    X_val, y_val = xy(val_rows)

    logreg = None
    if "logistic_regression" in cfg["score_fusion_methods"]:
        logreg = LogisticRegression()
        logreg.fit(X_dec, y_dec)

    report = {"target": args.target, "target_recall": cfg["target_recall"],
              "n_threshold_decision": len(decision_rows), "n_validation": len(val_rows),
              "methods": {}}
    models = {}

    for method in cfg["score_fusion_methods"]:
        dec_scored = [(fuse(method, x[0], x[1], logreg), label) for x, label in zip(X_dec, y_dec)]
        val_scored = [(fuse(method, x[0], x[1], logreg), label) for x, label in zip(X_val, y_val)]

        threshold, dec_recall = select_threshold(dec_scored, cfg["target_recall"])
        if threshold is None:
            report["methods"][method] = {"error": "threshold_decision split has no positives"}
            continue
        val_metrics = precision_recall_at(val_scored, threshold)

        report["methods"][method] = {
            "threshold": round(threshold, 4),
            "threshold_decision_recall_at_threshold": round(dec_recall, 4),
            "validation": val_metrics,
        }
        model = {"method": method, "threshold": round(threshold, 4)}
        if method == "logistic_regression":
            model["coef_yolo"] = round(float(logreg.coef_[0][0]), 6)
            model["coef_clip"] = round(float(logreg.coef_[0][1]), 6)
            model["intercept"] = round(float(logreg.intercept_[0]), 6)
        models[method] = model

    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "models.json", "w") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)

    write_run_manifest(out_dir, cfg, REPO_ROOT, extra={"stage": "stage4_fit_score_model"})

    print(f"compared {len(cfg['score_fusion_methods'])} methods "
          f"(decision={len(decision_rows)}, validation={len(val_rows)})")
    for method, m in report["methods"].items():
        if "error" in m:
            print(f"  {method}: {m['error']}")
        else:
            v = m["validation"]
            print(f"  {method}: threshold={m['threshold']} "
                  f"validation recall={v['recall']} precision={v['precision']}")
    print(f"-> {out_dir}/comparison_report.json, {out_dir}/models.json")


if __name__ == "__main__":
    main()
