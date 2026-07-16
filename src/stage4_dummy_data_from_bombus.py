#!/usr/bin/env python3
"""Stage4動作確認用ダミーデータアダプタ（骨組み構築フェーズのみ・本番運用では使わない）。

test_output/bombus_yolo_comparison/ の既存データ（crops_index.csv=YOLO confidence,
clip_scores.csv/master.csv=CLIPスコア, master.csv=gt_positive）を、
「バウトをビンと読み替えて」そのままstage4_fit_score_model.pyが読めるbin-table CSVに変換する。

これは本番の5秒固定ビンGTでも正式なvideo/day/site単位分割の代用でもない
（そもそも対象データがコオニユリではなくbombusで、バウトは可変長イベントでありビンではない）。
あくまで「4方式比較・閾値決定のコードパスが最後まで正しく動くか」を確認するための
プレースホルダー。splitはvideo名でソートした上で機械的に70/30分割するだけ
（video/day/site単位の意味のある分割ではない）。
"""
import argparse
import csv
import sys
from pathlib import Path

FPS_ASSUMED = 15.0  # crops_index.csv等にfps列がないための近似値（このデータセットの実測fpsに準拠）


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="/Users/ktak0119/Documents/20260709_InsectVisitPipeline/"
                                             "test_output/bombus_yolo_comparison",
                    help="crops_index.csv / clip_scores.csv / master.csv があるディレクトリ")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--decision-frac", type=float, default=0.7,
                    help="動画数のうち threshold_decision split に回す割合（残りはvalidation）")
    return p.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    src = Path(args.source_dir)

    crops_index = read_csv(src / "crops_index.csv")
    master = read_csv(src / "master.csv")

    yolo_by_key = {(r["video"], r["bout"]): float(r["confidence"]) for r in crops_index}

    videos = sorted({r["video"] for r in master})
    n_decision = max(1, int(len(videos) * args.decision_frac))
    decision_videos = set(videos[:n_decision])

    rows = []
    n_missing_yolo = 0
    for r in master:
        key = (r["video"], r["bout"])
        yolo_conf = yolo_by_key.get(key)
        if yolo_conf is None:
            n_missing_yolo += 1
            continue
        frame_start, frame_end = float(r["frame_start"]), float(r["frame_end"])
        crop_path = str((src / r["crop_path"]).resolve())
        rows.append({
            "video": r["video"],
            "bin_id": r["bout"],  # ダミー: バウト番号をbin_idとして読み替え
            "start_sec": round(frame_start / FPS_ASSUMED, 2),
            "end_sec": round(frame_end / FPS_ASSUMED, 2),
            "frame_idx": "",
            "t_sec": "",
            "yolo_conf": yolo_conf,
            "crop_path": crop_path,
            "clip_score": float(r["clip_score"]),
            "label": r["gt_positive"],
            "split": "threshold_decision" if r["video"] in decision_videos else "validation",
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "bin_id", "start_sec", "end_sec", "frame_idx", "t_sec",
                  "yolo_conf", "crop_path", "clip_score", "label", "split"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_dec = sum(1 for r in rows if r["split"] == "threshold_decision")
    n_val = len(rows) - n_dec
    print(f"wrote {len(rows)} dummy bin rows ({n_missing_yolo} skipped: no matching yolo confidence) "
          f"threshold_decision={n_dec} videos={len(decision_videos)}, validation={n_val} "
          f"videos={len(videos)-len(decision_videos)} -> {out_csv}")
    print("NOTE: this is placeholder data (bout read as bin, fps assumed=15.0, "
          "70/30 split by video order) for code-path smoke testing only.")


if __name__ == "__main__":
    main()
