#!/usr/bin/env python3
"""Stage2: YOLO検知＋ビンごとの代表クリップ抽出。

動画を1回だけ順次デコードしながらYOLO推論する（Stage1のMOG2フラグが有効なら
フラグ済みフレームのみ、無効なら全フレーム）。各フレームの最良ボックス（conf>=閾値、
1フレーム最大1件）をタイトクロップとしてその場でメモリ上に保持し、フレーム時刻が
あるビンの終了時刻を超えた時点でそのビンをクローズしてスコア上位K件のみをディスクに
書き出す（ストリーミング・ビンクローズ方式。3回目の動画デコードを避けるための設計、
詳細はplanのStage2節を参照）。

フラグ済みフレームは疎らに分布するためバッチ推論する（--yolo-batch-size）。
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.binning import bin_bounds, bin_id_for
from common.config import load_config
from common.provenance import should_skip, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--target", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-crops-dir", required=True)
    p.add_argument("--stage1-json", default=None,
                    help="Stage1出力(flagged_frames)。mog2_enabled=trueの場合は必須")
    p.add_argument("--limit-seconds", type=float, default=None, help="動作確認用の打ち切り秒数")
    return p.parse_args()


def tight_crop(frame, box_xyxy, pad_frac, pad_min_px):
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    pad_x = max(pad_min_px, int(w * pad_frac))
    pad_y = max(pad_min_px, int(h * pad_frac))
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x1c, y1c = min(W, int(x2 + pad_x)), min(H, int(y2 + pad_y))
    return frame[y0:y1c, x0:x1c].copy()


def close_bin(bin_id, candidates, top_k, video_stem, out_crops_dir, bin_sec, video_duration_sec, writer):
    start_sec, end_sec = bin_bounds(bin_id, bin_sec, video_duration_sec)
    candidates.sort(key=lambda c: -c["yolo_conf"])
    for c in candidates[:top_k]:
        crop_name = f"{video_stem}_bin{bin_id}_f{c['frame_idx']}.jpg"
        crop_path = out_crops_dir / crop_name
        cv2.imwrite(str(crop_path), c["crop_img"])
        writer.writerow({
            "video": video_stem,
            "bin_id": bin_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frame_idx": c["frame_idx"],
            "t_sec": c["t_sec"],
            "yolo_conf": c["yolo_conf"],
            "crop_path": str(crop_path),
        })


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_csv = Path(args.out_csv)
    out_crops_dir = Path(args.out_crops_dir)

    if should_skip(out_csv, cfg.get("resume", True)):
        print(f"SKIP (resume): {out_csv} already exists")
        return

    if cfg["mog2_enabled"] and not args.stage1_json:
        print("ERROR: mog2_enabled=true だが --stage1-json が指定されていない", file=sys.stderr)
        sys.exit(1)

    flagged_frames = None
    if cfg["mog2_enabled"]:
        import json
        with open(args.stage1_json) as f:
            s1 = json.load(f)
        flagged_frames = set(s1["flagged_frames"])

    from ultralytics import YOLO
    model = YOLO(cfg["yolo_model_path"])
    device = "mps"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    video_duration_sec = (total_frames / fps) if total_frames > 0 else None
    video_stem = Path(args.video).stem

    conf_threshold = cfg["yolo_conf_threshold"]
    top_k = cfg["top_k_per_bin"]
    batch_size = cfg["yolo_batch_size"]
    bin_sec = cfg["bin_sec"]
    pad_frac = cfg["tight_crop_pad_frac"]
    pad_min_px = cfg["tight_crop_pad_min_px"]

    out_crops_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "bin_id", "start_sec", "end_sec", "frame_idx", "t_sec", "yolo_conf", "crop_path"]

    open_bins = {}  # bin_id -> list of candidate dicts (with crop_img already extracted)
    max_frames = int(args.limit_seconds * fps) if args.limit_seconds else None

    def flush_batch(batch_frames, batch_idxs, writer):
        """batch_framesをYOLOにまとめて投入し、各フレームの最良ボックスをビンに追加する。
        処理後、確定的に閉じられるビン（最新フレームのbin_idより前のもの）を書き出す。"""
        if not batch_frames:
            return
        results = model.predict(batch_frames, conf=conf_threshold, device=device, verbose=False)
        for frame_idx, frame, r in zip(batch_idxs, batch_frames, results):
            t_sec = frame_idx / fps
            bin_id = bin_id_for(t_sec, bin_sec)
            if r.boxes is not None and len(r.boxes) > 0:
                confs = r.boxes.conf.tolist()
                best_i = max(range(len(confs)), key=lambda i: confs[i])
                best_conf = confs[best_i]
                xyxy = r.boxes.xyxy[best_i].tolist()
                crop_img = tight_crop(frame, xyxy, pad_frac, pad_min_px)
                open_bins.setdefault(bin_id, []).append({
                    "frame_idx": frame_idx, "t_sec": round(t_sec, 3),
                    "yolo_conf": round(best_conf, 4), "crop_img": crop_img,
                })
            # 現フレームより前の全ビンをクローズする（フレームは時刻昇順で処理されるため、
            # それらのビンに今後候補が追加されることはない）
            for closed_id in [bid for bid in open_bins if bid < bin_id]:
                close_bin(closed_id, open_bins.pop(closed_id), top_k, video_stem,
                          out_crops_dir, bin_sec, video_duration_sec, writer)

    t0 = time.time()
    processed = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch_frames, batch_idxs = [], []
        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
            processed += 1

            if flagged_frames is not None and frame_idx not in flagged_frames:
                continue

            batch_frames.append(frame)
            batch_idxs.append(frame_idx)
            if len(batch_frames) >= batch_size:
                flush_batch(batch_frames, batch_idxs, writer)
                batch_frames, batch_idxs = [], []

        flush_batch(batch_frames, batch_idxs, writer)
        cap.release()

        # 動画終端: 残っている全ビンをクローズする
        for bin_id in sorted(open_bins):
            close_bin(bin_id, open_bins[bin_id], top_k, video_stem,
                      out_crops_dir, bin_sec, video_duration_sec, writer)

    elapsed = time.time() - t0
    write_run_manifest(out_csv.parent, cfg, REPO_ROOT, extra={"stage": "stage2_yolo_bin", "video": args.video})
    n_yolo_frames = processed if flagged_frames is None else len(flagged_frames & set(range(processed)))
    print(f"video={video_stem} processed={processed} frames, yolo_ran_on~{n_yolo_frames} frames "
          f"in {elapsed:.1f}s; csv={out_csv}")


if __name__ == "__main__":
    main()
