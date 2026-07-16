#!/usr/bin/env python3
"""Stage1: MOG2動体検知（全フレーム連続実行、フラグのみ・クリップ非生成）。

Stage2(YOLO)へ渡すフレームを絞るための任意の前置フィルタ（仕様書§8）。背景差分は
「動きがあったかどうか」の粗い足切りのみを行い、どの動きが対象らしいかの判定は一切しない
（判定はStage2/3に委ねる）。クロップは一切生成せず、動体候補が1つでもあったフレームの
インデックスを記録するだけ。

MOG2は背景モデルを連続的に更新する必要があるため、常に全フレームに対して detect() を呼ぶ
（間引きは行わない）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# 1プロセス1動画のシングルスレッド前提（-Pでの並列実行時にOpenCVの内部スレッドプールが
# CPUを食い合わないようにする。run_stage1_series.py等の既存の慣習を踏襲）。
cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.motion_detector import MOG2Detector
from common.provenance import should_skip, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--target", required=True, help="configs/targets/<target>.yaml のキー")
    p.add_argument("--out-json", required=True)
    p.add_argument("--resize-width", type=int, default=640)
    p.add_argument("--limit-seconds", type=float, default=None, help="動作確認用の打ち切り秒数")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_path = Path(args.out_json)

    if should_skip(out_path, cfg.get("resume", True)):
        print(f"SKIP (resume): {out_path} already exists")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.resize_width / src_w
    rw, rh = args.resize_width, int(src_h * scale)

    detector = MOG2Detector(
        history=cfg["mog2_history"],
        var_threshold=cfg["mog2_var_threshold"],
        learning_rate=cfg["mog2_learning_rate"],
        min_area=cfg["motion_min_area"],
        max_area=cfg["motion_max_area"],
        mask_top_frac=cfg["mog2_mask_top_frac"],
        warmup_frames=cfg["mog2_warmup_frames"],
    )

    flagged_frames = []
    frame_idx = -1
    processed = 0
    max_frames = int(args.limit_seconds * fps) if args.limit_seconds else None
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
        processed += 1

        small = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        candidates = detector.detect(gray)
        if candidates:
            flagged_frames.append(frame_idx)

    cap.release()
    elapsed = time.time() - t0

    result = {
        "video": args.video,
        "fps": fps,
        "src_w": src_w,
        "src_h": src_h,
        "resize_width": args.resize_width,
        "processed_frames": processed,
        "elapsed_sec": round(elapsed, 1),
        "n_flagged_frames": len(flagged_frames),
        "flagged_frames": flagged_frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    write_run_manifest(out_path.parent, cfg, REPO_ROOT, extra={"stage": "stage1_mog2_flag", "video": args.video})

    print(f"video={Path(args.video).name} processed={processed} frames in {elapsed:.1f}s "
          f"-> {len(flagged_frames)} flagged frames ({100*len(flagged_frames)/max(1,processed):.1f}%); "
          f"json={out_path}")


if __name__ == "__main__":
    main()
