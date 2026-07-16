#!/usr/bin/env python3
"""Stage6-a: Stage5で陽性(TP)判定されたビンからクリップマニフェストを作り、
同時間帯・隣接するビンを機械的にマージする（意味的な訪花イベント統合ではない、
単なるファイル結合。merge_overlapping_clips.pyを移植）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--screening-csv", required=True, help="Stage5出力（screening列にTP/FP）")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def resolve_video_path(videos_dir: Path, video: str) -> str:
    name = video if video.endswith((".mp4", ".mov", ".avi")) else video + ".mp4"
    return str(videos_dir / name)


def main():
    args = parse_args()
    cfg = load_config(args.target)
    videos_dir = Path(args.videos_dir)

    import csv
    with open(args.screening_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    tp_rows = [r for r in rows if r["screening"] == "TP"]

    entries = []
    for r in tp_rows:
        entries.append({
            "video_path": resolve_video_path(videos_dir, r["video"]),
            "start_sec": float(r["start_sec"]),
            "end_sec": float(r["end_sec"]),
            "event_id": f"{Path(r['video']).stem}__bin{r['bin_id']}",
        })

    by_video = {}
    for e in entries:
        by_video.setdefault(e["video_path"], []).append(e)

    merge_gap_sec = cfg["merge_gap_sec"]
    merged = []
    for video_path, evs in by_video.items():
        evs.sort(key=lambda e: e["start_sec"])
        cur = None
        for e in evs:
            if cur is None:
                cur = {
                    "video_path": video_path,
                    "start_sec": e["start_sec"],
                    "end_sec": e["end_sec"],
                    "source_event_ids": [e["event_id"]],
                }
                continue
            gap = e["start_sec"] - cur["end_sec"]
            if gap <= merge_gap_sec:
                cur["end_sec"] = max(cur["end_sec"], e["end_sec"])
                cur["source_event_ids"].append(e["event_id"])
            else:
                merged.append(cur)
                cur = {
                    "video_path": video_path,
                    "start_sec": e["start_sec"],
                    "end_sec": e["end_sec"],
                    "source_event_ids": [e["event_id"]],
                }
        if cur is not None:
            merged.append(cur)

    for m in merged:
        m["event_id"] = m["source_event_ids"][0] + ("" if len(m["source_event_ids"]) == 1 else "_merged")
        m["n_merged"] = len(m["source_event_ids"])

    merged.sort(key=lambda m: (m["video_path"], m["start_sec"]))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    write_run_manifest(out_json.parent, cfg, REPO_ROOT, extra={"stage": "stage6_merge_bins"})

    n_merged_groups = sum(1 for m in merged if m["n_merged"] > 1)
    print(f"TP bins: {len(tp_rows)} -> merged clips: {len(merged)} ({n_merged_groups} groups merged) -> {out_json}")


if __name__ == "__main__":
    main()
