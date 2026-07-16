#!/usr/bin/env python3
"""Stage3: CLIPスコアリング（ビン代表画像、足切りなし・スコア付与のみ、仕様書§2）。

2026624_ClaudeDetection/work/aggregate_classify.py のCLIPスコアリングロジック
（open_clip ViT-B-32, MPS, 対象別ゼロショットプロンプト）をほぼ無改修で移植。
入力はStage2出力CSV(bin単位テーブル)、出力は同じ行に clip_score 列を追加したCSV。
足切り・manifest分離は行わない（configのclip_apply_cutoffは常にfalse運用の想定）。
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

import open_clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import load_config
from common.provenance import should_skip, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]

# CLIPのpos/negプロンプトはPythonコードでなくジョブconfig(clip_pos_prompts/clip_neg_prompts)
# 側に置く（対象ごとに変えてよい項目をジョブyaml1ファイルで完結させるため）。
# 書き方のガイドはconfigs/job_template.yamlのコメント参照。


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", required=True, help="Stage2出力CSV")
    p.add_argument("--target", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--batch-size", type=int, default=48)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.target)
    out_csv = Path(args.out_csv)

    if should_skip(out_csv, cfg.get("resume", True)):
        print(f"SKIP (resume): {out_csv} already exists")
        return

    with open(args.in_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(list(csv.DictReader(open(args.in_csv)).fieldnames or []) + ["clip_score"])
        print(f"input has 0 rows, wrote empty {out_csv}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["clip_model"], pretrained=cfg["clip_pretrained"])
    tokenizer = open_clip.get_tokenizer(cfg["clip_model"])
    model = model.to(device).eval()

    pos_prompts, neg_prompts = cfg["clip_pos_prompts"], cfg["clip_neg_prompts"]
    prompts = pos_prompts + neg_prompts
    n_pos = len(pos_prompts)
    with torch.no_grad():
        text_feats = model.encode_text(tokenizer(prompts).to(device))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    batch_size = args.batch_size
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            imgs, valid = [], []
            for row in batch:
                try:
                    imgs.append(preprocess(Image.open(row["crop_path"]).convert("RGB")))
                    valid.append(row)
                except Exception as e:
                    row["clip_score"] = ""
                    print(f"WARN: could not load {row['crop_path']}: {e}")
            if not imgs:
                continue
            stacked = torch.stack(imgs).to(device)
            img_feats = model.encode_image(stacked)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ text_feats.T) * 100.0
            probs = sims.softmax(dim=-1)
            pos_score = probs[:, :n_pos].sum(dim=-1)
            for row, ps in zip(valid, pos_score.tolist()):
                row["clip_score"] = round(ps, 4)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_run_manifest(out_csv.parent, cfg, REPO_ROOT, extra={"stage": "stage3_clip_score", "in_csv": args.in_csv})
    print(f"scored {len(rows)} crops (target={cfg['target']}) -> {out_csv}")


if __name__ == "__main__":
    main()
