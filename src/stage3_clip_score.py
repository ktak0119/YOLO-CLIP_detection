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

# 対象ごとに変えてよい3点のうちの1つ（CLIP文言）。aggregate_classify.pyのPROMPT_SETSを移植。
# "butterfly"のnegativeにはコオニユリの花誤認対策が含まれる（対象固有、他対象には転用しない）。
# "bombus"は現時点でbutterfly由来の汎用negativeのみ、bombus特有のnegativeは未検証（実データで
# 偽陽性を確認してから追加する）。
PROMPT_SETS = {
    "butterfly": {
        "pos": [
            "a photo of a butterfly on a flower",
            "a butterfly resting on a flower in a garden",
            "a close-up of a butterfly with open wings on a flower",
            "a moth or butterfly insect feeding on a flower",
        ],
        "neg": [
            "a photo of grass and leaves blowing in the wind",
            "a blurry photo of a flower with no insect",
            "an empty garden scene with plants",
            "a photo of bright sky glare through leaves",
            "a photo of a spider or small bug on a leaf",
            "a security camera photo of a plant stem with no animal",
            "a macro photo of an orange tiger lily petal with dark red-brown spots, no insect",
            "a curled-back lily flower petal that looks like a wing but is just a petal",
            "an orange Lilium flower with spotted petals and recurved curled edges, no butterfly",
            "a close-up of speckled flower petal texture, not an animal",
        ],
    },
    "bombus": {
        "pos": [
            "a photo of a bumblebee on a flower",
            "a bumblebee resting on a flower in a garden",
            "a close-up of a fuzzy bumblebee foraging on a flower",
            "a bee or bumblebee insect feeding on a flower",
        ],
        "neg": [
            "a photo of grass and leaves blowing in the wind",
            "a blurry photo of a flower with no insect",
            "an empty garden scene with plants",
            "a photo of bright sky glare through leaves",
            "a photo of a spider or small bug on a leaf",
            "a security camera photo of a plant stem with no animal",
        ],
    },
}


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

    prompt_set = PROMPT_SETS[cfg["clip_prompt_key"]]
    pos_prompts, neg_prompts = prompt_set["pos"], prompt_set["neg"]
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
    print(f"scored {len(rows)} crops (target={cfg['clip_prompt_key']}) -> {out_csv}")


if __name__ == "__main__":
    main()
