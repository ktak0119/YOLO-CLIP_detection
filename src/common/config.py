"""共通設定ローダー。

configs/pipeline.yaml（全対象共通の既定値）と configs/targets/<target>.yaml
（対象ごとに変えてよい3点: VLM探索対象文字列・CLIP文言キー・動体サイズ範囲、
および対象固有のモデルパス）をマージして1つの設定dictを返す。

対象ごとに変えてよい項目以外がtargets側で上書きされることを防ぐため、
両ファイルの間でキーが重複していたらエラーにする。
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(target: str) -> dict:
    with open(CONFIGS_DIR / "pipeline.yaml") as f:
        pipeline_cfg = yaml.safe_load(f) or {}
    target_path = CONFIGS_DIR / "targets" / f"{target}.yaml"
    if not target_path.exists():
        raise FileNotFoundError(f"no target config at {target_path}")
    with open(target_path) as f:
        target_cfg = yaml.safe_load(f) or {}

    overlap = set(pipeline_cfg) & set(target_cfg)
    if overlap:
        raise ValueError(
            f"configs/targets/{target}.yaml overrides common pipeline.yaml keys {sorted(overlap)}; "
            "only target-specific keys (VLM search string, CLIP prompt key, motion size range, "
            "model paths) belong in a target config -- shared numeric parameters live in pipeline.yaml"
        )

    cfg = {**pipeline_cfg, **target_cfg}
    cfg["target"] = target
    return cfg
