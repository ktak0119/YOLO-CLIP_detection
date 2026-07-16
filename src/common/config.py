"""共通設定ローダー。

configs/targets/<target>.yaml を1ファイルまるごと読み込んで設定dictを返す
（configs/job_template.yaml をコピーして作る「ジョブごとに1ファイル」形式。
以前あった pipeline.yaml との2ファイルマージ方式は廃止した）。
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(target: str) -> dict:
    target_path = CONFIGS_DIR / "targets" / f"{target}.yaml"
    if not target_path.exists():
        raise FileNotFoundError(
            f"no job config at {target_path} -- copy configs/job_template.yaml there first"
        )
    with open(target_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["target"] = target
    return cfg


def default_out_dir(target: str) -> Path:
    """out_dirがCLIにもconfigにも指定されなかった場合のフォールバック先。
    リポジトリ内の output/<target>/ 配下（.gitignore対象）。"""
    return REPO_ROOT / "output" / target
