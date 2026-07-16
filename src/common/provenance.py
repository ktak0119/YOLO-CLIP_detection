"""実行のprovenance記録とresume判定。

各ステージスクリプトの出力ディレクトリに run_manifest.json を書き、使用したconfig全体・
git commit sha・実行日時を記録する。--resume時は、対象動画の出力ファイルが既に存在すれば
その動画をスキップする（xargs -P並列実行と組み合わせる前提のシンプルな「動画単位」resume）。
"""
import json
import subprocess
import time
from pathlib import Path


def git_sha(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def write_run_manifest(out_dir: Path, config: dict, repo_root: Path, extra: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "git_sha": git_sha(repo_root),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        manifest.update(extra)
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def should_skip(out_path: Path, resume: bool) -> bool:
    """resume=Trueかつ既に出力ファイルが存在する場合、その動画の処理をスキップしてよいか"""
    return resume and out_path.exists()
