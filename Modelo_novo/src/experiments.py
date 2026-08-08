from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_DIR, REPO_ROOT


def provenance(config: dict, split_hash: str = "") -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        commit = ""
    return {"created_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "python": platform.python_version(), "platform": platform.platform(), "split_hash": split_hash, "config": config}


def save_provenance(run_dir: Path, config: dict, split_hash: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(provenance(config, split_hash), ensure_ascii=False, indent=2), encoding="utf-8")

