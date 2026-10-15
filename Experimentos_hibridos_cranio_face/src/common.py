from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:  # Os arquivos de configuracao tambem sao JSON valido.
    yaml = None


LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = LAB_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        cfg = (yaml.safe_load(fh) if yaml is not None else json.load(fh)) or {}
    cfg["_config_path"] = str(p)
    return cfg


def ensure_dirs() -> None:
    dirs = [
        "dataset/manifestos", "dataset/splits", "modelos", "logs", "resultados/auditoria",
        "resultados/cranios/mascaras", "resultados/cranios/previews", "resultados/faces/previews",
        "resultados/matching", "resultados/comparacoes", "resultados/relatorios_html",
    ]
    for rel in dirs:
        (LAB_ROOT / rel).mkdir(parents=True, exist_ok=True)


def read_image(path: str | Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image(path: str | Path, image) -> bool:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".png"
    ok, enc = cv2.imencode(ext, image)
    if ok:
        enc.tofile(str(p))
    return bool(ok)


def rel_repo(path: str | Path | float | None) -> str:
    if path is None or (isinstance(path, float) and np.isnan(path)) or not str(path):
        return ""
    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(p)


def abs_repo(path: str | Path | float | None) -> Path | None:
    if path is None or (isinstance(path, float) and np.isnan(path)) or not str(path):
        return None
    p = Path(str(path))
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            df.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, p)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return p


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().isin({"TRUE", "1", "SIM", "YES", "OK"})


def write_run_metadata(stage: str, config: dict, seed: int, extra: dict | None = None) -> None:
    payload = {
        "stage": stage, "timestamp": datetime.now().isoformat(), "seed": int(seed),
        "config": config.get("_config_path", ""), "pid": os.getpid(), **(extra or {}),
    }
    p = LAB_ROOT / "logs" / f"ultima_execucao_{stage}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
