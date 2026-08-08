from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .paths import PROJECT_DIR, repo_path


def _letterbox(source: Path, destination: Path, size: int, grayscale: bool = False) -> None:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("L" if grayscale else "RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        background = Image.new(image.mode, (size, size), 0 if grayscale else (0, 0, 0))
        background.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        background.save(destination, format="PNG", optimize=True)


def preprocess_manifest(metadata: pd.DataFrame, size: int = 224, force: bool = False, limit: int | None = None) -> pd.DataFrame:
    valid = metadata[metadata["usar_treino"].fillna(False)].copy()
    if limit:
        valid = valid.head(limit)
    root = PROJECT_DIR / "data/processed" / f"v1_{size}"
    rows = []
    for _, row in valid.iterrows():
        safe_id = hashlib.sha1(str(row["pair_key"]).encode()).hexdigest()[:16]
        outputs = {"face": root / "faces" / f"{safe_id}.png", "cranio": root / "cranios" / f"{safe_id}.png"}
        sources = {"face": repo_path(row["face_path"]), "cranio": repo_path(row["cranio_cf_path"])}
        rec = {"pair_key": row["pair_key"], "status": "OK", "erro": ""}
        try:
            for modality in ("face", "cranio"):
                if not sources[modality] or not sources[modality].exists():
                    raise FileNotFoundError(f"{modality}: {sources[modality]}")
                if force or not outputs[modality].exists():
                    _letterbox(sources[modality], outputs[modality], size, grayscale=modality == "cranio")
                rec[f"{modality}_processed"] = outputs[modality].relative_to(PROJECT_DIR.parent).as_posix()
        except Exception as exc:
            rec.update(status="FALHA", erro=str(exc))
        rows.append(rec)
    status = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    status.to_csv(root / "preprocess_status.csv", index=False)
    return status

