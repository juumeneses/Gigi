from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROJECT_DIR


def _allocate(group: pd.DataFrame, rng: np.random.Generator, ratios: tuple[float, float, float]) -> pd.DataFrame:
    idx = group.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    out = pd.Series("train", index=idx)
    out.loc[idx[:n_test]] = "test"
    out.loc[idx[n_test:n_test + n_val]] = "val"
    return out.to_frame("split")


def create_splits(metadata: pd.DataFrame, seed: int = 42, ratios=(0.70, 0.15, 0.15), output_dir: Path | None = None) -> pd.DataFrame:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("As proporcoes de split devem somar 1")
    valid = metadata[metadata["usar_treino"].fillna(False)].copy().reset_index(drop=True)
    if valid["pair_key"].duplicated().any():
        raise ValueError("pair_key duplicada")
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in valid.groupby(["ano", "sexo_face"], dropna=False, sort=True):
        pieces.append(_allocate(group, rng, ratios))
    labels = pd.concat(pieces).sort_index()["split"]
    valid["split"] = labels
    if valid.groupby("pair_key")["split"].nunique().max() != 1:
        raise AssertionError("Vazamento de pessoa entre splits")
    output_dir = output_dir or PROJECT_DIR / "data/splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        valid[valid["split"].eq(name)].to_csv(output_dir / f"{name}.csv", index=False)
    valid.to_csv(output_dir / "all_splits.csv", index=False)
    digest = hashlib.sha256(valid[["pair_key", "split"]].to_csv(index=False).encode()).hexdigest()
    summary = {"seed": seed, "ratios": ratios, "split_sha256": digest, "counts": valid["split"].value_counts().to_dict(), "by_year": valid.groupby(["split", "ano"]).size().unstack(fill_value=0).to_dict(orient="index")}
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return valid


def create_leave_one_year_out(metadata: pd.DataFrame, output_dir: Path | None = None, seed: int = 42, val_fraction: float = 0.15) -> list[Path]:
    valid = metadata[metadata["usar_treino"].fillna(False)].copy()
    output_dir = output_dir or PROJECT_DIR / "data/splits/leave_one_year_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in sorted(valid["ano"].astype(str).unique()):
        fold = valid.copy()
        fold["split"] = np.where(fold["ano"].astype(str).eq(year), "test", "train")
        # Validação interna vem apenas dos outros anos; o ano holdout permanece intocado.
        rng = np.random.default_rng(seed + int(year))
        remaining = fold[fold["split"].eq("train")]
        val_indices = []
        for _, group in remaining.groupby(["ano", "sexo_face"], dropna=False, sort=True):
            idx = group.index.to_numpy().copy(); rng.shuffle(idx)
            count = max(1, round(len(idx) * val_fraction)) if len(idx) >= 3 else 0
            val_indices.extend(idx[:count])
        fold.loc[val_indices, "split"] = "val"
        if set(fold[fold["ano"].astype(str).eq(year)]["split"]) != {"test"}:
            raise AssertionError(f"Vazamento no holdout {year}")
        path = output_dir / f"holdout_{year}.csv"
        fold.to_csv(path, index=False)
        summary = {"holdout_year":year,"seed":seed,"counts":fold["split"].value_counts().to_dict(),"test_years":sorted(fold[fold.split.eq("test")]["ano"].astype(str).unique().tolist())}
        path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths.append(path)
    return paths
