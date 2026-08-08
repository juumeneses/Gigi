from __future__ import annotations

import numpy as np
import pandas as pd

from .retrieval import retrieval_metrics
from .paths import REPO_ROOT


def evaluate_random(df: pd.DataFrame, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    scores = rng.random((len(df), len(df)))
    ids = df["pair_key"].astype(str).tolist()
    metrics, ranking = retrieval_metrics(scores, ids, ids)
    metrics["baseline"] = "aleatorio"
    return metrics, ranking


def evaluate_sex_only(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    face_m = df["sexo_face"].eq("MASCULINO").to_numpy(dtype=float)
    skull_m = pd.to_numeric(df.get("prob_masculino_cranio", 0.5), errors="coerce").fillna(0.5).to_numpy()
    scores = -np.abs(face_m[:, None] - skull_m[None, :])
    # Desempate deterministico, sem usar identidade.
    scores += np.arange(len(df))[None, :] * 1e-12
    ids = df["pair_key"].astype(str).tolist()
    metrics, ranking = retrieval_metrics(scores, ids, ids)
    metrics["baseline"] = "apenas_sexo"
    return metrics, ranking


def _shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Replica o baseline existente: melhor shift circular e orientação."""
    candidates = []
    for shift in range(len(b)):
        shifted = np.roll(b, shift, axis=0)
        candidates.append(np.sqrt(((a - shifted) ** 2).sum(axis=1)).mean())
        candidates.append(np.sqrt(((a - shifted[::-1]) ** 2).sum(axis=1)).mean())
    return float(min(candidates))


def evaluate_shape64(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    face = pd.read_csv(REPO_ROOT / "Treinamento_faces/resultados/features_matching_faces/shape_faces_64pts.csv")
    skull = pd.read_csv(REPO_ROOT / "Treinamento_cranios/resultados/features_matching_cranios/shape_cranios_64pts.csv")
    for frame in (face, skull): frame["ano"] = frame["ano"].astype(str)
    keys = df[["ano", "id_pessoa", "pair_key"]].copy(); keys["ano"] = keys["ano"].astype(str)
    f = keys.merge(face, on=["ano", "id_pessoa"], how="left")
    s = keys.merge(skull, on=["ano", "id_pessoa"], how="left")
    fcols = sorted([c for c in f if c.startswith("face_shape_")])
    scols = sorted([c for c in s if c.startswith("cranio_shape_")])
    fa = f[fcols].to_numpy(float).reshape(len(f), 64, 2)
    sk = s[scols].to_numpy(float).reshape(len(s), 64, 2)
    scores = np.empty((len(fa), len(sk)), dtype=float)
    for i in range(len(fa)):
        for j in range(len(sk)):
            scores[i, j] = -_shape_distance(fa[i], sk[j])
    ids = keys["pair_key"].astype(str).tolist()
    metrics, ranking = retrieval_metrics(scores, ids, ids)
    metrics["baseline"] = "shape64_atual"
    return metrics, ranking


def run_baselines(df: pd.DataFrame, output_dir=None) -> pd.DataFrame:
    from pathlib import Path
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "reports/baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, fn in (("aleatorio", evaluate_random), ("apenas_sexo", evaluate_sex_only), ("shape64_atual", evaluate_shape64)):
        metrics, ranking = fn(df)
        ranking.to_csv(output_dir / f"ranking_{name}.csv", index=False)
        results.append(metrics)
    out = pd.DataFrame(results)
    out.to_csv(output_dir / "metricas_baselines.csv", index=False)
    return out
