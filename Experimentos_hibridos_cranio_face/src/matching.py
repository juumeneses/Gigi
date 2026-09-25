from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import LAB_ROOT, REPO_ROOT, bool_series, read_csv, save_csv


COMMON = ["largura_nasal", "largura_zigomatica", "largura_mandibular", "orbita_mento"]


def build_paired_dataset(manifest: pd.DataFrame, splits: pd.DataFrame, skull: pd.DataFrame, face: pd.DataFrame) -> pd.DataFrame:
    base = manifest.merge(splits[["id_pessoa", "split"]], on="id_pessoa", how="left")
    s = skull.rename(columns={"largura_nasal_norm": "cranio_largura_nasal", "largura_zigomatica_norm": "cranio_largura_zigomatica", "largura_mandibular_norm": "cranio_largura_mandibular", "orbita_mento_norm": "cranio_orbita_mento"})
    f = face.rename(columns={"largura_nasal_norm": "face_largura_nasal", "largura_zigomatica_norm": "face_largura_zigomatica", "largura_mandibular_norm": "face_largura_mandibular", "olho_mento_norm": "face_orbita_mento"})
    keep_s = [c for c in ["id_pessoa", "status_cranio_geo", "topo_truncado", "regioes_centrais_detectadas", "cranio_largura_nasal", "cranio_largura_zigomatica", "cranio_largura_mandibular", "cranio_orbita_mento", "distancia_interorbital_norm"] if c in s]
    keep_f = [c for c in ["id_pessoa", "status_face_anatomica", "face_largura_nasal", "face_largura_zigomatica", "face_largura_mandibular", "face_orbita_mento", "distancia_interocular_norm"] if c in f]
    out = base.merge(s[keep_s], on="id_pessoa", how="left").merge(f[keep_f], on="id_pessoa", how="left")
    out["disponibilidade_regioes_core"] = out.get("regioes_centrais_detectadas", 0).fillna(0) / 4.0
    save_csv(out, LAB_ROOT / "dataset/manifestos/dataset_pareado_matching.csv")
    return out


def _vectors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [c for c in ["cranio_largura_nasal", "cranio_orbita_mento", "distancia_interorbital_norm", "face_largura_nasal", "face_orbita_mento", "distancia_interocular_norm"] if c in df]
    return df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float), columns


FEATURE_PAIRS = [("face_largura_nasal", "cranio_largura_nasal"), ("face_largura_zigomatica", "cranio_largura_zigomatica"), ("face_largura_mandibular", "cranio_largura_mandibular"), ("face_orbita_mento", "cranio_orbita_mento")]


def _calibration(paired: pd.DataFrame) -> dict:
    train = paired[(paired["split"] == "treino") & paired["usar_matching"]]
    stats = {}
    for fc, sc in FEATURE_PAIRS:
        if fc not in train or sc not in train: continue
        for col in [fc, sc]:
            values = pd.to_numeric(train[col], errors="coerce").dropna()
            stats[col] = (float(values.median()), float(max(values.quantile(.75)-values.quantile(.25), 1e-6))) if len(values) else (0.0, 1.0)
    return stats


def _shared_distance(face_row: pd.Series, skull_row: pd.Series, stats: dict) -> tuple[float, int]:
    errors = []
    for fc, sc in FEATURE_PAIRS:
        a, b = face_row.get(fc, np.nan), skull_row.get(sc, np.nan)
        if pd.notna(a) and pd.notna(b) and fc in stats and sc in stats:
            az = (float(a)-stats[fc][0])/stats[fc][1]; bz = (float(b)-stats[sc][0])/stats[sc][1]
            errors.append(abs(az-bz))
    return (float(np.mean(errors)), len(errors)) if errors else (999.0, 0)


def run_anatomical_ranking(paired: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = _calibration(paired)
    test = paired[(paired["split"] == "teste") & paired["usar_matching"]].copy().reset_index(drop=True)
    rows = []
    for _, q in test.iterrows():
        candidates = []
        for _, c in test.iterrows():
            d, n = _shared_distance(q, c, stats); score = 0.0 if n == 0 else 1.0 / (1.0 + d)
            candidates.append((str(c["id_pessoa"]), score, d, n))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for rank, (cid, score, dist, n) in enumerate(candidates, 1):
            rows.append({"query_id": q["id_pessoa"], "candidate_id": cid, "rank": rank, "score_anatomico": score, "distancia": dist, "regioes_comparadas": n, "correto": cid == str(q["id_pessoa"])})
    ranking = pd.DataFrame(rows); save_csv(ranking, LAB_ROOT / "resultados/matching/ranking_anatomico_teste.csv")
    positions = ranking[ranking["correto"]]["rank"].to_numpy() if not ranking.empty else np.array([])
    metrics = {"n_queries": int(len(test)), "recall_at_1": float(np.mean(positions <= 1)) if len(positions) else 0, "recall_at_5": float(np.mean(positions <= 5)) if len(positions) else 0,
               "recall_at_10": float(np.mean(positions <= 10)) if len(positions) else 0, "recall_at_20": float(np.mean(positions <= 20)) if len(positions) else 0,
               "mrr": float(np.mean(1/positions)) if len(positions) else 0, "posicao_mediana": float(np.median(positions)) if len(positions) else np.nan}
    (LAB_ROOT / "resultados/matching/metricas_ranking_anatomico.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return ranking, metrics


def build_pair_table(paired: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    eligible = paired[paired["usar_matching"] & paired["split"].notna()].copy()
    for split, group in eligible.groupby("split"):
        ids = group["id_pessoa"].astype(str).tolist()
        for _, r in group.iterrows():
            rid = str(r["id_pessoa"]); gender = str(r.get("genero_final", "INDEFINIDO")); same = group[(group["id_pessoa"].astype(str) != rid) & (group.get("genero_final", "").astype(str) == gender)]
            pool = same if len(same) else group[group["id_pessoa"].astype(str) != rid]
            if pool.empty: continue
            neg = pool.iloc[int(rng.integers(0, len(pool)))]
            rows.append({"split": split, "face_id": rid, "cranio_id": rid, "label": 1, "tipo_negativo": "POSITIVO"})
            rows.append({"split": split, "face_id": rid, "cranio_id": str(neg["id_pessoa"]), "label": 0, "tipo_negativo": "MESMO_GENERO" if len(same) else "ALEATORIO"})
    out = pd.DataFrame(rows); save_csv(out, LAB_ROOT / "dataset/manifestos/pares_matching.csv"); return out
