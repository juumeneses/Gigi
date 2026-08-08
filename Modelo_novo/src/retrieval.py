from __future__ import annotations

import numpy as np
import pandas as pd


def retrieval_metrics(similarity: np.ndarray, query_ids: list[str], gallery_ids: list[str]) -> tuple[dict, pd.DataFrame]:
    similarity = np.asarray(similarity)
    if similarity.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("Matriz de similaridade com dimensoes invalidas")
    rows, ranks = [], []
    for i, qid in enumerate(query_ids):
        order = np.argsort(-similarity[i])
        ordered_ids = np.asarray(gallery_ids)[order]
        hit = np.flatnonzero(ordered_ids == qid)
        rank = int(hit[0] + 1) if len(hit) else len(gallery_ids) + 1
        ranks.append(rank)
        for position, j in enumerate(order, 1):
            rows.append({"query_id": qid, "candidate_id": gallery_ids[j], "rank": position, "score": float(similarity[i, j]), "correto": gallery_ids[j] == qid})
    r = np.asarray(ranks)
    metrics = {"n_queries": len(query_ids), "n_candidates": len(gallery_ids), "top_1": float(np.mean(r <= 1)), "top_3": float(np.mean(r <= 3)), "top_5": float(np.mean(r <= 5)), "recall_at_10": float(np.mean(r <= 10)), "mrr": float(np.mean(1.0 / r)), "median_rank": float(np.median(r)), "mean_rank": float(np.mean(r))}
    return metrics, pd.DataFrame(rows)


def add_sex_penalty(similarity: np.ndarray, query_sex: list[str], gallery_prob_m: np.ndarray, weight: float) -> np.ndarray:
    if weight == 0:
        return np.asarray(similarity)
    probs = np.asarray(gallery_prob_m, dtype=float)
    result = np.asarray(similarity, dtype=float).copy()
    eps = 1e-6
    for i, sex in enumerate(query_sex):
        compatible = probs if sex == "MASCULINO" else (1 - probs if sex == "FEMININO" else np.full_like(probs, 0.5))
        result[i] += weight * np.log(np.clip(compatible, eps, 1.0))
    return result


def sampled_gallery_metrics(similarity: np.ndarray, gallery_size: int=10, repeats: int=50, seed: int=42) -> dict:
    """Avalia galerias aleatórias fixas; a diagonal contém os pares corretos."""
    similarity=np.asarray(similarity,float); n=len(similarity)
    if similarity.shape!=(n,n): raise ValueError("A matriz precisa ser quadrada")
    if not 2<=gallery_size<=n: raise ValueError("Tamanho de galeria inválido")
    rng=np.random.default_rng(seed); ranks=[]
    for query in range(n):
        wrong=np.delete(np.arange(n),query); correct=float(similarity[query,query])
        for _ in range(repeats):
            chosen=rng.choice(wrong,gallery_size-1,replace=False)
            ranks.append(1+int(np.sum(similarity[query,chosen]>correct)))
    values=np.asarray(ranks,float)
    return {"gallery10_top_1":float(np.mean(values<=1)),"gallery10_top_3":float(np.mean(values<=3)),"gallery10_top_5":float(np.mean(values<=5)),"gallery10_mrr":float(np.mean(1/values)),"gallery10_mean_rank":float(np.mean(values)),"gallery10_n":int(len(values))}
