# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common_faces import *


LEFT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def person_id_from_stem(stem: str) -> str:
    s = str(stem)
    for suffix in ["_FFN", "_FACE", "_face", "_frente", "_FRONTAL"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def resample_closed_contour(points: np.ndarray, n: int = 64) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("contorno insuficiente")
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if total <= 0:
        raise ValueError("contorno degenerado")
    targets = np.linspace(0, total, n + 1)[:-1]
    xs = np.interp(targets, dist, pts[:, 0])
    ys = np.interp(targets, dist, pts[:, 1])
    out = np.column_stack([xs, ys])
    c = out.mean(axis=0)
    out = out - c
    scale = max(out[:, 0].max() - out[:, 0].min(), out[:, 1].max() - out[:, 1].min(), 1e-9)
    return out / scale


def normalize_landmarks_for_pca(df: pd.DataFrame) -> np.ndarray:
    pts = df.sort_values("idx")[["x_px", "y_px"]].to_numpy(dtype=float)
    if pts.shape[0] < 468:
        raise ValueError("landmarks incompletos")
    pts = pts[:468]
    l_eye = pts[LEFT_EYE_RING].mean(axis=0)
    r_eye = pts[RIGHT_EYE_RING].mean(axis=0)
    center = (l_eye + r_eye) / 2.0
    pts = pts - center
    delta = r_eye - l_eye
    angle = np.arctan2(delta[1], delta[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = pts @ rot.T
    l_eye_r = (l_eye - center) @ rot.T
    r_eye_r = (r_eye - center) @ rot.T
    ipd = float(np.linalg.norm(r_eye_r - l_eye_r))
    if ipd <= 1e-9:
        raise ValueError("IPD invalida")
    pts = pts / ipd
    oval_y = pts[[i for i in FACE_OVAL_ORDER if i < len(pts)], 1]
    h = float(np.quantile(oval_y, 0.98) - np.quantile(oval_y, 0.02))
    if h <= 1e-9:
        raise ValueError("altura facial invalida")
    return (pts / h).reshape(-1)


def shape64_from_landmarks(df: pd.DataFrame) -> np.ndarray:
    idx = df.set_index("idx")
    order = [i for i in FACE_OVAL_ORDER if i in idx.index]
    if len(order) < 8:
        raise ValueError("contorno facial insuficiente")
    pts = idx.loc[order][["x_px", "y_px"]].to_numpy(dtype=float)
    return resample_closed_contour(pts, 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--target-evr", type=float, default=0.9995)
    ap.add_argument("--max-components", type=int, default=512)
    args = ap.parse_args()

    root = as_root(args.root)
    ensure_base_dirs(root)
    out = matching_features_dir(root)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    vectors = []
    shape_rows = []
    problems = []

    for y in discover_years(root, args.years):
        for p in sorted((marks_dir(root, y) / "landmarks_csv").glob("*_landmarks.csv")):
            stem = p.stem.replace("_landmarks", "")
            df = read_csv_safe(p)
            if df.empty:
                problems.append({"ano": y, "id_imagem": stem, "erro": "landmarks vazio"})
                continue
            try:
                vec = normalize_landmarks_for_pca(df)
                shp = shape64_from_landmarks(df)
            except Exception as e:
                problems.append({"ano": y, "id_imagem": stem, "erro": str(e)})
                continue
            row = {"ano": str(y), "id_imagem": stem, "id_pessoa": person_id_from_stem(stem)}
            rows.append(row)
            vectors.append(vec)
            srow = dict(row)
            for i, (x, yv) in enumerate(shp):
                srow[f"face_shape_{i:03d}_x"] = float(x)
                srow[f"face_shape_{i:03d}_y"] = float(yv)
            shape_rows.append(srow)

    base = pd.DataFrame(rows)
    if base.empty:
        safe_to_csv(pd.DataFrame(problems, columns=["ano", "id_imagem", "erro"]), out / "problemas_features_faces_final.csv")
        print("[ERRO] Nenhum landmark facial valido encontrado.")
        return

    X = np.vstack(vectors)
    n_comp = max(1, min(args.max_components, X.shape[0], X.shape[1]))
    probe = PCA(n_components=n_comp, svd_solver="randomized", random_state=42).fit(X)
    evr = np.cumsum(probe.explained_variance_ratio_)
    k = int(np.searchsorted(evr, args.target_evr, side="left") + 1)
    k = max(1, min(k, n_comp))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=42).fit(X)
    Z = pca.transform(X)
    pca_df = base.copy()
    for i in range(k):
        pca_df[f"face_pca_{i+1:03d}"] = Z[:, i]

    shape_df = pd.DataFrame(shape_rows)
    final = shape_df.merge(pca_df, on=["ano", "id_imagem", "id_pessoa"], how="left")

    ana = normalize_key_columns(load_all_analysis(root, discover_years(root, args.years)))
    if not ana.empty:
        keep = [c for c in ["ano", "id_imagem", "qualidade_landmarks", "motivo_qualidade", "razao_largura_altura_face", "razao_olhos_largura_face", "razao_boca_largura_face", "razao_bbox_contorno", "pca_face_1", "pca_face_2", "score_outlier_pca"] if c in ana.columns]
        final = final.merge(ana[keep].drop_duplicates(["ano", "id_imagem"]), on=["ano", "id_imagem"], how="left")

    genero = normalize_key_columns(read_csv_safe(genero_dir(root) / "genero_faces_curado.csv"))
    if not genero.empty:
        keep = [c for c in ["ano", "id_imagem", "genero_final", "origem_genero_final", "prob_masculino_fairface", "prob_feminino_fairface", "confianca_fairface", "usar_para_treino_genero"] if c in genero.columns]
        final = final.merge(genero[keep].drop_duplicates(["ano", "id_imagem"], keep="last"), on=["ano", "id_imagem"], how="left")

    final["usar_matching"] = final.get("qualidade_landmarks", pd.Series("OK", index=final.index)).astype(str).str.upper().isin(["OK", "OK_MANUAL", "OK_AUTO"])
    final["imagem_face"] = final.apply(lambda r: str(find_image_path(root, r["ano"], r["id_imagem"]) or ""), axis=1)

    safe_to_csv(shape_df, out / "shape_faces_64pts.csv")
    safe_to_csv(final, out / "features_matching_faces_final.csv")
    safe_to_csv(pd.DataFrame(problems, columns=["ano", "id_imagem", "erro"]), out / "problemas_features_faces_final.csv")
    joblib.dump({"pca": pca, "target_evr": args.target_evr, "feature": "468_landmarks_normalizados", "n_components": k}, out / "pca_faces_landmarks.pkl")
    print(f"[OK] faces final: {len(final)} linhas, PCA k={k}")


if __name__ == "__main__":
    main()
