# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common_cranios import *


def polygon_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def process_contour(path: Path) -> dict:
    df = read_csv_safe(path)
    if df.empty:
        raise ValueError("contorno vazio")
    pts = df[["x_px", "y_px"]].to_numpy(float)
    x, y = pts[:, 0], pts[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    area = polygon_area(x, y)
    per = float(np.sqrt(((pts - np.roll(pts, -1, axis=0)) ** 2).sum(axis=1)).sum())
    bw = float(x_max - x_min)
    bh = float(y_max - y_min)
    cx, cy = float(x.mean()), float(y.mean())
    left = pts[x < cx]
    right = pts[x >= cx]
    top = pts[y < cy]
    bottom = pts[y >= cy]
    assim_lr = abs(len(left) - len(right)) / max(len(pts), 1)
    assim_tb = abs(len(top) - len(bottom)) / max(len(pts), 1)
    comp = area / max(bw * bh, 1.0)
    circ = 4 * np.pi * area / max(per * per, 1.0)
    elong = max(bw, bh) / max(min(bw, bh), 1.0)
    img_path = str(df.get("imagem_cranio", pd.Series([""])).iloc[0])
    img = read_image(Path(img_path)) if img_path else None
    ih, iw = img.shape[:2] if img is not None else (1, 1)
    scale = max(bw, bh, 1.0)
    return {
        "ano": str(df["ano"].iloc[0]),
        "id_imagem": str(df["id_imagem"].iloc[0]),
        "id_pessoa": str(df.get("id_pessoa", pd.Series([person_id_from_stem(path.stem)])).iloc[0]),
        "imagem_cranio": img_path,
        "largura_cranio_px": bw,
        "altura_cranio_px": bh,
        "razao_largura_altura_cranio": bw / max(bh, 1.0),
        "area_contorno_cranio_px2": area,
        "perimetro_contorno_cranio_px": per,
        "area_relativa_imagem": area / max(iw * ih, 1),
        "assimetria_esq_dir": float(assim_lr),
        "assimetria_sup_inf": float(assim_tb),
        "centroide_x": cx,
        "centroide_y": cy,
        "bbox_w": bw,
        "bbox_h": bh,
        "bbox_x": x_min,
        "bbox_y": y_min,
        "bbox_x2": x_max,
        "bbox_y2": y_max,
        "bbox_ratio": bw / max(bh, 1.0),
        "imagem_w": iw,
        "imagem_h": ih,
        "compacidade": comp,
        "circularidade": circ,
        "elongacao": elong,
        "largura_cranio_norm": bw / scale,
        "altura_cranio_norm": bh / scale,
        "area_cranio_norm": area / max(scale * scale, 1.0),
        "perimetro_cranio_norm": per / scale,
        "num_pontos_contorno": len(pts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    rows, fails = [], []
    for y in discover_years(root, args.years):
        yrows = []
        for p in sorted((contornos_dir(root) / str(y)).glob("*_contorno.csv")):
            try:
                row = process_contour(p)
                rows.append(row)
                yrows.append(row)
            except Exception as e:
                fails.append({"ano": y, "id_imagem": p.stem.replace("_contorno", ""), "erro": str(e)})
        safe_to_csv(pd.DataFrame(yrows), medidas_dir(root) / f"medidas_cranio_{y}.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["razao_largura_altura_cranio", "area_cranio_norm", "perimetro_cranio_norm", "assimetria_esq_dir", "assimetria_sup_inf", "compacidade", "circularidade", "elongacao"]
        try:
            X = df[cols].replace([np.inf, -np.inf], np.nan).astype(float)
            X = X.fillna(X.median(numeric_only=True))
            if len(X) >= 3:
                Z = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
                df["pca_cranio_1"] = Z[:, 0]
                df["pca_cranio_2"] = Z[:, 1]
                d = np.sqrt((Z[:, 0] - np.median(Z[:, 0])) ** 2 + (Z[:, 1] - np.median(Z[:, 1])) ** 2)
                df["score_outlier_cranio_pca"] = np.round(d, 4)
            else:
                df["pca_cranio_1"] = 0.0
                df["pca_cranio_2"] = 0.0
                df["score_outlier_cranio_pca"] = 0.0
        except Exception:
            df["pca_cranio_1"] = 0.0
            df["pca_cranio_2"] = 0.0
            df["score_outlier_cranio_pca"] = 0.0
    safe_to_csv(df, medidas_dir(root) / "medidas_cranio_todos_anos.csv")
    safe_to_csv(pd.DataFrame(fails), medidas_dir(root) / "falhas_medidas_cranio.csv")
    print(f"[OK] medidas cranio: {len(df)}")


if __name__ == "__main__":
    main()
