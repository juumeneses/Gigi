from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from .common import LAB_ROOT, abs_repo, load_config, read_image, save_csv, write_image


def _load_mapping() -> dict:
    return load_config(LAB_ROOT / "configs/landmarks_face.yaml")


def _point(source: pd.DataFrame, indices: list[int]) -> tuple[float, float, float, float]:
    g = source[source["idx"].astype(int).isin(indices)]
    if g.empty or len(g) < len(set(indices)):
        return np.nan, np.nan, np.nan, np.nan
    return float(g["x_px"].mean()), float(g["y_px"].mean()), float(g["x"].mean()), float(g["y"].mean())


def _dist(points: dict, a: str, b: str) -> float:
    if a not in points or b not in points: return np.nan
    return float(np.hypot(points[a][2] - points[b][2], points[a][3] - points[b][3]))


def run_faces(manifest: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpath = LAB_ROOT / "resultados/faces/features_anatomicas_faces.csv"; ppath = LAB_ROOT / "resultados/faces/pontos_anatomicos_faces.csv"
    if fpath.exists() and ppath.exists() and not force:
        return pd.read_csv(fpath), pd.read_csv(ppath)
    mapping = _load_mapping(); feature_rows, point_rows = [], []
    for _, rec in manifest.iterrows():
        lm_path = abs_repo(rec.get("landmarks_face_468_path")); im_path = abs_repo(rec.get("imagem_face")); base = {"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "imagem_face": rec.get("imagem_face", "")}
        if not lm_path or not lm_path.exists():
            feature_rows.append({**base, "status_face_anatomica": "FALHA", "motivo_face_anatomica": "SEM_LANDMARKS_468"}); continue
        src = pd.read_csv(lm_path); points = {}
        for name, spec in mapping["landmarks"].items():
            xpx, ypx, xn, yn = _point(src, spec["indices"]); visible = np.isfinite(xn) and np.isfinite(yn)
            points[name] = (xpx, ypx, xn, yn)
            point_rows.append({**base, "ponto": name, "x_px": xpx, "y_px": ypx, "x_norm": xn, "y_norm": yn, "visivel": visible, "confianca": float(spec["peso"]), "regiao_cranio": spec["regiao_cranio"], "incerteza": spec["incerteza"], "metodo": "mediapipe_468_reduzido"})
        iod = _dist(points, "centro_olho_direito", "centro_olho_esquerdo")
        scale = iod if np.isfinite(iod) and iod > 1e-6 else 1.0
        row = {**base, "status_face_anatomica": "OK", "motivo_face_anatomica": "", "distancia_interocular_norm": iod,
               "largura_nasal_norm": _dist(points, "alare_direito", "alare_esquerdo") / scale,
               "largura_zigomatica_norm": _dist(points, "zigio_direito", "zigio_esquerdo") / scale,
               "largura_mandibular_norm": _dist(points, "gonio_direito", "gonio_esquerdo") / scale,
               "olho_mento_norm": np.nan}
        if all(k in points for k in ["centro_olho_direito", "centro_olho_esquerdo", "mento"]):
            ey = (points["centro_olho_direito"][3] + points["centro_olho_esquerdo"][3]) / 2
            row["olho_mento_norm"] = float((points["mento"][3] - ey) / scale)
            row["inclinacao_olhos_graus"] = float(np.degrees(np.arctan2(points["centro_olho_esquerdo"][3]-points["centro_olho_direito"][3], points["centro_olho_esquerdo"][2]-points["centro_olho_direito"][2])))
        feature_rows.append(row)
        if im_path and im_path.exists() and (im := read_image(im_path)) is not None:
            preview = im.copy()
            for name, (x, y, _, _) in points.items():
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(preview, (int(x), int(y)), max(2, min(im.shape[:2])//180), (0, 255, 255), -1)
                    cv2.putText(preview, name, (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1, cv2.LINE_AA)
            write_image(LAB_ROOT / "resultados/faces/previews" / f"{rec['id_pessoa']}_face.jpg", preview)
    fdf, pdf = pd.DataFrame(feature_rows), pd.DataFrame(point_rows); save_csv(fdf, fpath); save_csv(pdf, ppath); return fdf, pdf
