# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
FACES_SCRIPTS = REPO_ROOT / "Treinamento_faces" / "scripts"
CRANIOS_SCRIPTS = REPO_ROOT / "Treinamento_cranios" / "scripts"
for p in [str(FACES_SCRIPTS), str(CRANIOS_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common_cranios import largest_skull_contour, make_silhouette256, resample_closed_contour, contour_metrics, draw_contour_preview

try:
    import mediapipe as mp
except Exception:
    mp = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FACE_OVAL_ORDER = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def uploaded_to_temp(uploaded, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(name).write_bytes(uploaded.getbuffer())
    return Path(name)


def write_temp_image(img, suffix=".jpg") -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(name).unlink(missing_ok=True)
    ext = suffix if suffix.lower() in [".jpg", ".jpeg", ".png"] else ".jpg"
    ok, enc = cv2.imencode(ext, img)
    if ok:
        Path(name).write_bytes(enc.tobytes())
    return Path(name)


def detect_face_landmarks(image):
    if mp is None:
        raise RuntimeError("mediapipe indisponivel")
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise RuntimeError("face nao detectada")
    h, w = image.shape[:2]
    return np.array([(lm.x * w, lm.y * h) for lm in res.multi_face_landmarks[0].landmark], dtype=float)


def face_shape64(image) -> np.ndarray:
    pts = detect_face_landmarks(image)
    contour = pts[[i for i in FACE_OVAL_ORDER if i < len(pts)]]
    _, norm = resample_closed_contour(contour, 64, align_pca=True)
    return norm


def draw_face_shape64(image) -> tuple[np.ndarray, np.ndarray]:
    pts = detect_face_landmarks(image)
    contour = pts[[i for i in FACE_OVAL_ORDER if i < len(pts)]]
    xy, norm = resample_closed_contour(contour, 64, align_pca=True)
    out = image.copy()
    for x, y in contour.astype(int):
        cv2.circle(out, (int(x), int(y)), max(2, min(out.shape[:2]) // 220), (0, 220, 0), -1)
    cv2.polylines(out, [contour.astype(np.int32)], True, (255, 0, 0), max(2, min(out.shape[:2]) // 180))
    return norm, out


def cranio_shape64(image) -> tuple[np.ndarray, dict, np.ndarray]:
    contour = largest_skull_contour(image)
    if contour is None:
        raise RuntimeError("contorno de cranio nao encontrado")
    xy, norm = resample_closed_contour(contour.reshape(-1, 2), 64, align_pca=True)
    metrics = contour_metrics(contour, image.shape)
    return norm, metrics, xy


def cranio_shape64_preview(image) -> tuple[np.ndarray, dict, np.ndarray]:
    contour = largest_skull_contour(image)
    if contour is None:
        raise RuntimeError("contorno de cranio nao encontrado")
    xy, norm = resample_closed_contour(contour.reshape(-1, 2), 64, align_pca=True)
    metrics = contour_metrics(contour, image.shape)
    preview = draw_contour_preview(image, contour)
    for x, y in xy.astype(int):
        cv2.circle(preview, (int(x), int(y)), max(2, min(preview.shape[:2]) // 220), (0, 255, 255), -1)
    return norm, metrics, preview


def image_features_from_silhouette(sil, side: str = "CF", size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(sil, cv2.COLOR_BGR2GRAY) if sil.ndim == 3 else sil
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    side_vec = np.array([side == "CF", side == "CE", side == "CD"], dtype=np.float32)
    return np.concatenate([small.reshape(-1), side_vec])


def class_from_probs(pm: float, pf: float) -> tuple[str, str]:
    pmax = max(float(pm), float(pf))
    if pmax < 0.55:
        return "INDEFINIDO", "INDEFINIDA"
    conf = "ALTA" if pmax >= 0.80 else ("MEDIA" if pmax >= 0.65 else "BAIXA")
    return ("MASCULINO", conf) if pm >= pf else ("FEMININO", conf)


def predict_cranio_gender(image) -> dict:
    model_path = REPO_ROOT / "Treinamento_cranios" / "resultados" / "sexo_cranio" / "modelo_sexo_cranio_sklearn.pkl"
    if not model_path.exists():
        return {"genero_previsto_cranio": "INDEFINIDO", "confianca_cranio": "SEM_MODELO", "prob_masculino_cranio": np.nan, "prob_feminino_cranio": np.nan, "modelo_sexo_cranio": "nao_disponivel"}
    contour = largest_skull_contour(image)
    if contour is None:
        return {"genero_previsto_cranio": "INDEFINIDO", "confianca_cranio": "FALHA_CONTORNO", "prob_masculino_cranio": np.nan, "prob_feminino_cranio": np.nan, "modelo_sexo_cranio": "sklearn_logistic_silhueta_64_cf"}
    sil = make_silhouette256(image, contour)
    feat = image_features_from_silhouette(sil, "CF").reshape(1, -1)
    model = joblib.load(model_path)
    pm = float(model.predict_proba(feat)[0, 1])
    pf = 1.0 - pm
    classe, conf = class_from_probs(pm, pf)
    return {"genero_previsto_cranio": classe, "confianca_cranio": conf, "prob_masculino_cranio": pm, "prob_feminino_cranio": pf, "modelo_sexo_cranio": "sklearn_logistic_silhueta_64_cf"}


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1, 2)
    n = min(len(a), len(b))
    if n == 0:
        return 999.0
    a = a[:n]
    b = b[:n]
    candidates = []
    for shift in range(n):
        br = np.roll(b, shift, axis=0)
        candidates.append(float(np.sqrt(((a - br) ** 2).sum(axis=1)).mean()))
        candidates.append(float(np.sqrt(((a - br[::-1]) ** 2).sum(axis=1)).mean()))
    return min(candidates)


def score_from_distance(d: float) -> float:
    return float(1.0 / (1.0 + max(d, 0.0) * 6.0))


def sexo_score(face_genero: str, classe_cranio: str | None, prob_m: float | None = None, prob_f: float | None = None) -> float:
    fg = str(face_genero or "").upper()
    cc = str(classe_cranio or "").upper()
    if fg not in ["MASCULINO", "FEMININO"] or not cc or "INDEFINIDO" in cc:
        return 0.5
    if fg == "MASCULINO":
        return float(prob_m) if prob_m is not None else (0.75 if "MASCULINO" in cc else 0.25)
    return float(prob_f) if prob_f is not None else (0.75 if "FEMININO" in cc else 0.25)


def rank_uploaded(face_path: Path, cranio_paths: list[Path], face_genero: str = "INDEFINIDO") -> pd.DataFrame:
    face_img = read_image(face_path)
    if face_img is None:
        raise RuntimeError("face nao abriu")
    fshape, face_preview = draw_face_shape64(face_img)
    face_preview_path = write_temp_image(face_preview)
    rows = []
    for p in cranio_paths:
        img = read_image(p)
        if img is None:
            rows.append({"arquivo_cranio": p.name, "status": "FALHA", "erro": "imagem nao abriu"})
            continue
        try:
            cshape, metrics, cranio_preview = cranio_shape64_preview(img)
            d = shape_distance(fshape, cshape)
            s_shape = score_from_distance(d)
            pred = predict_cranio_gender(img)
            s_sexo = sexo_score(face_genero, pred.get("genero_previsto_cranio"), pred.get("prob_masculino_cranio"), pred.get("prob_feminino_cranio"))
            qual = float(metrics.get("area_relativa_imagem", 0))
            s_quality = max(0.0, min(1.0, 1.0 - abs(qual - 0.35) / 0.35))
            score = 0.72 * s_shape + 0.18 * s_sexo + 0.10 * s_quality
            concordancia = "INDEFINIDO"
            if str(face_genero).upper() in ["MASCULINO", "FEMININO"] and pred.get("genero_previsto_cranio") in ["MASCULINO", "FEMININO"]:
                concordancia = "CONCORDANTE" if str(face_genero).upper() == pred.get("genero_previsto_cranio") else "DIVERGENTE"
            cranio_preview_path = write_temp_image(cranio_preview)
            rows.append({
                "arquivo_cranio": p.name,
                "status": "OK",
                "score_total": round(score, 6),
                "score_shape": round(s_shape, 6),
                "score_genero": round(s_sexo, 6),
                "score_qualidade": round(s_quality, 6),
                "genero_face": str(face_genero).upper(),
                "genero_previsto_cranio": pred.get("genero_previsto_cranio"),
                "concordancia_genero": concordancia,
                "confianca_cranio": pred.get("confianca_cranio"),
                "prob_masculino_cranio_pct": round(float(pred.get("prob_masculino_cranio", np.nan)) * 100, 2) if pd.notna(pred.get("prob_masculino_cranio", np.nan)) else np.nan,
                "prob_feminino_cranio_pct": round(float(pred.get("prob_feminino_cranio", np.nan)) * 100, 2) if pd.notna(pred.get("prob_feminino_cranio", np.nan)) else np.nan,
                "dist_shape": round(d, 6),
                "area_relativa_imagem": round(float(metrics.get("area_relativa_imagem", 0)), 6),
                "bbox_ratio": round(float(metrics.get("bbox_ratio", 0)), 6),
                "preview_face": str(face_preview_path),
                "preview_cranio": str(cranio_preview_path),
            })
        except Exception as e:
            rows.append({"arquivo_cranio": p.name, "status": "FALHA", "erro": str(e)})
    df = pd.DataFrame(rows)
    if "score_total" in df.columns:
        df = df.sort_values(["status", "score_total"], ascending=[False, False], na_position="last")
    return df.reset_index(drop=True)
