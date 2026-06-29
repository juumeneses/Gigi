# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIDES = ("CF", "CE", "CD")


def as_root(root=".") -> Path:
    return Path(root).expanduser().resolve()


def dados_root(root: Path) -> Path:
    return root / "dados_originais" / "cranios"


def normalizados_root(root: Path) -> Path:
    return root / "cranios_normalizados_pb"


def resultados_root(root: Path) -> Path:
    return root / "resultados"


def side_dir(root: Path, year: str, side: str) -> Path:
    return dados_root(root) / str(year) / side.upper()


def normalized_side_dir(root: Path, year: str, side: str) -> Path:
    return normalizados_root(root) / str(year) / side.upper()


def checks_dir(root: Path) -> Path:
    return resultados_root(root) / "checks"


def contornos_dir(root: Path) -> Path:
    return resultados_root(root) / "contornos_cranio"


def landmarks_dir(root: Path) -> Path:
    return resultados_root(root) / "landmarks_cranio_auto"


def medidas_dir(root: Path) -> Path:
    return resultados_root(root) / "medidas_cranio"


def qualidade_dir(root: Path) -> Path:
    return resultados_root(root) / "qualidade_cranio"


def previews_dir(root: Path) -> Path:
    return resultados_root(root) / "previews_cranio"


def sexo_dir(root: Path) -> Path:
    return resultados_root(root) / "sexo_cranio"


def features_dir(root: Path) -> Path:
    return resultados_root(root) / "features_matching_cranios"


def logs_dir(root: Path) -> Path:
    return resultados_root(root) / "logs"


def ensure_base_dirs(root: Path) -> None:
    for d in [checks_dir(root), contornos_dir(root), landmarks_dir(root), medidas_dir(root), qualidade_dir(root), previews_dir(root), sexo_dir(root), features_dir(root), logs_dir(root)]:
        d.mkdir(parents=True, exist_ok=True)


def discover_years(root: Path, years: Iterable[str] | None = None) -> list[str]:
    if years:
        return sorted(str(y) for y in years)
    base = dados_root(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 4)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path))


def write_image(path: Path, image) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in [".jpg", ".jpeg", ".png"] else ".jpg"
    ok, enc = cv2.imencode(ext, image)
    if not ok:
        return False
    enc.tofile(str(path))
    return True


def use_normalized_images() -> bool:
    return str(os.environ.get("CRANIOS_USE_NORMALIZED", "")).strip().upper() in {"1", "TRUE", "SIM", "YES"}


def image_source_dir(root: Path, year: str, side: str) -> Path:
    norm = normalized_side_dir(root, year, side)
    if use_normalized_images() and norm.exists():
        return norm
    return side_dir(root, year, side)


def list_images(root: Path, year: str, side: str | None = None) -> list[Path]:
    dirs = [image_source_dir(root, year, side)] if side else [image_source_dir(root, year, s) for s in SIDES]
    out = []
    for d in dirs:
        if d.exists():
            out.extend(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return sorted(out)


def person_id_from_stem(stem: str) -> str:
    s = str(stem)
    for suffix in ["_CF", "_CE", "_CD", "_CRANIO", "_CRANIO_FRONTAL", "_CRANIO_ESQ", "_CRANIO_DIR", "_skull", "_cranio"]:
        if s.upper().endswith(suffix.upper()):
            s = s[: -len(suffix)]
            break
    s = re.sub(r"[_-]?(CF|CE|CD)$", "", s, flags=re.I)
    m = re.match(r"^(\d{4})_(\d+)_(.+)$", s)
    if m:
        year, num, rest = m.groups()
        rest = re.sub(r"[^A-Za-z0-9]+", "", rest)
        return f"{year}_{num}_{rest}"
    return s


def side_from_path(path: Path) -> str:
    name = path.parent.name.upper()
    if name in SIDES:
        return name
    stem = path.stem.upper()
    for side in SIDES:
        if stem.endswith("_" + side) or stem.endswith("-" + side):
            return side
    return ""


def id_imagem(path: Path) -> str:
    return path.stem


def build_triplet_index(root: Path, years: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    for y in discover_years(root, years):
        by_person: dict[str, dict[str, str]] = {}
        for side in SIDES:
            for p in list_images(root, y, side):
                pid = person_id_from_stem(p.stem)
                rec = by_person.setdefault(pid, {"ano": str(y), "id_pessoa": pid})
                rec[f"imagem_{side.lower()}"] = str(p)
                rec[f"id_{side.lower()}"] = p.stem
        for rec in by_person.values():
            rec["tem_tripleta"] = all(rec.get(f"imagem_{s.lower()}") for s in SIDES)
            rows.append(rec)
    return pd.DataFrame(rows)


def complete_triplets(root: Path, years: Iterable[str] | None = None) -> pd.DataFrame:
    trip = build_triplet_index(root, years)
    if trip.empty or "tem_tripleta" not in trip.columns:
        return pd.DataFrame()
    return trip[trip["tem_tripleta"].astype(bool)].copy()


def complete_person_ids(root: Path, years: Iterable[str] | None = None) -> set[tuple[str, str]]:
    trip = complete_triplets(root, years)
    if trip.empty:
        return set()
    return set(map(tuple, trip[["ano", "id_pessoa"]].astype(str).values.tolist()))


def list_complete_cf_images(root: Path, year: str) -> list[Path]:
    trip = complete_triplets(root, [year])
    if trip.empty:
        return []
    return sorted(Path(p) for p in trip["imagem_cf"].dropna().astype(str) if p)


def cleanup_generated_outputs(root: Path, years: Iterable[str] | None = None, keep_logs: bool = True) -> list[str]:
    removed = []
    for d in [contornos_dir(root), landmarks_dir(root), medidas_dir(root), qualidade_dir(root), previews_dir(root), features_dir(root), sexo_dir(root), checks_dir(root)]:
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d))
    if not keep_logs and logs_dir(root).exists():
        shutil.rmtree(logs_dir(root))
        removed.append(str(logs_dir(root)))
    ensure_base_dirs(root)
    return removed


def largest_skull_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray_blur = cv2.bilateralFilter(gray_eq, 7, 45, 45)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else None
    # CT renderings usually have a black background. Prefer foreground masks
    # over inverse thresholding, because inverse masks often select the image border.
    masks = []
    masks.append(cv2.inRange(gray, 12, 255))
    masks.append(cv2.inRange(gray_blur, 18, 255))
    if hsv is not None:
        hch, sch, vch = cv2.split(hsv)
        masks.append(cv2.bitwise_and(cv2.inRange(vch, 18, 255), cv2.inRange(sch, 8, 255)))
    best = None
    best_score = -1.0
    h, w = gray.shape[:2]
    img_area = float(h * w)
    border_margin = max(2, int(min(h, w) * 0.01))
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
        # Remove the outer frame band. This cuts off rectangular image borders
        # without deleting large skulls that sit close to the frame.
        band = max(2, int(min(h, w) * 0.015))
        mask[:band, :] = 0
        mask[-band:, :] = 0
        mask[:, :band] = 0
        mask[:, -band:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < img_area * 0.015 or area > img_area * 0.88:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            touches_border = x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin)
            if touches_border and area > img_area * 0.82:
                continue
            ratio = bw / max(bh, 1)
            if ratio < 0.30 or ratio > 1.90:
                continue
            center_penalty = 0.15 * abs((x + bw / 2) - w / 2) + 0.15 * abs((y + bh / 2) - h / 2)
            score = area - center_penalty
            if score > best_score:
                best = c
                best_score = score
    return best


def contour_metrics(contour, image_shape) -> dict:
    h, w = image_shape[:2]
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)
    pts = contour.reshape(-1, 2).astype(float)
    return {
        "area_contorno_px2": area,
        "perimetro_contorno_px": per,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "bbox_ratio": float(bw / max(bh, 1)),
        "area_relativa_imagem": float(area / max(h * w, 1)),
        "num_pontos_contorno": int(len(pts)),
    }


def save_contour_csv(root: Path, year: str, stem: str, image_path: Path, contour, side: str = "CF") -> dict:
    pts = contour.reshape(-1, 2).astype(float)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    scale = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), 1.0)
    df = pd.DataFrame({
        "ano": str(year),
        "id_imagem": stem,
        "id_pessoa": person_id_from_stem(stem),
        "lado": side,
        "imagem_cranio": str(image_path),
        "ordem": np.arange(len(pts)),
        "x_px": pts[:, 0],
        "y_px": pts[:, 1],
        "x_norm": (pts[:, 0] - cx) / scale,
        "y_norm": (pts[:, 1] - cy) / scale,
    })
    out = contornos_dir(root) / str(year) / f"{stem}_contorno.csv"
    safe_to_csv(df, out)
    return {"ano": str(year), "id_imagem": stem, "id_pessoa": person_id_from_stem(stem), "lado": side, "imagem_cranio": str(image_path), "contorno_csv": str(out)}


def draw_contour_preview(image, contour, shape_points: np.ndarray | None = None):
    out = image.copy()
    cv2.drawContours(out, [contour], -1, (0, 230, 0), max(2, int(min(out.shape[:2]) / 260)))
    if shape_points is not None and len(shape_points):
        for i, (x, y) in enumerate(shape_points):
            cv2.circle(out, (int(x), int(y)), max(2, int(min(out.shape[:2]) / 220)), (0, 0, 255), -1)
            if i % 8 == 0:
                cv2.putText(out, str(i), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)
    return out


def resample_closed_contour(points: np.ndarray, n: int = 64, align_pca: bool = True) -> tuple[np.ndarray, np.ndarray]:
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
    xy = np.column_stack([np.interp(targets, dist, pts[:, 0]), np.interp(targets, dist, pts[:, 1])])
    norm = xy - xy.mean(axis=0)
    if align_pca and len(norm) >= 3:
        _, _, vt = np.linalg.svd(norm, full_matrices=False)
        norm = norm @ vt.T
        if norm[:, 1].mean() > 0:
            norm[:, 1] *= -1
    scale = max(norm[:, 0].max() - norm[:, 0].min(), norm[:, 1].max() - norm[:, 1].min(), 1e-9)
    norm = norm / scale
    return xy, norm


def make_silhouette256(image, contour):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(contour)
    margin = int(max(bw, bh) * 0.12)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        crop = mask
    side = max(crop.shape[:2])
    canvas = np.zeros((side, side), dtype=np.uint8)
    yy = (side - crop.shape[0]) // 2
    xx = (side - crop.shape[1]) // 2
    canvas[yy:yy + crop.shape[0], xx:xx + crop.shape[1]] = crop
    sil = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(sil, cv2.COLOR_GRAY2RGB)


def confidence_label(prob: float) -> str:
    p = float(prob)
    if p >= 0.80:
        return "ALTA"
    if p >= 0.65:
        return "MEDIA"
    if p >= 0.55:
        return "BAIXA"
    return "INDEFINIDO"


def class_from_probs(prob_m: float, prob_f: float) -> tuple[str, str]:
    pmax = max(float(prob_m), float(prob_f))
    conf = confidence_label(pmax)
    if pmax < 0.55:
        return "INDEFINIDO", conf
    return ("MASCULINO_PROVAVEL", conf) if prob_m >= prob_f else ("FEMININO_PROVAVEL", conf)


def quality_from_row(row: pd.Series) -> tuple[str, bool, str, float]:
    motivos = []
    score = 100.0
    try:
        area = float(row.get("area_relativa_imagem", np.nan))
        npts = float(row.get("num_pontos_contorno", np.nan))
        ratio = float(row.get("bbox_ratio", np.nan))
        outlier = float(row.get("score_outlier_cranio_pca", 0.0))
        bbox_y = float(row.get("bbox_y", np.nan))
        bbox_x = float(row.get("bbox_x", np.nan))
        bbox_x2 = float(row.get("bbox_x2", np.nan))
        bbox_y2 = float(row.get("bbox_y2", np.nan))
        img_w = float(row.get("imagem_w", np.nan))
        img_h = float(row.get("imagem_h", np.nan))
        comp = float(row.get("compacidade", np.nan))
    except Exception:
        return "FALHA", False, "metricas_invalidas", 0.0
    if not np.isfinite(area) or area <= 0:
        return "FALHA", False, "contorno_ausente", 0.0
    if area < 0.04 or area > 0.92:
        motivos.append("area_relativa_atipica")
        score -= 25
    elif area > 0.875:
        motivos.append("area_relativa_muito_alta_revisar")
        score -= 12
    if npts < 60:
        motivos.append("poucos_pontos_contorno")
        score -= 20
    if ratio < 0.35 or ratio > 1.65:
        motivos.append("bbox_ratio_atipico")
        score -= 18
    if (
        np.isfinite(bbox_y)
        and np.isfinite(img_h)
        and np.isfinite(comp)
        and img_h > 1
        and bbox_y <= img_h * 0.02
        and area > 0.73
        and comp < 0.78
    ):
        motivos.append("contorno_colado_borda_superior")
        score -= 12
    if (
        np.isfinite(bbox_x)
        and np.isfinite(bbox_x2)
        and np.isfinite(bbox_y2)
        and np.isfinite(img_w)
        and np.isfinite(img_h)
        and img_w > 1
        and img_h > 1
        and area > 0.82
        and (bbox_x <= img_w * 0.02 or bbox_x2 >= img_w * 0.98 or bbox_y2 >= img_h * 0.98)
    ):
        motivos.append("contorno_colado_borda_lateral_ou_inferior")
        score -= 10
    if outlier > 6:
        motivos.append("outlier_pca")
        score -= 20
    status = "OK" if not motivos else "REVISAR"
    return status, status == "OK", ";".join(motivos) if motivos else "metricas_ok", max(0.0, score)
