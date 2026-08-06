# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common_cranios import *


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()


def gray_to_bgr(gray):
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def foreground_mask(gray):
    mask = cv2.inRange(gray, 10, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def percentile_clip_gray(gray, low=1.0, high=99.0):
    vals = gray[gray > 0]
    if vals.size < 20:
        vals = gray.reshape(-1)
    lo, hi = np.percentile(vals, [low, high])
    if hi <= lo:
        return gray.copy()
    out = np.clip(gray.astype(np.float32), lo, hi)
    out = (out - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_atual(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = foreground_mask(gray)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    return gray_to_bgr(gray)


def normalize_monografia_equalizacao(img):
    gray = to_gray(img)
    gray = cv2.equalizeHist(gray)
    return gray_to_bgr(gray)


def normalize_clahe_suave(img):
    gray = to_gray(img)
    gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
    return gray_to_bgr(gray)


def normalize_percentile_clip(img):
    gray = percentile_clip_gray(to_gray(img), 1.0, 99.0)
    return gray_to_bgr(gray)


def normalize_crop_pad_percentile(img, output_size: int = 256):
    contour = largest_skull_contour(img)
    gray = to_gray(img)
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        margin = int(max(w, h) * 0.12)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(gray.shape[1], x + w + margin), min(gray.shape[0], y + h + margin)
        crop = gray[y1:y2, x1:x2]
    else:
        crop = gray
    if crop.size == 0:
        crop = gray
    side = max(crop.shape[:2])
    canvas = np.zeros((side, side), dtype=np.uint8)
    yy = (side - crop.shape[0]) // 2
    xx = (side - crop.shape[1]) // 2
    canvas[yy:yy + crop.shape[0], xx:xx + crop.shape[1]] = crop
    canvas = percentile_clip_gray(canvas, 1.0, 99.0)
    out = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return gray_to_bgr(out)


NORMALIZERS = {
    "atual": normalize_atual,
    "monografia_equalizacao": normalize_monografia_equalizacao,
    "clahe_suave": normalize_clahe_suave,
    "percentile_clip": normalize_percentile_clip,
    "crop_pad_percentile": normalize_crop_pad_percentile,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--normalization-kind", default="atual", choices=["all", *NORMALIZATION_KINDS])
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    kinds = list(NORMALIZATION_KINDS) if args.normalization_kind == "all" else [args.normalization_kind]
    rows = []
    for y in discover_years(root, args.years):
        for side in SIDES:
            src_dir = side_dir(root, y, side)
            imgs = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if src_dir.exists() else []
            if args.limit:
                imgs = imgs[: args.limit]
            for kind in kinds:
                dst_dir = normalized_variant_side_dir(root, kind, y, side)
                dst_dir.mkdir(parents=True, exist_ok=True)
                legacy_dir = normalized_side_dir(root, y, side) if kind == "atual" else None
                if legacy_dir:
                    legacy_dir.mkdir(parents=True, exist_ok=True)
                for p in tqdm(imgs, desc=f"{y} {side} {kind}"):
                    out = dst_dir / p.name
                    if out.exists() and not args.force:
                        rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "normalizacao": kind, "origem": str(p), "saida": str(out), "status": "ja_existia"})
                        continue
                    img = read_image(p)
                    if img is None:
                        rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "normalizacao": kind, "origem": str(p), "saida": str(out), "status": "falha_abertura"})
                        continue
                    norm = NORMALIZERS[kind](img)
                    ok = write_image(out, norm)
                    if ok and legacy_dir:
                        write_image(legacy_dir / p.name, norm)
                    rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "normalizacao": kind, "origem": str(p), "saida": str(out), "status": "OK" if ok else "falha_gravacao"})
    safe_to_csv(pd.DataFrame(rows), checks_dir(root) / "preprocess_cranios_normalizados.csv")
    print(f"[OK] cranios normalizados: {sum(1 for r in rows if r.get('status') in ['OK', 'ja_existia'])}")


if __name__ == "__main__":
    main()
