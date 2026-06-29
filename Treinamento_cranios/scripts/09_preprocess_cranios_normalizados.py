# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common_cranios import *


def normalize_skull_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.inRange(gray, 10, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    rows = []
    for y in discover_years(root, args.years):
        for side in SIDES:
            src_dir = side_dir(root, y, side)
            dst_dir = normalized_side_dir(root, y, side)
            dst_dir.mkdir(parents=True, exist_ok=True)
            imgs = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if src_dir.exists() else []
            if args.limit:
                imgs = imgs[: args.limit]
            for p in tqdm(imgs, desc=f"{y} {side} PB"):
                out = dst_dir / p.name
                if out.exists() and not args.force:
                    rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "ja_existia"})
                    continue
                img = read_image(p)
                if img is None:
                    rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "falha_abertura"})
                    continue
                norm = normalize_skull_image(img)
                ok = write_image(out, norm)
                rows.append({"ano": y, "lado": side, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "OK" if ok else "falha_gravacao"})
    safe_to_csv(pd.DataFrame(rows), checks_dir(root) / "preprocess_cranios_normalizados.csv")
    print(f"[OK] cranios normalizados: {sum(1 for r in rows if r.get('status') in ['OK', 'ja_existia'])}")


if __name__ == "__main__":
    main()
