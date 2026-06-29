# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import cv2
import pandas as pd
from tqdm import tqdm

from common_faces import *


def normalize_face_image(img, denoise: bool = True, sharpen: float = 0.55):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    out = cv2.convertScaleAbs(out, alpha=1.04, beta=2)
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 2, 2, 7, 21)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.0 + float(sharpen), blur, -float(sharpen), 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--sharpen", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    rows = []
    for y in discover_years(root, args.years):
        out_dir = normalized_images_dir(root, y)
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted([p for p in images_dir(root, y).iterdir() if is_original_face_image(p)]) if images_dir(root, y).exists() else []
        if args.limit:
            imgs = imgs[: args.limit]
        for p in tqdm(imgs, desc=f"{y} faces normalizadas"):
            out = out_dir / p.name
            if out.exists() and not args.force:
                rows.append({"ano": y, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "ja_existia"})
                continue
            img = read_image(p)
            if img is None:
                rows.append({"ano": y, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "falha_abertura"})
                continue
            norm = normalize_face_image(img, denoise=not args.no_denoise, sharpen=args.sharpen)
            ok = write_image(out, norm)
            rows.append({"ano": y, "id_imagem": p.stem, "origem": str(p), "normalizada": str(out), "status": "OK" if ok else "falha_gravacao"})
    safe_to_csv(pd.DataFrame(rows), resultados_root(root) / "checks" / "preprocess_faces_normalizadas.csv")
    print(f"[OK] faces normalizadas: {sum(1 for r in rows if r.get('status') in ['OK', 'ja_existia'])}")


if __name__ == "__main__":
    main()
