# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

import pandas as pd
from tqdm import tqdm

from common_cranios import *


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use-normalized", action="store_true")
    args = ap.parse_args()
    if args.use_normalized:
        os.environ["CRANIOS_USE_NORMALIZED"] = "1"
    root = as_root(args.root)
    ensure_base_dirs(root)
    rows, fails = [], []

    for y in discover_years(root, args.years):
        imagens_cf = list_complete_cf_images(root, y)
        if not imagens_cf:
            print(f"[AVISO] {y}: nenhum individuo completo CF+CE+CD encontrado; pulando contornos.")
            continue
        for p in tqdm(imagens_cf, desc=f"{y} contornos CF completos"):
            out_csv = contornos_dir(root) / str(y) / f"{p.stem}_contorno.csv"
            prev = previews_dir(root) / str(y) / f"{p.stem}_contorno_preview.jpg"
            if out_csv.exists() and prev.exists() and not args.force:
                rows.append({"ano": y, "id_imagem": p.stem, "status_segmentacao": "ja_existia", "contorno_csv": str(out_csv)})
                continue
            img = read_image(p)
            if img is None:
                fails.append({"ano": y, "id_imagem": p.stem, "erro": "imagem nao abre", "imagem_cranio": str(p)})
                continue
            contour = largest_skull_contour(img)
            if contour is None:
                fails.append({"ano": y, "id_imagem": p.stem, "erro": "contorno nao encontrado", "imagem_cranio": str(p)})
                continue
            rec = save_contour_csv(root, y, p.stem, p, contour, "CF")
            rec.update(contour_metrics(contour, img.shape))
            rec.update({"status_segmentacao": "OK", "motivo_segmentacao": "maior_contorno_valido"})
            rows.append(rec)
            write_image(prev, draw_contour_preview(img, contour))

    safe_to_csv(pd.DataFrame(rows), contornos_dir(root) / "contornos_cranio_todos_anos.csv")
    safe_to_csv(pd.DataFrame(fails), contornos_dir(root) / "falhas_contornos_cranio.csv")
    print(f"[OK] contornos: {len(rows)} ok, {len(fails)} falhas")


if __name__ == "__main__":
    main()
