# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

import pandas as pd

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

    trip = build_triplet_index(root, args.years)
    for _, rec in trip.iterrows():
        if not bool(rec.get("tem_tripleta", False)):
            continue
        for side in SIDES:
            path = Path(rec.get(f"imagem_{side.lower()}", ""))
            out = sexo_dir(root) / "silhuetas_256" / str(rec["ano"]) / side / f"{path.stem}_silhueta256.png"
            if out.exists() and not args.force:
                rows.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "silhueta": str(out), "status": "ja_existia"})
                continue
            img = read_image(path)
            if img is None:
                fails.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "erro": "imagem nao abre"})
                continue
            contour = largest_skull_contour(img)
            if contour is None:
                fails.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "erro": "contorno nao encontrado"})
                continue
            sil = make_silhouette256(img, contour)
            if side == "CE":
                sil = cv2.flip(sil, 1)
            write_image(out, sil)
            rows.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "id_imagem": path.stem, "lado": side, "imagem": str(path), "silhueta": str(out), "status": "OK"})

    safe_to_csv(pd.DataFrame(rows), sexo_dir(root) / "silhuetas_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(fails), sexo_dir(root) / "falhas_silhuetas_sexo_cranio.csv")
    print(f"[OK] silhuetas sexo cranio: {len(rows)}")


if __name__ == "__main__":
    main()
