# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os

import pandas as pd

from common_cranios import *


def prepare_silhouettes_for_kind(root: Path, years, force: bool, norm_kind: str) -> tuple[list[dict], list[dict]]:
    if norm_kind == NORMALIZATION_ORIGINAL:
        os.environ.pop("CRANIOS_NORMALIZATION_KIND", None)
    else:
        os.environ["CRANIOS_NORMALIZATION_KIND"] = norm_kind
    rows, fails = [], []
    trip = build_triplet_index(root, years)
    for _, rec in trip.iterrows():
        if not bool(rec.get("tem_tripleta", False)):
            continue
        for side in SIDES:
            path = Path(rec.get(f"imagem_{side.lower()}", ""))
            out = sexo_dir(root) / "silhuetas_256" / norm_kind / str(rec["ano"]) / side / f"{path.stem}_silhueta256.png"
            if out.exists() and not force:
                rows.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "normalizacao": norm_kind, "silhueta": str(out), "status": "ja_existia"})
                continue
            img = read_image(path)
            if img is None:
                fails.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "normalizacao": norm_kind, "erro": "imagem nao abre"})
                continue
            contour = largest_skull_contour(img)
            if contour is None:
                fails.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "lado": side, "normalizacao": norm_kind, "erro": "contorno nao encontrado"})
                continue
            sil = make_silhouette256(img, contour)
            if side == "CE":
                sil = cv2.flip(sil, 1)
            write_image(out, sil)
            rows.append({"ano": rec["ano"], "id_pessoa": rec["id_pessoa"], "id_imagem": path.stem, "lado": side, "normalizacao": norm_kind, "imagem": str(path), "silhueta": str(out), "status": "OK"})
    return rows, fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use-normalized", action="store_true")
    ap.add_argument("--normalization-kind", default="original", choices=["all", NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS])
    args = ap.parse_args()
    if args.use_normalized:
        args.normalization_kind = "atual"
    root = as_root(args.root)
    ensure_base_dirs(root)
    rows, fails = [], []
    kinds = [NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS] if args.normalization_kind == "all" else [args.normalization_kind]
    for norm_kind in kinds:
        print(f"[INFO] preparando silhuetas: normalizacao={norm_kind}", flush=True)
        r, f = prepare_silhouettes_for_kind(root, args.years, args.force, norm_kind)
        rows.extend(r)
        fails.extend(f)

    safe_to_csv(pd.DataFrame(rows), sexo_dir(root) / "silhuetas_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(fails), sexo_dir(root) / "falhas_silhuetas_sexo_cranio.csv")
    print(f"[OK] silhuetas sexo cranio: {len(rows)}")


if __name__ == "__main__":
    main()
