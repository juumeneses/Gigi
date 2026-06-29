# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from common_cranios import *


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)

    rows, probs = [], []
    years = discover_years(root, args.years)
    if not years:
        probs.append({"ano": "", "id_imagem": "", "problema": "nenhum ano encontrado em dados_originais/cranios"})

    for y in years:
        for side in SIDES:
            d = side_dir(root, y, side)
            if not d.exists():
                probs.append({"ano": y, "id_imagem": "", "lado": side, "problema": "pasta ausente"})
                continue
            files = list_images(root, y, side)
            stems = Counter(p.stem for p in files)
            for p in files:
                img = read_image(p)
                status = "OK" if img is not None else "FALHA"
                if img is None:
                    probs.append({"ano": y, "id_imagem": p.stem, "lado": side, "problema": "imagem nao abre", "caminho": str(p)})
                if stems[p.stem] > 1:
                    probs.append({"ano": y, "id_imagem": p.stem, "lado": side, "problema": "nome duplicado", "caminho": str(p)})
                rows.append({
                    "ano": str(y),
                    "id_imagem": p.stem,
                    "id_pessoa": person_id_from_stem(p.stem),
                    "lado": side,
                    "arquivo": p.name,
                    "caminho": str(p),
                    "extensao": p.suffix.lower(),
                    "status": status,
                })

    trip = build_triplet_index(root, years)
    if not trip.empty:
        for _, r in trip.iterrows():
            if not bool(r.get("tem_tripleta", False)):
                faltando = [s for s in SIDES if not r.get(f"imagem_{s.lower()}")]
                probs.append({"ano": r.get("ano"), "id_pessoa": r.get("id_pessoa"), "problema": "tripleta incompleta", "faltando": ",".join(faltando)})
        safe_to_csv(trip, checks_dir(root) / "tripletas_cranios.csv")

    safe_to_csv(pd.DataFrame(rows), checks_dir(root) / "check_cranios.csv")
    safe_to_csv(pd.DataFrame(probs), checks_dir(root) / "problemas_cranios.csv")
    print(f"[OK] check cranios: {len(rows)} imagens")


if __name__ == "__main__":
    main()
