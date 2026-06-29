# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import pandas as pd

from common_cranios import *


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    med = read_csv_safe(medidas_dir(root) / "medidas_cranio_todos_anos.csv")
    if med.empty:
        print("[ERRO] Rode medidas antes.")
        return
    rows = []
    for _, r in med.iterrows():
        status, usar, motivo, score = quality_from_row(r)
        row = {
            "ano": str(r.get("ano", "")),
            "id_imagem": str(r.get("id_imagem", "")),
            "id_pessoa": str(r.get("id_pessoa", "")),
            "status_cranio": status,
            "usar_matching": bool(usar),
            "motivo_qualidade_cranio": motivo,
            "score_qualidade_cranio": round(score, 3),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    safe_to_csv(out, qualidade_dir(root) / "qualidade_cranios.csv")
    safe_to_csv(out.groupby(["status_cranio", "usar_matching"], dropna=False).size().reset_index(name="qtd"), qualidade_dir(root) / "resumo_qualidade_cranios.csv")
    print(f"[OK] qualidade cranios: {len(out)}")


if __name__ == "__main__":
    main()
