# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import pandas as pd

from common_cranios import *


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in ["ano", "id_imagem", "id_pessoa"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    med = normalize_keys(read_csv_safe(medidas_dir(root) / "medidas_cranio_todos_anos.csv"))
    shape = normalize_keys(read_csv_safe(features_dir(root) / "shape_cranios_64pts.csv"))
    qual = normalize_keys(read_csv_safe(qualidade_dir(root) / "qualidade_cranios.csv"))
    pred = normalize_keys(read_csv_safe(sexo_dir(root) / "predicoes_sexo_cranio.csv"))
    trip = normalize_keys(complete_triplets(root))
    if med.empty or shape.empty:
        print("[ERRO] Faltam medidas ou shape.")
        return
    df = med.merge(shape, on=["ano", "id_imagem", "id_pessoa"], how="left")
    if not qual.empty:
        df = df.merge(qual, on=["ano", "id_imagem", "id_pessoa"], how="left")
    if not pred.empty:
        df = df.merge(pred.drop(columns=[c for c in ["id_imagem_cranio"] if c in pred.columns]), on=["ano", "id_pessoa"], how="left")
    if not trip.empty:
        df = df.merge(trip, on=["ano", "id_pessoa"], how="left", suffixes=("", "_tripleta"))
        df = df[df["tem_tripleta"].fillna(False).astype(bool)].copy()
    else:
        df = df.iloc[0:0].copy()
    if "usar_matching" not in df.columns:
        df["usar_matching"] = True
    for c in ["prob_masculino_cranio", "prob_feminino_cranio"]:
        if c not in df.columns:
            df[c] = 0.5
    if "classe_cranio" not in df.columns:
        df["classe_cranio"] = "INDEFINIDO"
    if "confianca_cranio" not in df.columns:
        df["confianca_cranio"] = "INDEFINIDO"
    safe_to_csv(df, features_dir(root) / "features_matching_cranios_final.csv")
    print(f"[OK] export cranios final: {len(df)}")


if __name__ == "__main__":
    main()
