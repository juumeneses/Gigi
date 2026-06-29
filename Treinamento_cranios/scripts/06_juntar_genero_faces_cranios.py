# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re

import pandas as pd

from common_cranios import *


def col_or_default(df: pd.DataFrame, name: str, default) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def face_person_id(stem: str) -> str:
    s = str(stem)
    s = re.sub(r"[_-]?(FFN|FACE|FRONTAL|FOTO|IMG)$", "", s, flags=re.I)
    return person_id_from_stem(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--faces-root", default="../Treinamento_faces")
    args = ap.parse_args()
    root = as_root(args.root)
    faces_root = (root / args.faces_root).resolve() if not Path(args.faces_root).is_absolute() else Path(args.faces_root)
    ensure_base_dirs(root)

    genero = read_csv_safe(faces_root / "resultados" / "genero_faces" / "genero_faces_curado.csv")
    med = read_csv_safe(medidas_dir(root) / "medidas_cranio_todos_anos.csv")
    shape = read_csv_safe(features_dir(root) / "shape_cranios_64pts.csv")
    qual = read_csv_safe(qualidade_dir(root) / "qualidade_cranios.csv")
    sil = read_csv_safe(sexo_dir(root) / "silhuetas_sexo_cranio.csv")
    if genero.empty or med.empty or shape.empty or qual.empty:
        print("[ERRO] Faltam genero faces, medidas, shape ou qualidade.")
        return
    genero = genero.copy()
    for c in ["ano", "id_imagem"]:
        if c in genero.columns:
            genero[c] = genero[c].astype(str)
    genero["id_pessoa"] = genero["id_imagem"].astype(str).map(face_person_id)
    genero = genero[col_or_default(genero, "genero_final", "").astype(str).str.upper().isin(["MASCULINO", "FEMININO"])].copy()
    if "usar_para_treino_genero" in genero.columns:
        genero = genero[genero["usar_para_treino_genero"].astype(str).str.upper().isin(["TRUE", "1", "SIM", "YES"])]
    for frame in [med, shape, qual, sil]:
        for c in ["ano", "id_imagem", "id_pessoa"]:
            if c in frame.columns:
                frame[c] = frame[c].astype(str)
    df = med.merge(shape, on=["ano", "id_imagem", "id_pessoa"], how="left")
    df = df.merge(qual, on=["ano", "id_imagem", "id_pessoa"], how="left")
    df = df.merge(genero, on=["ano", "id_pessoa"], how="inner", suffixes=("_cranio", "_face"))
    if not sil.empty:
        counts = sil[sil["status"].astype(str).str.upper().isin(["OK", "JA_EXISTIA"])].groupby(["ano", "id_pessoa"])["lado"].nunique().reset_index(name="vistas_ok")
        df = df.merge(counts, on=["ano", "id_pessoa"], how="left")
    else:
        df["vistas_ok"] = 0
    ok_cranio = col_or_default(df, "status_cranio", "").astype(str).str.upper().eq("OK")
    usar_matching = col_or_default(df, "usar_matching", True).astype(str).str.upper().isin(["TRUE", "1", "SIM", "YES"])
    vistas_ok = col_or_default(df, "vistas_ok", 0).fillna(0).astype(float).ge(3)
    df = df[ok_cranio & usar_matching & vistas_ok].copy()
    safe_to_csv(df, sexo_dir(root) / "dataset_treino_sexo_cranio.csv")
    for old_name in [
        "metricas_modelo_sexo_cranio.csv",
        "matriz_confusao_sexo_cranio.csv",
        "predicoes_sexo_cranio_por_vista.csv",
        "predicoes_sexo_cranio.csv",
        "comparacao_genero_face_cranio.csv",
        "modelo_sexo_cranio_vgg16.keras",
        "modelo_sexo_cranio_vgg16_head.keras",
        "modelo_sexo_cranio_sklearn.pkl",
    ]:
        old_path = sexo_dir(root) / old_name
        if old_path.exists():
            old_path.unlink()
    print(f"[OK] dataset sexo cranio: {len(df)}")


if __name__ == "__main__":
    main()
