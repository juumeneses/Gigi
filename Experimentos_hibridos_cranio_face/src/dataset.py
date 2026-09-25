from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import LAB_ROOT, REPO_ROOT, abs_repo, bool_series, read_csv, rel_repo, save_csv, sha256_file


CRANIO_ROOT = REPO_ROOT / "Treinamento_cranios"
FACE_ROOT = REPO_ROOT / "Treinamento_faces"


def _person_from_face(stem: str) -> str:
    for suffix in ("_FFN", "_FN", "_FACE"):
        if stem.upper().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_sources() -> tuple[pd.DataFrame, ...]:
    trip = read_csv(CRANIO_ROOT / "resultados/checks/tripletas_cranios.csv")
    ff = read_csv(FACE_ROOT / "resultados/features_matching_faces/features_matching_faces_final.csv")
    fq = read_csv(FACE_ROOT / "resultados/curadoria_matching/dataset_matching_faces.csv")
    fg = read_csv(FACE_ROOT / "resultados/genero_faces/genero_faces_curado.csv")
    cq = read_csv(CRANIO_ROOT / "resultados/qualidade_cranio/qualidade_cranios.csv")
    cg = read_csv(CRANIO_ROOT / "resultados/sexo_cranio/predicoes_sexo_cranio.csv")
    return trip, ff, fq, fg, cq, cg


def build_master_manifest(years: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    out = LAB_ROOT / "dataset/manifestos/manifesto_mestre.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    trip, ff, fq, fg, cq, cg = _load_sources()
    if trip.empty:
        raise RuntimeError("tripletas_cranios.csv nao encontrado ou vazio")
    if years:
        trip = trip[trip["ano"].astype(int).isin([int(y) for y in years])].copy()

    base = trip.copy()
    for col in ["imagem_cf", "imagem_ce", "imagem_cd"]:
        if col in base:
            base[col] = base[col].map(rel_repo)
    if not ff.empty:
        keep = [c for c in ["id_pessoa", "id_imagem", "imagem_face", "usar_matching"] if c in ff]
        fx = ff[keep].rename(columns={"id_imagem": "id_imagem_face", "usar_matching": "usar_matching_face"})
        fx["imagem_face"] = fx["imagem_face"].map(rel_repo)
        base = base.merge(fx, on="id_pessoa", how="left")
    if not fq.empty:
        q = fq[[c for c in ["id_imagem", "status_final", "qualidade_landmarks"] if c in fq]].rename(
            columns={"id_imagem": "id_imagem_face", "status_final": "status_face", "qualidade_landmarks": "qualidade_face"})
        base = base.merge(q, on="id_imagem_face", how="left")
    if not fg.empty:
        g = fg[[c for c in ["id_imagem", "genero_final", "origem_genero_final", "confianca_fairface"] if c in fg]].rename(columns={"id_imagem": "id_imagem_face"})
        base = base.merge(g, on="id_imagem_face", how="left")
    if not cq.empty:
        q = cq[[c for c in ["id_pessoa", "status_cranio", "usar_matching", "score_qualidade_cranio"] if c in cq]].rename(columns={"usar_matching": "usar_matching_cranio"})
        base = base.merge(q, on="id_pessoa", how="left")
    if not cg.empty:
        g = cg[[c for c in ["id_pessoa", "prob_masculino_cranio", "prob_feminino_cranio", "classe_cranio", "confianca_cranio"] if c in cg]]
        base = base.merge(g, on="id_pessoa", how="left")

    landmark_files = {}
    for p in (FACE_ROOT / "dados_originais/faces").glob("*/marcacoes_landmarks/landmarks_csv/*_landmarks.csv"):
        landmark_files[p.stem.replace("_landmarks", "")] = rel_repo(p)
    base["landmarks_face_468_path"] = base.get("id_imagem_face", pd.Series(index=base.index, dtype=str)).map(landmark_files).fillna("")
    base["tem_landmarks_face_468"] = base["landmarks_face_468_path"].ne("")
    base["tem_face"] = base.get("imagem_face", "").astype(str).ne("")
    base["tem_cranio_cf"] = base["imagem_cf"].astype(str).ne("")
    face_ok = bool_series(base.get("usar_matching_face", pd.Series(False, index=base.index)))
    skull_ok = bool_series(base.get("usar_matching_cranio", pd.Series(False, index=base.index)))
    base["usar_matching"] = face_ok & skull_ok & base["tem_face"] & base["tem_cranio_cf"]
    base["motivo_exclusao"] = np.where(~base["tem_face"], "SEM_FACE", np.where(~base["tem_cranio_cf"], "SEM_CRANIO_CF", np.where(~face_ok, "FACE_NAO_ELEGIVEL", np.where(~skull_ok, "CRANIO_NAO_ELEGIVEL", ""))))

    for col in ["imagem_face", "imagem_cf", "imagem_ce", "imagem_cd"]:
        sizes, hashes, exists = [], [], []
        for value in base.get(col, pd.Series("", index=base.index)):
            p = abs_repo(value)
            ok = bool(p and p.exists())
            exists.append(ok); sizes.append(p.stat().st_size if ok else np.nan); hashes.append(sha256_file(p) if ok else "")
        base[f"{col}_existe"] = exists; base[f"{col}_bytes"] = sizes; base[f"{col}_sha256"] = hashes
    base = base.sort_values(["ano", "id_pessoa"]).drop_duplicates("id_pessoa").reset_index(drop=True)
    save_csv(base, out)
    return base


def _strata(df: pd.DataFrame) -> pd.Series:
    gender = df.get("genero_final", pd.Series("INDEFINIDO", index=df.index)).fillna("INDEFINIDO").astype(str)
    return df["ano"].astype(str) + "_" + gender


def build_locked_splits(manifest: pd.DataFrame, seed: int, force: bool = False) -> pd.DataFrame:
    out = LAB_ROOT / "dataset/splits/split_identidades.csv"
    if out.exists() and not force:
        saved = pd.read_csv(out)
        if saved["id_pessoa"].duplicated().any():
            raise RuntimeError("split salvo contem identidades duplicadas")
        return saved
    eligible = manifest[manifest["usar_matching"]].copy()
    strata = _strata(eligible)
    counts = strata.value_counts(); strata = strata.where(strata.map(counts) >= 3, eligible["ano"].astype(str))
    train, temp = train_test_split(eligible, test_size=0.30, random_state=seed, stratify=strata)
    tstrata = _strata(temp); tc = tstrata.value_counts(); tstrata = tstrata.where(tstrata.map(tc) >= 2, temp["ano"].astype(str))
    try:
        val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=tstrata)
    except ValueError:
        val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    rows = []
    for name, frame in [("treino", train), ("validacao", val), ("teste", test)]:
        part = frame[["id_pessoa", "ano"]].copy(); part["split"] = name; rows.append(part)
    result = pd.concat(rows).sort_values(["split", "ano", "id_pessoa"]).reset_index(drop=True)
    save_csv(result, out)
    return result


def create_link_tree(manifest: pd.DataFrame, force: bool = False) -> None:
    for _, row in manifest.iterrows():
        year = str(row["ano"])
        mapping = {
            LAB_ROOT / "dataset/faces" / year: row.get("imagem_face", ""),
            LAB_ROOT / "dataset/cranios" / year / "CF": row.get("imagem_cf", ""),
            LAB_ROOT / "dataset/cranios" / year / "CE": row.get("imagem_ce", ""),
            LAB_ROOT / "dataset/cranios" / year / "CD": row.get("imagem_cd", ""),
        }
        for folder, value in mapping.items():
            src = abs_repo(value)
            if not src or not src.exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / src.name
            if dst.exists() or dst.is_symlink():
                if not force:
                    continue
                dst.unlink()
            relative = os.path.relpath(src, start=folder)
            dst.symlink_to(relative)


def build_audit(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    joined = manifest.merge(splits[["id_pessoa", "split"]], on="id_pessoa", how="left")
    rows = []
    for (year, split), g in joined.groupby(["ano", "split"], dropna=False):
        rows.append({"ano": year, "split": split, "identidades": len(g), "elegiveis": int(g["usar_matching"].sum()), "tripletas": int(bool_series(g["tem_tripleta"]).sum())})
    return save_csv(pd.DataFrame(rows), LAB_ROOT / "resultados/auditoria/resumo_dataset.csv") and joined
