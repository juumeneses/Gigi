from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .paths import PROJECT_DIR, REPO_ROOT, ensure_project_dirs, relative_repo_path, repo_path


FACE_FEATURES = REPO_ROOT / "Treinamento_faces/resultados/features_matching_faces/features_matching_faces_com_genero.csv"
FACE_SEX = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv"
SKULL_FEATURES = REPO_ROOT / "Treinamento_cranios/resultados/features_matching_cranios/features_matching_cranios_final.csv"
SKULL_SEX = REPO_ROOT / "Treinamento_cranios/resultados/sexo_cranio/predicoes_sexo_cranio.csv"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio ausente: {path}")
    df = pd.read_csv(path)
    if "ano" in df:
        df["ano"] = df["ano"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def face_person_id(image_id: str) -> str:
    return re.sub(r"_(FFN|FN|FACE)$", "", str(image_id), flags=re.IGNORECASE)


def _sex_label(value: object) -> str:
    text = str(value or "").upper()
    if "MASCULINO" in text:
        return "MASCULINO"
    if "FEMININO" in text:
        return "FEMININO"
    return "INDEFINIDO"


def _existing_face_path(row: pd.Series) -> str:
    for key in ("imagem_corrigida", "imagem_original"):
        resolved = repo_path(row.get(key))
        if resolved and resolved.exists():
            return relative_repo_path(resolved)
    year, image_id = str(row["ano"]), str(row["id_imagem"])
    folder = REPO_ROOT / "Treinamento_faces/dados_originais/faces" / year / "images"
    matches = sorted(folder.glob(f"{image_id}.*"))
    return relative_repo_path(matches[0]) if matches else ""


def validate_metadata(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"pair_key", "ano", "id_pessoa", "face_path", "cranio_cf_path", "usar_treino"}
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"colunas ausentes: {missing}")
    if "pair_key" in df and df["pair_key"].duplicated().any():
        errors.append(f"pair_key duplicada: {int(df['pair_key'].duplicated().sum())}")
    if "usar_treino" in df:
        valid = df[df["usar_treino"].fillna(False)]
        for col in ("face_path", "cranio_cf_path"):
            absent = sum(not bool(repo_path(v) and repo_path(v).exists()) for v in valid[col])
            if absent:
                errors.append(f"{absent} arquivos ausentes em {col} entre pares validos")
    return errors


def build_metadata(output: Path | None = None) -> tuple[pd.DataFrame, dict]:
    ensure_project_dirs()
    faces = _read(FACE_FEATURES)
    curated = _read(FACE_SEX)
    skulls = _read(SKULL_FEATURES)
    skull_sex = _read(SKULL_SEX)

    faces["id_pessoa"] = faces["id_imagem"].map(face_person_id)
    # O CSV de features ja contem a curadoria, mas este merge torna a precedencia explicita.
    ccols = [c for c in ("ano", "id_imagem", "genero_final", "origem_genero_final", "status_genero_manual") if c in curated]
    faces = faces.drop(columns=[c for c in ("genero_final", "origem_genero_final") if c in faces]).merge(
        curated[ccols].drop_duplicates(["ano", "id_imagem"]), on=["ano", "id_imagem"], how="left"
    )
    faces["face_path"] = faces.apply(_existing_face_path, axis=1)
    faces["face_ok"] = faces.get("usar_matching", False).fillna(False).astype(bool)

    skulls = skulls.drop(columns=[c for c in ("prob_masculino_cranio", "prob_feminino_cranio", "classe_cranio", "confianca_cranio", "modelo_sexo_cranio") if c in skulls])
    skulls = skulls.merge(skull_sex.drop(columns=[c for c in ("id_imagem_cranio",) if c in skull_sex]), on=["ano", "id_pessoa"], how="left")
    skulls["cranio_ok"] = skulls.get("usar_matching", False).fillna(False).astype(bool)

    fkeep = [c for c in ("ano", "id_pessoa", "id_imagem", "face_path", "face_ok", "status_final", "status_manual", "qualidade_landmarks", "genero_final", "origem_genero_final", "prob_masculino_fairface", "prob_feminino_fairface") if c in faces]
    ckeep = [c for c in ("ano", "id_pessoa", "id_imagem", "imagem_cf", "imagem_ce", "imagem_cd", "cranio_ok", "status_cranio", "score_qualidade_cranio", "tem_tripleta", "classe_cranio", "confianca_cranio", "prob_masculino_cranio", "prob_feminino_cranio", "modelo_sexo_cranio") if c in skulls]
    merged = faces[fkeep].merge(skulls[ckeep], on=["ano", "id_pessoa"], how="outer", suffixes=("_face", "_cranio"), indicator=True)
    merged = merged.rename(columns={"id_imagem_face": "id_face", "id_imagem_cranio": "id_cranio_cf", "genero_final": "sexo_face", "classe_cranio": "sexo_cranio"})
    for source, target in (("imagem_cf", "cranio_cf_path"), ("imagem_ce", "cranio_ce_path"), ("imagem_cd", "cranio_cd_path")):
        merged[target] = merged.get(source, "").map(relative_repo_path)
    merged["pair_key"] = merged["ano"].astype(str) + "::" + merged["id_pessoa"].astype(str)
    merged["sexo_face"] = merged.get("sexo_face", "INDEFINIDO").map(_sex_label)
    merged["sexo_cranio"] = merged.get("sexo_cranio", "INDEFINIDO").map(_sex_label)
    merged["sexo_concordante"] = (merged["sexo_face"] == merged["sexo_cranio"]) & merged["sexo_face"].isin(["MASCULINO", "FEMININO"])
    merged["face_exists"] = merged["face_path"].map(lambda v: bool(repo_path(v) and repo_path(v).exists()))
    merged["cranio_cf_exists"] = merged["cranio_cf_path"].map(lambda v: bool(repo_path(v) and repo_path(v).exists()))
    face_ok = merged.get("face_ok", pd.Series(False, index=merged.index)).eq(True)
    cranio_ok = merged.get("cranio_ok", pd.Series(False, index=merged.index)).eq(True)
    merged["usar_treino"] = face_ok & cranio_ok & merged["face_exists"] & merged["cranio_cf_exists"] & merged["_merge"].eq("both")
    merged["exclude_reason"] = ""
    merged.loc[merged["_merge"].eq("left_only"), "exclude_reason"] = "SEM_CRANIO"
    merged.loc[merged["_merge"].eq("right_only"), "exclude_reason"] = "SEM_FACE"
    merged.loc[~face_ok & merged["exclude_reason"].eq(""), "exclude_reason"] = "FACE_NAO_APROVADA"
    merged.loc[~cranio_ok & merged["exclude_reason"].eq(""), "exclude_reason"] = "CRANIO_NAO_APROVADO"
    merged.loc[~merged["face_exists"] & merged["exclude_reason"].eq(""), "exclude_reason"] = "ARQUIVO_FACE_AUSENTE"
    merged.loc[~merged["cranio_cf_exists"] & merged["exclude_reason"].eq(""), "exclude_reason"] = "ARQUIVO_CRANIO_AUSENTE"
    merged = merged.drop(columns=[c for c in ("imagem_cf", "imagem_ce", "imagem_cd", "_merge") if c in merged]).sort_values(["ano", "id_pessoa"]).reset_index(drop=True)

    errors = validate_metadata(merged)
    digest = hashlib.sha256(merged.to_csv(index=False).encode()).hexdigest()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": digest,
        "rows": len(merged), "valid_pairs": int(merged["usar_treino"].sum()),
        "excluded": merged.loc[~merged["usar_treino"], "exclude_reason"].value_counts(dropna=False).to_dict(),
        "valid_by_year": merged[merged["usar_treino"]].groupby("ano").size().to_dict(),
        "validation_errors": errors,
    }
    output = output or PROJECT_DIR / "data/manifests/metadata_pares.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    output.with_suffix(".audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("Manifesto invalido: " + "; ".join(errors))
    return merged, report
