from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps, ImageStat

from .paths import PROJECT_DIR, REPO_ROOT, repo_path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

FACE_METRICS = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/metricas_modelo_genero_faces.csv"
FACE_CONFUSION = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/matriz_confusao_genero_faces.csv"
FACE_COMPARISON = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/comparacao_genero_fairface_modelo.csv"
FACE_CURATED = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv"
FACE_MATCHING = REPO_ROOT / "Treinamento_faces/resultados/curadoria_matching/dataset_matching_faces.csv"
FACE_FEATURES = REPO_ROOT / "Treinamento_faces/resultados/features_matching_faces/features_matching_faces_com_genero.csv"

SKULL_METRICS = REPO_ROOT / "Treinamento_cranios/resultados/sexo_cranio/metricas_modelo_sexo_cranio.csv"
SKULL_VIEW_METRICS = REPO_ROOT / "Treinamento_cranios/resultados/sexo_cranio/metricas_modelo_sexo_cranio_por_vista.csv"
SKULL_CONFUSION = REPO_ROOT / "Treinamento_cranios/resultados/sexo_cranio/matriz_confusao_sexo_cranio.csv"
SKULL_NORMALIZATIONS = REPO_ROOT / "Treinamento_cranios/resultados/sexo_cranio/comparativo_normalizacoes.csv"
SKULL_FEATURES = REPO_ROOT / "Treinamento_cranios/resultados/features_matching_cranios/features_matching_cranios_final.csv"

METADATA = PROJECT_DIR / "data/manifests/metadata_pares.csv"
SPLITS = PROJECT_DIR / "data/splits/all_splits.csv"
BASELINES = PROJECT_DIR / "reports/baselines/metricas_baselines.csv"
LOYO = PROJECT_DIR / "studies/leave_one_year_out/summary.csv"
VIEW_ABLATION = PROJECT_DIR / "studies/view_ablation/summary.csv"
GALLERY10 = PROJECT_DIR / "reports/gallery_10_ensemble_domain_normalized_hard_negatives/metrics.csv"
REPORT_MD = REPO_ROOT / "docs/relatorio_estado_repositorio_gigi.md"
REPORT_HTML = REPO_ROOT / "docs/relatorio_estado_repositorio_gigi_com_imagens.html"


PIPELINE_COMMANDS: dict[str, list[tuple[str, str, list[str]]]] = {
    "Face": [
        ("Checar dataset", "Treinamento_faces/resultados/checks/check_faces.csv", ["Treinamento_faces/scripts/00_checar_dataset_faces.py"]),
        ("Extrair landmarks", "Treinamento_faces/dados_originais/faces/2017/marcacoes_landmarks/previews", ["Treinamento_faces/scripts/01_extrair_landmarks_faces.py"]),
        ("Gerar medidas", "Treinamento_faces/resultados/analises_faces/analise_padroes_faces_todos_anos.csv", ["Treinamento_faces/scripts/02_gerar_medidas_faces.py"]),
        ("Analisar qualidade", "Treinamento_faces/resultados/curadoria_matching/dataset_matching_faces.csv", ["Treinamento_faces/scripts/curadoria_matching_faces.py"]),
        ("FairFace", "Treinamento_faces/resultados/genero_faces/predicoes_genero_fairface.csv", ["Treinamento_faces/scripts/06_predizer_genero_fairface_faces.py"]),
        ("Curadoria sexo estimado", "Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv", ["Treinamento_faces/scripts/07_curadoria_genero_faces.py"]),
        ("Treinar sexo da face", "Treinamento_faces/resultados/genero_faces/metricas_modelo_genero_faces.csv", ["Treinamento_faces/scripts/09_treinar_modelo_genero_faces.py"]),
        ("Exportar features", "Treinamento_faces/resultados/features_matching_faces/features_matching_faces_com_genero.csv", ["Treinamento_faces/scripts/11_exportar_matching_faces_com_genero.py"]),
    ],
    "Crânio": [
        ("Checar tripletas", "Treinamento_cranios/resultados/checks/tripletas_cranios.csv", ["Treinamento_cranios/scripts/00_checar_dataset_cranios.py"]),
        ("Extrair contorno", "Treinamento_cranios/resultados/contornos_cranio", ["Treinamento_cranios/scripts/01_extrair_contorno_cranios.py"]),
        ("Gerar shape64", "Treinamento_cranios/resultados/features_matching_cranios/shape_cranios_64pts.csv", ["Treinamento_cranios/scripts/02_gerar_shape_cranios_64pts.py"]),
        ("Gerar medidas", "Treinamento_cranios/resultados/medidas_cranio/medidas_cranios.csv", ["Treinamento_cranios/scripts/03_gerar_medidas_cranios.py"]),
        ("Analisar qualidade", "Treinamento_cranios/resultados/qualidade_cranio", ["Treinamento_cranios/scripts/04_analisar_qualidade_cranios.py"]),
        ("Silhuetas", "Treinamento_cranios/resultados/sexo_cranio/silhuetas_sexo_cranio.csv", ["Treinamento_cranios/scripts/05_preparar_silhuetas_sexo_cranio.py"]),
        ("Treinar sexo craniano", "Treinamento_cranios/resultados/sexo_cranio/metricas_modelo_sexo_cranio.csv", ["Treinamento_cranios/scripts/07_treinar_sexo_cranio.py"]),
        ("Exportar features", "Treinamento_cranios/resultados/features_matching_cranios/features_matching_cranios_final.csv", ["Treinamento_cranios/scripts/08_exportar_features_cranios_final.py"]),
    ],
    "Matching": [
        ("Construir metadata", "Modelo_novo/data/manifests/metadata_pares.csv", ["Modelo_novo/cli.py", "build-index"]),
        ("Criar splits", "Modelo_novo/data/splits/all_splits.csv", ["Modelo_novo/cli.py", "create-splits"]),
        ("Preprocessar", "Modelo_novo/data/processed", ["Modelo_novo/cli.py", "preprocess"]),
        ("Baselines", "Modelo_novo/reports/baselines/metricas_baselines.csv", ["Modelo_novo/cli.py", "baselines"]),
        ("Treinar two-tower", "Modelo_novo/runs", ["Modelo_novo/cli.py", "train"]),
        ("Gallery-10", "Modelo_novo/reports/gallery_10_ensemble_domain_normalized_hard_negatives/metrics.csv", ["Modelo_novo/cli.py", "simulate-gallery", "--size", "10"]),
    ],
}


LITERATURE_ROWS = [
    {
        "referencia": "MEPROCS Handbook / Framework",
        "aplicacao": "Protocolo, incerteza, qualidade, vistas e limites.",
        "teste_extraido": "Status por etapa, entrada/saída, relatório e checklist metodológico.",
        "onde": "Pipeline, Testes, Literatura, Relatórios",
    },
    {
        "referencia": "Relationships Between Skull and Face",
        "aplicacao": "Regiões automáticas úteis: órbitas, nariz, zigomático, mandíbula e contorno.",
        "teste_extraido": "Crop orbitonasal, mid-face, shape64 facial e contorno craniano.",
        "onde": "Extrações, Faces, Crânios, Erros",
    },
    {
        "referencia": "Craniofacial Superimposition Techniques",
        "aplicacao": "Pose, escala, orientação e preservação de proporção.",
        "teste_extraido": "Normalização, centralização, rotação e original vs normalizado.",
        "onde": "Testes, Imagens",
    },
    {
        "referencia": "Photographic Variables / Focus Distance / Perspective Distortion",
        "aplicacao": "Fotografia 2D altera escala, foco, perspectiva e registro.",
        "teste_extraido": "Blur, contraste, resolução, área útil, proporções e alerta de perspectiva.",
        "onde": "Testes, Faces, Crânios, Erros",
    },
    {
        "referencia": "Skull-Face Correspondence Criteria",
        "aplicacao": "Critérios clássicos viram diagnósticos automáticos, sem marcação manual.",
        "teste_extraido": "Contorno, assimetria, mid-face, mandíbula e forma geral.",
        "onde": "Extrações, Matching, Erros",
    },
    {
        "referencia": "Soft Tissue Thickness Reviews",
        "aplicacao": "Tecido mole varia por pessoa/população; não é regra rígida.",
        "teste_extraido": "Avisos e limitações; não usar espessura como decisão automática.",
        "onde": "Literatura, Relatórios",
    },
    {
        "referencia": "FaceNet / Metric Learning / Cross-modal Retrieval",
        "aplicacao": "Embeddings L2, two-tower, InfoNCE/Triplet e ranking.",
        "teste_extraido": "Top-K, MRR, CMC, hard negatives, baselines e sexo ligado/desligado.",
        "onde": "Modelo Novo, Métricas, Matching",
    },
]


EXTRACTION_ROWS = [
    ("Face original aprovada", "RGB, dimensão, resolução, blur, contraste, iluminação, centralização, área relativa e margem", "Auditoria, fallback, comparação antes/depois"),
    ("Landmarks automáticos", "MediaPipe, inclinação dos olhos, proporções, shape64, preview e status", "Alinhamento, qualidade, crop, baseline geométrico"),
    ("Face normalizada", "Alinhada, redimensionada, RGB/cinza, fundo reduzido quando possível", "Entrada principal da torre facial"),
    ("Crop central/orbitonasal", "Olhos, ponte/base nasal, região malar e terço médio", "Embedding local e diagnóstico de perspectiva/pose"),
    ("Crop FairFace", "Crop, classe, probabilidade M/F, confiança e origem", "Sexo estimado, filtro opcional, penalização e ablação"),
    ("Crânio CF", "RGB, dimensão, resolução, blur, contraste, iluminação, centralização e área útil", "Entrada principal craniana e auditoria"),
    ("Crânios CE/CD", "RGB, qualidade, status de vista, consistência e embeddings laterais", "Multivista e ablação CF vs CF+CE+CD"),
    ("Crânio normalizado", "Normalizado, cinza, fundo padronizado e variantes", "Torre craniana e robustez por ano"),
    ("Contorno craniano", "Maior contorno, shape64, bbox, centroide, área, perímetro, assimetria, compacidade", "Baseline shape64, feature híbrida e diagnóstico"),
    ("Silhueta craniana", "Máscara binária, silhueta 256, área preenchida e proporção fundo/crânio", "Sexo craniano, ablação e qualidade"),
    ("Sexo craniano", "Classe, probabilidade M/F, confiança, modelo e previsão por vista", "Diagnóstico, filtro opcional, penalização e ablação"),
]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def exists_rel(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


def file_info(rel_path: str) -> dict:
    path = REPO_ROOT / rel_path
    return {
        "arquivo": rel_path,
        "existe": path.exists(),
        "modificado": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)) if path.exists() else "",
        "bytes": path.stat().st_size if path.exists() and path.is_file() else "",
    }


def pipeline_status(section: str) -> pd.DataFrame:
    rows = []
    for name, output, command in PIPELINE_COMMANDS[section]:
        info = file_info(output)
        rows.append(
            {
                "etapa": name,
                "status": "OK" if info["existe"] else "FALTA",
                "saida": output,
                "modificado": info["modificado"],
                "comando": "python " + " ".join(command),
            }
        )
    return pd.DataFrame(rows)


def project_summary() -> dict:
    metadata = read_csv(METADATA)
    splits = read_csv(SPLITS)
    face = read_csv(FACE_MATCHING)
    skull = read_csv(SKULL_FEATURES)
    return {
        "metadata_rows": len(metadata),
        "valid_pairs": int(metadata["usar_treino"].fillna(False).sum()) if "usar_treino" in metadata else 0,
        "excluded_pairs": int((~metadata["usar_treino"].fillna(False)).sum()) if "usar_treino" in metadata else 0,
        "split_rows": len(splits),
        "face_ok": int(face["usar_matching"].fillna(False).sum()) if "usar_matching" in face else 0,
        "skull_ok": int(skull["usar_matching"].fillna(False).sum()) if "usar_matching" in skull else 0,
    }


def status_style(value: object) -> str:
    text = str(value).upper()
    if text in {"OK", "TRUE", "SIM"} or "CONCORDANTE" in text:
        return "background-color:#d1fae5;color:#065f46;font-weight:700"
    if "ATENCAO" in text or "REVISAR" in text or "FALTA" in text or "INDEFINIDO" in text or "DIVERG" in text:
        return "background-color:#fef3c7;color:#92400e;font-weight:700"
    if "ERRO" in text or "FALHA" in text or "FALSE" in text or "NAO" in text:
        return "background-color:#fee2e2;color:#991b1b;font-weight:700"
    return ""


def style_status_df(df: pd.DataFrame):
    if df.empty:
        return df

    def style_col(col: pd.Series) -> list[str]:
        if col.name.lower() in {"status", "existe", "usar_treino", "face_ok", "cranio_ok", "sexo_concordante"}:
            return [status_style(v) for v in col]
        return ["" for _ in col]

    return df.style.apply(style_col, axis=0)


def image_quality(path: Path | None) -> dict:
    if not path or not path.exists():
        return {"status": "ERRO", "motivo": "imagem ausente"}
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("L")
            width, height = image.size
            sample = image.copy()
            sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
            stat = ImageStat.Stat(sample)
            edges = sample.filter(ImageFilter.FIND_EDGES)
            edge_std = float(ImageStat.Stat(edges).stddev[0])
            contrast = float(stat.stddev[0])
            brightness = float(stat.mean[0])
        status = "OK"
        reason = "métricas básicas dentro do esperado"
        if width < 180 or height < 180 or contrast < 18 or edge_std < 8:
            status = "ATENCAO"
            reason = "resolução, contraste ou nitidez podem estar baixos"
        return {
            "status": status,
            "motivo": reason,
            "largura": width,
            "altura": height,
            "contraste": round(contrast, 2),
            "nitidez_borda": round(edge_std, 2),
            "brilho": round(brightness, 2),
        }
    except Exception as exc:
        return {"status": "ERRO", "motivo": str(exc)}


def normalized_manifest(size: int = 224) -> pd.DataFrame:
    return read_csv(PROJECT_DIR / "data/domain_normalized" / f"v1_{size}" / "normalized_manifest.csv")


def processed_manifest(size: int = 224) -> pd.DataFrame:
    return read_csv(PROJECT_DIR / "data/processed" / f"v1_{size}" / "preprocess_status.csv")


def face_preview_path(year: object, image_id: object) -> Path | None:
    path = REPO_ROOT / "Treinamento_faces/dados_originais/faces" / str(year) / "marcacoes_landmarks/previews" / f"{image_id}_preview.jpg"
    return path if path.exists() else None


def face_fairface_crop(year: object, image_id: object) -> Path | None:
    path = REPO_ROOT / "Treinamento_faces/resultados/genero_faces/crops_fairface" / f"{year}_{image_id}_crop.jpg"
    return path if path.exists() else None


def skull_preview_path(year: object, skull_id: object, kind: str = "contorno") -> Path | None:
    suffix = "contorno_preview" if kind == "contorno" else "shape64_preview"
    path = REPO_ROOT / "Treinamento_cranios/resultados/previews_cranio" / str(year) / f"{skull_id}_{suffix}.jpg"
    return path if path.exists() else None


def normalized_paths_for_pair(pair_key: str, size: int = 224) -> dict[str, Path | None]:
    manifest = normalized_manifest(size)
    if manifest.empty or "pair_key" not in manifest:
        return {}
    row = manifest[manifest["pair_key"].astype(str).eq(str(pair_key))]
    if row.empty:
        return {}
    rec = row.iloc[0]
    return {
        "face_normalizada": repo_path(rec.get("normalized_face_path")),
        "face_mascarada": repo_path(rec.get("normalized_face_masked_path")),
        "cranio_cf_normalizado": repo_path(rec.get("normalized_cranio_cf_path")),
        "cranio_ce_normalizado": repo_path(rec.get("normalized_cranio_ce_path")),
        "cranio_cd_normalizado": repo_path(rec.get("normalized_cranio_cd_path")),
    }


def select_metadata_rows(years: Iterable[str] | None = None, sex: str = "Todos", valid_only: bool = True) -> pd.DataFrame:
    df = read_csv(METADATA)
    if df.empty:
        return df
    df["ano"] = df["ano"].astype(str)
    if years:
        df = df[df["ano"].isin([str(y) for y in years])]
    if valid_only and "usar_treino" in df:
        df = df[df["usar_treino"].fillna(False)]
    if sex != "Todos" and "sexo_face" in df:
        df = df[df["sexo_face"].astype(str).str.upper().eq(sex.upper())]
    return df.reset_index(drop=True)


def audit_tests() -> pd.DataFrame:
    rows = []
    metadata = read_csv(METADATA)
    splits = read_csv(SPLITS)
    normalized = normalized_manifest()
    baselines = read_csv(BASELINES)
    checks = [
        ("metadata_pares", METADATA.exists(), "Modelo_novo/data/manifests/metadata_pares.csv", "Manifesto de pares existe"),
        ("pares_validos", "usar_treino" in metadata and int(metadata["usar_treino"].fillna(False).sum()) > 0, "metadata_pares.csv", "Há pares válidos para treino"),
        ("splits", SPLITS.exists() and not splits.empty, "Modelo_novo/data/splits/all_splits.csv", "Splits existem"),
        ("sem_vazamento_splits", _no_split_leakage(splits), "all_splits.csv", "Nenhuma pessoa aparece em mais de um split"),
        ("caminhos_existentes", _valid_paths(metadata), "metadata_pares.csv", "Faces e crânios válidos existem no disco"),
        ("normalizacao_visual", not normalized.empty and normalized.get("status", pd.Series()).astype(str).eq("OK").any(), "Modelo_novo/data/domain_normalized", "Há normalizações geradas"),
        ("baselines", BASELINES.exists() and not baselines.empty, "Modelo_novo/reports/baselines/metricas_baselines.csv", "Baselines existem"),
        ("relatorio_html", REPORT_HTML.exists(), "docs/relatorio_estado_repositorio_gigi_com_imagens.html", "Relatório HTML com imagens existe"),
    ]
    for name, ok, output, criterion in checks:
        rows.append(
            {
                "teste": name,
                "entrada": output,
                "saida": output,
                "criterio": criterion,
                "status": "OK" if ok else "ATENCAO",
                "referencia": _test_reference(name),
            }
        )
    return pd.DataFrame(rows)


def _no_split_leakage(splits: pd.DataFrame) -> bool:
    if splits.empty or not {"pair_key", "split"}.issubset(splits.columns):
        return False
    counts = splits.groupby("pair_key")["split"].nunique()
    return bool(counts.le(1).all())


def _valid_paths(metadata: pd.DataFrame) -> bool:
    if metadata.empty or "usar_treino" not in metadata:
        return False
    valid = metadata[metadata["usar_treino"].fillna(False)]
    for col in ("face_path", "cranio_cf_path"):
        if col not in valid:
            return False
        if any(not (repo_path(v) and repo_path(v).exists()) for v in valid[col]):
            return False
    return True


def _test_reference(name: str) -> str:
    if "split" in name or "metadata" in name or "pares" in name:
        return "MEPROCS + FaceNet/retrieval"
    if "normalizacao" in name or "caminhos" in name:
        return "Photographic variables / pose / perspectiva"
    if "baseline" in name:
        return "Metric learning + avaliação Top-K/MRR"
    return "Checklist metodológico"


def run_cli(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    command = [sys.executable, str(PROJECT_DIR / "cli.py"), *args]
    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def latest_logs(limit: int = 200) -> pd.DataFrame:
    candidates = []
    for root in [PROJECT_DIR / "runs", PROJECT_DIR / "studies", REPO_ROOT / "Treinamento_faces/resultados/logs", REPO_ROOT / "Treinamento_cranios/resultados/logs"]:
        if root.exists():
            candidates.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".log", ".json", ".csv"}])
    rows = []
    for p in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        rows.append(
            {
                "arquivo": p.relative_to(REPO_ROOT).as_posix(),
                "modificado": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                "bytes": p.stat().st_size,
            }
        )
    return pd.DataFrame(rows)
