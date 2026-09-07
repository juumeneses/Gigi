from __future__ import annotations

import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.baselines import run_baselines
from src.dashboard_utils import (
    BASELINES,
    FACE_COMPARISON,
    FACE_CONFUSION,
    FACE_CURATED,
    FACE_FEATURES,
    FACE_MATCHING,
    FACE_METRICS,
    GALLERY10,
    LOYO,
    METADATA,
    REPORT_HTML,
    REPORT_MD,
    SKULL_CONFUSION,
    SKULL_FEATURES,
    SKULL_METRICS,
    SKULL_NORMALIZATIONS,
    SKULL_VIEW_METRICS,
    SPLITS,
    VIEW_ABLATION,
    audit_tests,
    face_fairface_crop,
    face_preview_path,
    image_quality,
    latest_logs,
    normalized_paths_for_pair,
    pipeline_status,
    project_summary,
    read_csv,
    repo_path,
    select_metadata_rows,
    skull_preview_path,
    style_status_df,
)
from src.domain_preprocessing import audit_domains, normalize_domains
from src.domain_preprocessing import normalize_face as normalize_face_image
from src.index_builder import build_metadata
from src.inference import available_checkpoint_sets, load_ensemble, rank_uploaded_candidates
from src.paths import REPO_ROOT
from src.preprocess import preprocess_manifest
from src.splits import create_leave_one_year_out, create_splits


st.set_page_config(page_title="Gigi · Modelo Novo", page_icon="G", layout="wide")

VENV_PYTHON = REPO_ROOT / "Treinamento_faces/.venv/bin/python"
RUNNER_PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
RUNTIME_CONFIG = PROJECT_DIR / "resultados/app_runtime_config.yaml"

BUTTON_HELP = {
    "face_norm_missing": "Gera as imagens normalizadas que ainda faltam, sem apagar as que ja existem.",
    "face_norm_force": "Refaz todas as normalizacoes derivadas e sobrescreve as saidas ja geradas. Nao altera imagens originais.",
    "fairface_missing": "Roda o FairFace apenas nas faces que ainda nao tem previsao de sexo estimado.",
    "fairface_force": "Refaz as predicoes FairFace. Decisoes manuais devem continuar prevalecendo nos CSVs curados.",
    "face_gender_train": "Treina o modelo proprio de sexo estimado usando as faces curadas.",
    "face_gender_compare": "Compara FairFace com o modelo proprio e gera tabela de concordancias/divergencias.",
    "face_gender_export": "Junta curadoria, predicoes e medidas em um CSV final para matching.",
    "face_landmarks": "Extrai pontos automaticos da face e gera previews com marcacao.",
    "face_features": "Transforma landmarks e metricas de qualidade em caracteristicas numericas.",
    "skull_norm_missing": "Gera normalizacoes derivadas para faces e cranios que ainda faltam no Modelo_novo.",
    "skull_norm_force": "Refaz as normalizacoes derivadas do Modelo_novo. Nao mexe nos originais.",
    "skull_compare_norm": "Testa variantes de normalizacao do cranio e registra qual performa melhor para sexo craniano.",
    "skull_silhouettes": "Prepara silhuetas dos cranios para treinar/avaliar sexo craniano.",
    "skull_gender_train": "Treina o modelo de sexo craniano com as silhuetas e tripletas disponiveis.",
    "skull_gender_compare": "Compara normalizacoes e metricas do modelo de sexo craniano.",
    "skull_gender_export": "Exporta features cranianas finais com qualidade e sexo craniano para o matching.",
    "skull_contour": "Extrai o contorno automatico do cranio e salva previews para auditoria visual.",
    "skull_shape64": "Converte o contorno craniano em 64 pontos normalizados.",
    "skull_measures": "Gera medidas numericas do cranio, como area, perimetro, compacidade e assimetria.",
    "match_build": "Reconstroi o manifesto de pares face-cranio a partir dos resultados de face e cranio.",
    "match_splits": "Cria splits de treino, validacao, teste e holdout por ano sem vazamento de individuo.",
    "match_baselines": "Roda comparacoes simples antes do modelo neural, como baseline aleatorio/sexo/shape.",
    "match_train": "Treina o modelo two-tower de embeddings face-cranio.",
    "match_gallery10": "Avalia o ranking em galerias simuladas com 10 candidatos.",
    "match_loyo_run": "Treina e avalia deixando um ano inteiro como teste.",
    "match_hard": "Roda estudo mais pesado com multiplas sementes e negativos dificeis.",
}

st.markdown(
    """
    <style>
    .main .block-container{padding-top:1rem;max-width:1680px}
    div[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px}
    .step-box{border:1px solid #e5e7eb;border-radius:8px;padding:14px;background:#fff;margin-bottom:12px}
    .soft-note{color:#6b7280;font-size:.92rem}
    img{border-radius:8px}
    .log-box{max-height:520px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#0f172a;color:#e5e7eb}
    .table-scroll{max-height:520px;overflow:auto;border:1px solid #e5e7eb;border-radius:8px}
    .table-scroll table{border-collapse:collapse;width:100%;font-size:.88rem}
    .table-scroll th{position:sticky;top:0;background:#f8fafc;color:#374151;border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}
    .table-scroll td{border-bottom:1px solid #f1f5f9;padding:7px;vertical-align:top}
    .table-scroll tr:nth-child(even){background:#fbfdff}
    @media(max-width: 760px){
      .main .block-container{padding-left:.75rem;padding-right:.75rem}
      div[data-testid="stMetric"]{padding:9px}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def metadata_df() -> pd.DataFrame:
    return read_csv(METADATA)


@st.cache_data(show_spinner=False)
def splits_df() -> pd.DataFrame:
    return read_csv(SPLITS)


def clear_caches() -> None:
    st.cache_data.clear()


def show_table(df: pd.DataFrame, height: int = 360, key: str | None = None) -> None:
    if df.empty:
        st.info("Sem dados para mostrar.")
        return
    safe_df = df.copy()
    for column in safe_df.columns:
        if safe_df[column].dtype == "object":
            safe_df[column] = safe_df[column].map(lambda value: "" if value is None else str(value))
    html = safe_df.head(1000).to_html(index=False, escape=True)
    st.markdown(f'<div class="table-scroll" style="max-height:{height}px">{html}</div>', unsafe_allow_html=True)


def render_image(path: Path | None, caption: str) -> None:
    if path and path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Saida ainda nao gerada/disponivel: {caption}")


def face_visual_normalized_path(row: pd.Series, size: int = 224) -> Path | None:
    year = str(row.get("ano", ""))
    image_id = str(row.get("id_imagem") or row.get("id_face") or "")
    source = repo_path(row.get("imagem_original") or row.get("face_path"))
    if not year or not image_id or not source or not source.exists():
        return None
    output = PROJECT_DIR / "data/visual_previews/face_normalized" / year / f"{image_id}.png"
    if output.exists():
        return output
    try:
        image, _ = normalize_face_image(source, year, image_id, size=size, masked=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return output
    except Exception:
        return None


def face_masked_path(row: pd.Series) -> Path | None:
    pair_key = face_pair_key(row)
    normalized = normalized_paths_for_pair(pair_key)
    masked = normalized.get("face_mascarada") if normalized else None
    if masked and masked.exists():
        return masked
    return None


def compact_columns(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in df.columns]


def count_contains(series: pd.Series, text: str) -> int:
    if series.empty:
        return 0
    return int(series.astype(str).str.upper().str.contains(text.upper(), na=False).sum())


def stream_command(command: list[str], key: str, timeout: int) -> tuple[int, str]:
    started = time.time()
    status_box = st.empty()
    log_box = st.empty()
    output_lines: list[str] = []
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command_text = " ".join(command)
    status_box.info(f"Rodando: {command_text}")
    try:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception as exc:
        message = f"Falha ao iniciar comando: {exc}"
        log_box.code(message, language="text")
        return 1, message

    line_queue: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        if proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line_queue.put(raw_line.rstrip())

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_render = 0.0
    while True:
        while True:
            try:
                output_lines.append(line_queue.get_nowait())
            except queue.Empty:
                break
        elapsed = time.time() - started
        if elapsed > timeout and proc.poll() is None:
            proc.kill()
            output_lines.append(f"[ERRO] Tempo limite excedido ({timeout}s). Processo interrompido.")
        now = time.time()
        if now - last_render >= 0.25:
            tail = "\n".join(output_lines[-220:]) or "Aguardando primeira linha de log..."
            status_box.info(f"Rodando ha {elapsed:.0f}s: {command_text}")
            log_box.code(tail, language="text")
            last_render = now
        if proc.poll() is not None:
            reader.join(timeout=1)
            while True:
                try:
                    output_lines.append(line_queue.get_nowait())
                except queue.Empty:
                    break
            break
        time.sleep(0.1)

    elapsed = time.time() - started
    final_output = "\n".join(output_lines).strip()
    if "ModuleNotFoundError" in final_output:
        final_output += (
            "\n\n[AJUDA] O comando nao encontrou uma dependencia Python. "
            f"Este botao tentou usar: {command[0]}. "
            "Se continuar acontecendo, rode no terminal: "
            "source Treinamento_faces/.venv/bin/activate && pip install -r requirements.txt"
        )
    status_box.info(f"Finalizado em {elapsed:.0f}s: {command_text}")
    log_box.code(final_output or "(sem saída)", language="text")
    st.session_state[f"{key}_last_log"] = final_output
    return int(proc.returncode or 0), final_output


def merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(path: Path, seen: set[Path] | None = None) -> dict:
    target = path.resolve()
    seen = set() if seen is None else seen
    if target in seen:
        raise ValueError(f"Herança circular de configuração: {target}")
    seen.add(target)
    current = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    parent = current.pop("extends", None)
    if not parent:
        return current
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = target.parent / parent_path
    return merge_config(load_config_file(parent_path, seen), current)


def available_config_paths() -> list[Path]:
    paths = sorted((PROJECT_DIR / "configs").glob("*.yaml"))
    if RUNTIME_CONFIG.exists():
        paths = [RUNTIME_CONFIG, *paths]
    return paths


def config_label(path: Path) -> str:
    return "Parâmetros salvos no app" if path == RUNTIME_CONFIG else path.name


def training_config_panel() -> Path | None:
    paths = available_config_paths()
    if not paths:
        st.warning("Nenhum arquivo de configuração encontrado.")
        return None
    default_index = 0 if RUNTIME_CONFIG.exists() else next((i for i, p in enumerate(paths) if p.name == "default.yaml"), 0)
    selected = st.selectbox("Configuração base dos próximos treinos", paths, index=default_index, format_func=config_label)
    try:
        cfg = load_config_file(selected)
    except Exception as exc:
        st.error(f"Não consegui abrir a configuração: {exc}")
        return selected

    with st.expander("Parâmetros editáveis do treino", expanded=False):
        st.caption(
            "Esses valores valem para os próximos botões de treino/avaliação depois de salvar. "
            "Eles mudam como o modelo aprende e como o ranking é calculado, sem mexer nas imagens originais."
        )
        data_cfg = dict(cfg.get("data", {}))
        model_cfg = dict(cfg.get("model", {}))
        train_cfg = dict(cfg.get("training", {}))
        rank_cfg = dict(cfg.get("ranking", {}))
        exp_cfg = dict(cfg.get("experiment", {}))
        with st.form("matching_config_form"):
            exp_cfg["variant"] = st.text_input("Nome da variante", value=str(exp_cfg.get("variant", "app_runtime")))
            c1, c2, c3, c4 = st.columns(4)
            train_cfg["epochs"] = c1.number_input("Épocas", 1, 300, int(train_cfg.get("epochs", 20)))
            train_cfg["batch_size"] = c2.number_input("Batch size", 1, 128, int(train_cfg.get("batch_size", 16)))
            train_cfg["patience"] = c3.number_input("Paciência", 1, 80, int(train_cfg.get("patience", 7)))
            model_cfg["embedding_dim"] = c4.number_input("Dimensão embedding", 16, 512, int(model_cfg.get("embedding_dim", 64)), step=16)
            c5, c6, c7, c8 = st.columns(4)
            train_cfg["learning_rate"] = c5.number_input("Learning rate", 0.000001, 0.1, float(train_cfg.get("learning_rate", 0.001)), format="%.6f")
            train_cfg["backbone_learning_rate"] = c6.number_input("LR backbone", 0.000001, 0.1, float(train_cfg.get("backbone_learning_rate", 0.0001)), format="%.6f")
            train_cfg["temperature"] = c7.number_input("Temperatura InfoNCE", 0.01, 1.0, float(train_cfg.get("temperature", 0.07)), format="%.3f")
            train_cfg["margin"] = c8.number_input("Margem triplet/hard", 0.01, 2.0, float(train_cfg.get("margin", 0.2)), format="%.3f")
            c9, c10, c11, c12 = st.columns(4)
            train_cfg["loss"] = c9.selectbox("Loss", ["infonce", "infonce_hard_negatives"], index=0 if str(train_cfg.get("loss", "infonce")) == "infonce" else 1)
            model_cfg["view_fusion"] = c10.selectbox("Fusão das vistas do crânio", ["mean", "attention"], index=0 if str(model_cfg.get("view_fusion", "mean")) == "mean" else 1)
            data_cfg["use_domain_normalized"] = c11.checkbox("Usar normalizadas", value=bool(data_cfg.get("use_domain_normalized", False)))
            data_cfg["require_triplet"] = c12.checkbox("Exigir CF+CE+CD", value=bool(data_cfg.get("require_triplet", False)))
            data_cfg["skull_views"] = st.multiselect("Vistas cranianas usadas no treino", ["CF", "CE", "CD"], default=list(data_cfg.get("skull_views", ["CF", "CE", "CD"])))
            c13, c14 = st.columns(2)
            train_cfg["hard_negative_weight"] = c13.number_input("Peso hard negative", 0.0, 3.0, float(train_cfg.get("hard_negative_weight", 0.0)), format="%.3f")
            rank_cfg["sex_weight"] = c14.number_input("Peso/penalização de sexo no ranking", 0.0, 1.0, float(rank_cfg.get("sex_weight", 0.0)), format="%.3f")
            save = st.form_submit_button("Salvar parâmetros para próximos botões", type="primary")
        if save:
            cfg["experiment"] = exp_cfg
            cfg["data"] = data_cfg
            cfg["model"] = model_cfg
            cfg["training"] = train_cfg
            cfg["ranking"] = rank_cfg
            RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            RUNTIME_CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            st.session_state["matching_config_path"] = str(RUNTIME_CONFIG)
            st.success(f"Parâmetros salvos em {RUNTIME_CONFIG.relative_to(REPO_ROOT)}")
            return RUNTIME_CONFIG
        show_table(
            pd.DataFrame(
                [
                    {"grupo": "treino", "parâmetro": "epochs", "valor": train_cfg.get("epochs"), "impacto": "mais épocas podem aprender melhor, mas demoram e podem overfit"},
                    {"grupo": "treino", "parâmetro": "learning_rate", "valor": train_cfg.get("learning_rate"), "impacto": "controla o tamanho dos passos do aprendizado"},
                    {"grupo": "treino", "parâmetro": "loss", "valor": train_cfg.get("loss"), "impacto": "define o objetivo; hard negatives força separar casos difíceis"},
                    {"grupo": "crânio", "parâmetro": "skull_views", "valor": ", ".join(data_cfg.get("skull_views", [])), "impacto": "usa CF, CE e/ou CD na torre craniana"},
                    {"grupo": "ranking", "parâmetro": "sex_weight", "valor": rank_cfg.get("sex_weight"), "impacto": "penaliza divergência de sexo quando essa opção estiver ligada"},
                ]
            ),
            height=220,
            key="config_explain_table",
        )
    return Path(st.session_state.get("matching_config_path", selected))


def face_pair_key(row: pd.Series) -> str:
    person = str(row.get("id_pessoa") or row.get("id_imagem", "")).replace("_FFN", "")
    return f"{row.get('ano')}::{person}"


def skull_pair_key(row: pd.Series) -> str:
    person = str(row.get("id_pessoa") or row.get("id_imagem", "")).replace("_CF", "")
    return f"{row.get('ano')}::{person}"


def filter_by_year_status_sex(df: pd.DataFrame, area: str, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    if "ano" in view:
        view["ano"] = view["ano"].astype(str)
        years = sorted(view["ano"].dropna().unique())
        chosen = st.multiselect("Ano", years, default=years, key=f"{prefix}_years")
        if chosen:
            view = view[view["ano"].isin(chosen)]
    status_col = "status_final" if "status_final" in view else ("status_cranio" if "status_cranio" in view else None)
    if status_col:
        statuses = ["Todos"] + sorted(view[status_col].dropna().astype(str).unique())
        status = st.selectbox("Status", statuses, key=f"{prefix}_status")
        if status != "Todos":
            view = view[view[status_col].astype(str).eq(status)]
    sex_col = "genero_final" if area == "face" and "genero_final" in view else ("classe_cranio" if area == "cranio" and "classe_cranio" in view else None)
    if sex_col:
        sex_values = ["Todos"] + sorted(view[sex_col].dropna().astype(str).unique())
        sex = st.selectbox("Sexo estimado", sex_values, key=f"{prefix}_sex")
        if sex != "Todos":
            view = view[view[sex_col].astype(str).eq(sex)]
    return view


def run_model_step(label: str, args: list[str], key: str, timeout: int = 1800, config_path: Path | None = None) -> None:
    help_text = BUTTON_HELP.get(key, "")
    if help_text:
        st.caption(help_text)
    if st.button(label, key=key, width="stretch", help=help_text or None):
        if help_text:
            st.info(f"O que vai acontecer: {help_text}")
        command = [str(RUNNER_PYTHON), str(PROJECT_DIR / "cli.py")]
        if config_path:
            command.extend(["--config", str(config_path)])
        command.extend(args)
        code, _ = stream_command(command, key, timeout)
        clear_caches()
        if code == 0:
            st.success("Concluído.")
        else:
            st.error(f"Terminou com erro ({code}).")


def run_script_step(label: str, script: str, key: str, args: list[str] | None = None, timeout: int = 3600) -> None:
    help_text = BUTTON_HELP.get(key, "")
    if help_text:
        st.caption(help_text)
    if st.button(label, key=key, width="stretch", help=help_text or None):
        if help_text:
            st.info(f"O que vai acontecer: {help_text}")
        command = [str(RUNNER_PYTHON), str(REPO_ROOT / script), *(args or [])]
        code, _ = stream_command(command, key, timeout)
        clear_caches()
        if code == 0:
            st.success("Concluído.")
        else:
            st.error(f"Terminou com erro ({code}).")


def status_metrics(status: pd.DataFrame) -> None:
    ok = int(status["status"].eq("OK").sum()) if not status.empty and "status" in status else 0
    total = len(status)
    c1, c2, c3 = st.columns([1, 1, 3])
    c1.metric("Etapas OK", f"{ok}/{total}")
    c2.metric("Faltantes", total - ok)
    c3.progress(ok / max(total, 1), text=f"{ok}/{total} saídas encontradas")


def render_face_versions(row: pd.Series, versions: list[str]) -> None:
    if "Original" in versions:
        render_image(repo_path(row.get("imagem_original") or row.get("face_path")), "Original")
    if "Landmarks" in versions:
        render_image(face_preview_path(row.get("ano"), row.get("id_imagem") or row.get("id_face")), "Landmarks")
    if "Normalizada" in versions:
        render_image(face_visual_normalized_path(row), "Normalizada visual")
    if "Máscara técnica" in versions:
        render_image(face_masked_path(row), "Máscara técnica para modelo")
    if "Crop FairFace" in versions:
        render_image(face_fairface_crop(row.get("ano"), row.get("id_imagem") or row.get("id_face")), "Crop FairFace")
    caption = [
        f"id: {row.get('id_imagem') or row.get('id_face')}",
        f"ano: {row.get('ano')}",
        f"status: {row.get('status_final', '')}",
        f"sexo: {row.get('genero_final') or row.get('sexo_face', '')}",
        f"método: {row.get('origem_genero_final', '')}",
        f"confiança: {row.get('confianca_fairface', '')}",
    ]
    st.caption(" · ".join([item for item in caption if not item.endswith(": ")]))


def render_skull_versions(row: pd.Series, versions: list[str]) -> None:
    pair_key = skull_pair_key(row)
    normalized = normalized_paths_for_pair(pair_key)
    skull_id = row.get("id_imagem") or row.get("id_cranio_cf") or row.get("id_cf")
    if "CF" in versions:
        render_image(repo_path(row.get("imagem_cf") or row.get("cranio_cf_path") or row.get("imagem_cranio")), "CF")
    if "CE" in versions:
        render_image(repo_path(row.get("imagem_ce") or row.get("cranio_ce_path")), "CE")
    if "CD" in versions:
        render_image(repo_path(row.get("imagem_cd") or row.get("cranio_cd_path")), "CD")
    if "Normalizadas" in versions:
        render_image(normalized.get("cranio_cf_normalizado"), "CF normalizado")
        render_image(normalized.get("cranio_ce_normalizado"), "CE normalizado")
        render_image(normalized.get("cranio_cd_normalizado"), "CD normalizado")
    if "Contorno" in versions:
        render_image(skull_preview_path(row.get("ano"), skull_id, "contorno"), "Contorno")
    if "Shape64" in versions:
        render_image(skull_preview_path(row.get("ano"), skull_id, "shape64"), "Shape64")
    caption = [
        f"id: {skull_id}",
        f"ano: {row.get('ano')}",
        f"status: {row.get('status_cranio', '')}",
        f"sexo: {row.get('classe_cranio') or row.get('sexo_cranio', '')}",
        f"confiança: {row.get('confianca_cranio', '')}",
    ]
    st.caption(" · ".join([item for item in caption if not item.endswith(": ")]))


def render_pair_versions(row: pd.Series, versions: list[str]) -> None:
    if "Face" in versions:
        render_image(repo_path(row.get("face_path")), f"Face · {row.get('sexo_face', '')}")
    if "Crânio CF" in versions:
        render_image(repo_path(row.get("cranio_cf_path")), f"Crânio CF · {row.get('sexo_cranio', '')}")
    if "Crânio CE/CD" in versions:
        render_image(repo_path(row.get("cranio_ce_path")), "Crânio CE")
        render_image(repo_path(row.get("cranio_cd_path")), "Crânio CD")
    if "Normalizadas" in versions:
        normalized = normalized_paths_for_pair(str(row.get("pair_key")))
        render_image(face_visual_normalized_path(row), "Face normalizada visual")
        render_image(normalized.get("cranio_cf_normalizado"), "CF normalizado")
    if "Landmarks/contorno" in versions:
        render_image(face_preview_path(row.get("ano"), row.get("id_face")), "Landmarks")
        render_image(skull_preview_path(row.get("ano"), row.get("id_cranio_cf"), "contorno"), "Contorno")
    st.caption(
        f"{row.get('pair_key')} · face: {row.get('sexo_face')} · crânio: {row.get('sexo_cranio')} · concordante: {row.get('sexo_concordante')}"
    )


@st.dialog("Ver imagens", width="large")
def image_dialog(area: str, records: list[dict], default_versions: list[str]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        st.info("Sem imagens para abrir.")
        return
    version_options = {
        "face": ["Original", "Landmarks", "Normalizada", "Máscara técnica", "Crop FairFace"],
        "cranio": ["CF", "CE", "CD", "Normalizadas", "Contorno", "Shape64"],
        "matching": ["Face", "Crânio CF", "Crânio CE/CD", "Normalizadas", "Landmarks/contorno"],
    }[area]
    versions = st.multiselect("O que mostrar", version_options, default=default_versions)
    per_row = st.selectbox("Imagens por linha", [1, 2, 3, 4, 5], index=2)
    search = st.text_input("Buscar id/nome", "")
    if search:
        mask = pd.Series(False, index=df.index)
        for column in df.columns:
            mask = mask | df[column].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]
    if "ano" in df:
        years = ["Todos"] + sorted(df["ano"].astype(str).dropna().unique())
        year = st.selectbox("Ano", years)
        if year != "Todos":
            df = df[df["ano"].astype(str).eq(year)]
    limit = st.slider("Limite nesta visualização", 1, 100, min(24, len(df) if len(df) else 1))
    df = df.head(int(limit))
    if df.empty:
        st.info("Nenhum item bate com o filtro.")
        return
    for start in range(0, len(df), int(per_row)):
        cols = st.columns(int(per_row))
        for col, (_, row) in zip(cols, df.iloc[start : start + int(per_row)].iterrows()):
            with col:
                if area == "face":
                    render_face_versions(row, versions)
                elif area == "cranio":
                    render_skull_versions(row, versions)
                else:
                    render_pair_versions(row, versions)


def open_images_button(label: str, area: str, df: pd.DataFrame, default_versions: list[str], key: str) -> None:
    disabled = df.empty
    if st.button(label, key=key, width="stretch", disabled=disabled):
        image_dialog(area, df.head(300).to_dict("records"), default_versions)


def page_inicio() -> None:
    st.header("Início")
    st.caption("Aplicação local do Modelo_novo. O matching é ranking experimental de similaridade, não identificação pericial.")
    summary = project_summary()
    cols = st.columns(6)
    cols[0].metric("Metadata", summary["metadata_rows"])
    cols[1].metric("Pares válidos", summary["valid_pairs"])
    cols[2].metric("Excluídos", summary["excluded_pairs"])
    cols[3].metric("Splits", summary["split_rows"])
    cols[4].metric("Faces OK", summary["face_ok"])
    cols[5].metric("Crânios OK", summary["skull_ok"])

    st.subheader("Status por área")
    cards = st.columns(3)
    for col, section in zip(cards, ["Face", "Crânio", "Matching"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{section}**")
                status = pipeline_status(section)
                ok = int(status["status"].eq("OK").sum()) if not status.empty else 0
                total = len(status)
                st.metric("Etapas OK", f"{ok}/{total}")
                st.progress(ok / max(total, 1))

    meta = metadata_df()
    if not meta.empty:
        st.subheader("Resumo por ano")
        valid = meta[meta["usar_treino"].fillna(False)] if "usar_treino" in meta else meta
        by_year = valid.groupby("ano").agg(
            pares=("pair_key", "count"),
            faces_ok=("face_ok", "sum"),
            cranios_ok=("cranio_ok", "sum"),
            tripletas=("tem_tripleta", "sum"),
        ).reset_index()
        show_table(by_year, height=220, key="home_year")

    st.subheader("Alertas")
    show_table(audit_tests(), height=260, key="home_alerts")


def face_summary_table(df: pd.DataFrame, curated: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ano"] = df["ano"].astype(str)
    grouped = df.groupby("ano").agg(
        total=("id_imagem", "count"),
        ok_auto=("status_final", lambda s: count_contains(s, "OK_AUTO")),
        ok_manual=("status_final", lambda s: count_contains(s, "OK_MANUAL")),
        revisar=("qualidade_landmarks", lambda s: count_contains(s, "REVISAR")),
        usar_matching=("usar_matching", "sum"),
    ).reset_index()
    if not curated.empty:
        curated = curated.copy()
        curated["ano"] = curated["ano"].astype(str)
        gender = curated.groupby("ano").agg(
            genero_alta_confianca=("origem_genero_final", lambda s: count_contains(s, "ALTA")),
            genero_manual=("origem_genero_final", lambda s: count_contains(s, "MANUAL")),
            genero_definido=("genero_final", lambda s: int(s.notna().sum())),
        ).reset_index()
        grouped = grouped.merge(gender, on="ano", how="left")
    return grouped.fillna(0)


def face_page() -> None:
    st.header("Face")
    section = st.segmented_control("Etapa", ["Resumo", "Normalização", "Gênero", "Características"], default="Resumo")
    features = read_csv(FACE_FEATURES)
    matching = read_csv(FACE_MATCHING)
    curated = read_csv(FACE_CURATED)
    source = features if not features.empty else matching
    if source.empty:
        st.warning("Ainda não encontrei a tabela de faces/features.")
        return

    if section == "Resumo":
        view = filter_by_year_status_sex(source, "face", "face_summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Faces", len(view))
        c2.metric("Usar matching", int(view.get("usar_matching", pd.Series(dtype=bool)).fillna(False).sum()))
        c3.metric("OK auto", count_contains(view.get("status_final", pd.Series(dtype=str)), "OK_AUTO"))
        c4.metric("OK manual", count_contains(view.get("status_final", pd.Series(dtype=str)), "OK_MANUAL"))
        c5.metric("Revisar", count_contains(view.get("qualidade_landmarks", pd.Series(dtype=str)), "REVISAR"))
        show_table(face_summary_table(view, curated), height=240, key="face_summary_year")
        open_images_button("Ver imagens", "face", view, ["Original", "Landmarks"], "face_summary_images")
        show_table(view[compact_columns(["ano", "id_imagem", "qualidade_landmarks", "status_final", "genero_final", "origem_genero_final", "usar_matching"], view)].head(500), key="face_summary_table")

    elif section == "Normalização":
        status = pipeline_status("Matching")
        st.subheader("Ações")
        b1, b2, b3 = st.columns(3)
        with b1:
            run_model_step("Normalizar faltantes", ["normalize-images", "--size", "224"], "face_norm_missing")
        with b2:
            run_model_step("Refazer normalização com force", ["normalize-images", "--size", "224", "--force"], "face_norm_force")
        with b3:
            if st.button("Checar normalizações", width="stretch"):
                clear_caches()
                st.success("Dados recarregados.")
        st.subheader("Status")
        show_table(status[status["etapa"].astype(str).str.contains("Preprocessar|Construir|Criar", case=False, na=False) | status["etapa"].astype(str).str.contains("Preprocess", case=False, na=False)], height=220, key="face_norm_status")
        view = filter_by_year_status_sex(source, "face", "face_norm")
        open_images_button("Ver imagens", "face", view, ["Original", "Normalizada"], "face_norm_images")
        metrics_cols = compact_columns(
            ["ano", "id_imagem", "original_score", "original_area_face_normalizada", "original_margem_borda", "original_inclinacao_olhos_graus", "qualidade_landmarks", "motivo_qualidade"],
            view,
        )
        show_table(view[metrics_cols].head(500), height=360, key="face_norm_table")

    elif section == "Gênero":
        st.subheader("Ações")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            run_script_step("Rodar FairFace faltantes", "Treinamento_faces/scripts/06_predizer_genero_fairface_faces.py", "fairface_missing", ["--root", "Treinamento_faces"])
        with b2:
            run_script_step("Rodar FairFace force", "Treinamento_faces/scripts/06_predizer_genero_fairface_faces.py", "fairface_force", ["--root", "Treinamento_faces", "--force"])
        with b3:
            run_script_step("Treinar modelo próprio", "Treinamento_faces/scripts/09_treinar_modelo_genero_faces.py", "face_gender_train", ["--root", "Treinamento_faces"])
        with b4:
            run_script_step("Comparar FairFace x modelo", "Treinamento_faces/scripts/10_comparar_genero_fairface_modelo.py", "face_gender_compare", ["--root", "Treinamento_faces"])
        with b5:
            run_script_step("Exportar features", "Treinamento_faces/scripts/11_exportar_matching_faces_com_genero.py", "face_gender_export", ["--root", "Treinamento_faces"])
        view = curated if not curated.empty else source
        view = filter_by_year_status_sex(view, "face", "face_gender")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Definidos", int(view.get("genero_final", pd.Series(dtype=str)).notna().sum()))
        c2.metric("Masculino", count_contains(view.get("genero_final", pd.Series(dtype=str)), "MASCULINO"))
        c3.metric("Feminino", count_contains(view.get("genero_final", pd.Series(dtype=str)), "FEMININO"))
        c4.metric("Alta confiança", count_contains(view.get("origem_genero_final", pd.Series(dtype=str)), "ALTA"))
        c5.metric("Manual", count_contains(view.get("origem_genero_final", pd.Series(dtype=str)), "MANUAL"))
        show_table(read_csv(FACE_METRICS), height=170, key="face_gender_metrics")
        show_table(read_csv(FACE_CONFUSION), height=170, key="face_gender_confusion")
        open_images_button("Ver imagens", "face", view, ["Original", "Crop FairFace", "Landmarks"], "face_gender_images")
        cols = compact_columns(["ano", "id_imagem", "classe_fairface", "prob_masculino_fairface", "prob_feminino_fairface", "confianca_fairface", "genero_final", "origem_genero_final", "status_genero_manual", "status_genero_automatico"], view)
        show_table(view[cols].head(500), height=360, key="face_gender_table")

    else:
        st.subheader("Ações")
        b1, b2, b3 = st.columns(3)
        with b1:
            run_script_step("Extrair landmarks", "Treinamento_faces/scripts/01_extrair_landmarks_faces.py", "face_landmarks", ["--root", "Treinamento_faces"])
        with b2:
            run_script_step("Gerar medidas/features", "Treinamento_faces/scripts/02_gerar_medidas_faces.py", "face_features", ["--root", "Treinamento_faces"])
        with b3:
            if st.button("Checar features", width="stretch"):
                clear_caches()
                st.success("Features recarregadas.")
        view = filter_by_year_status_sex(source, "face", "face_features")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas", len(view))
        c2.metric("Landmarks OK", count_contains(view.get("qualidade_landmarks", pd.Series(dtype=str)), "OK"))
        c3.metric("Shape/LM cols", len([c for c in view.columns if c.startswith("lm_")]))
        c4.metric("NaN", int(view.isna().sum().sum()))
        open_images_button("Ver imagens", "face", view, ["Original", "Landmarks"], "face_features_images")
        feature_cols = compact_columns(["ano", "id_imagem", "qualidade_landmarks", "razao_largura_altura_face", "razao_olhos_largura_face", "razao_boca_largura_face", "pca_face_1", "pca_face_2", "score_outlier_pca"], view)
        show_table(view[feature_cols].head(500), height=420, key="face_features_table")


def skull_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("ano").agg(
        total=("id_imagem", "count"),
        pessoas=("id_pessoa", "nunique"),
        ok=("status_cranio", lambda s: count_contains(s, "OK")),
        tripletas=("tem_tripleta", "sum"),
        usar_matching=("usar_matching", "sum"),
        sexo_definido=("classe_cranio", lambda s: int(s.notna().sum())),
    ).reset_index()


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "-"


def summarize_ranking(path: Path) -> dict:
    rank = read_csv(path)
    if rank.empty or not {"query_id", "rank", "correto"}.issubset(rank.columns):
        return {}
    correct = rank[rank["correto"].fillna(False)].copy()
    queries = int(rank["query_id"].nunique())
    correct_ranks = pd.to_numeric(correct["rank"], errors="coerce")
    return {
        "arquivo": path.relative_to(PROJECT_DIR).as_posix(),
        "consultas": queries,
        "top_1": int(correct_ranks.le(1).sum()),
        "top_1_pct": correct_ranks.le(1).sum() / max(queries, 1),
        "top_3": int(correct_ranks.le(3).sum()),
        "top_3_pct": correct_ranks.le(3).sum() / max(queries, 1),
        "top_5": int(correct_ranks.le(5).sum()),
        "top_5_pct": correct_ranks.le(5).sum() / max(queries, 1),
        "mrr": float((1 / correct_ranks).mean()) if len(correct_ranks) else 0.0,
        "rank_mediano": float(correct_ranks.median()) if len(correct_ranks) else None,
        "rank_medio": float(correct_ranks.mean()) if len(correct_ranks) else None,
        "modificado": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)),
        "mtime": path.stat().st_mtime,
    }


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    for col in ["top_1", "top_3", "top_5", "recall_at_10", "mrr"]:
        if col in view:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(lambda value: pct(value) if pd.notna(value) else "-")
    return view


def gallery_summary_table() -> pd.DataFrame:
    frames = []
    for path in sorted((PROJECT_DIR / "reports").glob("gallery_10*/metrics.csv")):
        df = read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["resultado"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (result, scenario), group in data.groupby(["resultado", "scenario"], dropna=False):
        weights = pd.to_numeric(group.get("n", pd.Series([1] * len(group))), errors="coerce").fillna(1)
        row = {"resultado": result, "cenario": scenario, "n": int(weights.sum())}
        for col in ["top_1", "top_3", "top_5", "mrr", "mean_rank", "median_rank"]:
            if col in group:
                values = pd.to_numeric(group[col], errors="coerce")
                row[col] = float((values * weights).sum() / max(weights.sum(), 1)) if values.notna().any() else None
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["mrr", "top_1"], ascending=False, na_position="last")
    return format_metric_table(result)


def loyo_summary_table() -> pd.DataFrame:
    df = read_csv(LOYO)
    if df.empty:
        return df
    view = df.copy()
    if "run_dir" in view:
        view["existe_ranking"] = view["run_dir"].map(lambda p: (PROJECT_DIR / str(p) / "ranking_test.csv").exists())
    cols = compact_columns(["holdout_year", "variant", "best_epoch", "n_queries", "n_candidates", "top_1", "top_3", "top_5", "recall_at_10", "mrr", "median_rank", "mean_rank", "existe_ranking", "run_dir"], view)
    view = view[cols].sort_values(["holdout_year", "mrr"], ascending=[True, False])
    return format_metric_table(view)


def ranking_summary_raw() -> pd.DataFrame:
    paths = sorted((PROJECT_DIR / "runs").glob("*/ranking_test.csv"))
    paths += sorted((PROJECT_DIR / "studies").rglob("ranking_test.csv"))
    rows = [summarize_ranking(path) for path in paths if path.exists()]
    rows = [row for row in rows if row]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mtime", ascending=False)


def ranking_summary_table() -> pd.DataFrame:
    df = ranking_summary_raw()
    if df.empty:
        return df
    return format_metric_table(df.drop(columns=["mtime"], errors="ignore"))


def metric_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def ranking_selector(raw: pd.DataFrame, key: str) -> Path | None:
    if raw.empty:
        st.info("Nenhum ranking encontrado ainda.")
        return None
    best_idx = pd.to_numeric(raw["mrr"], errors="coerce").fillna(0).idxmax()
    latest_idx = raw["mtime"].idxmax() if "mtime" in raw else raw.index[0]
    labels: dict[str, str] = {}
    for idx, row in raw.iterrows():
        markers = []
        if idx == latest_idx:
            markers.append("mais recente")
        if idx == best_idx:
            markers.append("melhor")
        marker = f" ({', '.join(markers)})" if markers else ""
        labels[row["arquivo"]] = (
            f"{row['arquivo']}{marker} · Top-1 {pct(row.get('top_1_pct'))} · "
            f"Top-5 {pct(row.get('top_5_pct'))} · MRR {pct(row.get('mrr'))}"
        )
    options = raw["arquivo"].astype(str).tolist()
    default = options.index(str(raw.loc[latest_idx, "arquivo"]))
    selected = st.selectbox("Execução/ranking", options, index=default, format_func=lambda item: labels.get(item, item), key=key)
    return PROJECT_DIR / selected


def meta_by_pair_key() -> pd.DataFrame:
    meta = select_metadata_rows(valid_only=False)
    if meta.empty or "pair_key" not in meta:
        return pd.DataFrame()
    return meta.drop_duplicates("pair_key").set_index("pair_key", drop=False)


def render_candidate_skull(row: pd.Series, caption: str) -> None:
    paths = [("CF", row.get("cranio_cf_path")), ("CE", row.get("cranio_ce_path")), ("CD", row.get("cranio_cd_path"))]
    existing = [(label, repo_path(path)) for label, path in paths if repo_path(path) and repo_path(path).exists()]
    if not existing:
        st.info("Imagens do crânio não encontradas.")
        return
    cols = st.columns(len(existing))
    for col, (label, path) in zip(cols, existing):
        with col:
            render_image(path, label)
    st.caption(caption)


def render_ranking_visual(rank: pd.DataFrame, selected_query: str) -> None:
    meta = meta_by_pair_key()
    if meta.empty:
        st.info("Metadata não encontrada para abrir as imagens.")
        return
    query_rows = rank[rank["query_id"].astype(str).eq(str(selected_query))].copy()
    if query_rows.empty:
        st.info("Sem linhas para essa consulta.")
        return
    query_meta = meta.loc[selected_query] if selected_query in meta.index else None
    st.markdown("**Face de referência**")
    if query_meta is not None:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            render_image(repo_path(query_meta.get("face_path")), f"{selected_query} · sexo: {query_meta.get('sexo_face', '-')}")
        with c2:
            render_image(face_visual_normalized_path(query_meta), "Face normalizada")
        with c3:
            st.write(
                {
                    "sexo_face": query_meta.get("sexo_face"),
                    "origem": query_meta.get("origem_genero_final"),
                    "prob_masculino": query_meta.get("prob_masculino_fairface"),
                    "prob_feminino": query_meta.get("prob_feminino_fairface"),
                }
            )
    st.markdown("**Crânios no ranking**")
    top_n = st.slider("Quantos candidatos mostrar", 1, 10, min(10, len(query_rows)), key=f"rank_visual_top_{selected_query}")
    for _, item in query_rows.sort_values("rank").head(int(top_n)).iterrows():
        candidate_id = str(item.get("candidate_id"))
        with st.container(border=True):
            title = f"#{item.get('rank')} · {candidate_id}"
            if bool(item.get("correto", False)):
                title += " · correto"
            st.markdown(f"**{title}**")
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Score", f"{metric_float(item.get('score')):.4f}")
                st.write(f"Correto: {'sim' if bool(item.get('correto', False)) else 'não'}")
            with c2:
                if candidate_id in meta.index:
                    row = meta.loc[candidate_id]
                    caption = f"sexo crânio: {row.get('sexo_cranio', '-')} · confiança: {row.get('confianca_cranio', '-')}"
                    render_candidate_skull(row, caption)
                else:
                    st.info("Candidato não encontrado na metadata.")


def skull_view_sort_key(uploaded_file) -> tuple[int, str]:
    name = str(getattr(uploaded_file, "name", "")).upper()
    if "_CF" in name or "-CF" in name or " CF" in name:
        return (0, name)
    if "_CE" in name or "-CE" in name or " CE" in name:
        return (1, name)
    if "_CD" in name or "-CD" in name or " CD" in name:
        return (2, name)
    return (3, name)


def uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def normalize_sex_label(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("M"):
        return "MASCULINO"
    if text.startswith("F"):
        return "FEMININO"
    return "INDEFINIDO"


def row_image(path_value: object) -> Image.Image | None:
    path = repo_path(path_value)
    if path and path.exists():
        return Image.open(path).convert("RGB")
    return None


def candidate_views_from_row(row: pd.Series) -> list[dict]:
    views = []
    for label, column in [("CF", "cranio_cf_path"), ("CE", "cranio_ce_path"), ("CD", "cranio_cd_path")]:
        image = row_image(row.get(column))
        if image is not None:
            views.append({"label": label, "image": image})
    return views


def valid_random_meta() -> pd.DataFrame:
    meta = select_metadata_rows(valid_only=True)
    if meta.empty:
        return meta
    view = meta.copy()
    for column in ["face_path", "cranio_cf_path"]:
        view[f"{column}_exists_local"] = view[column].map(lambda value: bool(repo_path(value) and repo_path(value).exists()))
    view = view[view["face_path_exists_local"] & view["cranio_cf_path_exists_local"]]
    return view


def infer_row_from_uploaded_names(files: list, kind: str) -> pd.Series | None:
    if not files:
        return None
    meta = select_metadata_rows(valid_only=False)
    if meta.empty:
        return None
    haystack = " ".join(str(getattr(file, "name", "")).upper() for file in files if file is not None)
    if not haystack.strip():
        return None
    score_columns = ["id_pessoa"]
    if kind == "face":
        score_columns += ["id_face"]
    else:
        score_columns += ["id_cranio_cf"]
    best_score = 0
    best_row = None
    for _, row in meta.iterrows():
        score = 0
        for column in score_columns:
            value = str(row.get(column, "") or "").upper()
            if value and value in haystack:
                score += 2 if column != "id_pessoa" else 1
        if score > best_score:
            best_score = score
            best_row = row
    return best_row if best_score else None


def sex_option_index(value: object) -> int:
    sex = normalize_sex_label(value)
    return {"INDEFINIDO": 0, "MASCULINO": 1, "FEMININO": 2}.get(sex, 0)


def build_random_compare_sample(use_gender: bool, seed: int | None = None) -> dict | None:
    meta = valid_random_meta()
    if meta.empty:
        return None
    rng = random.Random(seed if seed is not None else time.time_ns())
    reference = meta.sample(n=1, random_state=rng.randint(0, 2_000_000_000)).iloc[0]
    reference_key = str(reference.get("pair_key"))
    reference_sex = normalize_sex_label(reference.get("sexo_face"))
    distractors = meta[meta["pair_key"].astype(str).ne(reference_key)].copy()
    count = min(9, len(distractors))
    if count < 9:
        return None
    chosen = distractors.sample(n=count, random_state=rng.randint(0, 2_000_000_000))
    candidates_df = pd.concat([reference.to_frame().T, chosen], ignore_index=True)
    candidates_df = candidates_df.sample(frac=1, random_state=rng.randint(0, 2_000_000_000)).reset_index(drop=True)
    candidates = []
    for _, row in candidates_df.iterrows():
        pair_key = str(row.get("pair_key"))
        candidates.append(
            {
                "name": pair_key,
                "pair_key": pair_key,
                "sex": normalize_sex_label(row.get("sexo_cranio")),
                "correct": pair_key == reference_key,
                "views": candidate_views_from_row(row),
                "sexo_face_referencia": reference_sex,
                "sexo_cranio": normalize_sex_label(row.get("sexo_cranio")),
                "confianca_cranio": row.get("confianca_cranio", ""),
                "prob_masculino_cranio": row.get("prob_masculino_cranio", ""),
                "prob_feminino_cranio": row.get("prob_feminino_cranio", ""),
                "contorno_path": str(skull_preview_path(row.get("ano"), row.get("id_cranio_cf"), "contorno")),
            }
        )
    return {
        "reference_pair_key": reference_key,
        "reference_sex": reference_sex,
        "face": row_image(reference.get("face_path")),
        "face_row": reference.to_dict(),
        "candidates": candidates,
        "use_gender": use_gender,
        "filtered_by_gender": False,
    }


def enrich_compare_result(result: pd.DataFrame, candidates: list[dict], reference_pair_key: str | None = None, reference_sex: str | None = None) -> pd.DataFrame:
    by_name = {item["name"]: item for item in candidates}
    enriched = result.copy()
    enriched["pair_key"] = enriched["candidato"].map(lambda name: by_name.get(str(name), {}).get("pair_key", ""))
    enriched["correto"] = enriched["candidato"].map(lambda name: bool(by_name.get(str(name), {}).get("correct", False)))
    if reference_pair_key:
        enriched["crânio_correto"] = enriched["pair_key"].astype(str).eq(str(reference_pair_key))
    ref_sex = normalize_sex_label(reference_sex or enriched.get("sexo_referencia", pd.Series(["INDEFINIDO"])).iloc[0])
    enriched["sexo_referencia"] = ref_sex
    enriched["genero_bate_referencia"] = enriched["candidato"].map(
        lambda name: (
            normalize_sex_label(by_name.get(str(name), {}).get("sexo_cranio")) == ref_sex
            if ref_sex != "INDEFINIDO" and normalize_sex_label(by_name.get(str(name), {}).get("sexo_cranio")) != "INDEFINIDO"
            else None
        )
    )
    enriched["genero_bate_referencia_texto"] = enriched["genero_bate_referencia"].map(
        lambda value: "sim" if value is True else ("nao" if value is False else "indefinido")
    )
    enriched["genero_cranio_correto"] = enriched["candidato"].map(
        lambda name: (
            normalize_sex_label(by_name.get(str(name), {}).get("sexo_cranio")) == ref_sex
            if bool(by_name.get(str(name), {}).get("correct", False)) and ref_sex != "INDEFINIDO"
            else None
        )
    )
    enriched["confianca_cranio"] = enriched["candidato"].map(lambda name: by_name.get(str(name), {}).get("confianca_cranio", ""))
    enriched["prob_masculino_cranio"] = enriched["candidato"].map(lambda name: by_name.get(str(name), {}).get("prob_masculino_cranio", ""))
    enriched["prob_feminino_cranio"] = enriched["candidato"].map(lambda name: by_name.get(str(name), {}).get("prob_feminino_cranio", ""))
    if "score_embedding" in enriched:
        enriched["posição_sem_filtro_genero"] = pd.to_numeric(enriched["score_embedding"], errors="coerce").rank(
            method="first",
            ascending=False,
        ).astype("Int64")
    return enriched


def render_random_sample(sample: dict) -> None:
    if not sample:
        return
    st.markdown("**Seleção carregada**")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if sample.get("face") is not None:
            st.image(sample["face"], caption=f"Face referência · {sample['reference_pair_key']}", width="stretch")
    with c2:
        row = pd.Series(sample.get("face_row", {}))
        render_image(face_preview_path(row.get("ano"), row.get("id_face")), "Landmarks da face")
    with c3:
        st.write(
            {
                "sexo_face": sample.get("reference_sex"),
                "usar_genero": sample.get("use_gender"),
                "candidatos_filtrados_por_genero": sample.get("filtered_by_gender"),
            }
        )
    st.markdown("**Candidatos escolhidos**")
    for candidate in sample.get("candidates", []):
        with st.container(border=True):
            label = candidate["name"]
            if candidate.get("correct"):
                label += " · crânio correto"
            st.markdown(f"**{label}**")
            cols = st.columns(max(1, len(candidate["views"])))
            for col, view in zip(cols, candidate["views"]):
                with col:
                    st.image(view["image"], caption=view["label"], width="stretch")
            contour_path = existing_candidate_contour(candidate)
            if contour_path:
                render_image(contour_path, "Contorno/pontos do crânio")
            st.caption(
                f"sexo crânio previsto: {candidate.get('sexo_cranio')} · confiança: {candidate.get('confianca_cranio')} · "
                f"prob M: {candidate.get('prob_masculino_cranio')} · prob F: {candidate.get('prob_feminino_cranio')}"
            )


def existing_candidate_contour(candidate: dict) -> Path | None:
    value = str(candidate.get("contorno_path", "") or "")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() and path.is_file() else None


def candidate_by_rank_label(row: pd.Series) -> str:
    sex_match = row.get("genero_bate_referencia_texto", "indefinido")
    correct = " · correto" if bool(row.get("correto", False)) else ""
    position = row.get("posição_ranking_incluido", row.get("posição", 0))
    return (
        f"#{int(position)} · {row.get('candidato')}{correct} · "
        f"score {metric_float(row.get('score_final')):.4f} · genero bate: {sex_match}"
    )


def json_safe(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def render_candidate_detail(result: pd.DataFrame, face: Image.Image, candidates: list[dict], context: dict | None = None, key_prefix: str = "compare") -> None:
    if result.empty:
        return
    context = context or {}
    rows = result.sort_values("posição").copy()
    options = rows["candidato"].astype(str).tolist()
    labels = {str(row.get("candidato")): candidate_by_rank_label(row) for _, row in rows.iterrows()}
    selected_name = st.selectbox(
        "Crânio para detalhar",
        options,
        index=0,
        format_func=lambda item: labels.get(item, item),
        key=f"{key_prefix}_detail_candidate_{hash(tuple(options))}",
    )
    selected_row = rows[rows["candidato"].astype(str).eq(str(selected_name))].iloc[0]
    by_name = {item["name"]: item for item in candidates}
    candidate = by_name.get(str(selected_name), {})
    face_row = pd.Series(context.get("face_row", {}))

    st.markdown("**Resumo da comparação selecionada**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rank incluído", f"#{int(selected_row.get('posição_ranking_incluido', selected_row.get('posição', 0)))}")
    m2.metric("Score final", f"{metric_float(selected_row.get('score_final')):.4f}")
    m3.metric("Embedding", f"{metric_float(selected_row.get('score_embedding')):.4f}")
    m4.metric("Score sexo", f"{metric_float(selected_row.get('score_sexo')):.4f}")
    m5.metric("Delta lider", f"{metric_float(selected_row.get('diferença_para_líder')):.4f}")

    sex_match = selected_row.get("genero_bate_referencia")
    if sex_match is True:
        st.success("Sexo/gênero do crânio bate com a face de referência.")
    elif sex_match is False:
        st.error("Sexo/gênero do crânio diverge da face de referência.")
    else:
        st.info("Sexo/gênero indefinido para essa comparação.")
    if bool(selected_row.get("correto", False)):
        st.success("Este é o crânio correto da amostra.")
    elif selected_row.get("pair_key"):
        st.caption("Este candidato não é o par correto conhecido da amostra.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Face referência**")
        st.image(face, caption=f"Entrada · sexo: {selected_row.get('sexo_referencia', context.get('reference_sex', '-'))}", width="stretch")
        if not face_row.empty:
            preview = face_preview_path(face_row.get("ano"), face_row.get("id_face") or face_row.get("id_imagem"))
            normalized = face_visual_normalized_path(face_row)
            small_cols = st.columns(2)
            with small_cols[0]:
                render_image(preview, "Landmarks automáticos")
            with small_cols[1]:
                render_image(normalized, "Face normalizada")
    with c2:
        st.markdown("**Crânio selecionado**")
        if candidate:
            cols = st.columns(max(1, min(3, len(candidate.get("views", [])))))
            for col, view in zip(cols, candidate.get("views", [])):
                with col:
                    st.image(view["image"], caption=view["label"], width="stretch")
            contour_path = existing_candidate_contour(candidate)
            if contour_path:
                render_image(contour_path, "Contorno/pontos do crânio")
        else:
            st.info("Imagens desse candidato não ficaram salvas no estado da comparação.")

    st.markdown("**Parâmetros e sinais usados**")
    detail = {
        "modelo": context.get("model_label") or context.get("model_key"),
        "tipo_de_modelo": context.get("model_source"),
        "uso_do_genero": context.get("sex_mode", "Desligado"),
        "peso_genero": context.get("sex_weight", 0.0),
        "sexo_face_referencia": selected_row.get("sexo_referencia", context.get("reference_sex")),
        "sexo_cranio_candidato": selected_row.get("sexo_candidato") or candidate.get("sexo_cranio"),
        "genero_bate_referencia": selected_row.get("genero_bate_referencia_texto"),
        "filtrado_por_genero": bool(selected_row.get("filtrado_por_sexo", False)),
        "vistas_usadas": selected_row.get("vistas_usadas"),
        "confianca_cranio": selected_row.get("confianca_cranio"),
        "prob_masculino_cranio": selected_row.get("prob_masculino_cranio"),
        "prob_feminino_cranio": selected_row.get("prob_feminino_cranio"),
    }
    st.json({key: json_safe(value) for key, value in detail.items()}, expanded=False)


def render_compare_result(result: pd.DataFrame, face: Image.Image, candidates: list[dict], context: dict | None = None, key_prefix: str = "compare") -> None:
    active_result = result.copy()
    if "filtrado_por_sexo" in active_result and bool(context and context.get("use_gender")):
        active_result = active_result[~active_result["filtrado_por_sexo"].fillna(False)].copy()
    if active_result.empty:
        active_result = result.copy()
        st.warning("Todos os candidatos foram filtrados por gênero; mostrando ranking completo para auditoria.")
    active_result = active_result.sort_values("score_final" if "score_final" in active_result else "posição", ascending=False if "score_final" in active_result else True).reset_index(drop=True)
    active_result["posição_ranking_incluido"] = range(1, len(active_result) + 1)
    st.success(f"Melhor candidato incluído: {active_result.iloc[0].candidato}")
    display = active_result.copy()
    for col in ["score_embedding", "score_sexo", "score_final", "similaridade_media", "diferença_para_líder"]:
        if col in display:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
    show_cols = compact_columns(
        [
            "posição_ranking_incluido",
            "posição",
            "candidato",
            "correto",
            "sexo_referencia",
            "sexo_candidato",
            "genero_bate_referencia_texto",
            "score_embedding",
            "score_sexo",
            "score_final",
            "diferença_para_líder",
            "vistas_usadas",
            "confianca_cranio",
            "prob_masculino_cranio",
            "prob_feminino_cranio",
            "filtrado_por_sexo",
            "posição_sem_filtro_genero",
        ],
        display,
    )
    show_table(display[show_cols], height=320, key=f"{key_prefix}_result_{time.time_ns()}")
    discarded = result[result.get("filtrado_por_sexo", pd.Series(False, index=result.index)).fillna(False)].copy() if "filtrado_por_sexo" in result else pd.DataFrame()
    if not discarded.empty:
        with st.expander(f"Descartados por gênero ({len(discarded)})", expanded=True):
            st.caption("Eles não entram no ranking principal com o filtro ligado, mas a coluna abaixo mostra onde ficariam usando só o embedding.")
            discarded = discarded.sort_values("posição_sem_filtro_genero" if "posição_sem_filtro_genero" in discarded else "score_embedding")
            discarded_display = discarded.copy()
            for col in ["score_embedding", "score_sexo", "score_final", "similaridade_media", "diferença_para_líder"]:
                if col in discarded_display:
                    discarded_display[col] = pd.to_numeric(discarded_display[col], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
            discarded_cols = compact_columns(
                [
                    "posição_sem_filtro_genero",
                    "candidato",
                    "correto",
                    "sexo_referencia",
                    "sexo_candidato",
                    "genero_bate_referencia_texto",
                    "score_embedding",
                    "confianca_cranio",
                    "prob_masculino_cranio",
                    "prob_feminino_cranio",
                ],
                discarded_display,
            )
            show_table(discarded_display[discarded_cols], height=220, key=f"{key_prefix}_discarded")
    if "score_final" in active_result:
        st.bar_chart(active_result.set_index("candidato")["score_final"])
    st.markdown("**Detalhar ranking**")
    render_candidate_detail(active_result, face, candidates, context, key_prefix)
    st.markdown("**Face de referência**")
    st.image(face, caption="Face enviada", width=260)
    st.markdown("**Ranking visual dos crânios**")
    by_name = {item["name"]: item for item in candidates}
    for _, row in active_result.iterrows():
        candidate = by_name.get(str(row["candidato"]))
        with st.container(border=True):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Posição", f"#{int(row['posição'])}")
                st.metric("Score final", f"{metric_float(row.get('score_final')):.4f}")
                if bool(row.get("correto", False)):
                    st.success("Crânio correto")
                sex_match = row.get("genero_bate_referencia")
                if sex_match is True:
                    st.success("Gênero bate com a face")
                elif sex_match is False:
                    st.error("Gênero diverge da face")
                else:
                    st.info("Gênero indefinido")
                if row.get("genero_cranio_correto") is not None:
                    st.write(f"Gênero do crânio correto bateu: {'sim' if bool(row.get('genero_cranio_correto')) else 'não'}")
                st.caption(
                    f"embedding {metric_float(row.get('score_embedding')):.4f} · "
                    f"sexo {row.get('sexo_candidato')} · vistas {row.get('vistas_usadas')} · "
                    f"confiança {row.get('confianca_cranio', '')}"
                )
                if bool(row.get("filtrado_por_sexo", False)):
                    st.warning("Filtrado por sexo")
            with c2:
                if candidate:
                    contour_path = existing_candidate_contour(candidate)
                    extra = 1 if contour_path else 0
                    cols = st.columns(max(1, len(candidate["views"]) + extra))
                    for col, image_info in zip(cols, candidate["views"]):
                        with col:
                            st.image(image_info["image"], caption=image_info["label"], width="stretch")
                    if contour_path:
                        with cols[-1]:
                            render_image(contour_path, "Contorno/pontos")
                else:
                    st.info("Imagens desse candidato não ficaram salvas no estado da comparação.")


def checkpoint_label(row: pd.Series, best_key: str, latest_key: str) -> str:
    markers = []
    if str(row.get("key")) == best_key:
        markers.append("melhor")
    if str(row.get("key")) == latest_key:
        markers.append("mais recente")
    marker = f" ({', '.join(markers)})" if markers else ""
    mrr_value = row.get("mrr")
    metric = f"MRR {pct(mrr_value)}" if pd.notna(mrr_value) else "sem métrica"
    return (
        f"{row.get('label')}{marker} · {metric} · Top-1 {pct(row.get('top_1'))} · "
        f"{row.get('n_checkpoints')} ckpt · {row.get('fonte')}"
    )


def compare_model_selector() -> str | None:
    checkpoints = available_checkpoint_sets()
    if checkpoints.empty:
        st.error("Nenhum checkpoint `best.pt` encontrado. Rode um treino two-tower antes de comparar.")
        return None
    best_key = str(checkpoints.iloc[0].get("key"))
    latest_idx = checkpoints["mtime"].idxmax()
    latest_key = str(checkpoints.loc[latest_idx].get("key"))
    options = checkpoints["key"].astype(str).tolist()
    labels = {str(row.get("key")): checkpoint_label(row, best_key, latest_key) for _, row in checkpoints.iterrows()}
    selected_key = st.selectbox("Modelo/treinamento usado no ranking", options, index=0, format_func=lambda key: labels.get(key, key))
    selected = checkpoints[checkpoints["key"].astype(str).eq(selected_key)].iloc[0]
    status = "ensemble completo" if bool(selected.get("completo")) else "checkpoint disponível"
    st.caption(
        f"Selecionado: {selected.get('label')} · {status} · "
        f"{selected.get('n_checkpoints')} checkpoint(s) · anos: {selected.get('anos')}"
    )
    st.info(
        "Gallery-10 é uma avaliação do ranking em galerias simuladas. Ele mostra se o modelo acertou Top-1/Top-5/MRR, "
        "mas não é um checkpoint sozinho. No comparar, só aparecem modelos com arquivo `best.pt` disponível."
    )
    with st.expander("Ver todos os modelos encontrados", expanded=False):
        table = checkpoints.drop(columns=["checkpoints"], errors="ignore").copy()
        table["modificado"] = table["mtime"].map(lambda value: time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value))))
        show_table(table[compact_columns(["fonte", "label", "variant", "anos", "n_checkpoints", "completo", "top_1", "top_3", "top_5", "mrr", "modificado", "key"], table)], height=260, key="compare_models_table")
    gallery = gallery_summary_table()
    if not gallery.empty:
        with st.expander("Métricas Gallery-10 disponíveis", expanded=False):
            st.caption("Use essa tabela para escolher qual familia de treino parece melhor. Para usar no comparar, o checkpoint correspondente precisa existir.")
            show_table(gallery.head(20), height=300, key="compare_gallery_metrics")
    return selected_key


def selected_model_context(selected_key: str | None) -> dict:
    if not selected_key:
        return {}
    checkpoints = available_checkpoint_sets()
    if checkpoints.empty or "key" not in checkpoints:
        return {"model_key": selected_key}
    selected = checkpoints[checkpoints["key"].astype(str).eq(str(selected_key))]
    if selected.empty:
        return {"model_key": selected_key}
    row = selected.iloc[0]
    return {
        "model_key": str(row.get("key")),
        "model_label": str(row.get("label")),
        "model_source": str(row.get("fonte")),
        "model_variant": str(row.get("variant")),
        "model_top_1": row.get("top_1"),
        "model_top_5": row.get("top_5"),
        "model_mrr": row.get("mrr"),
        "model_checkpoints": row.get("n_checkpoints"),
        "model_years": row.get("anos"),
    }


def skull_page() -> None:
    st.header("Crânio")
    section = st.segmented_control("Etapa", ["Resumo", "Normalização", "Gênero", "Características"], default="Resumo")
    features = read_csv(SKULL_FEATURES)
    if features.empty:
        st.warning("Ainda não encontrei a tabela de crânios/features.")
        return

    if section == "Resumo":
        view = filter_by_year_status_sex(features, "cranio", "skull_summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Registros", len(view))
        c2.metric("Pessoas", view.get("id_pessoa", pd.Series(dtype=str)).nunique())
        c3.metric("OK", count_contains(view.get("status_cranio", pd.Series(dtype=str)), "OK"))
        c4.metric("Tripletas", int(view.get("tem_tripleta", pd.Series(dtype=bool)).fillna(False).sum()))
        c5.metric("Usar matching", int(view.get("usar_matching", pd.Series(dtype=bool)).fillna(False).sum()))
        show_table(skull_summary_table(view), height=240, key="skull_summary_year")
        open_images_button("Ver imagens", "cranio", view, ["CF", "CE", "CD"], "skull_summary_images")
        cols = compact_columns(["ano", "id_imagem", "id_pessoa", "status_cranio", "classe_cranio", "confianca_cranio", "tem_tripleta", "usar_matching"], view)
        show_table(view[cols].head(500), height=360, key="skull_summary_table")

    elif section == "Normalização":
        st.subheader("Ações")
        b1, b2, b3 = st.columns(3)
        with b1:
            run_model_step("Normalizar faltantes", ["normalize-images", "--size", "224"], "skull_norm_missing")
        with b2:
            run_model_step("Refazer normalização com force", ["normalize-images", "--size", "224", "--force"], "skull_norm_force")
        with b3:
            run_script_step("Comparar normalizações", "Treinamento_cranios/scripts/10_comparar_normalizacoes_sexo_cranio.py", "skull_compare_norm", ["--root", "Treinamento_cranios"])
        view = filter_by_year_status_sex(features, "cranio", "skull_norm")
        open_images_button("Ver imagens", "cranio", view, ["CF", "Normalizadas"], "skull_norm_images")
        show_table(read_csv(SKULL_NORMALIZATIONS), height=260, key="skull_norm_metrics")
        cols = compact_columns(["ano", "id_imagem", "status_cranio", "score_qualidade_cranio", "area_relativa_imagem", "compacidade", "circularidade", "elongacao", "motivo_qualidade_cranio"], view)
        show_table(view[cols].head(500), height=360, key="skull_norm_table")

    elif section == "Gênero":
        st.subheader("Ações")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            run_script_step("Preparar silhuetas", "Treinamento_cranios/scripts/05_preparar_silhuetas_sexo_cranio.py", "skull_silhouettes", ["--root", "Treinamento_cranios"])
        with b2:
            run_script_step("Treinar modelo sexo craniano", "Treinamento_cranios/scripts/07_treinar_sexo_cranio.py", "skull_gender_train", ["--root", "Treinamento_cranios"])
        with b3:
            run_script_step("Comparar normalizações", "Treinamento_cranios/scripts/10_comparar_normalizacoes_sexo_cranio.py", "skull_gender_compare", ["--root", "Treinamento_cranios"])
        with b4:
            run_script_step("Exportar features", "Treinamento_cranios/scripts/08_exportar_features_cranios_final.py", "skull_gender_export", ["--root", "Treinamento_cranios"])
        view = filter_by_year_status_sex(features, "cranio", "skull_gender")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Masculino", count_contains(view.get("classe_cranio", pd.Series(dtype=str)), "MASCULINO"))
        c2.metric("Feminino", count_contains(view.get("classe_cranio", pd.Series(dtype=str)), "FEMININO"))
        c3.metric("Alta", count_contains(view.get("confianca_cranio", pd.Series(dtype=str)), "ALTA"))
        c4.metric("Média/baixa", count_contains(view.get("confianca_cranio", pd.Series(dtype=str)), "MEDIA") + count_contains(view.get("confianca_cranio", pd.Series(dtype=str)), "BAIXA"))
        show_table(read_csv(SKULL_METRICS), height=170, key="skull_gender_metrics")
        show_table(read_csv(SKULL_VIEW_METRICS), height=220, key="skull_gender_view")
        show_table(read_csv(SKULL_CONFUSION), height=170, key="skull_gender_confusion")
        open_images_button("Ver imagens", "cranio", view, ["CF", "Contorno", "Shape64"], "skull_gender_images")
        cols = compact_columns(["ano", "id_imagem", "classe_cranio", "prob_masculino_cranio", "prob_feminino_cranio", "confianca_cranio", "modelo_sexo_cranio", "status_cranio"], view)
        show_table(view[cols].head(500), height=360, key="skull_gender_table")

    else:
        st.subheader("Ações")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            run_script_step("Extrair contorno", "Treinamento_cranios/scripts/01_extrair_contorno_cranios.py", "skull_contour", ["--root", "Treinamento_cranios"])
        with b2:
            run_script_step("Gerar shape64", "Treinamento_cranios/scripts/02_gerar_shape_cranios_64pts.py", "skull_shape64", ["--root", "Treinamento_cranios"])
        with b3:
            run_script_step("Gerar medidas", "Treinamento_cranios/scripts/03_gerar_medidas_cranios.py", "skull_measures", ["--root", "Treinamento_cranios"])
        with b4:
            if st.button("Checar features", width="stretch"):
                clear_caches()
                st.success("Features recarregadas.")
        view = filter_by_year_status_sex(features, "cranio", "skull_features")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas", len(view))
        c2.metric("Shape64 cols", len([c for c in view.columns if c.startswith("cranio_shape_")]))
        c3.metric("Contornos OK", count_contains(view.get("status_cranio", pd.Series(dtype=str)), "OK"))
        c4.metric("NaN", int(view.isna().sum().sum()))
        open_images_button("Ver imagens", "cranio", view, ["CF", "Contorno", "Shape64"], "skull_features_images")
        cols = compact_columns(["ano", "id_imagem", "area_contorno_cranio_px2", "perimetro_contorno_cranio_px", "assimetria_esq_dir", "compacidade", "circularidade", "elongacao", "score_outlier_cranio_pca"], view)
        show_table(view[cols].head(500), height=420, key="skull_features_table")


def matching_page() -> None:
    st.header("Matching")
    st.warning("Ranking experimental de similaridade. Não é identificação pericial.")
    section = st.segmented_control("Área", ["Resumo", "Treino", "Comparar", "Erros"], default="Resumo")

    if section == "Resumo":
        summary = project_summary()
        cols = st.columns(5)
        cols[0].metric("Pares válidos", summary["valid_pairs"])
        cols[1].metric("Excluídos", summary["excluded_pairs"])
        cols[2].metric("Splits", summary["split_rows"])
        cols[3].metric("Features face", summary["face_ok"])
        cols[4].metric("Features crânio", summary["skull_ok"])
        st.subheader("Métricas principais")
        show_table(read_csv(BASELINES), height=170, key="match_base")
        show_table(read_csv(GALLERY10), height=240, key="match_gallery")
        show_table(read_csv(LOYO), height=300, key="match_loyo")
        meta = select_metadata_rows(valid_only=True)
        open_images_button("Ver pares/imagens", "matching", meta, ["Face", "Crânio CF"], "match_summary_images")

    elif section == "Treino":
        config_path = training_config_panel()
        st.subheader("Ações")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            run_model_step("Construir metadata", ["build-index"], "match_build", config_path=config_path)
        with b2:
            run_model_step("Criar splits", ["create-splits"], "match_splits", config_path=config_path)
        with b3:
            run_model_step("Rodar baselines", ["baselines"], "match_baselines", config_path=config_path)
        with b4:
            run_model_step("Treinar two-tower", ["train"], "match_train", timeout=7200, config_path=config_path)
        b5, b6, b7 = st.columns(3)
        with b5:
            run_model_step("Rodar gallery-10", ["simulate-gallery", "--size", "10"], "match_gallery10", config_path=config_path)
        with b6:
            year = st.selectbox("LOYO ano", ["2017", "2019", "2021", "2022", "2023"], key="loyo_year")
            run_model_step("Rodar LOYO", ["loyo", "--year", year], "match_loyo_run", timeout=7200, config_path=config_path)
        with b7:
            run_model_step("Hard negatives/multiseed", ["multiseed-loyo", "--years", "all"], "match_hard", timeout=14400, config_path=config_path)
        st.subheader("Status das saídas")
        status = pipeline_status("Matching")
        status_metrics(status)
        show_table(status, height=260, key="match_train_status")
        st.subheader("Resultados dos treinamentos")
        st.caption("Top-1 = acertou o crânio correto em primeiro. Top-3/Top-5 = o correto apareceu entre os primeiros candidatos. MRR resume quanto mais alto o correto ficou no ranking.")
        tabs = st.tabs(["Gallery-10", "LOYO por ano", "Rankings/casos"])
        with tabs[0]:
            gallery = gallery_summary_table()
            if not gallery.empty:
                best = gallery.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Melhor Top-1", best.get("top_1", "-"))
                c2.metric("Melhor Top-5", best.get("top_5", "-"))
                c3.metric("Melhor MRR", best.get("mrr", "-"))
                c4.metric("Cenário", str(best.get("cenario", "-")))
            show_table(gallery, height=360, key="match_gallery_results")
        with tabs[1]:
            loyo = loyo_summary_table()
            if not loyo.empty:
                numeric = read_csv(LOYO)
                if not numeric.empty and "mrr" in numeric:
                    best = numeric.sort_values("mrr", ascending=False).iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Melhor ano", str(best.get("holdout_year", "-")))
                    c2.metric("Top-1", pct(best.get("top_1")))
                    c3.metric("Top-5", pct(best.get("top_5")))
                    c4.metric("MRR", pct(best.get("mrr")))
            show_table(loyo, height=420, key="match_loyo_results")
        with tabs[2]:
            raw_rankings = ranking_summary_raw()
            rankings = format_metric_table(raw_rankings.drop(columns=["mtime"], errors="ignore")) if not raw_rankings.empty else pd.DataFrame()
            show_table(rankings.head(80), height=420, key="match_ranking_results")
            selected_path = ranking_selector(raw_rankings, "train_result_ranking")
            if selected_path:
                detail = read_csv(selected_path)
                if not detail.empty:
                    correct = detail[detail["correto"].fillna(False)] if "correto" in detail else pd.DataFrame()
                    c1, c2, c3, c4 = st.columns(4)
                    ranks = pd.to_numeric(correct.get("rank", pd.Series(dtype=float)), errors="coerce")
                    c1.metric("Consultas", detail["query_id"].nunique() if "query_id" in detail else len(detail))
                    c2.metric("Top-1", pct(ranks.le(1).mean()) if len(ranks) else "-")
                    c3.metric("Top-5", pct(ranks.le(5).mean()) if len(ranks) else "-")
                    c4.metric("MRR", pct((1 / ranks).mean()) if len(ranks) else "-")
                    query_options = sorted(detail["query_id"].dropna().astype(str).unique()) if "query_id" in detail else []
                    if query_options:
                        query = st.selectbox("Consulta para ver com imagens", query_options, key="train_result_query")
                        render_ranking_visual(detail, query)
                    only_correct = st.checkbox("Na tabela, mostrar só a linha do candidato correto", value=True)
                    table = detail[detail["correto"].fillna(False)] if only_correct and "correto" in detail else detail
                    show_table(table.head(1000), height=420, key="train_result_detail")

    elif section == "Comparar":
        st.subheader("Comparar uma face contra 10 crânios")
        selected_model_key = compare_model_selector()
        mode = st.segmented_control("Origem das imagens", ["Amostra aleatória do banco", "Upload manual"], default="Amostra aleatória do banco")
        use_gender = st.checkbox("Usar gênero/sexo estimado para filtrar candidatos", value=True)
        sex_mode_label = "Filtro rígido opcional" if use_gender else "Desligado"
        sex_weight = 0.05
        model_context = selected_model_context(selected_model_key)
        st.caption(
            "Com o checkbox ligado, o app usa a tabela existente: sexo da face como referência e sexo craniano previsto para os candidatos. "
            "Os candidatos de outro sexo ficam no fim do ranking pelo filtro rígido."
        )

        @st.cache_resource(show_spinner=False)
        def cached_ensemble(model_key: str | None):
            return load_ensemble(model_key)

        if mode == "Amostra aleatória do banco":
            c1, c2, c3 = st.columns([1, 1, 2])
            seed_value = c1.number_input("Semente aleatória", min_value=0, max_value=999999, value=int(time.time()) % 100000, step=1)
            if c2.button("Selecionar aleatoriamente", width="stretch"):
                sample = build_random_compare_sample(use_gender, int(seed_value))
                if sample is None:
                    st.error("Não consegui montar uma galeria com 1 face, o crânio correto e 9 candidatos.")
                else:
                    st.session_state["random_compare_sample"] = sample
                    st.success("Amostra selecionada. Confira as imagens e clique em comparar.")
            c3.info("A seleção pega a face de referência, inclui o crânio correto e sorteia 9 crânios de outros candidatos. Eles não precisam ser do mesmo ano.")
            sample = st.session_state.get("random_compare_sample")
            if sample:
                render_random_sample(sample)
            ready = bool(selected_model_key and sample and sample.get("face") is not None and len(sample.get("candidates", [])) == 10)
            if st.button("Comparar e ranquear amostra", type="primary", disabled=not ready):
                try:
                    with st.spinner("Normalizando imagens e calculando ranking..."):
                        candidates_payload = [(item["name"], [view["image"] for view in item["views"]]) for item in sample["candidates"]]
                        candidate_sexes = [item["sex"] for item in sample["candidates"]]
                        result = rank_uploaded_candidates(
                            sample["face"],
                            candidates_payload,
                            cached_ensemble(selected_model_key),
                            reference_sex=sample["reference_sex"] if use_gender else "INDEFINIDO",
                            candidate_sexes=candidate_sexes if use_gender else ["INDEFINIDO"] * len(candidate_sexes),
                            sex_mode=sex_mode_label,
                            sex_weight=sex_weight,
                        )
                        result = enrich_compare_result(result, sample["candidates"], sample["reference_pair_key"], sample["reference_sex"])
                    context = {
                        **model_context,
                        "origem": "amostra_aleatoria",
                        "use_gender": use_gender,
                        "sex_mode": sex_mode_label,
                        "sex_weight": sex_weight,
                        "reference_pair_key": sample["reference_pair_key"],
                        "reference_sex": sample["reference_sex"],
                        "face_row": sample.get("face_row", {}),
                    }
                    correct_row = result[result.get("correto", pd.Series(dtype=bool)).fillna(False)]
                    if not correct_row.empty:
                        rank_value = int(correct_row.iloc[0]["posição"])
                        gender_ok = correct_row.iloc[0].get("genero_cranio_correto")
                        st.info(f"Crânio correto ficou na posição #{rank_value}. Sexo craniano correto compatível com a face: {'sim' if bool(gender_ok) else 'não'}.")
                    st.session_state["last_compare_result"] = result
                    st.session_state["last_compare_face"] = sample["face"]
                    st.session_state["last_compare_candidates"] = sample["candidates"]
                    st.session_state["last_compare_context"] = context
                    render_compare_result(result, sample["face"], sample["candidates"], context, key_prefix="compare_random_now")
                except Exception as exc:
                    st.error(f"Não foi possível calcular o ranking: {exc}")
        else:
            face_file = st.file_uploader("Fotografia frontal da face", type=["png", "jpg", "jpeg"], key="face_query")
            inferred_face_row = infer_row_from_uploaded_names([face_file] if face_file else [], "face")
            inferred_reference_sex = normalize_sex_label(inferred_face_row.get("sexo_face")) if inferred_face_row is not None else "INDEFINIDO"
            reference_sex = st.selectbox(
                "Sexo conhecido da face de referência",
                ["INDEFINIDO", "MASCULINO", "FEMININO"],
                index=sex_option_index(inferred_reference_sex),
                key="compare_reference_sex",
                disabled=not use_gender,
            )
            if use_gender and inferred_face_row is not None:
                st.caption(f"Sexo da face puxado da tabela: {inferred_reference_sex} · {inferred_face_row.get('pair_key')}")
            uploads = []
            for index in range(10):
                with st.expander(f"Candidato {index + 1}", expanded=index == 0):
                    name = st.text_input("Identificação", value=f"Candidato {index + 1}", key=f"candidate_name_{index}")
                    multi_views = st.file_uploader(
                        "Enviar vistas do crânio juntas (CF/CE/CD, aceita 1 a 3 imagens)",
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=True,
                        key=f"skull_views_{index}",
                    )
                    inferred_skull_row = infer_row_from_uploaded_names(multi_views or [], "cranio")
                    inferred_skull_sex = normalize_sex_label(inferred_skull_row.get("sexo_cranio")) if inferred_skull_row is not None else "INDEFINIDO"
                    sex = st.selectbox(
                        "Sexo estimado/conhecido do crânio",
                        ["INDEFINIDO", "MASCULINO", "FEMININO"],
                        index=sex_option_index(inferred_skull_sex),
                        key=f"candidate_sex_{index}",
                        disabled=not use_gender,
                    )
                    if use_gender and inferred_skull_row is not None:
                        st.caption(f"Sexo craniano puxado da tabela: {inferred_skull_sex} · {inferred_skull_row.get('pair_key')}")
                    st.caption("Ideal: CF + CE + CD. O ranking aceita 1 a 3 vistas; com menos vistas, trate o resultado como menos robusto.")
                    uploads.append((name, sex, sorted(multi_views or [], key=skull_view_sort_key)[:3], inferred_skull_row))
            ready = bool(selected_model_key) and face_file is not None and all(len(views) >= 1 for _, _, views in uploads)
            if st.button("Comparar e ranquear uploads", type="primary", disabled=not ready):
                try:
                    with st.spinner("Normalizando imagens e consultando ensemble..."):
                        face = uploaded_image(face_file)
                        candidates_payload = []
                        candidates_visual = []
                        candidate_sexes = []
                        for index, (name, sex, views, inferred_skull_row) in enumerate(uploads):
                            inferred_key = inferred_skull_row.get("pair_key") if inferred_skull_row is not None else ""
                            display_name = str(inferred_key or f"{index + 1}. {name}")
                            images = [{"label": getattr(view, "name", f"vista {n + 1}"), "image": uploaded_image(view)} for n, view in enumerate(views)]
                            candidates_payload.append((display_name, [item["image"] for item in images]))
                            candidates_visual.append(
                                {
                                    "name": display_name,
                                    "pair_key": inferred_key,
                                    "sex": normalize_sex_label(sex),
                                    "views": images,
                                    "correct": bool(inferred_face_row is not None and inferred_key and inferred_key == inferred_face_row.get("pair_key")),
                                    "sexo_cranio": normalize_sex_label(sex),
                                    "confianca_cranio": inferred_skull_row.get("confianca_cranio", "") if inferred_skull_row is not None else "",
                                    "prob_masculino_cranio": inferred_skull_row.get("prob_masculino_cranio", "") if inferred_skull_row is not None else "",
                                    "prob_feminino_cranio": inferred_skull_row.get("prob_feminino_cranio", "") if inferred_skull_row is not None else "",
                                    "contorno_path": str(skull_preview_path(inferred_skull_row.get("ano"), inferred_skull_row.get("id_cranio_cf"), "contorno")) if inferred_skull_row is not None else "",
                                }
                            )
                            candidate_sexes.append(sex)
                        result = rank_uploaded_candidates(
                            face,
                            candidates_payload,
                            cached_ensemble(selected_model_key),
                            reference_sex=reference_sex if use_gender else "INDEFINIDO",
                            candidate_sexes=candidate_sexes if use_gender else ["INDEFINIDO"] * len(candidate_sexes),
                            sex_mode=sex_mode_label,
                            sex_weight=sex_weight,
                        )
                        reference_key = inferred_face_row.get("pair_key") if inferred_face_row is not None else None
                        result = enrich_compare_result(result, candidates_visual, reference_key, reference_sex)
                    context = {
                        **model_context,
                        "origem": "upload_manual",
                        "use_gender": use_gender,
                        "sex_mode": sex_mode_label,
                        "sex_weight": sex_weight,
                        "reference_pair_key": reference_key,
                        "reference_sex": reference_sex,
                        "face_row": inferred_face_row.to_dict() if inferred_face_row is not None else {},
                    }
                    st.session_state["last_compare_result"] = result
                    st.session_state["last_compare_face"] = face
                    st.session_state["last_compare_candidates"] = candidates_visual
                    st.session_state["last_compare_context"] = context
                    render_compare_result(result, face, candidates_visual, context, key_prefix="compare_upload_now")
                except Exception as exc:
                    st.error(f"Não foi possível calcular o ranking: {exc}")
            elif not ready:
                st.caption("Envie a face e pelo menos uma vista para cada um dos 10 candidatos para habilitar o ranking.")
        if "last_compare_result" in st.session_state:
            with st.expander("Última comparação nesta sessão", expanded=False):
                render_compare_result(
                    st.session_state["last_compare_result"],
                    st.session_state["last_compare_face"],
                    st.session_state["last_compare_candidates"],
                    st.session_state.get("last_compare_context", {}),
                    key_prefix="compare_last",
                )

    else:
        st.subheader("Casos difíceis")
        selected_path = ranking_selector(ranking_summary_raw(), "errors_ranking_selector")
        if not selected_path:
            st.info("Nenhum ranking neural encontrado.")
            return
        rank = read_csv(selected_path)
        only_errors = st.checkbox("Mostrar apenas quando correto não ficou em Top-1", value=True)
        view = rank.copy()
        if only_errors and {"correto", "rank"}.issubset(view.columns):
            view = view[view["correto"].fillna(False) & (pd.to_numeric(view["rank"], errors="coerce") > 1)]
        show_table(view.head(500), height=430, key="match_errors")
        if not view.empty and "query_id" in view:
            query = st.selectbox("Abrir erro/consulta com imagens", sorted(view["query_id"].dropna().astype(str).unique()), key="error_query_visual")
            render_ranking_visual(rank, query)


def reports_logs_page() -> None:
    st.header("Relatórios / Logs")
    section = st.segmented_control("Área", ["Relatórios", "Logs", "Testes"], default="Relatórios")
    if section == "Relatórios":
        rows = pd.DataFrame(
            [
                {"arquivo": REPORT_MD.relative_to(REPO_ROOT).as_posix(), "tipo": "Markdown", "existe": REPORT_MD.exists()},
                {"arquivo": REPORT_HTML.relative_to(REPO_ROOT).as_posix(), "tipo": "HTML com imagens", "existe": REPORT_HTML.exists()},
                {"arquivo": METADATA.relative_to(REPO_ROOT).as_posix(), "tipo": "CSV", "existe": METADATA.exists()},
                {"arquivo": BASELINES.relative_to(REPO_ROOT).as_posix(), "tipo": "CSV", "existe": BASELINES.exists()},
                {"arquivo": FACE_FEATURES.relative_to(REPO_ROOT).as_posix(), "tipo": "CSV", "existe": FACE_FEATURES.exists()},
                {"arquivo": SKULL_FEATURES.relative_to(REPO_ROOT).as_posix(), "tipo": "CSV", "existe": SKULL_FEATURES.exists()},
            ]
        )
        show_table(rows, height=260, key="reports")
        if REPORT_HTML.exists():
            st.download_button("Baixar HTML com imagens", REPORT_HTML.read_bytes(), file_name=REPORT_HTML.name, mime="text/html")
        if REPORT_MD.exists():
            st.download_button("Baixar Markdown", REPORT_MD.read_bytes(), file_name=REPORT_MD.name, mime="text/markdown")
    elif section == "Logs":
        logs = latest_logs()
        show_table(logs, height=520, key="logs_table")
        if not logs.empty:
            selected = st.selectbox("Abrir log/arquivo", logs["arquivo"].astype(str).tolist())
            path = REPO_ROOT / selected
            if path.exists() and path.stat().st_size < 500_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    st.markdown(f'<div class="log-box"><pre>{text[-12000:]}</pre></div>', unsafe_allow_html=True)
                except Exception:
                    st.info("Arquivo selecionado não é texto.")
    else:
        show_table(audit_tests(), height=320, key="tests")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Recriar metadata", width="stretch"):
                with st.spinner("Construindo metadata..."):
                    _, report = build_metadata()
                clear_caches()
                st.success("Metadata recriada.")
                st.json(report)
        with b2:
            if st.button("Criar splits", width="stretch"):
                with st.spinner("Criando splits..."):
                    split = create_splits(metadata_df())
                    create_leave_one_year_out(metadata_df())
                clear_caches()
                st.success("Splits criados.")
                st.write(split["split"].value_counts())
        with b3:
            if st.button("Auditar domínios", width="stretch"):
                with st.spinner("Auditando imagens..."):
                    _, summary = audit_domains(metadata_df())
                clear_caches()
                show_table(summary, height=260)
        with b4:
            if st.button("Baselines", width="stretch"):
                test = read_csv(PROJECT_DIR / "data/splits/test.csv") if (PROJECT_DIR / "data/splits/test.csv").exists() else splits_df()
                with st.spinner("Rodando baselines..."):
                    result = run_baselines(test)
                clear_caches()
                show_table(result, height=220)
        st.divider()
        limit = st.number_input("Limite para teste visual rápido", 1, 200, 12)
        force = st.checkbox("Force/refazer saídas derivadas", value=False)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Preprocessar amostra", width="stretch"):
                result = preprocess_manifest(metadata_df(), 224, force, int(limit))
                clear_caches()
                show_table(result, height=260)
        with c2:
            if st.button("Normalizar amostra", width="stretch"):
                result = normalize_domains(metadata_df(), 224, force, int(limit))
                clear_caches()
                show_table(result, height=260)


PAGES = {
    "Início": page_inicio,
    "Face": face_page,
    "Crânio": skull_page,
    "Matching": matching_page,
    "Relatórios / Logs": reports_logs_page,
}


st.title("Gigi · Modelo Novo")
st.caption("Painel local para face, crânio e matching por embeddings automáticos.")

with st.sidebar:
    st.subheader("Navegação")
    page = st.radio("Tela", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Poucas áreas. Os detalhes ficam dentro de cada etapa.")
    if st.button("Atualizar dados", width="stretch"):
        clear_caches()
        st.rerun()

PAGES[page]()
