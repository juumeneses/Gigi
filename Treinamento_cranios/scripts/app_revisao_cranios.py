# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from common_cranios import *


st.set_page_config(page_title="Treinamento Crânios", page_icon="☠️", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container{padding-top:1rem;max-width:1680px}
    div[data-testid="stMetric"]{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px}
    .caption-box{font-size:.84rem;color:#374151;line-height:1.35;margin:.35rem 0 1rem;padding:.45rem .55rem;border-left:3px solid #d1d5db;background:#f9fafb}
    .caption-box strong{color:#111827}
    .ok{color:#166534;font-weight:700}
    .review{color:#a16207;font-weight:700}
    .bad{color:#991b1b;font-weight:700}
    .flow-title{font-size:1.05rem;font-weight:750;color:#111827;margin-bottom:.15rem}
    .flow-help{font-size:.92rem;color:#4b5563;line-height:1.35;margin:.25rem 0 .75rem}
    .flow-tag{display:inline-block;border-radius:999px;padding:.12rem .5rem;font-size:.76rem;font-weight:750;margin-bottom:.35rem}
    .flow-required{background:#dcfce7;color:#166534}
    .flow-optional{background:#e0f2fe;color:#075985}
    .flow-emergency{background:#fee2e2;color:#991b1b}
    .eye-cell div[data-testid="stButton"]{display:flex;justify-content:center}
    .eye-cell button{margin-left:auto;margin-right:auto}
    </style>
    """,
    unsafe_allow_html=True,
)

SCRIPTS = {
    "pipeline": "run_pipeline_cranios.py",
    "preprocess": "09_preprocess_cranios_normalizados.py",
    "check": "00_checar_dataset_cranios.py",
    "contorno": "01_extrair_contorno_cranios.py",
    "shape": "02_gerar_shape_cranios_64pts.py",
    "medidas": "03_gerar_medidas_cranios.py",
    "qualidade": "04_analisar_qualidade_cranios.py",
    "silhuetas": "05_preparar_silhuetas_sexo_cranio.py",
    "dataset_sexo": "06_juntar_genero_faces_cranios.py",
    "treinar_sexo": "07_treinar_sexo_cranio.py",
    "exportar": "08_exportar_features_cranios_final.py",
    "comparar_normalizacoes": "10_comparar_normalizacoes_sexo_cranio.py",
}


def guess_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_python(root: Path) -> Path:
    for p in [root.parent / ".venv" / "bin" / "python", root / ".venv" / "bin" / "python", root.parent / "Treinamento_faces" / ".venv" / "bin" / "python"]:
        if p.exists():
            return p
    return Path(sys.executable)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_PROGRESS_PREFIX = "__EPOCH_PROGRESS__"
EPOCH_PROGRESS_CURRENT_PREFIX = "__EPOCH_PROGRESS_CURRENT__"
EPOCH_PROGRESS_DONE_PREFIX = "__EPOCH_PROGRESS_DONE__"


def stop_training_flag(root: Path) -> Path:
    return sexo_dir(root) / "parar_treino_sexo_cranio.flag"


def current_process_pid_path(root: Path) -> Path:
    return logs_dir(root) / "processo_atual_cranios.pid"


def training_lock_path(root: Path) -> Path:
    return sexo_dir(root) / "treino_sexo_cranio.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def force_stop_pid(pid: int) -> bool:
    if pid <= 0 or not _pid_alive(pid):
        return False
    stopped = False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        stopped = True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except Exception:
            pass
    time.sleep(0.7)
    if _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            stopped = True
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
                stopped = True
            except Exception:
                pass
    return stopped


def force_stop_training(root: Path) -> list[int]:
    stop_training_flag(root).parent.mkdir(parents=True, exist_ok=True)
    stop_training_flag(root).write_text(pd.Timestamp.now().isoformat(), encoding="utf-8")
    pids = []
    for p in [current_process_pid_path(root), training_lock_path(root)]:
        try:
            if p.exists():
                pid = int(p.read_text(encoding="utf-8", errors="ignore").strip())
                if pid not in pids:
                    pids.append(pid)
        except Exception:
            pass
    stopped = []
    for pid in pids:
        if force_stop_pid(pid):
            stopped.append(pid)
    for p in [current_process_pid_path(root), training_lock_path(root)]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    return stopped


def clean_log_line(line: str) -> str:
    text = ANSI_RE.sub("", line.replace("\r", "\n"))
    text = text.replace("━", "-").replace("█", "#").replace("▏", "|").replace("▎", "|").replace("▍", "|").replace("▌", "|").replace("▋", "|").replace("▊", "|").replace("▉", "|")
    return "\n".join(part.rstrip() for part in text.splitlines() if part.strip())


def save_command_log(root: Path, key: str, cmd: list[str], rc: int, logs: list[str], progress: list[str]) -> Path:
    path = logs_dir(root) / f"falha_{key}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f"timestamp={pd.Timestamp.now().isoformat()}",
        f"codigo_saida={rc}",
        "comando:",
        " ".join(cmd),
        "",
        "progresso:",
        *(progress or ["(sem progresso capturado)"]),
        "",
        "logs:",
        *(logs or ["(sem logs capturados)"]),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def run_cmd(root: Path, key: str, years=None, force=False, extra=None):
    cmd = [str(project_python(root)), str(root / "scripts" / SCRIPTS[key]), "--root", str(root)]
    if years:
        cmd += ["--years"] + [str(y) for y in years]
    if force:
        cmd.append("--force")
    if extra:
        cmd += extra
    if key == "treinar_sexo" and stop_training_flag(root).exists():
        stop_training_flag(root).unlink()
    st.code(" ".join(cmd), language="bash")
    progress_box = st.empty()
    box = st.empty()
    logs = []
    epoch_done_lines = []
    epoch_current_line = ""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(resultados_root(root) / "cache" / "matplotlib"))
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", start_new_session=True)
    current_process_pid_path(root).parent.mkdir(parents=True, exist_ok=True)
    current_process_pid_path(root).write_text(str(proc.pid), encoding="utf-8")
    for line in proc.stdout:
        cleaned = clean_log_line(line)
        if cleaned:
            for part in cleaned.splitlines():
                if part.startswith(EPOCH_PROGRESS_CURRENT_PREFIX):
                    epoch_current_line = part.replace(EPOCH_PROGRESS_CURRENT_PREFIX, "", 1).strip()
                    lines = epoch_done_lines[-6:] + ([epoch_current_line] if epoch_current_line else [])
                    progress_box.info("\n".join(lines))
                elif part.startswith(EPOCH_PROGRESS_DONE_PREFIX):
                    epoch_done_lines.append(part.replace(EPOCH_PROGRESS_DONE_PREFIX, "", 1).strip())
                    epoch_current_line = ""
                    lines = epoch_done_lines[-6:]
                    progress_box.info("\n".join(lines))
                elif part.startswith(EPOCH_PROGRESS_PREFIX):
                    epoch_current_line = part.replace(EPOCH_PROGRESS_PREFIX, "", 1).strip()
                    lines = epoch_done_lines[-6:] + ([epoch_current_line] if epoch_current_line else [])
                    progress_box.info("\n".join(lines))
                else:
                    logs.append(part)
        if logs:
            box.code("\n".join(logs[-260:]), language="text")
    rc = proc.wait()
    stopped_by_user = stop_training_flag(root).exists() and rc != 0
    try:
        if current_process_pid_path(root).exists() and current_process_pid_path(root).read_text(encoding="utf-8").strip() == str(proc.pid):
            current_process_pid_path(root).unlink()
    except Exception:
        pass
    if stopped_by_user:
        try:
            stop_training_flag(root).unlink()
        except Exception:
            pass
        st.warning("Execução interrompida pelo usuário.")
        return 130
    if rc == 0:
        st.success("Concluído.")
    else:
        full_progress = epoch_done_lines[-20:] + ([epoch_current_line] if epoch_current_line else [])
        log_path = save_command_log(root, key, cmd, rc, logs, full_progress)
        st.error(f"Falhou com código {rc}.")
        st.warning(f"Log completo salvo em: {log_path}")
        with st.expander("Log completo da falha", expanded=True):
            if full_progress:
                st.markdown("**Progresso capturado**")
                st.code("\n".join(full_progress), language="text")
            st.markdown("**Saída completa**")
            st.code("\n".join(logs) if logs else "Sem logs capturados.", language="text")
    return rc


def show_df(df: pd.DataFrame, height=320):
    if df is None or df.empty:
        st.info("Sem dados.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().isin(["TRUE", "1", "SIM", "YES", "OK"])


def col_or_empty(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index, dtype=str)


def load_tables(root: Path):
    check = read_csv_safe(checks_dir(root) / "check_cranios.csv")
    trip = read_csv_safe(checks_dir(root) / "tripletas_cranios.csv")
    qual = read_csv_safe(qualidade_dir(root) / "qualidade_cranios.csv")
    pred = read_csv_safe(sexo_dir(root) / "predicoes_sexo_cranio.csv")
    final = read_csv_safe(features_dir(root) / "features_matching_cranios_final.csv")
    for df in [check, trip, qual, pred, final]:
        for c in ["ano", "id_imagem", "id_pessoa"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
    return check, trip, qual, pred, final


def merged_quality(root: Path) -> pd.DataFrame:
    _, _, qual, pred, _ = load_tables(root)
    if qual.empty:
        return qual
    out = qual.copy()
    med = read_csv_safe(medidas_dir(root) / "medidas_cranio_todos_anos.csv")
    if not med.empty:
        for c in ["ano", "id_imagem", "id_pessoa"]:
            if c in med.columns:
                med[c] = med[c].astype(str)
        keep = [c for c in ["ano", "id_imagem", "id_pessoa", "imagem_cranio", "area_relativa_imagem", "bbox_ratio", "compacidade"] if c in med.columns]
        out = out.merge(med[keep], on=["ano", "id_imagem", "id_pessoa"], how="left")
    if not pred.empty:
        out = out.merge(pred, on=["ano", "id_pessoa"], how="left", suffixes=("", "_sexo"))
    out["tipo_status"] = np.where(col_or_empty(out, "status_manual").astype(str).str.contains("MANUAL", case=False, na=False), "MANUAL", "AUTO")
    out["status_exibicao"] = out["status_cranio"].astype(str) + " " + out["tipo_status"].astype(str)
    return out


def image_path_from_row(row) -> Path | None:
    path = str(row.get("caminho", "") or row.get("imagem_cranio", "") or "")
    if path and Path(path).exists():
        return Path(path)
    return None


def image_for_quality(root: Path, row, mode: str) -> Path | None:
    y = str(row.get("ano", ""))
    stem = str(row.get("id_imagem", ""))
    if mode == "contorno":
        p = previews_dir(root) / y / f"{stem}_contorno_preview.jpg"
        return p if p.exists() else None
    if mode == "shape64":
        p = previews_dir(root) / y / f"{stem}_shape64_preview.jpg"
        return p if p.exists() else None
    path = str(row.get("imagem_cranio", ""))
    return Path(path) if path and Path(path).exists() else None


def render_image(path: Path | None, caption: str):
    if path and path.exists():
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Imagem ausente: {caption}")


def find_person_side_image(root: Path, year: str, person_id: str, side: str, normalized: bool = False, normalization_kind: str = "atual") -> Path | None:
    if normalized:
        base = normalized_kind_dir(root, normalization_kind, year, side)
    else:
        base = side_dir(root, year, side)
    if not base.exists():
        return None
    for ext in IMAGE_EXTS:
        candidate = base / f"{person_id}_{side}{ext}"
        if candidate.exists():
            return candidate
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and person_id_from_stem(p.stem) == person_id:
            return p
    return None


def person_images(root: Path, year: str, person_id: str, normalization_kind: str = "atual") -> list[dict[str, Path | str | None]]:
    rows = []
    for side in SIDES:
        rows.append({
            "lado": side,
            "original": find_person_side_image(root, year, person_id, side, normalized=False),
            "normalizada": find_person_side_image(root, year, person_id, side, normalized=True, normalization_kind=normalization_kind),
        })
    return rows


def divergence_review_path(root: Path) -> Path:
    return resultados_root(root) / "diagnosticos" / "curadoria_divergencias_cranios.csv"


def read_divergence_review(root: Path) -> pd.DataFrame:
    path = divergence_review_path(root)
    df = read_csv_safe(path)
    expected = ["ano", "id_pessoa", "decisao_divergencia", "usar_no_retreino", "observacao", "revisado_em"]
    if df.empty:
        return pd.DataFrame(columns=expected)
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df["ano"] = df["ano"].astype(str)
    df["id_pessoa"] = df["id_pessoa"].astype(str)
    return df[expected]


def save_divergence_review(root: Path, row: dict):
    path = divergence_review_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = read_divergence_review(root)
    ano = str(row.get("ano", ""))
    id_pessoa = str(row.get("id_pessoa", ""))
    mask = df["ano"].astype(str).eq(ano) & df["id_pessoa"].astype(str).eq(id_pessoa)
    clean = {k: row.get(k, "") for k in ["ano", "id_pessoa", "decisao_divergencia", "usar_no_retreino", "observacao", "revisado_em"]}
    if mask.any():
        for k, v in clean.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([clean])], ignore_index=True)
    df.to_csv(path, index=False)


def latest_divergence_diagnostic(root: Path) -> pd.DataFrame:
    diag_root = resultados_root(root) / "diagnosticos"
    files = sorted(diag_root.glob("**/diagnostico_divergencias_resumo.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for path in files:
        df = read_csv_safe(path)
        if not df.empty and {"ano", "id_pessoa"}.issubset(df.columns):
            df["ano"] = df["ano"].astype(str)
            df["id_pessoa"] = df["id_pessoa"].astype(str)
            keep = [c for c in ["ano", "id_pessoa", "flags", "n_flags", "ce_direcao", "cd_direcao", "ce_orient_conf", "cd_orient_conf"] if c in df.columns]
            return df[keep]
    return pd.DataFrame()


def current_divergences(root: Path) -> pd.DataFrame:
    comp = read_csv_safe(sexo_dir(root) / "comparacao_genero_face_cranio.csv")
    if comp.empty or "resultado_genero" not in comp.columns:
        return pd.DataFrame()
    comp["ano"] = comp["ano"].astype(str)
    comp["id_pessoa"] = comp["id_pessoa"].astype(str)
    div = comp[comp["resultado_genero"].astype(str).str.upper().eq("DIVERGENCIA")].copy()
    if div.empty:
        return div
    diag = latest_divergence_diagnostic(root)
    if not diag.empty:
        div = div.merge(diag, on=["ano", "id_pessoa"], how="left")
    review = read_divergence_review(root)
    if not review.empty:
        div = div.merge(review, on=["ano", "id_pessoa"], how="left")
    for col in ["decisao_divergencia", "usar_no_retreino", "observacao", "flags"]:
        if col not in div.columns:
            div[col] = ""
    div["decisao_divergencia"] = div["decisao_divergencia"].fillna("PENDENTE").replace("", "PENDENTE")
    div["usar_no_retreino"] = div["usar_no_retreino"].fillna("SIM").replace("", "SIM")
    div["observacao"] = div["observacao"].fillna("")
    div["flags"] = div["flags"].fillna("sem_diagnostico")
    if "prob_masculino_cranio_pct" not in div.columns and "prob_masculino_cranio" in div.columns:
        div["prob_masculino_cranio_pct"] = (pd.to_numeric(div["prob_masculino_cranio"], errors="coerce") * 100).round(2)
    return div


@st.dialog("Visualizar crânio")
def show_person_images_dialog(root: Path, year: str, person_id: str, normalization_kind: str | None = None):
    normalization_kind = normalization_kind or st.session_state.get("sexo_normalization_kind", "atual")
    if normalization_kind not in NORMALIZATION_KINDS:
        normalization_kind = "atual"
    st.write(f"**Ano:** {year}")
    st.write(f"**ID pessoa:** `{person_id}`")
    st.caption(f"Comparação das imagens originais com a normalização `{normalization_kind}`, quando disponível.")
    any_image = False
    for item in person_images(root, str(year), str(person_id), str(normalization_kind)):
        side = str(item["lado"])
        st.subheader(side)
        c1, c2 = st.columns(2)
        with c1:
            render_image(item["original"], f"{side} original")
        with c2:
            render_image(item["normalizada"], f"{side} normalizada")
        any_image = any_image or bool(item["original"]) or bool(item["normalizada"])
    if not any_image:
        st.warning("Nenhuma imagem original ou normalizada foi encontrada para esta pessoa.")
    if st.button("Fechar", key=f"close_modal_{year}_{person_id}"):
        st.rerun()


def render_comparison_with_view_buttons(root: Path, df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Sem dados.")
        return
    hidden_cols = {
        "prob_masculino_cranio",
        "prob_feminino_cranio",
        "modelo_sexo_cranio",
        "threshold_masculino",
        "estrategia_voto",
    }
    text_cols = {
        "genero_definido_face",
        "genero_previsto_cranio",
        "resultado_genero",
        "confianca_cranio",
        "classe_cranio",
        "normalizacao",
    }
    display_cols = [c for c in df.columns if c not in hidden_cols]
    ordered = ["ano", "id_pessoa"] + [c for c in display_cols if c not in {"ano", "id_pessoa"}]
    labels = ["ano", "Ver"] + [c for c in ordered if c != "ano"]
    weights = [0.65, 0.55, 1.55] + [1.45] * max(len(labels) - 3, 0)
    st.caption(f"Mostrando {len(df)} registro(s). Clique no olho para abrir as imagens original e normalizada.")
    header = st.columns(weights)
    for col, label in zip(header, labels):
        col.markdown(f"**{label}**")
    for idx, row in df.reset_index(drop=True).iterrows():
        row_cols = st.columns(weights)
        year = str(row.get("ano", ""))
        person_id = str(row.get("id_pessoa", ""))
        row_cols[0].write(year)
        with row_cols[1]:
            st.markdown('<div class="eye-cell">', unsafe_allow_html=True)
            if st.button("👁", help="Visualizar imagens original e normalizada", key=f"ver_cranio_cmp_{year}_{person_id}_{idx}"):
                show_person_images_dialog(root, year, person_id, str(row.get("normalizacao", "") or ""))
            st.markdown("</div>", unsafe_allow_html=True)
        for col, name in zip(row_cols[2:], [c for c in ordered if c != "ano"]):
            value = row.get(name, "")
            if isinstance(value, float):
                col.write(f"{value:.2f}")
            elif name in text_cols:
                col.write(str(value).replace("_", " ").lower().capitalize())
            else:
                col.write(value)


def metric_value(row: pd.Series, names: list[str], pct_value: bool = False) -> str:
    for name in names:
        if name in row.index:
            try:
                value = float(row.get(name))
                if np.isfinite(value):
                    if pct_value:
                        return f"{value:.2f}%" if value > 1 else f"{value * 100:.2f}%"
                    return f"{value:.4f}"
            except Exception:
                pass
    return "-"


def show_training_metric_explanations(root: Path):
    metrics_df = read_csv_safe(sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
    if metrics_df.empty:
        st.info("Métricas do treinamento ainda não disponíveis. Rode o treino para preencher accuracy, loss e validação.")
        return
    row = metrics_df.iloc[-1]
    with st.container(border=True):
        st.markdown("**Métricas do treinamento**")
        st.caption("Use estas métricas para avaliar se o treino está aprendendo e se generaliza bem. Loss/validation loss passam a ser salvas nas próximas execuções do treino.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", metric_value(row, ["train_accuracy_pct", "accuracy_pct", "train_accuracy", "accuracy"], pct_value=True))
        m1.caption("Percentual de acerto. Bom quando fica alto; para este dataset, busque acima de 85% junto com boa acurácia balanceada.")
        m2.metric("Loss", metric_value(row, ["train_loss", "best_train_loss"]))
        m2.caption("Erro do modelo no treino. Bom quando é baixo e cai ao longo das épocas; sozinho não garante bom resultado.")
        m3.metric("Validation accuracy", metric_value(row, ["val_accuracy_pct", "melhor_val_accuracy_pct", "val_accuracy", "best_val_accuracy"], pct_value=True))
        m3.caption("Acerto em dados de validação. Bom quando acompanha o treino sem ficar muito abaixo dele.")
        m4.metric("Validation loss", metric_value(row, ["val_loss", "best_val_loss"]))
        m4.caption("Erro na validação. Bom quando é baixo; se sobe enquanto a loss do treino cai, pode indicar overfitting.")


def pct(v) -> str:
    try:
        value = float(v)
        if np.isnan(value):
            return "-"
        return f"{value * 100:.1f}%"
    except Exception:
        return "-"


def gender_label(row) -> str:
    classe = str(row.get("classe_cranio", "") or row.get("classe_cranio_sexo", "") or "INDEFINIDO")
    conf = str(row.get("confianca_cranio", "") or "INDEFINIDO")
    pm = pct(row.get("prob_masculino_cranio", np.nan))
    pf = pct(row.get("prob_feminino_cranio", np.nan))
    return f"{classe} · {conf} · M {pm} · F {pf}"


def _move_page(key: str, delta: int, last_page: int):
    st.session_state[key] = max(1, min(last_page, int(st.session_state.get(key, 1)) + delta))


def gallery_navigation(total: int, key: str, default_size: int = 24):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    page_size = c1.selectbox("Imagens por página", [12, 24, 48, 96], index=[12, 24, 48, 96].index(default_size), key=f"{key}_size")
    last_page = max(1, int(np.ceil(total / int(page_size))))
    page_key = f"{key}_page"
    st.session_state[page_key] = max(1, min(last_page, int(st.session_state.get(page_key, 1))))
    page = int(st.session_state[page_key])
    c2.button("Anterior", use_container_width=True, disabled=page <= 1, key=f"{key}_prev", on_click=_move_page, args=(page_key, -1, last_page))
    c3.markdown(f"<p style='text-align:center;margin:.55rem 0 0'>Página <b>{page}</b> de <b>{last_page}</b></p>", unsafe_allow_html=True)
    c4.button("Próxima", use_container_width=True, disabled=page >= last_page, key=f"{key}_next", on_click=_move_page, args=(page_key, 1, last_page))
    return page, int(page_size)


def render_gallery(root: Path, df: pd.DataFrame, per_row: int, key: str):
    page, page_size = gallery_navigation(len(df), key)
    view = df.iloc[(page - 1) * page_size : page * page_size].reset_index(drop=True)
    for start in range(0, len(view), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (_, row) in zip(cols, view.iloc[start : start + per_row].iterrows()):
            with col:
                path = image_path_from_row(row)
                render_image(path, f"{row.get('ano', '')} · {row.get('lado', '')} · {row.get('id_imagem', '')}")
                status = str(row.get("status", ""))
                complete = str(row.get("tem_tripleta", "")).upper() in ["TRUE", "1", "SIM", "YES"]
                trip = "tripleta completa" if complete else "tripleta incompleta"
                css = "ok" if complete else "review"
                st.markdown(
                    f"<div class='caption-box'><strong>{row.get('id_pessoa', '')}</strong>"
                    f"<br>{row.get('lado', '')} · {status} · <span class='{css}'>{trip}</span>"
                    f"<br>Sexo craniano: <strong>{gender_label(row)}</strong></div>",
                    unsafe_allow_html=True,
                )


def render_landmark_gallery(root: Path, df: pd.DataFrame, per_row: int, key: str):
    page, page_size = gallery_navigation(len(df), key)
    view = df.iloc[(page - 1) * page_size : page * page_size].reset_index(drop=True)
    for start in range(0, len(view), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, (_, row) in zip(cols, view.iloc[start : start + per_row].iterrows()):
            with col:
                render_image(image_for_quality(root, row, "shape64"), f"{row.get('ano', '')} · {row.get('id_imagem', '')}")
                status = str(row.get("status_exibicao", row.get("status_cranio", ""))).upper()
                css = "ok" if "OK" in status else ("bad" if "EXCLUIR" in status else "review")
                st.markdown(
                    f"<div class='caption-box'><strong>{row.get('id_pessoa', '')}</strong>"
                    f"<br><span class='{css}'>{status}</span> · score {row.get('score_qualidade_cranio', '-')}"
                    f"<br>Sexo craniano: <strong>{gender_label(row)}</strong></div>",
                    unsafe_allow_html=True,
                )


def _move_review(key: str, total: int, delta: int):
    idx_key = f"{key}_idx"
    st.session_state[idx_key] = max(0, min(total - 1, int(st.session_state.get(idx_key, 0)) + delta))


def review_select(df: pd.DataFrame, key: str) -> pd.Series | None:
    if df.empty:
        st.info("Nenhum item nesta lista.")
        return None
    df = df.reset_index(drop=True)
    idx_key = f"{key}_idx"
    st.session_state[idx_key] = max(0, min(len(df) - 1, int(st.session_state.get(idx_key, 0))))
    opts = [
        f"{r.get('ano', '')} | {r.get('id_imagem', '')} | {r.get('status_exibicao', r.get('status_cranio', ''))}"
        for _, r in df.iterrows()
    ]
    selected = st.selectbox("Item", opts, index=st.session_state[idx_key], key=f"{key}_select")
    st.session_state[idx_key] = opts.index(selected)
    idx = int(st.session_state[idx_key])
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.button("Anterior", use_container_width=True, disabled=idx <= 0, key=f"{key}_prev", on_click=_move_review, args=(key, len(df), -1))
    c2.markdown(f"<p style='text-align:center;margin:.55rem 0 0'>Item <b>{idx + 1}</b> de <b>{len(df)}</b></p>", unsafe_allow_html=True)
    c3.button("Próximo", use_container_width=True, disabled=idx >= len(df) - 1, key=f"{key}_next", on_click=_move_review, args=(key, len(df), 1))
    return df.iloc[idx]


def save_quality_decision(root: Path, row, decision: str, obs: str):
    qual_path = qualidade_dir(root) / "qualidade_cranios.csv"
    qual = read_csv_safe(qual_path)
    if qual.empty:
        return False, "Arquivo de qualidade vazio."
    for c in ["ano", "id_imagem"]:
        qual[c] = qual[c].astype(str)
    mask = qual["ano"].eq(str(row["ano"])) & qual["id_imagem"].eq(str(row["id_imagem"]))
    if not mask.any():
        return False, "Item não encontrado na qualidade."
    qual.loc[mask, "status_manual"] = decision
    qual.loc[mask, "observacao_manual"] = obs
    if decision == "OK_MANUAL":
        qual.loc[mask, "status_cranio"] = "OK"
        qual.loc[mask, "usar_matching"] = True
    elif decision == "EXCLUIR_MANUAL":
        qual.loc[mask, "status_cranio"] = "EXCLUIR"
        qual.loc[mask, "usar_matching"] = False
    elif decision == "REVISAR_MANUAL":
        qual.loc[mask, "status_cranio"] = "REVISAR"
        qual.loc[mask, "usar_matching"] = False
    elif decision == "LIMPAR_MANUAL":
        qual.loc[mask, "status_manual"] = ""
        qual.loc[mask, "observacao_manual"] = ""
        med = read_csv_safe(medidas_dir(root) / "medidas_cranio_todos_anos.csv")
        if not med.empty:
            med["ano"] = med["ano"].astype(str)
            med["id_imagem"] = med["id_imagem"].astype(str)
            mm = med["ano"].eq(str(row["ano"])) & med["id_imagem"].eq(str(row["id_imagem"]))
            if mm.any():
                status, usar, motivo, score = quality_from_row(med.loc[mm].iloc[0])
                qual.loc[mask, "status_cranio"] = status
                qual.loc[mask, "usar_matching"] = usar
                qual.loc[mask, "motivo_qualidade_cranio"] = motivo
                qual.loc[mask, "score_qualidade_cranio"] = round(score, 3)
    safe_to_csv(qual, qual_path)
    return True, "Decisão salva."


root = Path(st.sidebar.text_input("Pasta Treinamento_cranios", str(guess_root()))).expanduser().resolve()
ensure_base_dirs(root)
years = discover_years(root)
st.sidebar.caption(f"Anos: {', '.join(years) if years else 'nenhum'}")

page = st.sidebar.radio(
    "Página",
    ["Painel", "Pipeline", "Matching/Curadoria", "Gênero", "Divergências", "Imagens", "Métricas", "Logs"],
)

if page == "Painel":
    st.title("Treinamento Crânios")
    check, trip, qual, pred, final = load_tables(root)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Imagens", len(check))
    c2.metric("Indivíduos", len(trip))
    c3.metric("Tripletas completas", int(trip.get("tem_tripleta", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not trip.empty else 0)
    c4.metric("OK matching", int(truthy(qual.get("usar_matching", pd.Series(dtype=str))).sum()) if not qual.empty else 0)
    c5.metric("Sexo estimado", len(pred))
    show_df(read_csv_safe(qualidade_dir(root) / "resumo_qualidade_cranios.csv"), 220)
    if not trip.empty:
        st.subheader("Tripletas incompletas ignoradas")
        miss = trip[~trip["tem_tripleta"].fillna(False).astype(bool)].copy()
        show_df(miss[[c for c in ["ano", "id_pessoa", "imagem_cf", "imagem_ce", "imagem_cd"] if c in miss.columns]], 220)

elif page == "Pipeline":
    st.title("Pipeline")
    st.caption("Pipeline geométrico: check, contorno, shape 64, medidas, qualidade e exportação. O sexo craniano fica na aba própria.")
    sel = st.multiselect("Anos", years, default=years)
    force = st.checkbox("Refazer outputs gerados (--force)")
    use_norm = st.checkbox("Usar imagens normalizadas no pipeline", value=False, help="Quando marcado, contorno e silhuetas leem cranios_normalizados_pb se existir.")
    st.subheader("Pré-processamento")
    cpre1, cpre2 = st.columns([1, 3])
    force_norm = cpre1.checkbox("Refazer normalizadas", key="cranio_force_norm")
    if cpre2.button("Gerar cranios_normalizados_pb", type="primary"):
        run_cmd(root, "preprocess", sel, force_norm)
    norm_log = read_csv_safe(checks_dir(root) / "preprocess_cranios_normalizados.csv")
    if not norm_log.empty:
        show_df(norm_log.tail(200), 220)
    extra_norm = ["--use-normalized"] if use_norm else None
    if st.button("Rodar pipeline completo", type="primary"):
        run_cmd(root, "pipeline", sel, force, extra=extra_norm)
    etapa = st.selectbox("Etapa individual", ["check", "contorno", "shape", "medidas", "qualidade", "exportar"])
    if st.button("Rodar etapa"):
        extra = extra_norm if etapa == "contorno" else None
        run_cmd(root, etapa, sel if etapa in ["check", "contorno", "shape", "medidas"] else None, force, extra=extra)

elif page == "Imagens":
    st.title("Imagens")
    st.caption("Mostra somente o crânio frontal (CF) com landmarks/shape 64. A curadoria manual fica na aba Matching/Curadoria.")
    view = merged_quality(root)
    if view.empty:
        st.info("Rode contorno, shape e qualidade primeiro.")
    else:
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        ano = f1.selectbox("Ano", ["TODOS"] + years, key="img_ano")
        lista = f2.selectbox("Qualidade/status", ["TODAS", "REVISAR", "OK_AUTO", "OK_MANUAL", "EXCLUIDAS"], key="img_lista")
        genero = f3.selectbox("Gênero previsto", ["TODOS", "MASCULINO", "FEMININO", "INDEFINIDO"], key="img_genero")
        visual = f4.selectbox("Visualização", ["lado a lado", "marcação", "original"], key="img_visual")
        per_row = int(f5.selectbox("Por fileira", [3, 4, 5], index=1, key="img_cols"))
        busca = f6.text_input("Buscar id/iniciais", key="img_busca")
        if ano != "TODOS":
            view = view[view["ano"].astype(str).eq(str(ano))].copy()
        if lista == "REVISAR":
            view = view[view["status_cranio"].astype(str).str.upper().eq("REVISAR")].copy()
        elif lista == "OK_AUTO":
            view = view[view["status_cranio"].astype(str).str.upper().eq("OK") & ~col_or_empty(view, "status_manual").astype(str).str.contains("MANUAL", case=False, na=False)].copy()
        elif lista == "OK_MANUAL":
            view = view[col_or_empty(view, "status_manual").astype(str).str.upper().eq("OK_MANUAL")].copy()
        elif lista == "EXCLUIDAS":
            view = view[view["status_cranio"].astype(str).str.upper().eq("EXCLUIR")].copy()
        if genero != "TODOS":
            if genero == "INDEFINIDO":
                view = view[~col_or_empty(view, "classe_cranio").astype(str).str.upper().str.contains("MASCULINO|FEMININO", regex=True, na=False)].copy()
            else:
                view = view[col_or_empty(view, "classe_cranio").astype(str).str.upper().str.contains(genero, na=False)].copy()
        if busca:
            needle = busca.strip().lower()
            view = view[view.astype(str).agg(" ".join, axis=1).str.lower().str.contains(needle, regex=False, na=False)].copy()
        view = view.sort_values(["ano", "id_imagem"]).reset_index(drop=True)
        st.caption(f"{len(view)} crânio(s) frontal(is) neste filtro.")
        resumo_cols = [c for c in ["ano", "id_pessoa", "id_imagem", "status_exibicao", "motivo_qualidade_cranio", "score_qualidade_cranio", "classe_cranio", "confianca_cranio", "prob_masculino_cranio", "prob_feminino_cranio"] if c in view.columns]
        show_df(view[resumo_cols], 240)
        page_num, page_size = gallery_navigation(len(view), "img_cf_gallery")
        page_df = view.iloc[(page_num - 1) * page_size : page_num * page_size].reset_index(drop=True)
        for start in range(0, len(page_df), per_row):
            cols = st.columns(per_row, gap="medium")
            for col, (_, row) in zip(cols, page_df.iloc[start : start + per_row].iterrows()):
                with col:
                    if visual == "lado a lado":
                        a, b = st.columns(2)
                        with a:
                            render_image(image_for_quality(root, row, "original"), "Original")
                        with b:
                            render_image(image_for_quality(root, row, "shape64"), "Marcação")
                    elif visual == "marcação":
                        render_image(image_for_quality(root, row, "shape64"), "Marcação")
                    else:
                        render_image(image_for_quality(root, row, "original"), "Original")
                    status = str(row.get("status_exibicao", row.get("status_cranio", ""))).upper()
                    css = "ok" if "OK" in status else ("bad" if "EXCLUIR" in status else "review")
                    st.markdown(
                        f"<div class='caption-box'><strong>{row.get('ano', '')} | {row.get('id_imagem', '')}</strong>"
                        f"<br><span class='{css}'>{status}</span> · score {row.get('score_qualidade_cranio', '-')}"
                        f"<br>Gênero: <strong>{gender_label(row)}</strong></div>",
                        unsafe_allow_html=True,
                    )

elif page == "Matching/Curadoria":
    st.title("Matching/Curadoria")
    st.caption("Revise os contornos/shape 64, aprove manualmente ou exclua do matching. OK_AUTO vem da análise automática; OK_MANUAL é sua decisão.")
    qual = merged_quality(root)
    if qual.empty:
        st.info("Rode qualidade primeiro.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        ano = f1.selectbox("Ano", ["TODOS"] + years, key="cur_ano")
        lista = f2.selectbox("Lista", ["REVISAR", "TODAS", "OK_AUTO", "OK_MANUAL", "EXCLUIDAS"], key="cur_lista")
        visual = f3.selectbox("Visualização", ["lado a lado", "shape64", "contorno", "original"], key="cur_visual")
        busca = f4.text_input("Buscar id/iniciais", key="cur_busca")
        df = qual.copy()
        if ano != "TODOS":
            df = df[df["ano"].astype(str).eq(str(ano))].copy()
        if lista == "REVISAR":
            df = df[df["status_cranio"].astype(str).str.upper().eq("REVISAR")].copy()
        elif lista == "OK_AUTO":
            df = df[df["status_cranio"].astype(str).str.upper().eq("OK") & ~col_or_empty(df, "status_manual").astype(str).str.contains("MANUAL", case=False, na=False)].copy()
        elif lista == "OK_MANUAL":
            df = df[col_or_empty(df, "status_manual").astype(str).str.upper().eq("OK_MANUAL")].copy()
        elif lista == "EXCLUIDAS":
            df = df[df["status_cranio"].astype(str).str.upper().eq("EXCLUIR")].copy()
        if busca:
            needle = busca.strip().lower()
            df = df[df.astype(str).agg(" ".join, axis=1).str.lower().str.contains(needle, regex=False, na=False)].copy()
        st.caption(f"{len(df)} item(ns) nesta lista.")
        row = review_select(df.sort_values(["ano", "id_imagem"]).reset_index(drop=True), "cur_landmarks")
        if row is not None:
            c1, c2 = st.columns([1.25, 1])
            with c1:
                if visual == "lado a lado":
                    a, b, c = st.columns(3)
                    with a:
                        render_image(image_for_quality(root, row, "original"), "Original")
                    with b:
                        render_image(image_for_quality(root, row, "contorno"), "Contorno")
                    with c:
                        render_image(image_for_quality(root, row, "shape64"), "Shape 64")
                else:
                    render_image(image_for_quality(root, row, visual), visual)
            with c2:
                st.subheader(str(row.get("id_imagem", "")))
                st.write(f"Status: **{row.get('status_exibicao', row.get('status_cranio', ''))}**")
                st.write(f"Motivo: `{row.get('motivo_qualidade_cranio', '')}`")
                st.write(f"Score: `{row.get('score_qualidade_cranio', '')}`")
                st.write(f"Sexo craniano: **{gender_label(row)}**")
                show_df(pd.DataFrame([row.to_dict()]), 220)
                decisao = st.selectbox("Decisão", ["OK_MANUAL", "REVISAR_MANUAL", "EXCLUIR_MANUAL", "LIMPAR_MANUAL"], key=f"dec_{row.get('ano')}_{row.get('id_imagem')}")
                obs = st.text_input("Observação", value=str(row.get("observacao_manual", "") or ""), key=f"obs_{row.get('ano')}_{row.get('id_imagem')}")
                if st.button("Salvar decisão", type="primary"):
                    ok, msg = save_quality_decision(root, row, decisao, obs)
                    st.success(msg) if ok else st.warning(msg)
                    time.sleep(0.15)
                    st.rerun()
                if st.button("Salvar e exportar final"):
                    ok, msg = save_quality_decision(root, row, decisao, obs)
                    st.success(msg) if ok else st.warning(msg)
                    if ok:
                        run_cmd(root, "exportar")
                        time.sleep(0.15)
                        st.rerun()
        st.subheader("Tabela filtrada")
        show_df(df, 360)

elif page == "Gênero":
    st.title("Gênero")
    st.caption("Fluxo separado: as silhuetas CF/CE/CD do crânio são treinadas usando como rótulo o gênero final definido/curado nas faces.")
    faces_root = st.text_input("Treinamento_faces", str(root.parent / "Treinamento_faces"), key="sexo_faces_root")
    sel = st.multiselect("Anos para silhuetas", years, default=years, key="sexo_years")
    norm_options = ["original", *NORMALIZATION_KINDS]
    norm_silhouette_options = ["todas as normalizações", *norm_options]
    norm_generation_options = ["Todas", *NORMALIZATION_KINDS]

    st.subheader("Fluxo recomendado")
    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-required">Obrigatório para comparar normalizações</span>
            <div class="flow-title">1. Gerar imagens normalizadas</div>
            <div class="flow-help">
            Cria cópias dos crânios em pastas separadas para cada estratégia de pré-processamento.
            Os originais não são alterados. Rode antes de treinar com uma variante nova ou antes de comparar todas.
            </div>
            """,
            unsafe_allow_html=True,
        )
        n1, n2, n3 = st.columns([1.5, 1.0, 1.0])
        gerar_norm = n1.selectbox("Normalização", norm_generation_options, index=0, key="sexo_generate_normalization_kind")
        force_norm_genero = n2.checkbox("Refazer imagens normalizadas", key="sexo_force_norm_variantes")
        if n3.button("Gerar normalizadas", type="primary", key="btn_sexo_gerar_normalizadas"):
            kind_arg = "all" if gerar_norm == "Todas" else gerar_norm
            run_cmd(root, "preprocess", sel, force_norm_genero, extra=["--normalization-kind", kind_arg])

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-required">Obrigatório quando imagens/silhuetas mudarem</span>
            <div class="flow-title">2. Preparar silhuetas</div>
            <div class="flow-help">
            Gera as silhuetas binárias CF/CE/CD que serão usadas pelo treino de sexo craniano.
            Rode quando adicionar imagens, trocar anos, normalizar crânios ou quiser recriar as silhuetas de uma variante.
            </div>
            """,
            unsafe_allow_html=True,
        )
        s1, s2 = st.columns([1, 2])
        force = s1.checkbox(
            "Refazer silhuetas existentes",
            help="Opcional. Sobrescreve silhuetas já geradas; use quando mudou imagens, normalização ou quer limpar resultados antigos.",
            key="sexo_force",
        )
        sil_norm_kind = s2.selectbox(
            "Normalização para silhuetas",
            norm_silhouette_options,
            index=0,
            help="Escolha original para usar as imagens brutas ou uma variante normalizada já gerada.",
            key="sexo_silhueta_normalization_kind",
        )
        if st.button("Preparar silhuetas", type="primary", key="btn_sexo_silhuetas"):
            sil_kind_arg = "all" if sil_norm_kind == "todas as normalizações" else sil_norm_kind
            run_cmd(root, "silhuetas", sel, force, extra=["--normalization-kind", sil_kind_arg])

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-required">Obrigatório quando a curadoria das faces mudar</span>
            <div class="flow-title">3. Juntar gênero das faces</div>
            <div class="flow-help">
            Copia o <code>genero_final</code> curado no treinamento de faces para montar o dataset de treino do crânio.
            Rode novamente sempre que corrigir gênero nas faces ou alterar a pasta de faces usada como referência.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Fonte dos rótulos: `{faces_root}`")
        if st.button("Juntar gênero das faces", key="btn_sexo_dataset"):
            run_cmd(root, "dataset_sexo", extra=["--faces-root", faces_root])

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-required">Obrigatório para atualizar predições</span>
            <div class="flow-title">4. Treinar modelo</div>
            <div class="flow-help">
            Treina o classificador de sexo craniano com as silhuetas e os rótulos vindos das faces.
            Use <strong>VGG16 fine-tuning</strong> para o melhor modelo atual; <strong>VGG16 modo artigo</strong>
            para reproduzir a ideia por silhuetas; <strong>VGG16 embeddings</strong> para um teste intermediário;
            e <strong>Rápido</strong> apenas para validação curta. Se selecionar <strong>todas</strong>, o app treina
            uma run por normalização e gera o comparativo final.
            </div>
            """,
            unsafe_allow_html=True,
        )
        t0, t1 = st.columns([1, 1])
        train_norm_kind = t0.selectbox("Treinar com normalização", ["todas", *norm_options], index=2, key="sexo_normalization_kind")
        backend = t1.selectbox("Modelo", ["VGG16 fine-tuning", "VGG16 modo artigo", "VGG16 embeddings", "Rápido"], index=0, key="sexo_backend")
        t2, t3, t4, t5, t6 = st.columns(5)
        default_epochs = 20 if backend == "VGG16 fine-tuning" else (18 if backend == "VGG16 modo artigo" else 8)
        epochs = t2.number_input("Épocas", min_value=1, max_value=80, value=default_epochs, step=1, key="sexo_epochs")
        image_size = t3.selectbox("Tamanho", [128, 160, 224, 256], index=3 if backend in {"VGG16 fine-tuning", "VGG16 modo artigo"} else 0, key="sexo_image_size")
        batch_size = t4.selectbox("Batch", [4, 8, 16, 32], index=1, key="sexo_batch_size")
        weights = t5.selectbox("Pesos", ["imagenet", "none"], index=0, key="sexo_weights")
        learning_rate = t6.selectbox("Learning rate", [0.00001, 0.000005, 0.000002, 0.000001], index=0 if backend == "VGG16 fine-tuning" else 2, format_func=lambda x: f"{x:g}", key="sexo_learning_rate")
        if backend == "Rápido":
            treino_extra = ["--backend", "sklearn"]
        elif backend == "VGG16 modo artigo":
            treino_extra = ["--backend", "article_vgg16", "--epochs", str(int(epochs)), "--image-size", str(int(image_size)), "--batch-size", str(int(batch_size)), "--weights", weights, "--learning-rate", str(float(learning_rate))]
        elif backend == "VGG16 fine-tuning":
            treino_extra = ["--backend", "vgg16_finetune", "--epochs", str(int(epochs)), "--image-size", str(int(image_size)), "--batch-size", str(int(batch_size)), "--weights", weights, "--learning-rate", str(float(learning_rate))]
        else:
            treino_extra = ["--backend", "vgg16", "--epochs", str(int(epochs)), "--image-size", str(int(image_size)), "--batch-size", str(int(batch_size)), "--weights", weights]
        if st.button("Treinar modelo", key="btn_sexo_treinar"):
            if train_norm_kind == "todas":
                compare_extra = [
                    "--normalization-kinds",
                    "original",
                    *NORMALIZATION_KINDS,
                    "--backend",
                    treino_extra[1],
                    "--epochs",
                    str(int(epochs)),
                    "--image-size",
                    str(int(image_size)),
                    "--batch-size",
                    str(int(batch_size)),
                    "--weights",
                    weights,
                    "--learning-rate",
                    str(float(learning_rate)),
                ]
                if force:
                    compare_extra.append("--force-silhouettes")
                run_cmd(root, "comparar_normalizacoes", sel, extra=compare_extra)
            else:
                rc = run_cmd(root, "silhuetas", sel, force, extra=["--normalization-kind", train_norm_kind])
                if rc == 0:
                    rc = run_cmd(root, "treinar_sexo", extra=treino_extra + ["--normalization-kind", train_norm_kind])
                if rc == 0:
                    run_cmd(root, "exportar")

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-required">Obrigatório após treinar ou revisar</span>
            <div class="flow-title">5. Exportar final</div>
            <div class="flow-help">
            Atualiza os arquivos finais usados pelo matching. Rode depois de treinar, alterar curadoria
            ou quando quiser garantir que o matching está lendo a versão mais recente.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Exportar final", key="btn_sexo_exportar"):
            run_cmd(root, "exportar")

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-emergency">Opcional / emergência</span>
            <div class="flow-title">Parar treino</div>
            <div class="flow-help">
            Interrompe um treino em andamento ou travado e limpa o lock/sinal antigo. Use apenas quando
            o treino não responde, foi iniciado sem querer ou precisa liberar uma nova execução.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Parar treino", key="btn_sexo_parar"):
            stopped = force_stop_training(root)
            if stopped:
                st.error(f"Treino interrompido imediatamente. PID(s): {', '.join(map(str, stopped))}")
            else:
                st.warning("Parada solicitada, mas nenhum processo ativo foi encontrado. O lock/sinal antigo foi limpo.")
    if stop_training_flag(root).exists():
        st.warning("Há uma solicitação de parada ativa para o treino craniano.")

    with st.container(border=True):
        st.markdown(
            """
            <span class="flow-tag flow-optional">Opcional / avançado</span>
            <div class="flow-title">Fluxo automático completo</div>
            <div class="flow-help">
            Roda as etapas 1 → 4 em sequência: preparar silhuetas, juntar rótulos das faces, treinar e exportar.
            Use quando quiser refazer o processo inteiro de ponta a ponta. Para investigar problemas, prefira rodar
            os cards separadamente.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Rodar fluxo de gênero/sexo completo", key="btn_sexo_fluxo_completo"):
            full_norm_kind = st.session_state.get("sexo_normalization_kind", "crop_pad_percentile")
            if full_norm_kind == "todas":
                full_norm_kind = "crop_pad_percentile"
            rc = run_cmd(root, "silhuetas", sel, force, extra=["--normalization-kind", full_norm_kind])
            if rc == 0:
                rc = run_cmd(root, "dataset_sexo", extra=["--faces-root", faces_root])
            if rc == 0:
                rc = run_cmd(root, "treinar_sexo", extra=treino_extra + ["--normalization-kind", full_norm_kind])
            if rc == 0:
                run_cmd(root, "exportar")
    st.divider()
    st.info("VGG16 modo artigo treina sobre silhuetas 2D com augmentation flip/rotação ±5° e avalia por voto CF/CE/CD. VGG16 embeddings é o modo anterior, mais rápido.")
    comparativo_norm = read_csv_safe(sexo_dir(root) / "comparativo_normalizacoes.csv")
    if not comparativo_norm.empty:
        st.subheader("Comparativo de normalizações")
        comp_norm = comparativo_norm.copy()
        sort_cols = [c for c in ["balanced_accuracy_pct", "f1_pct", "acertos_pct"] if c in comp_norm.columns]
        if sort_cols:
            comp_norm = comp_norm.sort_values(sort_cols, ascending=False).reset_index(drop=True)
            melhor = str(comp_norm.iloc[0].get("normalizacao", ""))
            st.success(f"Melhor normalização atual: {melhor}")
        cols_norm = [c for c in ["melhor_normalizacao", "normalizacao", "status", "acertos_pct", "accuracy_pct", "balanced_accuracy_pct", "f1_pct", "precision_pct", "recall_pct", "val_accuracy_pct", "val_loss", "run_id"] if c in comp_norm.columns]
        show_df(comp_norm[cols_norm] if cols_norm else comp_norm, 300)
    dataset = read_csv_safe(sexo_dir(root) / "dataset_treino_sexo_cranio.csv")
    predicoes = read_csv_safe(sexo_dir(root) / "predicoes_sexo_cranio.csv")
    comparacao = read_csv_safe(sexo_dir(root) / "comparacao_genero_face_cranio.csv")
    if not dataset.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rótulos vindos das faces", len(dataset))
        c2.metric("Masculino face", int(dataset.get("genero_final", pd.Series(dtype=str)).astype(str).str.upper().eq("MASCULINO").sum()))
        c3.metric("Feminino face", int(dataset.get("genero_final", pd.Series(dtype=str)).astype(str).str.upper().eq("FEMININO").sum()))
        if comparacao.empty and not predicoes.empty:
            comparacao = dataset[["ano", "id_pessoa", "genero_final"]].drop_duplicates().merge(predicoes, on=["ano", "id_pessoa"], how="left")
            comparacao = comparacao.rename(columns={"genero_final": "genero_definido_face"})
            comparacao["genero_previsto_cranio"] = comparacao.get("classe_cranio", pd.Series("", index=comparacao.index)).astype(str).str.upper().map(
                lambda x: "MASCULINO" if "MASCULINO" in x else ("FEMININO" if "FEMININO" in x else "INDEFINIDO")
            )
            comparacao["resultado_genero"] = np.where(
                comparacao["genero_previsto_cranio"].eq("INDEFINIDO"),
                "INDEFINIDO",
                np.where(comparacao["genero_definido_face"].astype(str).str.upper().eq(comparacao["genero_previsto_cranio"]), "ACERTO", "DIVERGENCIA"),
            )
        if "prob_masculino_cranio" in comparacao.columns and "prob_masculino_cranio_pct" not in comparacao.columns:
            comparacao["prob_masculino_cranio_pct"] = (comparacao["prob_masculino_cranio"].astype(float) * 100).round(2)
        if "prob_feminino_cranio" in comparacao.columns and "prob_feminino_cranio_pct" not in comparacao.columns:
            comparacao["prob_feminino_cranio_pct"] = (comparacao["prob_feminino_cranio"].astype(float) * 100).round(2)
        if not comparacao.empty:
            st.subheader("Acerto do gênero craniano vs gênero definido na face")
            res = comparacao.get("resultado_genero", pd.Series("", index=comparacao.index)).astype(str).str.upper()
            total_cmp = max(len(comparacao), 1)
            acertos = int(res.eq("ACERTO").sum())
            divergencias = int(res.eq("DIVERGENCIA").sum())
            indefinidos = int(res.eq("INDEFINIDO").sum())
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Comparados", len(comparacao))
            r2.metric("Acertos", f"{acertos} ({acertos / total_cmp * 100:.1f}%)")
            r3.metric("Divergências", f"{divergencias} ({divergencias / total_cmp * 100:.1f}%)")
            r4.metric("Indefinidos", f"{indefinidos} ({indefinidos / total_cmp * 100:.1f}%)")
            show_training_metric_explanations(root)
            f1, f2, f3 = st.columns(3)
            filtro_resultado = f1.selectbox("Resultado", ["TODOS", "ACERTO", "DIVERGENCIA", "INDEFINIDO"], key="cmp_sexo_resultado")
            filtro_genero = f2.selectbox("Gênero definido na face", ["TODOS", "MASCULINO", "FEMININO"], key="cmp_sexo_genero")
            busca_cmp = f3.text_input("Buscar id", key="cmp_sexo_busca")
            comp_view = comparacao.copy()
            if filtro_resultado != "TODOS":
                comp_view = comp_view[comp_view["resultado_genero"].astype(str).str.upper().eq(filtro_resultado)].copy()
            if filtro_genero != "TODOS":
                comp_view = comp_view[comp_view["genero_definido_face"].astype(str).str.upper().eq(filtro_genero)].copy()
            if busca_cmp:
                comp_view = comp_view[comp_view.astype(str).agg(" ".join, axis=1).str.lower().str.contains(busca_cmp.strip().lower(), regex=False, na=False)].copy()
            cols_cmp = [c for c in ["ano", "id_pessoa", "normalizacao", "genero_definido_face", "genero_previsto_cranio", "resultado_genero", "confianca_cranio", "prob_masculino_cranio_pct", "prob_feminino_cranio_pct", "prob_masculino_cranio", "prob_feminino_cranio", "modelo_sexo_cranio"] if c in comp_view.columns]
            render_comparison_with_view_buttons(root, comp_view[cols_cmp])
    for name, path in [
        ("Silhuetas", sexo_dir(root) / "silhuetas_sexo_cranio.csv"),
        ("Dataset treino (rótulos vindos das faces)", sexo_dir(root) / "dataset_treino_sexo_cranio.csv"),
        ("Métricas", sexo_dir(root) / "metricas_modelo_sexo_cranio.csv"),
        ("Métricas por vista", sexo_dir(root) / "metricas_modelo_sexo_cranio_por_vista.csv"),
        ("Grid threshold validação", sexo_dir(root) / "grid_threshold_validacao_sexo_cranio.csv"),
        ("Grid threshold teste", sexo_dir(root) / "grid_threshold_teste_sexo_cranio.csv"),
        ("Matriz confusão", sexo_dir(root) / "matriz_confusao_sexo_cranio.csv"),
        ("Comparação face x crânio", sexo_dir(root) / "comparacao_genero_face_cranio.csv"),
        ("Predições por vista", sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv"),
        ("Predições", sexo_dir(root) / "predicoes_sexo_cranio.csv"),
    ]:
        st.subheader(name)
        show_df(read_csv_safe(path), 220)

elif page == "Divergências":
    st.title("Divergências")
    st.caption("Revisão dos casos em que o gênero craniano divergiu do gênero definido na face. As decisões ficam salvas em um CSV de curadoria, sem alterar os dados originais.")
    div = current_divergences(root)
    if div.empty:
        st.info("Nenhuma divergência encontrada. Rode o treino/exportação de gênero craniano para preencher a comparação.")
    else:
        total_div = len(div)
        reviewed = int((~div["decisao_divergencia"].astype(str).str.upper().eq("PENDENTE")).sum())
        high_conf = int(div.get("confianca_cranio", pd.Series("", index=div.index)).astype(str).str.upper().eq("ALTA").sum())
        no_alert = int(div.get("flags", pd.Series("", index=div.index)).astype(str).eq("sem_alerta_automatico").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Divergências", total_div)
        c2.metric("Revisadas", reviewed)
        c3.metric("Alta confiança", high_conf)
        c4.metric("Sem alerta automático", no_alert)

        st.markdown(
            """
            <div class="caption-box">
            <strong>Como usar:</strong> revise primeiro os casos de alta confiança e os casos com imagens ausentes/borradas.
            Classifique o motivo provável da divergência e escolha se o caso deve continuar no retreino. Isso cria uma fila
            limpa para corrigirmos dados antes de mexer novamente nos hiperparâmetros.
            </div>
            """,
            unsafe_allow_html=True,
        )

        f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1.4])
        filtro_status = f1.selectbox("Status revisão", ["TODOS", "PENDENTE", "ERRO_IMAGEM", "ERRO_ROTULO_FACE", "LADO_SUSPEITO", "QUALIDADE_RUIM", "CASO_AMBIGUO", "OK_MANTER", "EXCLUIR_RETREINO"], key="div_status")
        filtro_confianca = f2.selectbox("Confiança", ["TODAS", "ALTA", "MEDIA", "BAIXA", "INDEFINIDO"], key="div_conf")
        filtro_tipo = f3.selectbox("Tipo", ["TODOS", "FEMININO→MASCULINO", "MASCULINO→FEMININO"], key="div_tipo")
        filtro_ano = f4.selectbox("Ano", ["TODOS"] + sorted(div["ano"].dropna().astype(str).unique().tolist()), key="div_ano")
        busca = f5.text_input("Buscar id/flag/observação", key="div_busca")

        view = div.copy()
        if filtro_status != "TODOS":
            view = view[view["decisao_divergencia"].astype(str).str.upper().eq(filtro_status)].copy()
        if filtro_confianca != "TODAS" and "confianca_cranio" in view.columns:
            view = view[view["confianca_cranio"].astype(str).str.upper().eq(filtro_confianca)].copy()
        if filtro_tipo == "FEMININO→MASCULINO":
            view = view[
                view["genero_definido_face"].astype(str).str.upper().eq("FEMININO")
                & view["genero_previsto_cranio"].astype(str).str.upper().eq("MASCULINO")
            ].copy()
        elif filtro_tipo == "MASCULINO→FEMININO":
            view = view[
                view["genero_definido_face"].astype(str).str.upper().eq("MASCULINO")
                & view["genero_previsto_cranio"].astype(str).str.upper().eq("FEMININO")
            ].copy()
        if filtro_ano != "TODOS":
            view = view[view["ano"].astype(str).eq(filtro_ano)].copy()
        if busca:
            view = view[view.astype(str).agg(" ".join, axis=1).str.lower().str.contains(busca.strip().lower(), regex=False, na=False)].copy()

        st.caption(f"Mostrando {len(view)} de {total_div} divergência(s).")
        table_cols = [
            c
            for c in [
                "ano",
                "id_pessoa",
                "genero_definido_face",
                "genero_previsto_cranio",
                "confianca_cranio",
                "prob_masculino_cranio_pct",
                "decisao_divergencia",
                "usar_no_retreino",
                "n_flags",
                "flags",
                "observacao",
            ]
            if c in view.columns
        ]
        show_df(view[table_cols].sort_values([c for c in ["confianca_cranio", "n_flags", "ano", "id_pessoa"] if c in view.columns], ascending=[True, False, True, True][: len([c for c in ["confianca_cranio", "n_flags", "ano", "id_pessoa"] if c in view.columns])]), 260)

        if view.empty:
            st.info("Nenhum caso atende aos filtros atuais.")
        else:
            view = view.reset_index(drop=True)
            def _case_label(i: int) -> str:
                row = view.iloc[i]
                prob = row.get("prob_masculino_cranio_pct", "")
                prob_txt = f"{float(prob):.1f}%" if str(prob) not in {"", "nan", "None"} else "-"
                return f"{row.get('ano')} | {row.get('id_pessoa')} | face {row.get('genero_definido_face')} → crânio {row.get('genero_previsto_cranio')} | M {prob_txt}"

            selected_idx = st.selectbox("Caso para revisar", list(range(len(view))), format_func=_case_label, key="div_selected_case")
            row = view.iloc[int(selected_idx)]
            year = str(row.get("ano", ""))
            person_id = str(row.get("id_pessoa", ""))
            normalization_kind = str(row.get("normalizacao", "") or "atual")
            if normalization_kind not in NORMALIZATION_KINDS:
                normalization_kind = "atual"

            st.subheader(f"{year} | {person_id}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Face", str(row.get("genero_definido_face", "-")).replace("_", " ").lower().capitalize())
            k2.metric("Crânio", str(row.get("genero_previsto_cranio", "-")).replace("_", " ").lower().capitalize())
            k3.metric("Masculino", f"{float(row.get('prob_masculino_cranio_pct', 0)):.2f}%")
            k4.metric("Confiança", str(row.get("confianca_cranio", "-")).replace("_", " ").lower().capitalize())

            st.markdown("**Alertas automáticos**")
            st.code(str(row.get("flags", "sem_diagnostico")), language="text")

            for item in person_images(root, year, person_id, normalization_kind):
                side = str(item["lado"])
                st.markdown(f"**{side}**")
                p1, p2 = st.columns(2)
                with p1:
                    render_image(item["original"], f"{side} original")
                with p2:
                    render_image(item["normalizada"], f"{side} normalizada ({normalization_kind})")

            decision_options = ["PENDENTE", "ERRO_IMAGEM", "ERRO_ROTULO_FACE", "LADO_SUSPEITO", "QUALIDADE_RUIM", "CASO_AMBIGUO", "OK_MANTER", "EXCLUIR_RETREINO"]
            current_decision = str(row.get("decisao_divergencia", "PENDENTE") or "PENDENTE").upper()
            current_retrain = str(row.get("usar_no_retreino", "SIM") or "SIM").upper()
            if current_decision not in decision_options:
                current_decision = "PENDENTE"
            st.divider()
            st.markdown("**Curadoria da divergência**")
            d1, d2 = st.columns([1, 1])
            decision = d1.selectbox("Motivo provável", decision_options, index=decision_options.index(current_decision), key=f"div_decision_{year}_{person_id}")
            use_retrain = d2.selectbox("Usar no retreino", ["SIM", "NAO"], index=0 if current_retrain != "NAO" else 1, key=f"div_retrain_{year}_{person_id}")
            obs = st.text_area("Observação", value=str(row.get("observacao", "") or ""), key=f"div_obs_{year}_{person_id}", height=90)
            b1, b2 = st.columns([1, 4])
            if b1.button("Salvar revisão", type="primary", key=f"div_save_{year}_{person_id}"):
                save_divergence_review(
                    root,
                    {
                        "ano": year,
                        "id_pessoa": person_id,
                        "decisao_divergencia": decision,
                        "usar_no_retreino": use_retrain,
                        "observacao": obs,
                        "revisado_em": pd.Timestamp.now().isoformat(),
                    },
                )
                st.success("Revisão salva.")
                st.rerun()
            with b2:
                st.caption(f"Arquivo de curadoria: `{divergence_review_path(root)}`")

elif page == "Métricas":
    st.title("Métricas")
    tabs = st.tabs(["Runs", "Landmarks", "Gênero", "Exportações"])
    with tabs[0]:
        runs = read_csv_safe(resultados_root(root) / "runs" / "historico_treinos_cranios.csv")
        if not runs.empty:
            st.subheader("Histórico de treinos do crânio")
            show_df(runs.sort_values("timestamp", ascending=False) if "timestamp" in runs.columns else runs, 320)
            chart_cols = [c for c in ["balanced_accuracy_pct", "accuracy_pct", "f1_pct", "precision_pct", "recall_pct"] if c in runs.columns]
            if chart_cols:
                chart = runs.copy()
                chart["run"] = chart.get("run_id", pd.Series(range(len(chart)), index=chart.index)).astype(str)
                st.bar_chart(chart.set_index("run")[chart_cols])
        else:
            st.info("Ainda não há runs de treino registrados.")
    with tabs[1]:
        qual = read_csv_safe(qualidade_dir(root) / "qualidade_cranios.csv")
        resumo = read_csv_safe(qualidade_dir(root) / "resumo_qualidade_cranios.csv")
        show_df(resumo, 220)
        if not qual.empty and "status_cranio" in qual.columns:
            q = qual.groupby(["ano", "status_cranio"], dropna=False).size().reset_index(name="qtd")
            st.bar_chart(q.pivot(index="ano", columns="status_cranio", values="qtd").fillna(0))
        show_df(qual, 360)
    with tabs[2]:
        comp = read_csv_safe(sexo_dir(root) / "comparacao_genero_face_cranio.csv")
        mets = read_csv_safe(sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
        show_df(mets, 180)
        if not comp.empty:
            res = comp.get("resultado_genero", pd.Series("", index=comp.index)).astype(str).str.upper()
            total = max(len(comp), 1)
            cards = st.columns(4)
            cards[0].metric("Comparados", len(comp))
            cards[1].metric("Acertos", f"{int(res.eq('ACERTO').sum())} ({res.eq('ACERTO').sum()/total*100:.1f}%)")
            cards[2].metric("Divergências", f"{int(res.eq('DIVERGENCIA').sum())} ({res.eq('DIVERGENCIA').sum()/total*100:.1f}%)")
            cards[3].metric("Indefinidos", f"{int(res.eq('INDEFINIDO').sum())} ({res.eq('INDEFINIDO').sum()/total*100:.1f}%)")
            st.bar_chart(res.value_counts())
        show_df(comp, 420)
    with tabs[3]:
        st.subheader("Features finais usadas no matching")
        show_df(read_csv_safe(features_dir(root) / "features_matching_cranios_final.csv"), 420)
        st.subheader("Silhuetas")
        show_df(read_csv_safe(sexo_dir(root) / "silhuetas_sexo_cranio.csv"), 260)
    if st.button("Exportar novamente", type="primary"):
        run_cmd(root, "exportar")

else:
    st.title("Logs")
    files = sorted(logs_dir(root).glob("*"))
    if not files:
        st.info("Sem logs.")
    else:
        f = st.selectbox("Arquivo", files, format_func=lambda p: p.name)
        st.code(f.read_text(encoding="utf-8", errors="replace")[-20000:])
