# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
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
}


def guess_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_python(root: Path) -> Path:
    for p in [root.parent / ".venv" / "bin" / "python", root / ".venv" / "bin" / "python", root.parent / "Treinamento_faces" / ".venv" / "bin" / "python"]:
        if p.exists():
            return p
    return Path(sys.executable)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_log_line(line: str) -> str:
    text = ANSI_RE.sub("", line.replace("\r", "\n"))
    text = text.replace("━", "-").replace("█", "#").replace("▏", "|").replace("▎", "|").replace("▍", "|").replace("▌", "|").replace("▋", "|").replace("▊", "|").replace("▉", "|")
    return "\n".join(part.rstrip() for part in text.splitlines() if part.strip())


def run_cmd(root: Path, key: str, years=None, force=False, extra=None):
    cmd = [str(project_python(root)), str(root / "scripts" / SCRIPTS[key]), "--root", str(root)]
    if years:
        cmd += ["--years"] + [str(y) for y in years]
    if force:
        cmd.append("--force")
    if extra:
        cmd += extra
    st.code(" ".join(cmd), language="bash")
    box = st.empty()
    logs = []
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(resultados_root(root) / "cache" / "matplotlib"))
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        cleaned = clean_log_line(line)
        if cleaned:
            logs.extend(cleaned.splitlines())
        box.code("\n".join(logs[-260:]), language="text")
    rc = proc.wait()
    if rc == 0:
        st.success("Concluído.")
    else:
        st.error(f"Falhou com código {rc}.")
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
    ["Painel", "Pipeline", "Matching/Curadoria", "Gênero", "Imagens", "Métricas", "Logs"],
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
    force = st.checkbox("Refazer silhuetas existentes", key="sexo_force")
    use_norm_sexo = st.checkbox("Usar crânios normalizados nas silhuetas", value=False, key="sexo_use_norm")
    with st.expander("Configuração do treino", expanded=True):
        t1, t2, t3, t4, t5 = st.columns(5)
        backend = t1.selectbox("Modelo", ["VGG16 modo artigo", "VGG16 embeddings", "Rápido"], index=0, key="sexo_backend")
        epochs = t2.number_input("Épocas", min_value=1, max_value=80, value=18 if backend == "VGG16 modo artigo" else 8, step=1, key="sexo_epochs")
        image_size = t3.selectbox("Tamanho", [128, 160, 224, 256], index=3 if backend == "VGG16 modo artigo" else 0, key="sexo_image_size")
        batch_size = t4.selectbox("Batch", [4, 8, 16, 32], index=1, key="sexo_batch_size")
        weights = t5.selectbox("Pesos", ["imagenet", "none"], index=0, key="sexo_weights")
        if backend == "Rápido":
            treino_extra = ["--backend", "sklearn"]
        elif backend == "VGG16 modo artigo":
            treino_extra = ["--backend", "article_vgg16", "--epochs", str(int(epochs)), "--image-size", str(int(image_size)), "--batch-size", str(int(batch_size)), "--weights", weights, "--learning-rate", "0.000002"]
        else:
            treino_extra = ["--backend", "vgg16", "--epochs", str(int(epochs)), "--image-size", str(int(image_size)), "--batch-size", str(int(batch_size)), "--weights", weights]
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("1. Preparar silhuetas", type="primary"):
        run_cmd(root, "silhuetas", sel, force, extra=["--use-normalized"] if use_norm_sexo else None)
    if c2.button("2. Juntar gênero das faces"):
        run_cmd(root, "dataset_sexo", extra=["--faces-root", faces_root])
    if c3.button("3. Treinar modelo"):
        run_cmd(root, "treinar_sexo", extra=treino_extra)
    if c4.button("4. Exportar final"):
        run_cmd(root, "exportar")
    st.divider()
    if st.button("Rodar fluxo de gênero/sexo completo"):
        rc = run_cmd(root, "silhuetas", sel, force, extra=["--use-normalized"] if use_norm_sexo else None)
        if rc == 0:
            rc = run_cmd(root, "dataset_sexo", extra=["--faces-root", faces_root])
        if rc == 0:
            rc = run_cmd(root, "treinar_sexo", extra=treino_extra)
        if rc == 0:
            run_cmd(root, "exportar")
    st.divider()
    st.info("VGG16 modo artigo treina sobre silhuetas 2D com augmentation flip/rotação ±5° e avalia por voto CF/CE/CD. VGG16 embeddings é o modo anterior, mais rápido.")
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
            cols_cmp = [c for c in ["ano", "id_pessoa", "genero_definido_face", "genero_previsto_cranio", "resultado_genero", "confianca_cranio", "prob_masculino_cranio_pct", "prob_feminino_cranio_pct", "prob_masculino_cranio", "prob_feminino_cranio", "modelo_sexo_cranio"] if c in comp_view.columns]
            show_df(comp_view[cols_cmp], 360)
    for name, path in [
        ("Silhuetas", sexo_dir(root) / "silhuetas_sexo_cranio.csv"),
        ("Dataset treino (rótulos vindos das faces)", sexo_dir(root) / "dataset_treino_sexo_cranio.csv"),
        ("Métricas", sexo_dir(root) / "metricas_modelo_sexo_cranio.csv"),
        ("Métricas por vista", sexo_dir(root) / "metricas_modelo_sexo_cranio_por_vista.csv"),
        ("Matriz confusão", sexo_dir(root) / "matriz_confusao_sexo_cranio.csv"),
        ("Comparação face x crânio", sexo_dir(root) / "comparacao_genero_face_cranio.csv"),
        ("Predições por vista", sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv"),
        ("Predições", sexo_dir(root) / "predicoes_sexo_cranio.csv"),
    ]:
        st.subheader(name)
        show_df(read_csv_safe(path), 220)

elif page == "Métricas":
    st.title("Métricas")
    tabs = st.tabs(["Runs", "Landmarks", "Gênero", "Exportações"])
    with tabs[0]:
        runs = read_csv_safe(resultados_root(root) / "runs" / "historico_treinos_cranios.csv")
        if not runs.empty:
            st.subheader("Histórico de treinos do crânio")
            show_df(runs.sort_values("timestamp", ascending=False) if "timestamp" in runs.columns else runs, 320)
            chart_cols = [c for c in ["accuracy_pct", "f1_pct", "precision_pct", "recall_pct"] if c in runs.columns]
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
