# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd


st.set_page_config(page_title="Gigi · Painel Geral", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container{padding-top:1rem;max-width:1800px}
    iframe{border:1px solid #e5e7eb;border-radius:8px;background:white}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parents[1]

APPS = {
    "Face": {
        "root": ROOT / "Treinamento_faces",
        "script": ROOT / "Treinamento_faces" / "scripts" / "app_revisao_faces.py",
        "port": 8511,
    },
    "Crânio": {
        "root": ROOT / "Treinamento_cranios",
        "script": ROOT / "Treinamento_cranios" / "scripts" / "app_revisao_cranios.py",
        "port": 8512,
    },
    "Matching": {
        "root": ROOT / "Matching_cranio_face",
        "script": ROOT / "Matching_cranio_face" / "scripts" / "app_matching_cranio_face.py",
        "port": 8513,
    },
}


def python_bin() -> Path:
    for p in [ROOT / ".venv" / "bin" / "python", ROOT / "Treinamento_faces" / ".venv" / "bin" / "python"]:
        if p.exists():
            return p
    return Path(sys.executable)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def start_app(name: str):
    app = APPS[name]
    port = int(app["port"])
    if port_open(port):
        return True, f"{name} já está rodando."
    if not Path(app["script"]).exists():
        return False, f"Script não encontrado: {app['script']}"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_bin()),
        "-m",
        "streamlit",
        "run",
        str(app["script"]),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(cmd, cwd=str(app["root"]), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.session_state.setdefault("child_processes", {})[name] = proc.pid
    for _ in range(40):
        if port_open(port):
            return True, f"{name} iniciado na porta {port}."
        time.sleep(0.25)
    return False, f"{name} foi iniciado, mas ainda não respondeu na porta {port}."


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def show_df(df: pd.DataFrame, height: int = 320):
    if df.empty:
        st.info("Sem dados.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def add_pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and f"{c}_pct" not in out.columns:
            out[f"{c}_pct"] = (pd.to_numeric(out[c], errors="coerce") * 100).round(2)
    return out


def render_comparativos():
    st.subheader("Histórico e Comparativos")
    faces_runs = read_csv(ROOT / "Treinamento_faces" / "resultados" / "runs" / "historico_treinos_faces.csv")
    cranio_runs = read_csv(ROOT / "Treinamento_cranios" / "resultados" / "runs" / "historico_treinos_cranios.csv")
    face_metricas = add_pct_cols(read_csv(ROOT / "Treinamento_faces" / "resultados" / "genero_faces" / "metricas_modelo_genero_faces.csv"), ["accuracy", "precision_masculino", "recall_masculino", "f1_masculino"])
    cranio_metricas = add_pct_cols(read_csv(ROOT / "Treinamento_cranios" / "resultados" / "sexo_cranio" / "metricas_modelo_sexo_cranio.csv"), ["accuracy", "f1_masculino"])
    cranio_comp = read_csv(ROOT / "Treinamento_cranios" / "resultados" / "sexo_cranio" / "comparacao_genero_face_cranio.csv")
    face_land = read_csv(ROOT / "Treinamento_faces" / "resultados" / "analises_faces" / "resumo_analise_padroes_faces.csv")
    cranio_land = read_csv(ROOT / "Treinamento_cranios" / "resultados" / "qualidade_cranio" / "resumo_qualidade_cranios.csv")

    tabs = st.tabs(["Runs", "Gênero", "Landmarks", "Arquivos"])
    with tabs[0]:
        st.markdown("**Treinos de gênero da face**")
        show_df(add_pct_cols(faces_runs, ["accuracy", "precision_masculino", "recall_masculino", "f1_masculino"]), 300)
        if not faces_runs.empty and "accuracy" in faces_runs.columns:
            chart = faces_runs.copy()
            chart["label"] = chart.get("run_id", chart.index.astype(str)).astype(str) + " · " + chart.get("modelo", "").astype(str)
            st.bar_chart(chart.set_index("label")[[c for c in ["accuracy", "f1_masculino"] if c in chart.columns]])
        st.markdown("**Treinos de gênero/sexo do crânio**")
        show_df(add_pct_cols(cranio_runs, ["accuracy", "f1_masculino"]), 300)
        if not cranio_runs.empty and "accuracy" in cranio_runs.columns:
            chart = cranio_runs.copy()
            chart["label"] = chart.get("run_id", chart.index.astype(str)).astype(str) + " · " + chart.get("modelo", "").astype(str)
            st.bar_chart(chart.set_index("label")[[c for c in ["accuracy", "f1_masculino"] if c in chart.columns]])
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Face: métricas do último treino**")
            show_df(face_metricas, 260)
        with c2:
            st.markdown("**Crânio: métricas do último treino**")
            show_df(cranio_metricas, 260)
        st.markdown("**Crânio: previsto x definido na face**")
        if not cranio_comp.empty and "resultado_genero" in cranio_comp.columns:
            total = max(len(cranio_comp), 1)
            res = cranio_comp["resultado_genero"].astype(str).str.upper()
            cards = st.columns(4)
            cards[0].metric("Comparados", len(cranio_comp))
            for col, name in zip(cards[1:], ["ACERTO", "DIVERGENCIA", "INDEFINIDO"]):
                n = int(res.eq(name).sum())
                col.metric(name.title(), f"{n} ({n / total * 100:.1f}%)")
            show_df(cranio_comp, 420)
            st.bar_chart(res.value_counts().rename("qtd"))
        else:
            st.info("Ainda sem comparação de gênero crânio x face.")
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Landmarks da face**")
            show_df(face_land, 320)
            if not face_land.empty:
                num_cols = [c for c in face_land.columns if c != "ano" and pd.api.types.is_numeric_dtype(face_land[c])]
                if "ano" in face_land.columns and num_cols:
                    st.bar_chart(face_land.set_index("ano")[num_cols[:4]])
        with c2:
            st.markdown("**Landmarks/contornos do crânio**")
            show_df(cranio_land, 320)
            if not cranio_land.empty and "status_cranio" in cranio_land.columns and "qtd" in cranio_land.columns:
                st.bar_chart(cranio_land.set_index("status_cranio")["qtd"])
    with tabs[3]:
        files = pd.DataFrame([
            {"area": "face_runs", "arquivo": str(ROOT / "Treinamento_faces" / "resultados" / "runs" / "historico_treinos_faces.csv"), "existe": (ROOT / "Treinamento_faces" / "resultados" / "runs" / "historico_treinos_faces.csv").exists()},
            {"area": "cranio_runs", "arquivo": str(ROOT / "Treinamento_cranios" / "resultados" / "runs" / "historico_treinos_cranios.csv"), "existe": (ROOT / "Treinamento_cranios" / "resultados" / "runs" / "historico_treinos_cranios.csv").exists()},
            {"area": "cranio_comparacao", "arquivo": str(ROOT / "Treinamento_cranios" / "resultados" / "sexo_cranio" / "comparacao_genero_face_cranio.csv"), "existe": (ROOT / "Treinamento_cranios" / "resultados" / "sexo_cranio" / "comparacao_genero_face_cranio.csv").exists()},
        ])
        show_df(files, 220)


st.title("Gigi · Painel Geral")

selected = st.radio("Aplicativo", ["Comparativos"] + list(APPS.keys()), horizontal=True, label_visibility="collapsed")

if selected == "Comparativos":
    render_comparativos()
    st.stop()

app = APPS[selected]
url = f"http://localhost:{app['port']}"

c1, c2, c3 = st.columns([1, 1, 3])
if c1.button(f"Abrir {selected}", type="primary", use_container_width=True):
    ok, msg = start_app(selected)
    st.success(msg) if ok else st.error(msg)
if c2.button("Atualizar status", use_container_width=True):
    st.rerun()

running = port_open(int(app["port"]))
c3.markdown(
    f"**Status:** {'rodando' if running else 'parado'} · "
    f"**URL:** [{url}]({url}) · "
    f"**Script:** `{app['script']}`"
)

st.code(
    f"cd {ROOT}\n.venv/bin/python -m streamlit run {app['script']} --server.address 0.0.0.0 --server.port {app['port']}",
    language="bash",
)

if running:
    components.iframe(url, height=920, scrolling=True)
else:
    st.info("Clique em Abrir para iniciar este app dentro do painel geral.")
