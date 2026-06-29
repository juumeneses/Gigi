# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

from common_matching import rank_uploaded, uploaded_to_temp


st.set_page_config(page_title="Matching Crânio-Face", page_icon="🔎", layout="wide")

st.title("Matching Crânio-Face")
st.caption("V1: ranking por shape 64 e score auxiliar de sexo/gênero. A arquitetura two-tower fica preparada para treino posterior.")

face = st.file_uploader("Face frontal", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
cranios = st.file_uploader("Crânios candidatos", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"], accept_multiple_files=True)
genero = st.selectbox("Gênero/sexo aparente da face (opcional)", ["INDEFINIDO", "MASCULINO", "FEMININO"])

if st.button("Gerar ranking", type="primary", disabled=not face or not cranios):
    try:
        face_path = uploaded_to_temp(face, Path(face.name).suffix or ".jpg")
        cranio_paths = [uploaded_to_temp(c, Path(c.name).suffix or ".jpg") for c in cranios]
        df = rank_uploaded(face_path, cranio_paths, genero)
        st.success("Ranking gerado.")
        st.dataframe(df.drop(columns=[c for c in ["preview_face", "preview_cranio"] if c in df.columns]), use_container_width=True, height=360)
        ok = df[df["status"].astype(str).eq("OK")].head(12) if "status" in df.columns else df.head(12)
        st.subheader("Prévia dos melhores candidatos")
        for _, row in ok.iterrows():
            c1, c2, c3 = st.columns([1, 1, 1.2])
            with c1:
                p = Path(str(row.get("preview_face", "")))
                if p.exists():
                    st.image(Image.open(p), caption="Face com landmarks", use_container_width=True)
            with c2:
                p = Path(str(row.get("preview_cranio", "")))
                if p.exists():
                    st.image(Image.open(p), caption=f"Crânio {row.get('arquivo_cranio', '')}", use_container_width=True)
            with c3:
                st.metric("Score total", f"{float(row.get('score_total', 0))*100:.1f}%")
                st.write(f"Shape: **{float(row.get('score_shape', 0))*100:.1f}%**")
                st.write(f"Gênero face: **{row.get('genero_face', 'INDEFINIDO')}**")
                st.write(f"Gênero crânio: **{row.get('genero_previsto_cranio', 'INDEFINIDO')}**")
                st.write(f"Concordância: **{row.get('concordancia_genero', 'INDEFINIDO')}**")
        st.download_button("Baixar CSV", df.to_csv(index=False).encode("utf-8-sig"), "ranking_matching_cranio_face.csv", "text/csv")
    except Exception as e:
        st.error(str(e))

st.divider()
st.subheader("Two-tower neural")
st.info("Preparado para uma próxima etapa: VGG16 sobre crânio/silhueta + MLP sobre PCA facial, similaridade cosseno e treino binário com pares positivos/negativos.")
