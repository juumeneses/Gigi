from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.baselines import run_baselines
from src.index_builder import build_metadata
from src.paths import repo_path
from src.preprocess import preprocess_manifest
from src.splits import create_leave_one_year_out, create_splits
from src.domain_preprocessing import audit_domains, normalize_domains
from src.inference import load_ensemble, rank_uploaded_candidates


st.set_page_config(page_title="Modelo novo face-crânio", page_icon="🔬", layout="wide")
st.title("Matching face–crânio")
st.caption("Pipeline experimental: auditoria, baselines, retrieval e two-tower sem vazamento de indivíduos.")


@st.cache_data(show_spinner=False)
def load_metadata():
    path = PROJECT_DIR / "data/manifests/metadata_pares.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


page = st.sidebar.radio("Área", ["Comparar 10 crânios", "Resumo", "Dados e pares", "Pré-processamento", "Treinamento", "Avaliação", "Ranking", "Análise de erros", "Logs"])
df = load_metadata()

if page == "Comparar 10 crânios":
    st.header("Comparação de uma face com 10 crânios")
    st.warning("Ferramenta experimental de pesquisa. O resultado é um ranking de similaridade aprendida, não uma probabilidade de identidade nem uma conclusão pericial.")
    st.write("Use uma fotografia frontal, próxima e já recortada ao redor da cabeça. Para cada candidato, envie as vistas frontal (CF), lateral esquerda (CE) e lateral direita (CD).")
    face_file=st.file_uploader("Fotografia frontal da face",type=["png","jpg","jpeg"],key="face_query")
    uploads=[]
    for index in range(10):
        with st.expander(f"Candidato {index+1}",expanded=index==0):
            name=st.text_input("Identificação",value=f"Candidato {index+1}",key=f"candidate_name_{index}")
            columns=st.columns(3)
            cf=columns[0].file_uploader("CF — frontal",type=["png","jpg","jpeg"],key=f"cf_{index}")
            ce=columns[1].file_uploader("CE — esquerda",type=["png","jpg","jpeg"],key=f"ce_{index}")
            cd=columns[2].file_uploader("CD — direita",type=["png","jpg","jpeg"],key=f"cd_{index}")
            uploads.append((name,(cf,ce,cd)))
    ready=face_file is not None and all(all(view is not None for view in views) for _,views in uploads)
    if st.button("Calcular ranking",type="primary",disabled=not ready):
        try:
            with st.spinner("Normalizando imagens e consultando os cinco modelos..."):
                face=Image.open(face_file).convert("RGB")
                candidates=[(name,[Image.open(view).convert("RGB") for view in views]) for name,views in uploads]
                @st.cache_resource(show_spinner=False)
                def cached_ensemble(): return load_ensemble()
                result=rank_uploaded_candidates(face,candidates,cached_ensemble())
            st.success(f"Maior similaridade: {result.iloc[0].candidato}")
            display=result.copy(); display["similaridade_media"]=display.similaridade_media.map(lambda value:f"{value:.4f}"); display["diferença_para_líder"]=display["diferença_para_líder"].map(lambda value:f"{value:.4f}")
            st.dataframe(display,use_container_width=True,hide_index=True)
            st.bar_chart(result.set_index("candidato")["similaridade_media"])
            st.caption("Pontuações só podem ser comparadas dentro deste conjunto de dez candidatos. Uma pontuação alta não comprova que o crânio pertence à pessoa.")
        except Exception as exc:
            st.error(f"Não foi possível calcular o ranking: {exc}")
    elif not ready:
        st.caption("O botão será habilitado depois que a face e as três vistas dos dez candidatos forem enviadas.")

elif page == "Resumo":
    if st.button("Atualizar índice", type="primary"):
        with st.spinner("Auditando fontes..."):
            _, report = build_metadata()
        st.cache_data.clear(); st.success(f"Índice criado: {report['valid_pairs']} pares válidos."); st.rerun()
    if df.empty:
        st.warning("Índice ainda não criado.")
    else:
        valid = df[df.usar_treino.fillna(False)]
        cols = st.columns(4); cols[0].metric("Pares válidos", len(valid)); cols[1].metric("Excluídos", len(df)-len(valid)); cols[2].metric("Masculino", int(valid.sexo_face.eq("MASCULINO").sum())); cols[3].metric("Feminino", int(valid.sexo_face.eq("FEMININO").sum()))
        st.bar_chart(valid.groupby("ano").size())
        audit = PROJECT_DIR / "data/manifests/metadata_pares.audit.json"
        if audit.exists(): st.json(json.loads(audit.read_text(encoding="utf-8")))

elif page == "Dados e pares":
    if df.empty: st.warning("Crie o índice na tela Resumo.")
    else:
        years = st.multiselect("Ano", sorted(df.ano.astype(str).unique()), default=sorted(df.ano.astype(str).unique()))
        only = st.selectbox("Exibir", ["Todos", "Válidos", "Problemas", "Divergência de sexo"])
        view=df[df.ano.astype(str).isin(years)]
        if only == "Válidos": view=view[view.usar_treino]
        if only == "Problemas": view=view[~view.usar_treino]
        if only == "Divergência de sexo": view=view[view.usar_treino & ~view.sexo_concordante]
        st.dataframe(view, use_container_width=True, height=420)
        if len(view):
            key=st.selectbox("Visualizar par", view.pair_key.tolist()); row=view[view.pair_key.eq(key)].iloc[0]; c1,c2,c3,c4=st.columns(4)
            for col,path_col,label in ((c1,"face_path","Face"),(c2,"cranio_cf_path","CF"),(c3,"cranio_ce_path","CE"),(c4,"cranio_cd_path","CD")):
                p=repo_path(row.get(path_col));
                if p and p.exists(): col.image(str(p), caption=label, use_container_width=True)

elif page == "Pré-processamento":
    size=st.select_slider("Tamanho", [128,160,224,256,320], value=224); limit=st.number_input("Limite (0 = todos)",0,10000,0)
    force=st.checkbox("Recriar arquivos existentes")
    if st.button("Rodar faltantes", disabled=df.empty):
        result=preprocess_manifest(df,size,force,int(limit) or None); st.success(result.status.value_counts().to_dict()); st.dataframe(result)
    st.divider(); st.subheader("Normalização entre anos")
    if st.button("Auditar diferenças por ano",disabled=df.empty):
        details,summary=audit_domains(df); st.success(f"{len(details)} imagens auditadas"); st.dataframe(summary,use_container_width=True)
    if st.button("Gerar imagens alinhadas e padronizadas",disabled=df.empty):
        result=normalize_domains(df,int(size),force,int(limit) or None); st.success(result.status.value_counts().to_dict()); st.dataframe(result,use_container_width=True)
    summary_path=PROJECT_DIR/"reports/domain_audit/year_summary.csv"
    if summary_path.exists(): st.write("Resumo da auditoria por ano e modalidade"); st.dataframe(pd.read_csv(summary_path),use_container_width=True)

elif page == "Treinamento":
    st.info("O treinamento roda pela CLI para não bloquear o Streamlit: `python Modelo_novo/cli.py train`. O painel abaixo atualiza o progresso da execução mais recente.")
    if st.button("Criar/recriar splits", disabled=df.empty):
        splits=create_splits(df); create_leave_one_year_out(df); st.success(splits.split.value_counts().to_dict())
    st.code((PROJECT_DIR / "configs/default.yaml").read_text(encoding="utf-8"), language="yaml")
    def show_live_progress():
        runs=sorted(list((PROJECT_DIR / "runs").glob("*/progress.json")) + list((PROJECT_DIR / "studies").rglob("progress.json")), key=lambda p:p.stat().st_mtime, reverse=True)
        if not runs:
            st.caption("Nenhuma execução instrumentada iniciada ainda."); return
        state=json.loads(runs[0].read_text(encoding="utf-8")); st.subheader(f"Execução {state.get('run_id','')}")
        status=state.get("status","desconhecido"); st.write(f"Status: **{status}**")
        if status == "treinando":
            epoch=int(state.get("epoch",0)); maximum=int(state.get("max_epochs",1)); st.progress(epoch/maximum, text=f"Época {epoch} de {maximum}")
            c1,c2,c3,c4=st.columns(4); c1.metric("Loss",f"{state.get('train_loss',0):.4f}"); c2.metric("MRR validação",f"{state.get('val_mrr',0):.4f}"); c3.metric("Top-5",f"{state.get('val_top_5',0)*100:.2f}%"); c4.metric("Melhor época",state.get("best_epoch","—"))
            st.caption(f"Etapa: {state.get('stage')} · época: {state.get('epoch_seconds',0):.0f}s · estimativa restante: {state.get('eta_seconds',0)/60:.1f} min")
        elif status == "concluido":
            target=state.get("evaluation_split","test"); st.success(f"Treinamento concluído. Melhor época: {state.get('best_epoch')}"); st.json(state.get(target,{}))
        log=runs[0].with_name("train.log")
        if log.exists(): st.code("\n".join(log.read_text(encoding="utf-8").splitlines()[-30:]), language="text")
    if hasattr(st, "fragment"):
        st.fragment(run_every="2s")(show_live_progress)()
    else:
        show_live_progress(); st.caption("Use o botão Rerun do Streamlit para atualizar este painel.")

elif page == "Avaliação":
    test=PROJECT_DIR / "data/splits/test.csv"; metrics=PROJECT_DIR / "reports/baselines/metricas_baselines.csv"
    if st.button("Executar baselines", disabled=not test.exists()): st.dataframe(run_baselines(pd.read_csv(test)))
    elif metrics.exists(): st.dataframe(pd.read_csv(metrics), use_container_width=True)
    else: st.warning("Crie os splits e execute os baselines.")
    st.divider(); st.subheader("Estudos experimentais")
    view_summaries=[PROJECT_DIR / "studies/view_ablation/summary.csv"] if (PROJECT_DIR / "studies/view_ablation/summary.csv").exists() else []
    if view_summaries:
        study=pd.read_csv(view_summaries[-1]); st.write("Ablação de vistas — selecionada exclusivamente pela validação")
        st.dataframe(study,use_container_width=True); st.bar_chart(study.set_index("combination")[["mrr","top_1","top_5","recall_at_10"]])
    else: st.caption("Ablação ainda não executada. Use `python Modelo_novo/cli.py ablate-views --combination CF` para iniciar uma combinação.")
    loyo=PROJECT_DIR / "studies/leave_one_year_out/summary.csv"
    if loyo.exists():
        annual=pd.read_csv(loyo); st.write("Generalização leave-one-year-out"); st.dataframe(annual,use_container_width=True); st.bar_chart(annual.set_index("holdout_year")[["mrr","top_1","top_5","recall_at_10"]])
    else: st.caption("Holdouts anuais ainda não executados. Exemplo: `python Modelo_novo/cli.py loyo --year 2017`.")
    gallery=PROJECT_DIR/"reports/gallery_10_ensemble_domain_normalized_hard_negatives/metrics.csv"
    if not gallery.exists(): gallery=PROJECT_DIR/"reports/gallery_10_domain_normalized_hard_negatives/metrics.csv"
    if not gallery.exists(): gallery=PROJECT_DIR/"reports/gallery_10_domain_normalized/metrics.csv"
    if not gallery.exists(): gallery=PROJECT_DIR/"reports/gallery_10/metrics.csv"
    if gallery.exists():
        st.divider(); st.subheader("Galeria final com 10 candidatos"); gm=pd.read_csv(gallery); st.dataframe(gm,use_container_width=True)
        overall=gm[gm.year.astype(str).eq("TODOS")].set_index("scenario")
        if len(overall): st.bar_chart(overall[["top_1","top_3","top_5","mrr"]])

elif page == "Ranking":
    st.info("Use a área ‘Comparar 10 crânios’ para enviar imagens novas. Rankings conhecidos continuam salvos por experimento.")
elif page == "Análise de erros":
    rankings=sorted((PROJECT_DIR / "runs").glob("*/ranking_test.csv"))
    if rankings:
        rank=pd.read_csv(rankings[-1]); errors=rank[rank.correto & (rank["rank"]>1)]; st.dataframe(errors, use_container_width=True)
    else: st.warning("Nenhum ranking neural disponível.")
else:
    files=sorted((PROJECT_DIR / "runs").glob("*/*"), key=lambda p:p.stat().st_mtime, reverse=True)
    st.dataframe(pd.DataFrame([{"arquivo":str(p.relative_to(PROJECT_DIR)),"bytes":p.stat().st_size} for p in files[:200]]))
