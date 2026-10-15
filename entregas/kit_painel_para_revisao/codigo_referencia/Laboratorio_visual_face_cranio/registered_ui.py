"""Stored results by modality and a read-only consolidated comparison."""
import pandas as pd
import streamlit as st
from .registered import table, result_summary
from .core import ARTIFACTS, load


def results(modality=None):
    try:
        frame=table()
    except (ValueError, pd.errors.MergeError) as exc:
        st.error(str(exc))
        return
    if frame.empty:
        st.info('Nenhum resultado disponível.')
        return
    scope=modality or 'all'
    models={'FairFace': 'estimativa_fairface', 'Modelo próprio facial': 'previsao_face', 'Modelo craniano': 'previsao_cranio'}
    if modality=='face':
        models.pop('Modelo craniano')
        choice=st.segmented_control('Resultados faciais',['Comparar ambos']+list(models),default='Comparar ambos',key='facial_result_model')
        if choice!='Comparar ambos':
            models={choice:models[choice]}
        st.caption('Resultados faciais já registrados · FairFace e modelo próprio')
        copies=sorted((ARTIFACTS/'facial_recorded_results').glob('*/sources.json'),reverse=True)
        if copies:
            st.caption('Referências, resultados e métricas copiados para o laboratório · '+copies[0].parent.name)
        with st.expander('Arquivos faciais e referência curada'):
            from .registered import snapshot_facial_results, snapshot_curated
            if st.button('Atualizar cópia dos resultados existentes',key='copy_facial_records'):
                snapshot_curated()
                snapshot_facial_results()
                st.rerun()
            if copies:
                source=load(copies[0])
                for name in source['files']:
                    st.download_button(name,(copies[0].parent/name).read_bytes(),name,key='source_'+name)
    elif modality=='cranio':
        models={'Modelo craniano':'previsao_cranio'}
    cols=st.columns(3)
    years=sorted(frame.ano.dropna().astype(int).unique().tolist())
    year=cols[0].selectbox('Ano dos resultados',['Todos']+years,key=scope+'_result_year')
    origin=cols[1].selectbox('Referência da comparação',['Toda a curadoria','Revisada manualmente','Preenchida automaticamente'],key=scope+'_result_origin')
    search=cols[2].text_input('Pessoa',key=scope+'_result_person')
    if year!='Todos':
        frame=frame[pd.to_numeric(frame.ano,errors='coerce').eq(year)]
    if search:
        frame=frame[frame.id_pessoa.astype(str).str.contains(search,regex=False,case=False)]
    if origin=='Revisada manualmente':
        frame=frame[frame.origem_valor_curado.isin(['CORRIGIDO_MANUALMENTE','FAIRFACE_CONFIRMADO_MANUALMENTE'])]
    elif origin=='Preenchida automaticamente':
        frame=frame[frame.origem_valor_curado.eq('FAIRFACE_ALTA_CONFIANCA')]
    cols=st.columns(4)
    cols[0].metric('Total de pessoas',len(frame))
    cols[1].metric('Masculino · referência',int(frame.valor_curado.eq('MASCULINO').sum()))
    cols[2].metric('Feminino · referência',int(frame.valor_curado.eq('FEMININO').sum()))
    cols[3].metric('Sem referência',int((~frame.valor_curado.isin(['MASCULINO','FEMININO'])).sum()))
    st.dataframe(result_summary(frame,models),hide_index=True,width='stretch',column_config={
        'Concordância (%)':st.column_config.NumberColumn(format='%.1f%%'),
        'Divergência (%)':st.column_config.NumberColumn(format='%.1f%%')})
    display=frame[['id_pessoa','ano','valor_curado','origem_valor_curado']].rename(columns={
        'id_pessoa':'Pessoa','ano':'Ano','valor_curado':'Referência curada','origem_valor_curado':'Origem da referência'}).copy()
    for label,column in models.items():
        values=frame[column].astype('string').str.upper().str.replace('_PROVAVEL','',regex=False)
        display[label]=values
        comparable=values.isin(['MASCULINO','FEMININO']) & frame.valor_curado.isin(['MASCULINO','FEMININO'])
        display[label+' · avaliação']='SEM COMPARAÇÃO'
        display.loc[comparable,label+' · avaliação']='DIVERGE'
        display.loc[comparable & values.eq(frame.valor_curado),label+' · avaliação']='CONCORDA'
    st.dataframe(display,hide_index=True,width='stretch')
    st.caption('Percentuais entre registros comparáveis. Referência curada inclui valores automáticos do FairFace; concordância não equivale a acurácia independente.')
    with st.expander('Probabilidades, modelos e procedência'):
        allowed=['id_pessoa','valor_curado','origem_valor_curado']
        if modality in (None,'face'):
            allowed += [c for c in frame if 'fairface' in c or 'face' in c or c in ('modelo_metricas','prob_masculino_modelo','prob_feminino_modelo')]
        if modality in (None,'cranio'):
            allowed += [c for c in frame if 'cranio' in c or c in ('model','views','probability','referencia_da_execucao')]
        st.dataframe(frame[list(dict.fromkeys(allowed))],hide_index=True)
    st.download_button('Baixar resultados exibidos',display.to_csv(index=False).encode('utf-8-sig'),
                       f'resultados_{scope}.csv',key=scope+'_download_results')
