from pathlib import Path
import pandas as pd
import streamlit as st
from .core import load

def filter_ranking(ranking, labels, query_sex, field='sexo_documentado'):
    if query_sex not in ('MASCULINO', 'FEMININO'):
        raise ValueError('Informe uma referencia de sexo valida')
    kept, excluded = [], []
    for row in ranking:
        record = dict(row)
        record[field] = labels.get(row['person'], 'NAO INFORMADO')
        if record[field] in ('MASCULINO', 'FEMININO') and record[field] != query_sex:
            excluded.append(record)
        else:
            kept.append(record)
    return kept, excluded

def matching_controls(query):
    from Laboratorio_visual_face_cranio.core import ARTIFACTS
    from Laboratorio_visual_face_cranio.references_io import references
    with st.expander('Sexo informado e filtro exploratorio', expanded=True):
        query_sex = st.selectbox('Sexo provavel da consulta (informado por voce)',
                                ['NAO INFORMADO','MASCULINO','FEMININO'], key='query_sex_'+query['person'])
        source = st.radio('Referencia dos cranios', ['Documento', 'Estimativa craniana'], horizontal=True)
        labels = {}
        if source == 'Documento':
            paths = sorted((ARTIFACTS/'sex_references').glob('*/reference.csv'), reverse=True)
            paths = [p for p in paths if load(p.parent/'config.json').get('reference_kind', 'DOCUMENTAL') == 'DOCUMENTAL']
            if paths:
                ref = st.selectbox('Fonte documental', [p.parent for p in paths], format_func=lambda p:p.name)
                labels = references(ref)['sexo'].to_dict()
            else:
                st.info('Nenhuma referencia documental importada.')
        else:
            paths = sorted((ARTIFACTS/'cranial_sex_predictions').glob('*/predictions.json'), reverse=True)
            if paths:
                ref = st.selectbox('Execucao da estimativa craniana', paths, format_func=lambda p:p.parent.name)
                labels = {r['person']:r['predicted'] for r in load(ref) if r.get('status') == 'OK'}
            else:
                st.info('Treine e aplique o modelo craniano na aba Sexo registrado. Nao ha estimativas novas disponiveis.')
        enabled = st.checkbox('Mostrar ranking considerando o sexo informado', value=False)
        st.session_state['sex_filter_settings'] = dict(query=query, query_sex=query_sex, source=source, labels=labels, enabled=enabled)

def matching_filter(result):
    original = result['ranking']
    settings = st.session_state.get('sex_filter_settings', {})
    if settings.get('query') != result['query']:
        st.caption('Ranking sem filtro de sexo: os controles iniciais pertencem a outra consulta.')
        return original
    query_sex, source, labels, enabled = (settings[k] for k in ('query_sex','source','labels','enabled'))
    with st.container():
        if not enabled:
            return original
        if query_sex == 'NAO INFORMADO':
            st.warning('Informe o sexo da consulta; ranking original mantido.')
            return original
        field = 'sexo_documentado' if source == 'Documento' else 'sexo_estimado_cranio'
        kept, excluded = filter_ranking(original, labels, query_sex, field)
        for i,row in enumerate(kept, 1):
            row['posicao_original'] = row['rank']
            row['rank'] = i if row['score'] is not None else None
        st.warning('Filtro exploratorio, nao confirmacao de identidade. Uma estimativa errada pode excluir o correto. Candidatos sem referencia sao mantidos.')
        st.caption(f'{len(excluded)} excluidos; {sum(r[ field ] == "NAO INFORMADO" for r in kept)} sem referencia. Scores visuais nao foram recalculados. O comparador permanece no ranking visual original.')
        with st.expander('Ranking sem considerar sexo'):
            st.dataframe(pd.DataFrame(original), hide_index=True)
        with st.expander('Excluidos e suas posicoes originais', expanded=bool(excluded)):
            st.dataframe(pd.DataFrame(excluded), hide_index=True)
        if any(r['person'] == result['query']['person'] for r in excluded):
            st.error('O filtro removeu o candidato correto nesta consulta.')
        return kept
