"""Display existing annotations and manifest-linked pairs without inference."""
import json
import pandas as pd
import streamlit as st
from .region_pilot import RUNS, REPO_ROOT, image_input, key, verified, render_overlay
from .region_pilot_ui import display_frame


def show_item(run, item):
    folder = run/key(item)
    original, status = image_input(REPO_ROOT/item['path'])
    if original is None:
        st.error(f'Original indisponível: {status}')
        return
    left, right = st.columns(2)
    left.image(display_frame(original), channels='BGR', width='stretch', caption='Original')
    paths = [folder/'extracao/data.json', folder/'regioes/data.json']
    if not all(verified(p) for p in paths):
        right.warning('Marcações indisponíveis ou arquivos alterados.')
        return
    ex, regions = [json.loads(p.read_text()) for p in paths]
    marked = render_overlay(original, ex, regions, ['Seleção regional', 'Propostas anteriores'])
    right.image(display_frame(marked), channels='BGR', width='stretch', caption='Marcações existentes · não validadas')
    missing = [str(p['numero']) for p in ex.get('proposals', []) if p.get('x') is None or p.get('y') is None]
    if missing:
        st.warning('Pontos ausentes nas propostas: ' + ', '.join(missing))
    with st.expander('Detalhes técnicos · ' + item['modalidade']):
        st.write({'id': item['id'], 'status': ex.get('status'), 'motivo': ex.get('reason'),
                  'pontos_detectados': len(ex.get('points', [])), 'arquivo': item['path']})
        st.dataframe(pd.DataFrame(regions.get('regions', {})).T)


def render():
    st.title('Piloto de imagens')
    st.caption('Marcações experimentais · sem validação anatômica independente')
    from .bank_audit import render_report
    render_report()
    area = st.radio('Área', ['Faces', 'Crânios', 'Comparação', 'Matching'], horizontal=True)
    if area == 'Matching':
        st.info('Busca de identidade face–crânio indisponível nesta interface de auditoria.')
        st.caption('Os arquivos e resultados anteriores foram preservados. Nenhum treinamento ou ranking é executado aqui.')
        return
    if area in ('Faces', 'Crânios'):
        source = st.radio('Imagens exibidas', ['Piloto com marcações', 'Todos os arquivos originais'], horizontal=True)
        if source == 'Todos os arquivos originais':
            from .file_browser_ui import render_files
            render_files('face' if area == 'Faces' else 'cranio')
            return
    runs = sorted([p for p in RUNS.glob('*') if (p/'entradas.json').is_file()], reverse=True)
    if not runs:
        st.info('Nenhuma execução do piloto encontrada.')
        return
    with st.expander('Execução e diagnóstico'):
        run = st.selectbox('Execução', runs, format_func=lambda p: p.name)
        if st.button('Verificar integridade'):
            from .technical_audit import audit_run
            with st.spinner('Verificando arquivos...'):
                checks = pd.DataFrame(audit_run(run))
            st.dataframe(checks[checks.status.ne('OK_TECNICO')], hide_index=True)
            st.download_button('Baixar diagnóstico', checks.to_csv(index=False).encode(), 'diagnostico.csv')
    items = json.loads((run/'entradas.json').read_text())
    modalities = ['face', 'cranio'] if area == 'Comparação' else ['face' if area == 'Faces' else 'cranio']
    ids = sorted({i['id'] for i in items if i['modalidade'] in modalities})
    if not ids:
        st.info('Não há imagens para esta seleção.')
        return
    identity = st.selectbox('Pessoa no manifesto' if area == 'Comparação' else 'Imagem', ids)
    if area == 'Comparação':
        st.caption('Par já vinculado no manifesto; esta tela não verifica identidade.')
    for modality in modalities:
        item = next((i for i in items if i['id'] == identity and i['modalidade'] == modality), None)
        st.subheader('Face' if modality == 'face' else 'Crânio')
        if item is None:
            st.warning('Imagem ausente no manifesto.')
        else:
            show_item(run, item)
