"""Draft collection, intentionally separate from automatic point reviews."""
import cv2
import base64
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .common import LAB_ROOT, REPO_ROOT, read_image
from .reference_batch import save_draft
from .image_click import map_click

click_image = components.declare_component('cranial_point_click', path=str(LAB_ROOT/'src/click_image'))


def render_reference_batch():
    batches = sorted((LAB_ROOT/'dataset/referencia_anatomica').glob('*/lote.json'), reverse=True)
    if not batches:
        st.info('Lote de referência ainda não preparado.')
        return
    folder = st.selectbox('Lote de referência', batches, format_func=lambda p: p.parent.name).parent
    images = pd.read_csv(folder/'imagens.csv', keep_default_na=False)
    schema = pd.read_csv(folder/'pontos_artigo.csv')
    st.warning('Rascunhos não são referências validadas. Treinamento bloqueado até verificação anatômica independente.')
    with st.expander('Resumo do lote'):
        st.dataframe(images.groupby(['ano','status_CF']).size().rename('imagens').reset_index(), hide_index=True)
    year = st.selectbox('Ano do lote', ['Todos'] + sorted(images.ano.unique().tolist()))
    filtered = images if year == 'Todos' else images[images.ano.eq(year)]
    only_proposed = st.checkbox('Somente imagens com rascunho', value=True)
    if only_proposed:
        filtered = filtered[filtered.id_pessoa.map(lambda identity: bool(list((folder/'rascunhos'/identity).glob('*.csv'))))]
    if filtered.empty:
        st.info('Nenhum rascunho neste filtro. Desmarque o filtro para ver as demais imagens.')
        return
    identity = st.selectbox('Imagem do lote', filtered.id_pessoa.tolist())
    row = images[images.id_pessoa.eq(identity)].iloc[0]
    if row.status_CF != 'OK':
        st.error(f'CF indisponível: {row.status_CF}')
        return
    image_path = REPO_ROOT/row.imagem_CF
    image = read_image(image_path)
    if image is None:
        st.error('Imagem original não pôde ser aberta.')
        return
    with st.expander('Protocolo do lote'):
        protocol_path = folder/'protocolo.md'
        if not protocol_path.exists():
            protocol_path = LAB_ROOT/'docs/protocolo_referencia_artigo.md'
        st.markdown(protocol_path.read_text())
    with st.expander('Figura de referência do artigo'):
        st.image(str(LAB_ROOT/'docs/figura_1_landmarks_2026.jpg'), caption='Figura 1; referência anatômica, não marcação desta imagem', width='stretch')
    with st.expander('Vistas laterais de apoio'):
        for column, view in zip(st.columns(2), ['CE','CD']):
            with column:
                if row['status_'+view] == 'OK':
                    st.image(str(REPO_ROOT/row['imagem_'+view]), caption=view)
                else:
                    st.info(f'{view}: {row["status_"+view]}')
    drafts = sorted((folder/'rascunhos'/identity).glob('*.csv'))
    initial = pd.read_csv(drafts[-1]) if drafts else pd.DataFrame(dict(
        ponto=schema.ponto, x_px=float('nan'), y_px=float('nan'), estado='PENDENTE', observacao=''))
    initial['observacao'] = initial.observacao.fillna('')
    draft_key = f'working_{folder.name}_{identity}'
    revision_key = draft_key+'_revision'
    if draft_key not in st.session_state:
        st.session_state[draft_key] = initial[['ponto','x_px','y_px','estado','observacao']].copy()
        st.session_state[revision_key] = 0
    initial = st.session_state[draft_key].copy()
    available = initial[initial.estado.isin(['PROPOSTA','LOCALIZADO']) & initial.x_px.notna() & initial.y_px.notna()].ponto.tolist()
    options = available + [p for p in schema.ponto if p not in available]
    point = st.selectbox('Ponto em análise', options, key=f'point_{folder.name}_{identity}',
        format_func=lambda p: f'{p} — com coordenada' if p in available else f'{p} — sem coordenada')
    definition = schema[schema.ponto.eq(point)].iloc[0]
    st.info(f'{definition.nome}: {definition.definicao} Cuidado: {definition.cuidado}')
    visual = st.container()
    with st.expander('Editar coordenadas e estados (avançado)', expanded=False):
        edited = st.data_editor(initial[['ponto','x_px','y_px','estado','observacao']], hide_index=True,
            disabled=['ponto'], key=f'editor_{draft_key}_{st.session_state[revision_key]}',
            column_config={'estado': st.column_config.SelectboxColumn(options=['PENDENTE','PROPOSTA','LOCALIZADO','NAO_VISIVEL','DUVIDA']),
                           'x_px': st.column_config.NumberColumn(min_value=0, max_value=image.shape[1]-1),
                           'y_px': st.column_config.NumberColumn(min_value=0, max_value=image.shape[0]-1)})
    st.session_state[draft_key] = edited.copy()
    overlay = image.copy()
    for p in edited.to_dict('records'):
        x, y = pd.to_numeric(p['x_px'], errors='coerce'), pd.to_numeric(p['y_px'], errors='coerce')
        if p['estado'] in ['LOCALIZADO','PROPOSTA'] and pd.notna(x) and pd.notna(y) and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            radius = max(8, round(max(image.shape[:2])*.007))
            color = (0,255,255) if p['ponto']==point else (0,160,255)
            cv2.circle(overlay, (round(x),round(y)), radius+3, (0,0,0), -1)
            cv2.circle(overlay, (round(x),round(y)), radius, color, 3)
            cv2.drawMarker(overlay, (round(x),round(y)), color, cv2.MARKER_CROSS, radius*3, 2)
    with visual:
        choice = edited[edited.ponto.eq(point)].iloc[0]
        x, y = pd.to_numeric(choice.x_px, errors='coerce'), pd.to_numeric(choice.y_px, errors='coerce')
        st.caption(f'{identity} · {choice.estado} · {len(available)} pontos com coordenada / {len(schema)}')
        mode = st.radio('Visualização', ['Região ampliada','Imagem inteira'], horizontal=True, key=f'view_{identity}')
        if mode == 'Região ampliada' and pd.notna(x) and pd.notna(y) and 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            radius = max(70, round(min(image.shape[:2])*.16))
            left, top = max(0,round(x)-radius), max(0,round(y)-radius)
            right, bottom = min(image.shape[1],round(x)+radius), min(image.shape[0],round(y)+radius)
            original_view, marked_view = image[top:bottom,left:right], overlay[top:bottom,left:right]
        else:
            original_view, marked_view = image, overlay
            left, top, right, bottom = 0, 0, image.shape[1], image.shape[0]
            if pd.isna(x) or pd.isna(y):
                st.warning('Este ponto ainda não tem coordenada proposta.')
        # Keep the marked region prominent even in a narrow browser panel.
        bounds = [left, top, right, bottom]
        context = json.dumps([draft_key, point, bounds, st.session_state[revision_key]])
        ok, encoded = cv2.imencode('.png', marked_view)
        if not ok:
            st.error('Falha ao preparar imagem para marcar.')
            return
        event = click_image(image='data:image/png;base64,'+base64.b64encode(encoded).decode(),
                            context=context, point=point, key=f'canvas_{draft_key}_{point}', default=None)
        if event and event.get('event_id') != st.session_state.get(draft_key+'_last_click'):
            st.session_state[draft_key+'_last_click'] = event.get('event_id')
            if event.get('context') == context:
                x_new, y_new = map_click(event['u'], event['v'], bounds)
                updated = edited.copy()
                updated.loc[updated.ponto.eq(point), ['x_px','y_px','estado']] = [x_new, y_new, 'LOCALIZADO']
                st.session_state[draft_key] = updated
                st.session_state[revision_key] += 1
                st.rerun()
        st.caption(f'{point} · {choice.estado} · X: {x if pd.notna(x) else "—"} · Y: {y if pd.notna(y) else "—"}')
        actions = st.columns(3)
        for column, label, state, icon in zip(actions, ['Limpar ponto','Não visível','Dúvida'],
                ['PENDENTE','NAO_VISIVEL','DUVIDA'], [':material/delete:',':material/visibility_off:',':material/help:']):
            if column.button(label, icon=icon, key=f'{draft_key}_{state}'):
                updated = edited.copy()
                updated.loc[updated.ponto.eq(point), ['x_px','y_px','estado']] = [float('nan'),float('nan'),state]
                st.session_state[draft_key] = updated
                st.session_state[revision_key] += 1
                st.rerun()
        with st.expander('Comparar com original sem marcação'):
            st.image(original_view, channels='BGR', caption='Mesma região, sem marcação', width='stretch')
    reviewer = st.text_input('Responsável pelo rascunho')
    if st.button('Salvar rascunho', icon=':material/save:'):
        try:
            save_draft(folder, identity, edited, schema, image_path, row.hash_CF, reviewer)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success('Nova revisão salva. Nenhum dado foi liberado para treino.')
    st.download_button('Baixar tabela atual', edited.to_csv(index=False).encode('utf-8-sig'), file_name=f'{identity}_rascunho.csv')
