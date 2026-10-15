"""Browse original files independently of identity manifests."""
import pandas as pd
import streamlit as st

from .common import REPO_ROOT, IMAGE_EXTS
from .region_pilot import image_input
from .region_pilot_ui import display_frame


def inventory(modality, root=REPO_ROOT):
    base = root/('Treinamento_faces/dados_originais/faces' if modality == 'face'
                 else 'Treinamento_cranios/dados_originais/cranios')
    rows = []
    for path in sorted(base.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if modality == 'face' and path.parent.name != 'images':
            continue
        parts = path.relative_to(base).parts
        rows.append(dict(arquivo=path.name, ano=parts[0],
                         vista='face' if modality == 'face' else path.parent.name,
                         caminho=str(path)))
    return pd.DataFrame(rows, columns=['arquivo', 'ano', 'vista', 'caminho'])


def render_files(modality):
    frame = inventory(modality)
    st.caption('Arquivos das pastas de originais, sem depender do manifesto. Nomes parecidos não são associados automaticamente.')
    if frame.empty:
        st.info('Nenhum arquivo de imagem encontrado nesta pasta.')
        return
    st.write(f'{len(frame)} arquivos encontrados.')
    year = st.selectbox('Ano da pasta', ['Todos'] + sorted(frame.ano.unique()), key='file_year_'+modality)
    view = st.selectbox('Vista da pasta', ['Todas'] + sorted(frame.vista.unique()), key='file_view_'+modality)
    search = st.text_input('Buscar no nome do arquivo', key='file_search_'+modality).strip()
    filtered = frame
    if year != 'Todos':
        filtered = filtered[filtered.ano.eq(year)]
    if view != 'Todas':
        filtered = filtered[filtered.vista.eq(view)]
    if search:
        filtered = filtered[filtered.arquivo.str.contains(search, case=False, regex=False)]
    st.download_button('Baixar lista de arquivos', filtered.to_csv(index=False).encode(), 'arquivos_originais.csv', key='files_download_'+modality)
    if filtered.empty:
        st.info('Nenhum arquivo corresponde aos filtros. Tente o número do registro ou parte do nome.')
        return
    options = filtered.caminho.tolist()
    state_key = 'file_selected_'+modality
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = options[0]
    labels = dict(zip(filtered.caminho, filtered.ano+' · '+filtered.vista+' · '+filtered.arquivo))
    path = st.selectbox('Arquivo original', options, format_func=labels.get, key=state_key)
    index = options.index(path)
    def move(delta):
        st.session_state[state_key] = options[index+delta]
    left, right = st.columns(2)
    left.button('Anterior', disabled=index == 0, on_click=move, args=(-1,), key='files_prev_'+modality)
    right.button('Próximo', disabled=index == len(options)-1, on_click=move, args=(1,), key='files_next_'+modality)
    st.caption(f'{index+1} de {len(options)} arquivos filtrados')
    image, status = image_input(path)
    if image is None:
        st.error(f'Arquivo encontrado, mas não foi possível abrir: {status}')
    else:
        st.image(display_frame(image), channels='BGR', width='stretch', caption=labels[path])
        st.caption(f'Resolução original: {image.shape[1]} × {image.shape[0]} pixels. Sem marcações nesta visualização.')
    st.code(path, language=None)
