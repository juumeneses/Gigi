from __future__ import annotations

import html
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Laboratorio_visual_face_cranio.core import (ARTIFACTS, GROUPS, REPO, ROOT, image,
                                                digest, letterbox, load, new_run, overlay,
                                                regions_for, save, sha, view_frame)
from Laboratorio_visual_face_cranio.data import inventory, legacy_sources, reference_runs

st.set_page_config(page_title='Laboratório de pontos e regiões', layout='wide')
st.markdown('<style>h1 {font-size:1.65rem !important;letter-spacing:0 !important;} '
            '.stMainBlockContainer {padding-top:2rem;} '
            '[data-testid="stImage"] img {border-radius:4px;}</style>', unsafe_allow_html=True)


@st.cache_data(ttl=30)
def bank_files():
    return inventory()


def runs(kind):
    return sorted((ARTIFACTS / kind).glob('*/config.json'), reverse=True)


def processing_label(value):
    if value in ('Referências existentes', 'Referências'):
        return value
    path = Path(value)
    action = load(path/'config.json')['action']
    labels = {'normalize': 'Normalização', 'apply': 'Pontos previstos', 'regions': 'Regiões das referências',
              'mesh': 'MediaPipe', 'contours': 'Contornos candidatos'}
    return labels.get(action, action) + ' · ' + path.name


def launch(action, args):
    process = st.session_state.get('worker')
    if process and process.poll() is None:
        st.warning('Uma execução já está em andamento.')
        return
    job = new_run('jobs')
    save(job / 'request.json', dict(action=action, args=args))
    save(job / 'status.json', dict(state='INICIANDO'))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(ARTIFACTS / 'mpl-cache'))
    with (job / 'stdout.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'cli.py'),
                                    '--request', str(job / 'request.json')],
                                   cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    st.session_state.worker = process
    st.session_state.worker_job = str(job)
    st.session_state.job = str(job)
    st.rerun()


@st.fragment(run_every=1)
def job_panel():
    name = st.session_state.get('job')
    if not name:
        return
    job = Path(name)
    status = load(job / 'status.json')
    worker = st.session_state.get('worker')
    if worker and st.session_state.get('worker_job') == name and worker.poll() is not None and status['state'] in ('INICIANDO', 'RODANDO'):
        status = dict(state='ERRO', error=f'Processo encerrou com código {worker.returncode}; veja o log.')
        save(job / 'status.json', status)
    with st.expander('Execução e logs · ' + status['state'], expanded=True):
        st.write(status)
        if (job/'process.json').exists():
            process_info = load(job/'process.json')
            if 'started_at' in process_info:
                elapsed = status.get('duration_seconds', time.time()-process_info['started_at'])
                st.caption(f'PID {process_info["pid"]} · {elapsed:.1f} segundos')
        logfile = job / 'stdout.log'
        if logfile.exists():
            with logfile.open('rb') as f:
                f.seek(max(0, logfile.stat().st_size - 24000))
                text = f.read().decode(errors='replace')
            components.html('<pre id="log" style="height:220px;overflow:auto;white-space:pre-wrap;font-size:13px">' +
                            html.escape(text) + '</pre><script>let e=document.getElementById("log");e.scrollTop=e.scrollHeight;</script>', height=240)
        a, b = st.columns(2)
        if a.button('Parar após etapa atual', disabled=status['state'] not in ('INICIANDO', 'RODANDO')):
            (job / 'stop.flag').touch()
        if b.button('Atualizar resultados'):
            st.rerun()


def choose_references():
    paths = reference_runs()
    with st.sidebar.expander('Referências e histórico', expanded=not bool(paths)):
        sources = legacy_sources()
        if sources:
            preferred = next((i for i, p in enumerate(sources) if p.parent.name == 'piloto_20260929_v3'), len(sources)-1)
            source = st.selectbox('Origem das marcações', sources, index=preferred, format_func=lambda p: p.parent.name)
            if st.button('Preparar referências', help='Importa pontos, regiões, hashes e splits em uma nova execução.'):
                launch('prepare', dict(source=str(source)))
        if not paths:
            st.info('Prepare as referências para habilitar o treinamento.')
            return None, []
        pending_reference = st.session_state.pop('next_reference', None)
        if pending_reference and Path(pending_reference) in paths:
            st.session_state['active_reference'] = Path(pending_reference)
        path = st.selectbox('Referências importadas', paths, format_func=lambda p: p.parent.name, key='active_reference')
        data = load(path)
        st.write(f'{len(data["samples"])} imagens · {len(data["failures"])} falhas')
        if data['failures']:
            st.dataframe(pd.DataFrame(data['failures']), hide_index=True)
        jobs = sorted((ARTIFACTS / 'jobs').glob('*/status.json'), reverse=True)[:15]
        if jobs:
            previous = st.selectbox('Histórico de comandos', jobs, format_func=lambda p: p.parent.name)
            if st.button('Abrir log desta execução'):
                st.session_state.job = str(previous.parent)
                st.rerun()
        return path, data['samples']


def reference_for(row, refs):
    return next((r for r in refs if row['key'] == r['key'] or row['key'] in r['bank_keys']), None)


def all_rows(refs):
    rows = list(bank_files())
    known = {r['path'] for r in rows}
    for r in refs:
        if r['path'] not in known:
            rows.append(dict(key=r['key'], path=r['path'], filename=Path(r['path']).name,
                             person=r['person'], year=r['year'], view=r['view'], modality=r['modality']))
    return rows


def training_runs(modality, view):
    return [p.parent for p in runs('training')
            if (load(p)['modality'], load(p)['view']) == (modality, view)]


def result_selector(modality, view, reference):
    available = training_runs(modality, view)
    if not available:
        return None
    eligible = []
    for r in available:
        if (r / 'evaluation.json').exists() and (r / 'best.pt').exists():
            config = load(r / 'config.json')
            if reference and config['reference_fingerprint'] == load(reference)['fingerprint']:
                eligible.append(r)
    best = min(eligible, key=lambda r: load(r / 'evaluation.json')['best_val_loss']) if eligible else None
    selected = st.selectbox('Modelo / execução', available,
                            index=available.index(best) if best else 0,
                            format_func=lambda p: p.name + (' (melhor validação desta referência)' if p == best else ''),
                            key='model_' + modality + view)
    if (selected / 'evaluation.json').exists():
        metrics = load(selected / 'evaluation.json')
        with st.expander('Resultados do modelo selecionado'):
            st.caption(metrics['interpretation'])
            cols = st.columns(3)
            cols[0].metric('Pessoas na validação', metrics['n_validation'])
            median = metrics['median_diagonal']
            cols[1].metric('Deslocamento mediano / diagonal', f'{median:.2%}' if median is not None else 'Indisponível')
            cols[2].metric('Pontos avaliados', f'{metrics["n_evaluated"]}/{metrics["n_reference_points"]}')
            st.write({'P95 / diagonal': metrics['p95_diagonal'],
                      'Baseline médio / diagonal': metrics['baseline_median_diagonal'],
                      'Época selecionada': metrics['checkpoint_epoch'], 'Falhas': metrics['failures']})
            st.dataframe(pd.DataFrame(metrics['points']), hide_index=True)
            st.dataframe(pd.DataFrame(metrics['regions']), hide_index=True)
            st.download_button('Baixar métricas', (selected/'evaluation.json').read_bytes(), 'avaliacao.json')
            if (selected/'history.json').exists():
                st.line_chart(pd.DataFrame(load(selected/'history.json')).set_index('epoch'))
    return selected


def training_controls(modality, view, reference, refs, selected):
    with st.expander('Treinar e avaliar pontos'):
        samples = [r for r in refs if r['modality'] == modality and r['view'] == view]
        nt = sum(r['split'] == 'treino' for r in samples)
        nv = sum(r['split'] == 'validacao' for r in samples)
        st.write(f'{nt} imagens de treino · {nv} de validação · referências de IA')
        ready = reference is not None and nt > 0 and nv > 0
        if not ready:
            st.info('Esta vista precisa de referências próprias nos splits de treino e validação.')
        cols = st.columns(3)
        epochs = cols[0].number_input('Total de épocas', 1, 10000, 100)
        lr = cols[1].number_input('Taxa de aprendizado', .000001, .1, .001, format='%.6f')
        batch = cols[2].number_input('Imagens por lote', 1, 128, 8)
        cols = st.columns(2)
        size = cols[0].selectbox('Tamanho de entrada', [128, 224, 256])
        seed = cols[1].number_input('Semente', 0, 100000, 42)
        args = dict(reference=str(reference), modality=modality, view=view, epochs=epochs,
                    lr=lr, batch_size=batch, size=size, seed=seed)
        a, b, c = st.columns(3)
        if a.button('Treinar pontos', disabled=not ready, type='primary', help='Treina uma nova execução com os pseudorrótulos disponíveis.'):
            launch('train', args)
        can_resume = selected is not None and (selected/'last.pt').exists()
        if b.button('Retomar treinamento', disabled=not can_resume,
                    help='Mantém dados e parâmetros da execução selecionada; usa o total de épocas acima.'):
            old = load(selected/'config.json')
            args = {k: old[k] for k in ('reference', 'modality', 'view', 'lr', 'batch_size', 'size', 'seed')}
            args.update(epochs=epochs, resume=str(selected))
            launch('train', args)
        if c.button('Avaliar execução', disabled=selected is None or not (selected/'best.pt').exists()):
            launch('evaluate', dict(run=str(selected)))


def processing_controls(rows, current, modality, view, selected, reference):
    with st.expander('Processar imagens'):
        scope = st.radio('Lote', ['Imagem selecionada', 'Todos os arquivos filtrados'], horizontal=True)
        chosen = [current] if scope == 'Imagem selecionada' else rows
        size = st.selectbox('Tamanho da normalização', [256, 512, 1024])
        a, b, c = st.columns(3)
        action = None
        if a.button('Normalizar imagens', help='Redimensiona proporcionalmente e registra a transformação em uma cópia.'):
            action = 'normalize'
        if b.button('Aplicar modelo', disabled=selected is None or not (selected/'best.pt').exists()):
            action = 'apply'
        if c.button('Gerar regiões', help='Gera boxes pelos pontos de referência; a aplicação do modelo também gera suas regiões.'):
            action = 'regions'
        a, b = st.columns(2)
        if a.button('Extrair MediaPipe', disabled=modality != 'face' or view not in ('FFN', 'FFS')):
            action = 'mesh'
        if b.button('Gerar contorno candidato', help='Segmentação de primeiro plano por Otsu; requer inspeção em fundos complexos.'):
            action = 'contours'
        if action:
            launch('process', dict(rows=chosen, action=action, size=size,
                                   training_run=str(selected) if selected else None,
                                   reference=str(reference) if reference else None))
        previous = [p.parent for p in runs('processing')]
        if previous:
            p = st.selectbox('Execução de processamento para retomar', previous, format_func=lambda x: x.name)
            if st.button('Retomar processamento'):
                old = load(p/'config.json')
                launch('process', dict(rows=load(p/'inputs.json'), action=old['action'],
                                       size=old['size'], training_run=old['training_run'],
                                       reference=old['reference'], run=str(p), resume=True))


def show_images(row, refs, original_only=False):
    try:
        im = image(row['path'])
    except ValueError as exc:
        st.error(str(exc))
        return
    if original_only:
        st.image(letterbox(im)[0], channels='BGR', width=600, caption=row['filename'])
        st.caption(f'{im.shape[1]} × {im.shape[0]} pixels · {row["view"]}')
        return
    ref = reference_for(row, refs)
    current_hash = sha(row['path'])
    if ref and ref['hash'] != current_hash:
        st.error('O arquivo mudou desde a importação; marcações anteriores não serão sobrepostas.')
        ref = None
    outputs = [p.parent for p in runs('processing') if (p.parent/row['key']/'result.json').exists()]
    source = st.selectbox('Camada exibida', ['Referências existentes'] + [str(p) for p in outputs],
                          format_func=processing_label,
                          key='layer_' + row['key'])
    data = dict(points=ref['points'] if ref else [], regions=ref['regions'] if ref else {}, contours=[])
    if source != 'Referências existentes':
        data = load(Path(source)/row['key']/'result.json')
        expected = data.get('input_hash')
        if expected is not None and expected != current_hash:
            st.error('Saída pertence a outra versão desta imagem.')
            return
        st.caption(data['status'] + (' · ' + data['reason'] if data['reason'] else ''))
    opts = st.multiselect('Camadas', ['Pontos', 'Regiões', 'Contornos', 'Nomes', 'Referências IA'],
                          default=['Pontos', 'Regiões'], key='layers_' + row['key'])
    region = st.selectbox('Região', ['Imagem inteira'] + list(GROUPS), key='region_' + row['key'])
    box = data.get('regions', {}).get(region)
    if region != 'Imagem inteira' and (not box or 'box' not in box):
        st.info('Região sem box disponível; use o zoom manual para inspecionar.')
        box = None
    with st.expander('Zoom manual'):
        zoom = st.slider('Ampliação', 1., 8., 1., .1, key='zoom_' + row['key'])
        cx = st.slider('Centro horizontal (%)', 0, 100, 50, key='cx_' + row['key'])
        cy = st.slider('Centro vertical (%)', 0, 100, 50, key='cy_' + row['key'])
    if zoom > 1:
        h, w = im.shape[:2]
        half = min(w, h)/zoom/2
        box = dict(box=[max(0, w*cx/100-half), max(0, h*cy/100-half),
                        min(w, w*cx/100+half), min(h, h*cy/100+half)])
    marked = overlay(im, data.get('points', []) if 'Pontos' in opts else [],
                     data.get('regions', {}) if 'Regiões' in opts else {},
                     data.get('contours', []) if 'Contornos' in opts else [], 'Nomes' in opts)
    if ref and 'Referências IA' in opts:
        marked = overlay(marked, ref['points'], point_color=(255, 220, 0))
        st.caption('Amarelo: camada selecionada · ciano: referências anteriores de IA')
    a, b = st.columns(2)
    a.image(view_frame(im, box), channels='BGR', caption='Original', width='stretch')
    if data.get('action') == 'normalize' and data.get('image'):
        b.image(letterbox(image(data['image']))[0], channels='BGR', caption='Normalizada', width='stretch')
    else:
        b.image(view_frame(marked, box), channels='BGR', caption='Marcações e regiões propostas', width='stretch')
    st.caption(row['filename'])
    if ref:
        st.caption(ref['origin'] + ' · ' + ref['split'])
    missing = [p['numero'] for p in data.get('points', []) if p.get('x') is None or p.get('y') is None]
    if missing:
        st.warning('Pontos ausentes: ' + ', '.join(map(str, missing)))
    if not data.get('points'):
        st.info('Esta imagem ainda não tem marcações vinculadas.')
    with st.expander('Pontos, regiões e revisão'):
        st.dataframe(pd.DataFrame(data.get('points', [])), hide_index=True)
        st.dataframe(pd.DataFrame(data.get('regions', {})).T)
        with st.form('review_' + row['key']):
            author = st.text_input('Autor')
            state = st.selectbox('Visibilidade', ['NAO_AVALIADO', 'VISIVEL', 'OCULTO', 'CORTADO', 'INDETERMINADO'])
            reason = st.text_area('Motivo')
            if st.form_submit_button('Salvar revisão'):
                if not author.strip():
                    st.error('Informe o autor.')
                else:
                    target = new_run('reviews')/'review.json'
                    save(target, dict(key=row['key'], source=source, image_hash=sha(row['path']),
                                      region=region, state=state, reason=reason, author=author.strip(),
                                      annotation_hash=digest(dict(points=data.get('points', []),
                                                                  regions=data.get('regions', {}))),
                                      reference=str(ref['source']) if ref else None))
                    st.success('Revisão registrada.')


def modality_page(modality, rows, refs, reference):
    rows = [r for r in rows if r['modality'] == modality]
    if not rows:
        st.info('Nenhum arquivo encontrado.')
        return
    sex_section = 'Gênero / Sexo facial' if modality == 'face' else 'Sexo craniano'
    section = st.segmented_control('Seção', ['Originais', 'Marcações', 'Treinamento', sex_section],
                                   default='Marcações', key='section_' + modality)
    if section == sex_section:
        from Laboratorio_visual_face_cranio.registered_ui import results
        if modality == 'cranio':
            from Laboratorio_visual_face_cranio.workflows_ui import sex_panel
            sex_panel(all_rows(refs), launch)
        results(modality)
        return
    views = sorted({r['view'] for r in rows})
    if section == 'Treinamento':
        view = st.selectbox('Vista do treinamento', views,
                            index=views.index('FFN' if modality == 'face' else 'CF'))
        selected = result_selector(modality, view, reference)
        training_controls(modality, view, reference, refs, selected)
        return
    cols = st.columns(3)
    year = cols[0].selectbox('Ano', ['Todos'] + sorted({r['year'] for r in rows}))
    view = cols[1].selectbox('Vista', sorted({r['view'] for r in rows}),
                           index=sorted({r['view'] for r in rows}).index('FFN' if modality == 'face' else 'CF'))
    only_refs = cols[2].checkbox('Somente com referências', value=section != 'Originais',
                                 key='onlyrefs_' + modality + str(section))
    rows = [r for r in rows if r['view'] == view and (year == 'Todos' or r['year'] == year)
            and (not only_refs or reference_for(r, refs))]
    selected = result_selector(modality, view, reference) if section != 'Originais' else None
    if not rows:
        st.info('Nenhuma imagem neste filtro. Desmarque “Somente com referências” para ver o Banco.')
        return
    options = {r['key']: r for r in rows}
    selected_key = 'image_' + modality + view
    if st.session_state.get(selected_key) not in options:
        st.session_state[selected_key] = next(iter(options))
    key = st.selectbox('Imagem', list(options), format_func=lambda k: options[k]['filename'], key=selected_key)
    current = options[key]
    if section != 'Originais':
        processing_controls(rows, current, modality, view, selected, reference)
    show_images(current, refs, original_only=section == 'Originais')
    position = list(options).index(key)
    def move(delta):
        st.session_state[selected_key] = list(options)[position+delta]
    a, b = st.columns(2)
    a.button('Anterior', disabled=position == 0, on_click=move, args=(-1,))
    b.button('Próxima', disabled=position == len(options)-1, on_click=move, args=(1,))
    st.caption(f'{position+1} de {len(rows)} imagens filtradas')


def comparison(rows, refs):
    people = sorted({r['person'] for r in rows if r['modality'] == 'face'} &
                    {r['person'] for r in rows if r['modality'] == 'cranio'})
    if not people:
        st.info('Nenhum par com ambas as modalidades cadastrado.')
        return
    only_refs = st.checkbox('Somente pares com referências', value=True)
    if only_refs:
        known = {r['person'] for r in refs}
        people = [p for p in people if p in known]
    if not people:
        st.info('Nenhum par com referências importadas neste filtro.')
        return
    person = st.selectbox('Pessoa cadastrada', people)
    orientation = st.radio('Vista da comparação', ['Frontal', 'Lateral'], horizontal=True)
    faces = [r for r in rows if r['person'] == person and r['view'] in (['FFN', 'FFS'] if orientation == 'Frontal' else ['FL'])]
    skulls = [r for r in rows if r['person'] == person and r['view'] in (['CF'] if orientation == 'Frontal' else ['CE', 'CD'])]
    if not faces or not skulls:
        st.warning('Vista ausente para este par.')
        return
    a, b = st.columns(2)
    face = a.selectbox('Fotografia', faces, format_func=lambda r: r['filename'])
    skull = b.selectbox('Crânio', skulls, format_func=lambda r: r['filename'])
    if orientation == 'Lateral':
        st.caption('O lado da fotografia FL não está confirmado pelo nome. Verifique a orientação; não há espelhamento automático.')
    layers = st.multiselect('Camadas da comparação', ['Pontos', 'Regiões'], default=['Pontos'])
    canvases = []
    for column, row in ((a, face), (b, skull)):
        try:
            im = image(row['path'])
        except ValueError as exc:
            column.error(str(exc))
            return
        ref = reference_for(row, refs)
        if ref and ref['hash'] != sha(row['path']):
            ref = None
            column.warning('Hash diferente da referência; marcações omitidas.')
        points = ref['points'] if ref else []
        regions = ref['regions'] if ref else {}
        available = [p.parent for p in runs('processing') if (p.parent/row['key']/'result.json').exists()]
        source = column.selectbox('Marcações · ' + row['modality'], ['Referências'] + [str(p) for p in available],
                                   format_func=processing_label)
        if source != 'Referências':
            output = load(Path(source)/row['key']/'result.json')
            if output.get('input_hash', sha(row['path'])) == sha(row['path']):
                points, regions = output.get('points', []), output.get('regions', {})
            else:
                points, regions = [], {}
                column.warning('Saída pertence a outra versão da imagem.')
        column.image(letterbox(im)[0], channels='BGR', caption='Original', width='stretch')
        marked = overlay(im, points if 'Pontos' in layers else [], regions if 'Regiões' in layers else {})
        canvas = letterbox(marked)[0]
        column.image(canvas, channels='BGR', caption='Marcada' if points else 'Sem pontos disponíveis', width='stretch')
        canvases.append(canvas)
    with st.expander('Ajustar sobreposição', expanded=True):
        if st.button('Restaurar enquadramento'):
            for key in ['alpha', 'rotation', 'scale', 'dx', 'dy']:
                st.session_state.pop(key, None)
            st.rerun()
        cols = st.columns(3)
        alpha = cols[0].slider('Opacidade da face', 0., 1., .5, .05, key='alpha')
        rotation = cols[1].slider('Rotação da face', -90., 90., 0., .5, key='rotation')
        scale = cols[2].slider('Escala da face', .2, 3., 1., .05, key='scale')
        dx = st.slider('Deslocamento horizontal', -400, 400, 0, key='dx')
        dy = st.slider('Deslocamento vertical', -400, 400, 0, key='dy')
        matrix = cv2.getRotationMatrix2D((300, 300), rotation, scale)
        matrix[:, 2] += [dx, dy]
        moved = cv2.warpAffine(canvases[0], matrix, (600, 600))
        merged = cv2.addWeighted(moved, alpha, canvases[1], 1-alpha, 0)
        st.image(merged, channels='BGR', width=600, caption='Sobreposição visual ajustada manualmente')
        st.download_button('Baixar sobreposição', cv2.imencode('.png', merged)[1].tobytes(), 'sobreposicao.png')
        if st.button('Salvar ajuste'):
            target = new_run('overlays')/'view.json'
            save(target, dict(face=face, skull=skull, matrix=matrix.tolist(), alpha=alpha, layers=layers))
            st.success('Ajuste salvo.')


def registered_sex():
    from Laboratorio_visual_face_cranio.registered_ui import results
    results()


def legacy_tables():
    sources = {
        'Curadoria facial': REPO/'Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv',
        'Modelo facial anterior': REPO/'Treinamento_faces/resultados/genero_faces/predicoes_genero_metricas_faces.csv',
        'Crânio por vista': REPO/'Treinamento_cranios/resultados/sexo_cranio/predicoes_sexo_cranio_por_vista.csv',
        'Comparação anterior': REPO/'Treinamento_cranios/resultados/sexo_cranio/comparacao_genero_face_cranio.csv',
    }
    source = st.selectbox('Fonte', list(sources))
    path = sources[source]
    if not path.exists():
        st.info('Fonte indisponível.')
        return
    frame = pd.read_csv(path)
    search = st.text_input('Filtrar registro')
    if search:
        frame = frame[frame.astype(str).apply(lambda col: col.str.contains(search, regex=False, case=False)).any(axis=1)]
    st.dataframe(frame, hide_index=True)
    st.download_button('Baixar tabela exibida', frame.to_csv(index=False).encode(), path.name)
    st.caption(str(path.relative_to(REPO)))
    st.warning('A matriz de confusão facial antiga foi calculada após reajuste com todos os dados; não é avaliação independente.')


st.title('Laboratório de pontos e regiões')
st.caption('Pseudorrótulos assistidos por IA · validação técnica e revisão visual')
reference, refs = choose_references()
area = st.radio('Área', ['Faces', 'Crânios', 'Matching crânio–face', 'Gênero / Sexo registrado'], horizontal=True)
job_panel()
rows = all_rows(refs)
if area in ('Faces', 'Crânios'):
    from Laboratorio_visual_face_cranio.workflows_ui import expansion_panel
    expansion_panel(reference, refs, launch, modality='face' if area == 'Faces' else 'cranio')
    modality_page('face' if area == 'Faces' else 'cranio', rows, refs, reference)
elif area == 'Matching crânio–face':
    from Laboratorio_visual_face_cranio.workflows_ui import matching_panel
    matching_panel(rows, launch, lambda: comparison(rows, refs))
else:
    registered_sex()
