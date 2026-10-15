"""Compact Streamlit workflows; heavy actions use the persisted CLI job runner."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .core import ARTIFACTS, GROUPS, POINT_NAMES, atomic, coordinates, digest, image, letterbox, load, new_run, overlay, regions_for, save, sha, view_frame
from .references import paired


def model_choices(modality, label, key):
    paths = [p.parent for p in sorted((ARTIFACTS/'training').glob('*/config.json'), reverse=True)
             if load(p)['modality'] == modality and load(p)['view'] == ('FFN' if modality == 'face' else 'CF') and (p.parent/'best.pt').exists()]
    if not paths:
        st.info('Treine um localizador frontal de ' + modality + ' primeiro.')
        return None
    return st.selectbox(label, paths, format_func=lambda p: p.name, key=key)


def expansion_panel(reference, refs, launch, modality='face'):
    with st.expander('Ampliar e revisar referências de pontos'):
        notice = st.session_state.pop('review_saved_notice', None)
        if notice:
            st.success(notice)
        a, b = st.columns(2)
        with a:
            face = model_choices('face', 'Localizador facial do lote', 'batch_face')
        with b:
            skull = model_choices('cranio', 'Localizador craniano do lote', 'batch_skull')
        count = st.number_input('Pessoas no lote', 2, 1000, 50, key='batch_count')
        seed = st.number_input('Semente do lote', 0, 100000, 42, key='batch_seed')
        if st.button('Preparar lote de propostas', disabled=not (reference and face and skull), help='Seleciona pessoas de treino/validação e gera propostas; a revisão decide quais entram no treino.'):
            launch('expand', dict(reference=str(reference), face_run=str(face), skull_run=str(skull), count=count, seed=seed))
        batches = sorted((ARTIFACTS/'annotation_batches').glob('*/config.json'), reverse=True)
        if not batches:
            return
        batch = st.selectbox('Lote para revisão', [p.parent for p in batches], format_func=lambda p: p.name)
        refined = load(batch/'config.json').get('refinement')
        if refined:
            st.caption('Lote refinado · veja a procedência dos ajustes em cada imagem · revisão independente pendente')
        if (batch/'report.html').exists():
            st.download_button('Baixar revisão visual antes/depois', (batch/'report.html').read_bytes(),
                               'revisao_nasal.html', mime='text/html')
        if st.button('Retomar propostas do lote', disabled=bool(refined) or not (face and skull and reference)):
            config = load(batch/'config.json')
            source = {k: config[k] for k in ['reference', 'face_run', 'skull_run', 'count', 'seed']}
            launch('expand', dict(source, resume=str(batch)))
        all_proposals = sorted(batch.glob('*/proposal.json'), key=lambda p: (load(p)['person'], load(p)['modality']))
        selected_modality = st.selectbox('Modalidade do lote', ['face', 'cranio'], index=0 if modality == 'face' else 1,
                                         format_func=lambda m: 'Faces' if m == 'face' else 'Crânios', key='batch_modality_'+modality)
        proposals = [p for p in all_proposals if load(p)['modality'] == selected_modality]
        from .references import current_reviews
        decisions = current_reviews(batch)
        approved = {str(p) for p in proposals if str(p) in decisions}
        ready = {str(p) for p in proposals if str(p) in decisions and any(
            r.get('source') and Path(r['source']).resolve() == decisions[str(p)][0].resolve()
            and r.get('source_hash') == sha(decisions[str(p)][0]) for r in refs)}
        totals = st.columns(3)
        totals[0].metric('Pendentes de revisão', len(proposals)-len(approved))
        totals[1].metric('Confirmadas', len(approved))
        totals[2].metric('Prontas para treinamento', len(ready))
        queue = st.segmented_control('Fila de imagens', ['Pendentes', 'Prontas para treinamento', 'Todas'],
                                     default='Pendentes', key='review_queue_'+modality)
        if queue == 'Pendentes':
            proposals = [p for p in proposals if str(p) not in approved]
        elif queue == 'Prontas para treinamento':
            proposals = [p for p in proposals if str(p) in ready]
        if not proposals:
            st.info('Nenhuma imagem nesta fila.')
            return
        if proposals:
            selection_key = 'review_image_'+batch.name+'_'+selected_modality
            if st.session_state.get(selection_key) not in proposals:
                st.session_state.pop(selection_key, None)
            selected = st.selectbox('Imagem do lote', proposals, format_func=lambda p: load(p)['filename'], key=selection_key)
            row = load(selected)
            if row['status'] == 'ERRO':
                st.error(row['error'])
            else:
                st.caption(row['person'] + ' · ' + row['split'])
                if row.get('visual_audit'):
                    st.caption(row['visual_audit']['summary'])
                if (selected.parent/'nose_comparison.jpg').exists():
                    with st.expander('Nariz: antes e depois'):
                        st.image(str(selected.parent/'nose_comparison.jpg'), width='stretch')
                original = image(row['path'])
                frame = pd.DataFrame([dict(numero=p['numero'], ponto=POINT_NAMES[p['numero']-1], x=p.get('x'), y=p.get('y'),
                                           visibility=p.get('visibility', 'NAO_AVALIADO')) for p in row['points']])
                previous = [p for p in sorted((ARTIFACTS/'point_reviews').glob('*/review.json'), reverse=True) if load(p)['proposal'] == str(selected)]
                if previous:
                    decision = load(previous[0])
                    frame = pd.DataFrame([dict(numero=p['numero'], ponto=POINT_NAMES[p['numero']-1], x=p.get('x'), y=p.get('y'), visibility=p['visibility']) for p in decision['points']])
                    frame.attrs['reviewed'] = True
                    frame.attrs['review_hash'] = sha(previous[0])
                from .point_editor import editor
                points = editor(selected, row, original, frame)
                with st.expander('Detalhes da revisão (opcional)'):
                    author = st.text_input('Autor da revisão dos pontos', key='point_review_author')
                    reason = st.text_input('Observações da revisão', key='point_review_reason')
                author = author.strip() or 'Usuário do painel'
                st.caption('Ao confirmar a imagem, os pontos com coordenadas são aceitos. Ocultos, cortados e indeterminados continuam omitidos.')
                if st.button('OK · salvar e preparar para treinamento', type='primary', disabled=reference is None,
                             icon=':material/check_circle:'):
                    from .references import confirm_image_points, review, promote
                    try:
                        decision = review(selected, confirm_image_points(points), author, reason)
                        new_reference = promote(reference, batch, log=lambda message: None)/'references.json'
                        st.session_state['next_reference'] = str(new_reference)
                        st.session_state['review_saved_notice'] = row['filename']+' confirmada e incorporada às referências de treinamento.'
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                if st.button('Salvar pontos revisados'):
                    from .references import review
                    try:
                        review(selected, points, author, reason)
                        st.success('Revisão salva. Incorpore as revisões para criar outra versão das referências.')
                    except ValueError as exc:
                        st.error(str(exc))
            st.caption(f'{len(proposals)} imagens nesta modalidade · {len(all_proposals)} imagens no lote completo')
            def select_proposal(target):
                st.session_state[selection_key] = target
            index = proposals.index(selected)
            a, b = st.columns(2)
            a.button('Anterior no lote', disabled=index == 0, on_click=select_proposal, args=(proposals[max(0, index-1)],))
            b.button('Próxima no lote', disabled=index == len(proposals)-1, on_click=select_proposal, args=(proposals[min(len(proposals)-1, index+1)],))
            if st.button('Incorporar revisões em novas referências', disabled=reference is None):
                launch('promote', dict(reference=str(reference), batch=str(batch)))


def sex_panel(rows, launch):
    from .cranial_sex import import_reference, references
    with st.expander('Novo treinamento de sexo craniano · CF/CE/CD', expanded=True):
        uploaded = st.file_uploader('Referências de sexo (CSV)', type=['csv'], key='sex_reference_upload')
        st.caption('Colunas: id_pessoa, sexo, fonte, tipo_referencia. Tipo: DOCUMENTAL ou ESPECIALISTA.')
        author = st.text_input('Responsável pela importação')
        confirmed = st.checkbox('Conferi a origem documental ou especializada dos valores deste arquivo')
        if st.button('Importar referências de sexo', disabled=uploaded is None):
            try:
                import_reference(pd.read_csv(uploaded, dtype=str), author, confirmed)
                st.success('Referências importadas.')
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
        paths = sorted((ARTIFACTS/'sex_references').glob('*/reference.csv'), reverse=True)
        ref = st.selectbox('Referências para sexo craniano', [p.parent for p in paths], format_func=lambda p: p.name + ' | ' + load(p/'config.json').get('reference_kind', 'DOCUMENTAL')) if paths else None
        if not ref:
            st.info('Prepare uma referencia curada experimental acima ou importe referencias documentadas com fonte identificada.')
        else:
            if load(ref/'config.json').get('reference_kind') == 'CURADORIA_EXPLORATORIA':
                st.warning('Treino experimental com curadoria: metricas medem concordancia com esses rotulos, nao validacao independente de sexo biologico.')
            with st.expander('Conferir referencias selecionadas'):
                st.dataframe(references(ref).reset_index(), hide_index=True)
        C = st.number_input('Regularização do modelo craniano (C)', .001, 100., 1., key='sex_C')
        a, b, c = st.columns(3)
        low = a.number_input('Menor limiar craniano', .01, .99, .3, key='sex_low')
        high = b.number_input('Maior limiar craniano', .01, .99, .7, key='sex_high')
        seed = c.number_input('Seed do modelo craniano', 0, 100000, 42, key='sex_seed')
        if st.button('Iniciar novo treino craniano · CF/CE/CD', disabled=ref is None, type='primary',
                     help='Cria outra execução usando as referências selecionadas, preservando os modelos anteriores.'):
            launch('train_cranial_sex', dict(rows=rows, reference=str(ref), C=C, threshold_low=low, threshold_high=high, seed=seed))
        models = sorted((ARTIFACTS/'cranial_sex').glob('*/config.json'), reverse=True)
        models = [p for p in models if (p.parent/'evaluation.json').exists() and (p.parent/'model.joblib').exists()]
        if models:
            run = st.selectbox('Modelo craniano / execução', [p.parent for p in models], format_func=lambda p: p.name)
            st.write(load(run/'evaluation.json'))
            a, b = st.columns(2)
            if a.button('Aplicar modelo craniano ao banco'):
                launch('apply_cranial_sex', dict(run=str(run), items=paired(rows)))
            if b.button('Avaliar sexo craniano no teste congelado'):
                launch('apply_cranial_sex', dict(run=str(run), items=[p for p in paired(rows) if p['split'] == 'teste'], evaluate_test=True))
        predictions = sorted((ARTIFACTS/'cranial_sex_predictions').glob('*/predictions.json'), reverse=True)
        if predictions:
            path = st.selectbox('Resultados cranianos novos', predictions, format_func=lambda p: p.parent.name)
            output = pd.DataFrame(load(path))
            st.dataframe(output, hide_index=True)
            st.download_button('Baixar novas estimativas cranianas', output.to_csv(index=False).encode(), 'estimativas_cranianas.csv')
            if (path.parent/'metrics.json').exists():
                st.write(load(path.parent/'metrics.json'))


def uploaded_row(upload, view, person):
    content = upload.getvalue()
    import hashlib
    token = hashlib.sha256(content).hexdigest()
    target = ARTIFACTS/'uploads'/token/(view+Path(upload.name).suffix.lower())
    if not target.exists():
        atomic(target, content)
    image(target)
    return dict(key=token, path=str(target), hash=token, person=person, view=view, filename=upload.name, modality='face' if view.startswith('F') else 'cranio')


def gallery_preview(query, candidates):
    entries = [('Face de referência', query, 'FFN')] + [(r['person'], r, 'CF') for r in candidates]
    for i, (title, item, view) in enumerate(entries):
        if i % 3 == 0:
            columns = st.columns(3)
        try:
            columns[i % 3].image(letterbox(image(item['views'][view]['path']), 300)[0], channels='BGR', width='stretch', caption=title)
        except (ValueError, KeyError) as exc:
            columns[i % 3].error(str(exc))


def matching_run_controls(rows, launch, epochs):
    available = [p.parent for p in sorted((ARTIFACTS/'matching').glob('*/config.json'), reverse=True) if (p.parent/'best.pt').exists()]
    if not available:
        st.info('Treine uma execução de matching para habilitar o ranking.')
        return None
    family = st.selectbox('Família do modelo', sorted({load(p/'config.json')['kind'] for p in available}))
    family_runs = [p for p in available if load(p/'config.json')['kind'] == family]
    protocol = load(family_runs[0]/'config.json')['input_fingerprint']
    comparable = [p for p in family_runs if load(p/'config.json')['input_fingerprint'] == protocol and (p/'evaluation.json').exists()]
    best = max(comparable, key=lambda p: load(p/'evaluation.json')['mrr']) if comparable else family_runs[0]
    run = st.selectbox('Execução de matching', family_runs, index=family_runs.index(best),
                       format_func=lambda p: p.name + (' (melhor validação neste manifesto)' if p == best else ''))
    st.write(load(run/'evaluation.json'))
    if (run/'regional_diagnostic.json').exists():
        with st.expander('Diagnóstico geométrico: pares corretos e incorretos no treino'):
            st.dataframe(pd.DataFrame(load(run/'regional_diagnostic.json')), hide_index=True)
    if (run/'history.json').exists():
        st.line_chart(pd.DataFrame(load(run/'history.json')).set_index('epoch'))
    with st.expander('Retomar e avaliar a execução selecionada'):
        if st.button('Retomar matching', disabled=not (run/'last.pt').exists()):
            old = load(run/'config.json')
            resumed = {k: old[k] for k in ['face_run', 'cranio_run', 'kind', 'loss', 'lr', 'batch_size', 'seed', 'dimension', 'temperature', 'margin', 'patience', 'multiview', 'regional', 'ridge_alpha']}
            launch('train_matching', dict(resumed, rows=load(run/'inputs_rows.json'), epochs=epochs, resume=str(run)))
        split = st.selectbox('Split da avaliação', ['validacao', 'teste'])
        test_confirmed = st.checkbox('Configuração congelada para avaliar teste final') if split == 'teste' else True
        if st.button('Avaliar ranking e Gallery-10', disabled=not test_confirmed):
            launch('evaluate_matching', dict(run=str(run), rows=rows, split=split))
        reports = [p for p in sorted((ARTIFACTS/'matching_evaluations').glob('*/config.json'), reverse=True) if load(p)['run'] == str(run)]
        if reports:
            report = st.selectbox('Avaliação anterior', reports, format_func=lambda p: p.parent.name)
            st.write(load(report.parent/'metrics.json'))
            st.dataframe(pd.DataFrame(load(report.parent/'cases.json')), hide_index=True)
    return run


def matching_panel(rows, launch, comparison):
    from .matching import sample_gallery
    with st.expander('Treinamento e avaliação do matching'):
        kind = st.selectbox('Experimento', ['image', 'geometry', 'hybrid'], format_func=lambda s: {'image': 'Duas torres visuais', 'geometry': 'Baseline geométrico aprendido', 'hybrid': 'Imagem + geometria'}[s])
        loss = st.selectbox('Perda', ['infonce', 'triplet'])
        a, b = st.columns(2)
        with a:
            face = model_choices('face', 'Localizador para face', 'match_face')
        with b:
            skull = model_choices('cranio', 'Localizador para crânio', 'match_skull')
        a, b, c = st.columns(3)
        epochs = a.number_input('Épocas do matching', 1, 10000, 100)
        batch = b.number_input('Lote de identidades', 2, 128, 16)
        seed = c.number_input('Seed do treinamento', 0, 100000, 42)
        a, b, c = st.columns(3)
        lr = a.number_input('Learning rate do matching', .000001, .1, .0003, format='%.6f')
        temp = b.number_input('Temperatura InfoNCE', .001, 1., .07)
        margin = c.number_input('Margem triplet', .01, 2., .2)
        multiview = st.checkbox('Usar vistas adicionais disponíveis')
        regional = st.checkbox('Usar recortes regionais', disabled=kind == 'geometry')
        alpha = st.number_input('Regularização Ridge', .001, 10000., 10.)
        args = dict(rows=rows, face_run=str(face) if face else None, cranio_run=str(skull) if skull else None,
                    kind=kind, loss=loss, epochs=epochs, lr=lr, batch_size=batch, seed=seed, temperature=temp, margin=margin,
                    multiview=multiview, regional=regional if kind != 'geometry' else False, ridge_alpha=alpha)
        if st.button('Treinar matching', type='primary'):
            launch('train_matching', args)
    run = matching_run_controls(rows, launch, epochs)
    mode = st.segmented_control('Comparação', ['Entrega final', 'Teste conhecido'], default='Teste conhecido')
    query, candidates = None, []
    if mode == 'Teste conhecido':
        pool_split = st.selectbox('Pessoas disponíveis', ['validacao', 'treino'], format_func=lambda s: s + (' · demonstração' if s == 'treino' else ''))
        pool = [p for p in paired(rows) if p['split'] == pool_split]
        if len(pool) < 2:
            st.warning('Split com menos de duas pessoas disponíveis.')
            return
        count = st.number_input('Quantidade de crânios', 2, len(pool), min(10, len(pool)), key='known_count')
        gallery_seed = st.number_input('Semente do sorteio', 0, 100000, 42, key='gallery_seed')
        pool_fingerprint = digest([p['person'] for p in pool])
        if st.button('Sortear outra galeria') or 'gallery' not in st.session_state:
            st.session_state.gallery = sample_gallery(pool, count, gallery_seed)
            st.session_state.gallery_pool = pool_split
            st.session_state.gallery_fingerprint = pool_fingerprint
        sample = st.session_state.gallery
        if st.session_state.get('gallery_pool') != pool_split or st.session_state.get('gallery_fingerprint') != pool_fingerprint:
            st.info('Clique em Sortear outra galeria para usar o split selecionado.')
        query, candidates = sample['query'], sample['candidates']
    else:
        front = st.file_uploader('Face frontal', type=['jpg', 'jpeg', 'png'], key='upload_face')
        lateral = st.file_uploader('Face lateral opcional', type=['jpg', 'jpeg', 'png'], key='upload_face_lateral')
        count = st.number_input('Candidatos cranianos', 2, 100, 10, key='upload_count')
        if front:
            try:
                query = dict(person='FACE_ENVIADA', views={'FFN': uploaded_row(front, 'FFN', 'FACE_ENVIADA')})
                if lateral:
                    query['views']['FL'] = uploaded_row(lateral, 'FL', 'FACE_ENVIADA')
            except ValueError as exc:
                st.error(str(exc))
        for i in range(count):
            person = f'CANDIDATO_{i+1:03d}'
            with st.expander(person):
                cols = st.columns(3)
                views = {}
                for col, view in zip(cols, ['CF', 'CE', 'CD']):
                    uploaded = col.file_uploader(view + (' obrigatória' if view == 'CF' else ' opcional'), type=['jpg', 'jpeg', 'png'], key=f'upload_{person}_{view}')
                    if uploaded:
                        try:
                            views[view] = uploaded_row(uploaded, view, person)
                        except ValueError as exc:
                            col.error(str(exc))
                if 'CF' in views:
                    candidates.append(dict(person=person, views=views))
        if len(candidates) != count:
            st.info(f'{len(candidates)}/{count} candidatos com CF enviada.')
    reference_sex = None
    candidate_sexes = {}
    use_sex = st.checkbox('Filtrar por sexo informado/documentado')
    if use_sex:
        reference_sex = st.selectbox('Sexo informado da referência', ['INDEFINIDO', 'MASCULINO', 'FEMININO'])
        paths = sorted((ARTIFACTS/'sex_references').glob('*/reference.csv'), reverse=True)
        documented = {}
        if paths:
            from .cranial_sex import references
            documented = references(paths[0].parent).sexo.to_dict()
        for r in candidates:
            options = ['INDEFINIDO', 'MASCULINO', 'FEMININO']
            candidate_sexes[r['person']] = st.selectbox('Sexo registrado · '+r['person'], options,
                                                        index=options.index(documented.get(r['person'], 'INDEFINIDO')), key='sex_candidate_'+r['person'])
    if query and candidates:
        with st.expander('Imagens selecionadas', expanded=True):
            gallery_preview(query, candidates)
        if st.button('Comparar e ranquear', type='primary', disabled=run is None or (mode == 'Entrega final' and len(candidates) != count)):
            launch('compare_matching', dict(run=str(run), query=query, candidates=candidates,
                                           reference_sex=reference_sex, candidate_sexes=candidate_sexes))
    results = sorted((ARTIFACTS/'matching_comparisons').glob('*/result.json'), reverse=True)
    if results:
        result_path = st.selectbox('Resultado da comparação', results, format_func=lambda p: p.parent.name)
        result = load(result_path)
        request = load(result_path.parent/'inputs.json')
        st.caption('Modelo e galeria deste resultado: ' + result['run'])
        frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ('points', 'failures', 'views')} for r in result['ranking']])
        st.dataframe(frame, hide_index=True)
        st.download_button('Baixar ranking', frame.to_csv(index=False).encode(), 'ranking.csv')
        if result.get('correct_excluded'):
            st.error('O filtro excluiu o candidato correto.')
        elif result.get('correct_rank'):
            st.metric('Posição do crânio correto', result['correct_rank'])
        selected = st.selectbox('Inspecionar candidato na ordem original', result['ranking'], format_func=lambda r: f'{r["original_rank"] or "—"} · {r["person"]}', key='candidate_'+result_path.parent.name)
        candidate = next(r for r in request['candidates'] if r['person'] == selected['person'])
        inspect_pair(request['query'], candidate, result['query_points'], selected['points'], result_path.parent.name)
    with st.expander('Comparação e sobreposição de pares cadastrados'):
        comparison()


def inspect_pair(query, candidate, face_points, skull_points, key):
    key = digest([key, query['person'], candidate['person']])[:24]
    for item in [query, candidate]:
        for row in item['views'].values():
            if row.get('hash') and row['hash'] != sha(row['path']):
                st.error('Imagem alterada desde o ranking; execute outra comparação.')
                return
    a, b = st.columns(2)
    for col, item, points, primary in [(a, query, face_points, 'FFN'), (b, candidate, skull_points, 'CF')]:
        view = col.selectbox('Vista · '+item['person'], list(item['views']), key=key+item['person'])
        try:
            im = image(item['views'][view]['path'])
            ps = points.get(view, [])
            col.image(letterbox(im)[0], channels='BGR', caption='Original · '+view, width='stretch')
            col.image(letterbox(overlay(im, ps, regions_for(ps, im.shape[1], im.shape[0])))[0], channels='BGR', caption='Pontos e regiões · '+view, width='stretch')
            col.caption('Pontos previstos: ' + ', '.join(str(p['numero']) for p in ps if p.get('x') is not None))
        except ValueError as exc:
            col.error(str(exc))
    f = coordinates(face_points.get('FFN', []))
    s = coordinates(skull_points.get('CF', []))
    common = np.isfinite(f).all(1) & np.isfinite(s).all(1)
    st.caption(f'{int(common.sum())}/11 números de pontos disponíveis nas duas imagens; correspondência anatômica requer revisão.')
    from .matching import PAIRS, geometry
    try:
        fi = image(query['views']['FFN']['path'])
        si = image(candidate['views']['CF']['path'])
    except ValueError as exc:
        st.error(str(exc))
        return
    fv, fm = geometry(face_points.get('FFN', []), fi.shape[1], fi.shape[0])
    sv, sm = geometry(skull_points.get('CF', []), si.shape[1], si.shape[0])
    with st.expander('Relações numéricas entre os pontos'):
        measures = [dict(pontos=f'{a+1}–{b+1}', face_distancia_diagonal=float(fv[i]) if fm[i] else None,
                         cranio_distancia_diagonal=float(sv[i]) if sm[i] else None,
                         disponibilidade='AMBOS' if fm[i] and sm[i] else 'AUSENTE') for i, (a, b) in enumerate(PAIRS)]
        st.dataframe(pd.DataFrame(measures), hide_index=True)
        angles = []
        for a, b, c in [(4, 0, 5), (2, 1, 3), (7, 6, 8)]:
            record = dict(pontos=f'{a+1}–{b+1}–{c+1}')
            for modality, points in [('face', f), ('cranio', s)]:
                u, v = points[a]-points[b], points[c]-points[b]
                den = np.linalg.norm(u)*np.linalg.norm(v)
                record[modality+'_angulo_graus'] = float(np.degrees(np.arccos(np.clip(u@v/den, -1, 1)))) if np.isfinite(den) and den > 0 else None
            angles.append(record)
        st.dataframe(pd.DataFrame(angles), hide_index=True)
    if st.checkbox('Revisei as correspondências para o alinhamento', key='align_confirm_'+key):
        numbers = st.multiselect('Pares de pontos revisados (mesmo número)', [i+1 for i in np.where(common)[0]], key='align_points_'+key)
        if len(numbers) >= 3:
            idx = np.array(numbers)-1
            if np.linalg.matrix_rank(f[idx]-f[idx].mean(0)) < 2 or np.linalg.matrix_rank(s[idx]-s[idx].mean(0)) < 2:
                st.warning('Pontos colineares: escolha correspondências que definam duas dimensões.')
            else:
                # Similarity fit in original pixels; no anatomy or depth is inferred.
                src, dst = f[idx], s[idx]
                X = np.zeros((2*len(idx), 4))
                X[::2] = np.c_[src[:, 0], -src[:, 1], np.ones(len(idx)), np.zeros(len(idx))]
                X[1::2] = np.c_[src[:, 1], src[:, 0], np.zeros(len(idx)), np.ones(len(idx))]
                aa, bb, tx, ty = np.linalg.lstsq(X, dst.reshape(-1), rcond=None)[0]
                matrix = np.array([[aa, -bb, tx], [bb, aa, ty]])
                mapped = np.c_[src, np.ones(len(idx))]@matrix.T
                st.write({'resíduos_px': np.linalg.norm(mapped-dst, axis=1).tolist(), 'transformação': matrix.tolist()})
                original_face = image(query['views']['FFN']['path'])
                original_skull = image(candidate['views']['CF']['path'])
                moved = cv2.warpAffine(original_face, matrix, (original_skull.shape[1], original_skull.shape[0]))
                alpha = st.slider('Opacidade da face alinhada', 0., 1., .5, key='aligned_alpha_'+key)
                st.image(letterbox(cv2.addWeighted(moved, alpha, original_skull, 1-alpha, 0))[0], channels='BGR', width=600)
        else:
            st.info('Selecione pelo menos três correspondências revisadas.')
