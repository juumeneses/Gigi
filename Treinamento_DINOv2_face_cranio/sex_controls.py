"""Documented labels and independent cranial workflow; no facial inference."""
from collections import Counter
from .portable import Path

import pandas as pd
import streamlit as st

from .core import ARTIFACTS as DINO_ARTIFACTS, REPO, new_run, save, atomic, sha, load


def copy_curated():
    source = REPO/'Treinamento_faces/resultados/genero_faces/genero_faces_curado.csv'
    token = sha(source)
    for path in (DINO_ARTIFACTS/'curated_sex').glob('*/source.json'):
        if load(path)['sha256'] == token and sha(path.parent/source.name) == token:
            return path.parent
    folder = new_run('curated_sex')
    atomic(folder/source.name, source.read_bytes())
    frame = pd.read_csv(source)
    save(folder/'source.json', dict(source=str(source), sha256=token, rows=len(frame),
        origins=frame.origem_genero_final.value_counts().to_dict(),
        usage='comparacao_com_curadoria', documented=False))
    return folder


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
    from Laboratorio_visual_face_cranio.cranial_sex import references
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


def cranial_module():
    import importlib
    from Laboratorio_visual_face_cranio import cranial_sex
    # Streamlit can retain the old imported module after a source update.
    if not hasattr(cranial_sex, 'import_curated_reference'):
        cranial_sex = importlib.reload(cranial_sex)
    return cranial_sex


def cranial_controls(launch, busy=False):
    from Laboratorio_visual_face_cranio.core import ARTIFACTS
    from Laboratorio_visual_face_cranio.data import inventory
    from Laboratorio_visual_face_cranio.references import paired
    module = cranial_module()
    step = st.segmented_control('Etapa do modelo craniano',
        ['1 · Referências', '2 · Treinar', '3 · Aplicar e consultar'],
        default='1 · Referências', key='cranial_step')
    st.caption('Modelo independente: silhuetas dos crânios CF/CE/CD e regressão logística.')
    if step == '1 · Referências':
        source = st.radio('Origem dos rótulos', ['Curadoria existente', 'Importar CSV com fonte'], horizontal=True)
        if source == 'Curadoria existente':
            st.write('Use o arquivo já disponível no projeto. Não é necessário enviar outro arquivo.')
            manual = st.checkbox('Usar somente os casos revisados manualmente', value=True)
            author = st.text_input('Responsável pela curadoria experimental', key='curated_author')
            st.caption('A avaliação medirá concordância com essa curadoria. Valores automáticos mantêm sua origem identificada.')
            if st.button('Preparar referência curada', type='primary', disabled=busy):
                try:
                    folder = copy_curated()
                    ref = module.import_curated_reference(folder/'genero_faces_curado.csv', author, manual)
                    st.success('Referência preparada. Abra a etapa 2 · Treinar.')
                    st.caption(ref.name)
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        else:
            upload = st.file_uploader('Referências de sexo (CSV)', type=['csv'])
            st.caption('Colunas: id_pessoa, sexo, fonte, tipo_referencia. Tipo: DOCUMENTAL ou ESPECIALISTA.')
            author = st.text_input('Responsável pela importação')
            confirmed = st.checkbox('Conferi a origem documental ou especializada dos valores deste arquivo')
            if st.button('Importar referências', type='primary', disabled=upload is None or busy):
                try:
                    module.import_reference(pd.read_csv(upload, dtype=str), author, confirmed)
                    st.success('Referência importada. Abra a etapa 2 · Treinar.')
                except (ValueError, pd.errors.ParserError) as exc:
                    st.error(str(exc))
        return

    if step == '3 · Aplicar e consultar':
        models = [p.parent for p in sorted((ARTIFACTS/'cranial_sex').glob('*/config.json'), reverse=True)
                  if (p.parent/'model.joblib').exists() and (p.parent/'evaluation.json').exists()]
        if models:
            run = st.selectbox('Modelo craniano / execução', models, format_func=lambda p:p.name)
            config = load(run/'config.json')
            st.caption('Referência: '+config.get('reference_kind', 'DOCUMENTAL'))
            validation = load(run/'evaluation.json')['validation']
            a, b = st.columns(2)
            a.metric('Concordância na validação', f"{validation['accuracy']:.1%}")
            b.metric('Pessoas na validação', validation['n'])
            rows, ready = cranial_rows(inventory())
            if st.button('Aplicar modelo craniano ao banco', type='primary', disabled=busy or not ready):
                launch('apply_cranial_sex', dict(run=str(run), items=paired(rows)))
            with st.expander('Avaliação final e detalhes do modelo'):
                st.json(load(run/'evaluation.json'), expanded=False)
                if st.button('Avaliar sexo craniano no teste congelado', disabled=busy or not ready):
                    launch('apply_cranial_sex', dict(run=str(run), items=[p for p in paired(rows) if p['split'] == 'teste'], evaluate_test=True))
        else:
            st.info('Nenhum modelo craniano concluído. Prepare a referência e treine nas etapas 1 e 2.')
        paths = sorted((ARTIFACTS/'cranial_sex_predictions').glob('*/predictions.json'), reverse=True)
        if paths:
            path = st.selectbox('Resultados cranianos novos', paths, format_func=lambda p:p.parent.name)
            frame = pd.DataFrame(load(path))
            status = st.selectbox('Exibir', ['Todos', 'OK', 'ERRO'])
            if status != 'Todos' and 'status' in frame:
                frame = frame[frame.status.eq(status)]
            columns = [c for c in ('person', 'split', 'predicted', 'status', 'reference_kind', 'error') if c in frame]
            st.dataframe(frame[columns].rename(columns={'person':'Pessoa', 'split':'Grupo', 'predicted':'Estimativa',
                'status':'Situação', 'reference_kind':'Origem da referência', 'error':'Problema'}), hide_index=True, width='stretch')
            with st.expander('Probabilidades e procedência completas'):
                st.dataframe(frame, hide_index=True)
            st.download_button('Baixar estimativas exibidas', frame.to_csv(index=False).encode('utf-8-sig'), 'estimativas_cranianas.csv')
        return

    paths = sorted((ARTIFACTS/'sex_references').glob('*/reference.csv'), reverse=True)
    if not paths:
        st.info('Prepare uma referência na etapa 1 antes de treinar.')
        return
    ref = st.selectbox('Referências para sexo craniano', [p.parent for p in paths],
        format_func=lambda p:p.name+' · '+load(p/'config.json').get('reference_kind', 'DOCUMENTAL'))
    labels = module.references(ref)
    kind = load(ref/'config.json').get('reference_kind', 'DOCUMENTAL')
    if kind == 'CURADORIA_EXPLORATORIA':
        st.info('Referência experimental: os resultados medem concordância com a curadoria.')
    rows, ready = cranial_rows(inventory())
    items = [p for p in paired(rows) if p['person'] in labels.index and all(v in p['views'] for v in ('CF','CE','CD'))] if ready else []
    coverage = pd.DataFrame([dict(Grupo=split,
        Feminino=sum(p['split'] == split and labels.loc[p['person'], 'sexo'] == 'FEMININO' for p in items),
        Masculino=sum(p['split'] == split and labels.loc[p['person'], 'sexo'] == 'MASCULINO' for p in items))
        for split in ('treino', 'validacao', 'teste')])
    st.dataframe(coverage, hide_index=True, width='stretch')
    sufficient = bool((coverage.iloc[:2][['Feminino','Masculino']] > 0).all().all())
    if not sufficient:
        st.warning('São necessárias referências das duas classes no treino e na validação, com CF/CE/CD disponíveis.')
    with st.expander('Ajustes avançados e referências'):
        st.dataframe(labels.reset_index(), hide_index=True)
        C = st.number_input('Regularização (C)', .001, 100., 1.)
        a, b = st.columns(2)
        low = a.number_input('Menor limiar craniano', .01, .99, .3)
        high = b.number_input('Maior limiar craniano', .01, .99, .7)
        seed = st.number_input('Seed do modelo craniano', 0, 100000, 42)
    valid_thresholds = low <= high
    if not valid_thresholds:
        st.error('O menor limiar deve ser menor ou igual ao maior limiar.')
    if st.button('Iniciar novo treino craniano · CF/CE/CD', type='primary', disabled=busy or not ready or not sufficient or not valid_thresholds):
        launch('train_cranial_sex', dict(rows=rows, reference=str(ref), C=C, threshold_low=low, threshold_high=high, seed=seed))


def cranial_rows(rows):
    """Require an explicit choice before dropping ambiguous people."""
    counts = Counter((r['person'], r['view']) for r in rows)
    ambiguous = {key for key, count in counts.items() if count > 1 and key[1] in ('FFN','CF','CE','CD')}
    if not ambiguous:
        return rows, True
    st.warning('Há pessoas com vistas duplicadas. Escolha como tratar esses casos antes de executar.')
    with st.expander('Conferir duplicações'):
        st.dataframe(pd.DataFrame(sorted(ambiguous), columns=['Pessoa', 'Vista']), hide_index=True)
    excluded = st.checkbox('Excluir desta execução as pessoas com vistas duplicadas', value=False)
    if not excluded:
        return rows, False
    people = {person for person, view in ambiguous}
    st.caption(f'{len(people)} pessoas excluídas somente desta execução. Os originais são preservados.')
    return [r for r in rows if r['person'] not in people], True
