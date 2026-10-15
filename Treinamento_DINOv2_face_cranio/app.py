from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Treinamento_DINOv2_face_cranio.portable import Path
from Treinamento_DINOv2_face_cranio import file_lock as fcntl
from Treinamento_DINOv2_face_cranio.core import (ARTIFACTS, ROOT, REPO, EXPERIMENTS, LABELS,
    atomic, digest, letterbox, load, new_run, read_image, save, sha, visual_overlay)
from Treinamento_DINOv2_face_cranio.backbone import MODEL_DIR, MODELS, model_dir
from Treinamento_DINOv2_face_cranio.data import cohort
from Treinamento_DINOv2_face_cranio.engine import paired_comparison, sample_gallery
from Treinamento_DINOv2_face_cranio.log_view import log_html
from Treinamento_DINOv2_face_cranio.visuals import overlay as sensitivity_overlay
from Treinamento_DINOv2_face_cranio.reporting import metadata, comparison_rows
from Treinamento_DINOv2_face_cranio.ui_help import HELP, table, metrics_help
from Treinamento_DINOv2_face_cranio.matching_summary import query_summary, history_label
from Treinamento_DINOv2_face_cranio.core import uses_ce

st.set_page_config(page_title='DINOv2 | Face e cranio', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stMainBlockContainer{padding-top:1.4rem;max-width:1320px}
 .stApp{background:#f6f8fb;color:#172b40}
 [data-testid="stHeader"]{background:transparent}
 [data-testid="stAppDeployButton"]{display:none}
 [data-testid="stBaseButton-primary"]{background:#17675e;border-color:#17675e}
 [data-testid="stRadio"] label:has(input[name="main_area"]){padding:6px 10px;border-radius:9px}
 [data-testid="stMetric"]{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:16px}
 [data-testid="stExpander"]{background:white;border-radius:12px}
 [data-testid="stRadio"]>div[role="radiogroup"]{background:white;padding:10px 16px;border:1px solid #e2e8f0;border-radius:14px;gap:1.3rem}
 .home-intro{padding:26px 30px;background:#173c47;color:white;border-radius:20px;margin:16px 0 24px}
 .home-intro h2{color:white;font-size:1.8rem;margin:0 0 8px}
 .home-intro p{color:#d9e9e8;max-width:650px;margin:0;line-height:1.6}
h1{font-size:1.6rem!important;letter-spacing:0!important}
h2,h3{letter-spacing:0!important}
[data-testid="stImage"] img{border-radius:4px;object-fit:contain;max-height:440px}
@media(max-width:700px){
.stMainBlockContainer{padding:1rem .75rem}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap}
[data-testid="stColumn"]{min-width: min(260px,100%)!important;flex:1 1 260px!important}
[data-testid="stImage"] img{max-height:340px}
button,p,label{overflow-wrap:anywhere}
}
</style>''', unsafe_allow_html=True)


def worker_busy():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def launch(action, args):
    if worker_busy():
        st.warning('Ja existe uma execucao em andamento. Acompanhe o progresso; nenhum novo treinamento foi iniciado.')
        return
    job = new_run('jobs')
    save(job/'request.json', dict(action=action, args=args))
    save(job/'status.json', dict(state='INICIANDO'))
    with (job/'stdout.log').open('w') as out:
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'cli.py'), '--request', str(job/'request.json')],
             cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    st.session_state['job'] = str(job)
    st.session_state['process'] = proc
    st.rerun()


@st.fragment(run_every=2)
def progress():
    jobs = sorted((ARTIFACTS/'jobs').glob('*/request.json'), reverse=True)
    if not jobs:
        return
    paths = [str(p.parent) for p in jobs]
    running = [p for p in paths if (Path(p)/'status.json').exists() and load(Path(p)/'status.json').get('state') == 'RODANDO']
    follow = st.checkbox('Acompanhar execucao atual automaticamente', value=True, key='follow_job')
    active = running[0] if running and worker_busy() else st.session_state.get('job')
    if follow and not running:
        meaningful = [p for p in paths if (Path(p)/'status.json').exists() and
                      load(Path(p)/'status.json').get('state') != 'OCUPADO' and
                      'Resource temporarily unavailable' not in load(Path(p)/'status.json').get('error', '')]
        active = meaningful[0] if meaningful else paths[0]
    path = st.selectbox('Execucao', paths, index=paths.index(active) if active in paths else 0,
                       format_func=lambda s: Path(s).name, key='job_selector_'+digest(active)[:12], disabled=follow)
    job = Path(path)
    status = load(job/'status.json') if (job/'status.json').exists() else {'state': 'INICIANDO'}
    terminal = status['state'] in ('CONCLUIDO', 'ERRO', 'INTERROMPIDO', 'OCUPADO')
    completion = str(job)+':'+status['state']
    if follow and terminal and st.session_state.get('displayed_completion') != completion:
        # A fragment refresh does not update results outside the fragment.
        st.session_state['displayed_completion'] = completion
        st.rerun(scope='app')
    proc = st.session_state.get('process')
    if proc and proc.poll() is not None and status['state'] in ('INICIANDO', 'RODANDO'):
        # The process may be from another selected job; compare the stored PID first.
        meta = load(job/'process.json') if (job/'process.json').exists() else {}
        if meta.get('pid') == proc.pid:
            status = dict(state='ERRO', error='Processo encerrado; consulte o log')
            save(job/'status.json', status)
    if status['state'] == 'RODANDO':
        st.info('Em andamento. Este acompanhamento atualiza automaticamente a cada 2 segundos enquanto a pagina esta ativa.')
        spec = load(job/'request.json')
        pipeline_dir = spec['args'].get('run_dir') if spec['action'] in ('improvement_pipeline','retrieval_pipeline') else None
        if pipeline_dir and (Path(pipeline_dir)/'status.json').exists():
            stage = load(Path(pipeline_dir)/'status.json')
            st.progress(min(stage.get('done',0)/max(stage.get('total',1),1),1.),text='Pipeline: '+str(stage.get('stage','preparando')))
    elif status['state'] == 'CONCLUIDO':
        st.success('Execucao concluida. Resultados e mapas disponiveis foram atualizados.')
    elif status.get('error', '').find('Resource temporarily unavailable') >= 0 or status['state'] == 'OCUPADO':
        st.warning('Esta tentativa duplicada nao iniciou: outra execucao estava em andamento. Selecione a anterior no historico; nao houve falha do modelo.')
    else:
        st.write(status)
    log = job/'stdout.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size-12000))
            content = f.read().decode(errors='replace')
            counts = re.findall(r'(Caracteristicas face|Caracteristicas skull|Epoca|Pipeline) (\d+)/(\d+)', content)
            if status['state'] == 'CONCLUIDO':
                st.progress(1., text='Execucao concluida')
            elif counts:
                phase, done, total = counts[-1]
                st.progress(min(int(done)/max(int(total), 1), 1.), text=f'{phase}: {done}/{total}')
            auto_scroll = st.toggle('Rolar logs automaticamente', value=True, key='log_auto_scroll')
            snapshot_key = 'log_snapshot_'+digest(str(job))[:12]
            if auto_scroll:
                st.session_state.pop(snapshot_key, None)
                shown = content
            else:
                if snapshot_key not in st.session_state:
                    st.session_state[snapshot_key] = log.read_text(errors='replace')
                shown = st.session_state[snapshot_key]
                st.caption('Leitura pausada. O treinamento continua; reative a rolagem para voltar as linhas mais recentes.')
            st.iframe(log_html(shown, auto_scroll, digest(str(job))), height=244)
            st.download_button('Baixar log completo', log.read_bytes(), job.name+'.log', 'text/plain', key='download_job_log')
    a, b, c = st.columns(3)
    if a.button('Interromper', icon=':material/stop:', disabled=status['state'] not in ('RODANDO', 'INICIANDO')):
        (job/'stop.flag').touch()
    if b.button('Retomar', icon=':material/play_arrow:', disabled=worker_busy() or status['state'] not in ('INTERROMPIDO', 'ERRO')):
        spec = load(job/'request.json')
        args = spec['args'].copy()
        output = status.get('output')
        if spec['action'] == 'suite' and output:
            args['resume_dir'] = output
        elif spec['action'] in ('train_components', 'train_combinations', 'improvement_pipeline', 'retrieval_pipeline') and output:
            args['run_dir'] = output
        elif spec['action'] in ('train', 'extract') and output:
            args.update(run_dir=output, resume=True)
        launch(spec['action'], args)
    if c.button('Atualizar resultados', icon=':material/refresh:'):
        st.rerun()


def all_runs():
    paths = (list((ARTIFACTS/'runs').glob('*/config.json')) + list((ARTIFACTS/'suites').glob('*/*/config.json'))
             + list((ARTIFACTS/'fusion_training').glob('*/*/config.json'))
             + list((ARTIFACTS/'retrieval_pipelines').glob('*/trials/*/seed*/*/config.json')))
    return sorted([p.parent for p in paths if (p.parent/'evaluation.json').exists()], reverse=True)


def run_label(run):
    c = load(run/'config.json'); e = load(run/'evaluation.json')
    run_id = c.get('backbone', 'dinov2_vits14')+' | '+(run.parent.name+'/'+run.name if run.parent.parent.name == 'suites' else run.name)
    return f'{c["experiment"]} | {metadata(run)["Data (Sao Paulo)"]} | {c.get("loss", "direto")} | seed {c["seed"]} | {run_id} | Recall@1 {e["full"]["recall_1"]:.1%} | MRR {e["full"]["mrr"]:.3f}' + (' | teste tecnico' if c['smoke'] else '') + (' | LEGADO com CE' if uses_ce(run) else ' | sem CE')


def picture(row, caption, size=420):
    if not row:
        st.info(caption+' ausente')
        return
    try:
        im = read_image(row['path'])
        st.image(letterbox(im, size)[0], caption=caption, width='stretch')
    except Exception as exc:
        st.error(str(exc))


def choose_run(key):
    runs = all_runs()
    if key == 'matching_run':
        runs = [p for p in runs if not uses_ce(p)]
    if not runs:
        st.info('Nenhuma execucao avaliada.')
        return None
    real = [p for p in runs if not load(p/'config.json')['smoke']]
    values = real or runs
    latest = values[0]
    group = load(latest/'config.json')['cohort']
    comparable = [p for p in values if load(p/'config.json')['cohort'] == group]
    best = max(comparable, key=lambda p: (load(p/'evaluation.json')['full']['recall_1'], load(p/'evaluation.json')['full']['mrr']))
    return st.selectbox('Modelo / execucao', values, index=values.index(best), key=key,
                        format_func=lambda p: run_label(p)+(' (melhor Recall@1 nesta coorte)' if p == best else ''))


def bank_page(manifest_path):
    from Treinamento_DINOv2_face_cranio.quality import controls as quality_controls
    quality_controls(launch, worker_busy)
    if st.button('Atualizar inventario', icon=':material/refresh:'):
        launch('inventory', dict(resolutions=load(manifest_path).get('resolutions', {}) if manifest_path else {}))
    if not manifest_path:
        return
    source = load(manifest_path)
    df = pd.DataFrame([r for r in source['rows'] if r['view'] != 'CE'])
    if df.empty:
        st.info('Nenhuma imagem com vista reconhecida no inventario.'); return
    a, b, c = st.columns(3)
    year = a.selectbox('Ano', ['Todos']+sorted(df.year.unique()))
    split = b.selectbox('Split', ['Todos']+sorted(df.split.unique()))
    problem = c.checkbox('Somente problemas')
    selected = df[(df.year == year) if year != 'Todos' else np.ones(len(df), bool)]
    if split != 'Todos': selected = selected[selected.split == split]
    if problem:
        conflict_ids = {pid for values in source['identity_hash_collisions'].values() for pid in values}
        conflict_ids.update(p['person'] for p in source['people'] if p['conflicts'])
        selected = selected[(selected.status != 'OK') | (selected.split == 'pendente') | selected.person.isin(conflict_ids)]
    st.caption(f'{len(selected)} imagens | {selected.person.nunique()} pessoas no filtro')
    if len(selected):
        key = st.selectbox('Imagem', selected.index.tolist(), format_func=lambda i: f'{df.loc[i, "person"]} · {df.loc[i, "view"]}')
        row = df.loc[key].to_dict()
        normalized = st.checkbox('Ver entrada normalizada', help='RGB, orientacao EXIF e redimensionamento proporcional com preenchimento. Nao alinha anatomia.')
        if normalized:
            try: st.image(letterbox(read_image(row['path']),224)[0], caption='Entrada 224 x 224', width=420)
            except (OSError,ValueError) as exc: st.warning(str(exc))
        else: picture(row, 'Original')
    with st.expander('Inventario e download'):
        table(selected[['person', 'year', 'view', 'split', 'status', 'error']].rename(columns={'person':'Pessoa','year':'Ano','view':'Vista','split':'Divisao','status':'Estado','error':'Problema'}))
        st.download_button('Inventario CSV', (manifest_path.parent/'inventory.csv').read_bytes(), 'inventario.csv')
    conflicts = [(p, v, paths) for p in source['people'] for v, paths in p['conflicts'].items()]
    if source['identity_hash_collisions']:
        st.warning('Identidades com arquivos compartilhados ficam fora do treino e da avaliacao ate revisao do vinculo.')
        with st.expander('Conflitos de identidade'): st.json(source['identity_hash_collisions'])
    if conflicts:
        with st.expander(f'Resolver vistas duplicadas ({len(conflicts)})'):
            index = st.selectbox('Conflito', range(len(conflicts)), format_func=lambda i: conflicts[i][0]['person']+' '+conflicts[i][1])
            p, view, paths = conflicts[index]
            path = st.selectbox('Arquivo a usar', paths)
            picture(dict(path=path), view)
            if st.button('Registrar escolha e gerar inventario'):
                resolutions = source.get('resolutions', {}).copy()
                resolutions[p['person']+'/'+view] = path
                launch('inventory', dict(resolutions=resolutions))


def train_page(manifest_path):
    with st.expander('Acompanhamento e logs', expanded=worker_busy()): progress()
    from Treinamento_DINOv2_face_cranio.retrieval_controls import controls as retrieval_controls
    retrieval_controls(manifest_path,launch,worker_busy)
    from Treinamento_DINOv2_face_cranio.pipeline_controls import controls as pipeline_controls
    pipeline_controls(manifest_path,launch,worker_busy)
    from Treinamento_DINOv2_face_cranio.weight_controls import search_controls
    search_controls(launch, worker_busy)
    from Treinamento_DINOv2_face_cranio.profile_ui import blend_controls
    blend_controls(launch, worker_busy)
    backbone = st.selectbox('Extrator visual', list(MODELS), format_func=lambda k: MODELS[k][0])
    a, b = st.columns(2)
    a.metric(MODELS[backbone][0], 'Disponivel' if (model_dir(backbone)/'model.json').exists() else 'Pendente')
    if b.button('Preparar modelo', icon=':material/download:', help='Baixa e verifica os pesos do extrator selecionado. Execute uma vez por extrator; nao treina o matching.'):
        launch('prepare', dict(backbone=backbone))
    if not manifest_path:
        st.info('Inventario pendente.')
        return
    options = {'frontal': 'Neutra: face frontal / cranio frontal',
               'sorriso': 'Sorrindo: face sorrindo / cranio frontal',
               'lateral': 'Lateral direita: face FL / cranio CD',
               'todos': 'Treinar os tres modelos',
               'sete': 'Treinar os tres + avaliar as 7 combinacoes'}
    selection = st.selectbox('O que treinar', list(options), format_func=options.get)
    bundles = sorted((ARTIFACTS/'fusion_training').glob('*/request.json'), reverse=True)
    st.caption('Mesmas pessoas com vistas completas para comparar os modelos. O teste final permanece reservado.')
    st.caption('Prioridade dos novos treinos: Recall@1 na validacao; MRR desempata a escolha do checkpoint.')
    try:
        c = cohort(manifest_path, 'E6D', 'matched')
    except ValueError as exc:
        st.error(str(exc)); return
    cols = st.columns(3)
    for col, split in zip(cols, ('treino', 'validacao', 'teste')):
        col.metric(split.capitalize(), sum(p['split'] == split for p in c['people']))
    with st.expander('Exclusoes da coorte'):
        st.dataframe(c['excluded'], hide_index=True)
    a, b = st.columns(2)
    loss = a.selectbox('Loss', ['infonce', 'triplet'], help=HELP['Loss'])
    seed = b.number_input('Seed', 0, 999999, 42, help=HELP['Seed'])
    with st.expander('Parametros avancados'):
        a, b, d = st.columns(3)
        epochs = a.number_input('Epocas', 1, 1000, 100, help=HELP['Epocas limite'])
        lr = b.number_input('Learning rate', .000001, .1, .001, format='%.6f', help=HELP['Learning rate'])
        batch = d.number_input('Batch de identidades', 2, 256, 32, help=HELP['Batch'])
        a, b, d = st.columns(3)
        patience = a.number_input('Paciencia', 1, 100, 15, help=HELP['Paciencia'])
        wd = b.number_input('Weight decay', 0., .1, .0001, format='%.6f', help=HELP['Weight decay'])
        size = d.selectbox('Resolucao', [224, 336, 448], help=HELP['Resolucao'])
        a, b, d = st.columns(3)
        temperature = a.number_input('Temperatura InfoNCE', .001, 1., .07, format='%.3f')
        margin = b.number_input('Margem triplet', .01, 2., .2)
        device = d.selectbox('Dispositivo', ['auto', 'cpu', 'mps', 'cuda'])
    params = dict(manifest=str(manifest_path), seed=seed, backbone=backbone, selection_metric='recall_1',
                  epochs=epochs, lr=lr, batch=batch, patience=patience, weight_decay=wd,
                  size=size, loss=loss, temperature=temperature, margin=margin, device=device)
    a, b = st.columns(2)
    if a.button('Extrair caracteristicas', icon=':material/image_search:', help='Calcula e guarda vetores das imagens para reutilizar nos treinos; nao ajusta pesos.'):
        launch('extract', dict(manifest=str(manifest_path), size=size, device=device, backbone=backbone))
    if b.button('Iniciar treinamento selecionado', type='primary', icon=':material/play_arrow:', disabled=worker_busy() or not (model_dir(backbone)/'model.json').exists(), help='Sempre inicia um NOVO treinamento em uma nova pasta, com os parametros acima. Prepare o extrator antes. Nao retoma nem reutiliza checkpoints anteriores.'):
        args = params.copy()
        if selection == 'sete':
            launch('train_combinations', args)
        else:
            launch('train_components', dict(args, component=selection))
    with st.expander('Combinar modelos ja treinados'):
        compatible = [p.parent for p in bundles if load(p).get('view_policy') == 'without_CE_v1']
        target = st.selectbox('Sequencia apenas para combinar resultados', [None]+compatible,
                              format_func=lambda p: 'Selecione uma sequencia concluida' if p is None else p.name)
        st.caption('Sete opcoes: neutra; sorriso; lateral; neutra+sorriso; neutra+lateral; sorriso+lateral; todas. Cada modelo e treinado uma vez; as combinacoes usam pesos iguais entre os modelos selecionados.')
        if target:
            ready = all((target/name/'evaluation.json').exists() for name in ('frontal', 'sorriso', 'lateral'))
            st.write('Componentes concluidos: '+', '.join(name for name in ('frontal', 'sorriso', 'lateral') if (target/name/'evaluation.json').exists()))
            a, b, c = st.columns(3)
            wf = a.number_input('Peso frontal', 0., 10., .5, .1, key='fusion_frontal_50_30_20')
            ws = b.number_input('Peso sorriso', 0., 10., .3, .1, key='fusion_sorriso_50_30_20')
            wl = c.number_input('Peso lateral', 0., 10., .2, .1, key='fusion_lateral_50_30_20')
            st.caption('Proposta para avaliar: neutra 50%, sorrindo 30%, lateral 20%.')
            st.caption('Pesos relativos fixos, normalizados pela soma. Scores padronizados usando somente pares diferentes do treino. Ajustar pesos olhando a validacao e exploratorio; nao e ganho independente confirmado.')
            if st.button('Avaliar combinacao dos scores', disabled=not ready or worker_busy() or wf+ws+wl <= 0):
                launch('fuse', dict(bundle=str(target), weights=[wf, ws, wl]))
            if st.button('Avaliar as 7 combinacoes dos modelos ja treinados', disabled=not ready or worker_busy()):
                launch('all_combinations', dict(bundle=str(target)))
    with st.expander('Historico e avaliacao'):
        run = choose_run('train_run')
        if run:
            a, b, d = st.columns(3)
            if a.button('Avaliar validacao'): launch('evaluate', dict(run=str(run)))
            if b.button('Congelar configuracao'): launch('freeze', dict(run=str(run)))
            if d.button('Avaliar teste final', disabled=not (run/'frozen.json').exists()):
                launch('evaluate', dict(run=str(run), split='teste'))


def show_maps(path, key):
    explanation = load(Path(path)/'result.json')
    st.caption('Vermelho: ocultar reduziu o score. Azul: ocultar aumentou. Cor nao significa acerto ou erro.')
    a,b = st.columns(2)
    faces = [m for m in explanation['maps'] if m['modality']=='face']
    skulls = [m for m in explanation['maps'] if m['modality']=='skull']
    selected = []
    for col, rows, label in [(a,faces,'Face'),(b,skulls,'Cranio')]:
        with col:
            if not rows:
                st.info('Sem mapa desta modalidade.'); continue
            index = st.selectbox('Vista '+label, list(range(len(rows))), format_func=lambda i, rows=rows:rows[i]['view'], key=key+'_'+label)
            row = rows[index]
            values = np.asarray(row['values'])
            im = read_image(row['path'])
            st.image(letterbox(sensitivity_overlay(im,values),420)[0],caption=row['view'],width='stretch')
            st.caption(f'{values.shape[0]} x {values.shape[1]} blocos | {int((values>0).sum())} vermelhos | {int((values<0).sum())} azuis')
            selected.append(row)
    with st.expander('Significado e valores dos blocos'):
        st.write('Delta = score original menos score apos ocultar o bloco. Grade de testes posteriores ao treino, nao pontos anatomicos. Intensidade relativa a cada imagem; os efeitos nao somam 100%. Um mapa todo vermelho e possivel.')
        for row in selected:
            st.write(row['view']); table(pd.DataFrame(row['values']))
        table(explanation['removed_views'])


def visual_examples(run, key):
    st.subheader('Exemplos visuais: onde o score mudou')
    paths = [run/'visual_examples.json']
    cfg = load(run/'config.json')
    if 'bundle' in cfg:
        paths.append(Path(cfg['bundle'])/'visual_examples.json')
    records = next((load(p) for p in paths if p.exists()), [])
    available = [r for r in records if r.get('path') and (Path(r['path'])/'result.json').exists()]
    if available:
        selected = st.selectbox('Exemplo com mapa', list(range(len(available))), key=key+'_example',
            format_func=lambda i: Path(available[i]['model']).name+' | '+available[i]['person']+
            (' | correto em primeiro' if available[i]['correct'] else ' | primeiro candidato incorreto'))
        item = available[selected]
        inputs = load(Path(item['path'])/'inputs.json')
        st.write('Face: '+inputs['query']['person']+' | Cranio comparado: '+inputs['candidate']['person'])
        st.caption('Mapa do modelo individual deste exemplo, nao de uma combinacao de scores. Amostra ilustrativa de validacao, nao todos os casos.')
        show_maps(item['path'], key)
    else:
        st.info('Esta execucao ainda nao tem mapas visuais salvos.')
    if any('error' in r for r in records):
        with st.expander('Falhas na geracao dos mapas'): st.json([r for r in records if 'error' in r])
    if st.button('Gerar exemplos visuais desta execucao', key=key+'_generate', disabled=worker_busy()):
        launch('visual_report', dict(run=str(run)))


def results_page():
    runs = all_runs()
    if not runs:
        st.info('Sem resultados ainda.'); return
    run = choose_run('result_run')
    if not run: return
    result = load(run/'evaluation.json')
    metrics_help()
    with st.expander('Quando rodou e parametros do treino selecionado'):
        st.dataframe([metadata(run)], hide_index=True)
        st.caption('Data de criacao da execucao, em Sao Paulo. Retomadas mantem essa data. Temperatura aplica-se a InfoNCE; margem, a triplet.')
        st.json(load(run/'config.json'), expanded=False)
    if load(run/'config.json').get('smoke'):
        st.warning('Teste tecnico com poucos casos. Estes numeros nao medem o desempenho no banco completo.')
    st.caption('Resultados de validacao: pessoas separadas do treino. Nao representam o teste final.')
    general, simple, technical, weight_results, pipeline_results, retrieval_results = st.tabs(['Visao geral', 'Resumo simples', 'Detalhes tecnicos', 'Pesos das vistas', 'Pipeline de melhorias', 'Pipeline Top-10'])
    with general:
        general_results(runs)
    with simple:
        simple_results(run, result)
    with technical:
        technical_results(runs, run, result)
    with weight_results:
        from Treinamento_DINOv2_face_cranio.weight_controls import search_results
        search_results()
    with pipeline_results:
        from Treinamento_DINOv2_face_cranio.pipeline_controls import results as show_pipeline
        show_pipeline()
    with retrieval_results:
        from Treinamento_DINOv2_face_cranio.retrieval_controls import results as show_retrieval
        show_retrieval()
    with st.expander('Acompanhamento e logs', expanded=worker_busy()): progress()


def general_results(runs):
    a, b = st.columns(2)
    include_smoke = a.checkbox('Incluir testes tecnicos', value=False, key='general_smoke')
    gallery = b.radio('Galeria dos graficos', ['Completa', 'Gallery reduzida'], horizontal=True, key='general_gallery')
    rows = comparison_rows(runs, [p.parent for p in (ARTIFACTS/'fusions').glob('*/results.json')],
                          gallery='full' if gallery == 'Completa' else 'gallery', include_smoke=include_smoke)
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info('Sem resultados neste filtro.'); return
    frame['Seed'] = frame['Seed'].astype(str)
    a, b = st.columns(2)
    losses = a.multiselect('Filtrar loss', sorted(frame['Loss'].dropna().astype(str).unique()))
    protocol = b.selectbox('Protocolo comparavel', list(frame['Protocolo'].unique()),
                           help=HELP['Protocolo'])
    if losses: frame = frame[frame['Loss'].astype(str).isin(losses)]
    frame = frame[frame['Protocolo'] == protocol]
    mode = st.segmented_control('Visualizar resultados', ['Tabela', 'Graficos', 'Combinacoes'], default='Tabela')
    st.caption('Validacao, nao teste final. Datas mais recentes primeiro. Ganhos descritivos, sem garantia estatistica.')
    if mode == 'Combinacoes':
        fusion_results(); return
    if frame.empty:
        st.info('Sem execucoes neste protocolo e loss.'); return
    if mode == 'Graficos':
        metric = st.selectbox('Metrica do grafico', ['Recall@1 (%)', 'MRR', 'Recall@3 (%)', 'Recall@5 (%)',
                                                   'Recall@10 (%)', 'Ganho MRR vs melhor anterior'])
        plotted = frame.copy()
        plotted['Treino'] = plotted['Extrator']+' | '+plotted['Modelo']+' | '+plotted['Loss'].astype(str)+' | seed '+plotted['Seed'].astype(str)+' | '+plotted['Data (Sao Paulo)']
        st.bar_chart(plotted.set_index('Treino')[[metric]], height=340)
        st.caption(HELP.get(metric, 'Recall: de 0% a 100%; maior e melhor.'))
    else:
        details = st.checkbox('Mostrar parametros e todas as colunas', value=False)
        columns = ['Data (Sao Paulo)', 'Extrator', 'Modelo', 'Loss', 'Seed', 'Recall@1 (%)', 'Recall@3 (%)',
                   'Recall@5 (%)', 'Recall@10 (%)', 'MRR', 'Evolucao Top-1', 'Ganho Recall@1 vs melhor anterior (p.p.)']
        table(frame if details else frame[columns])
    st.download_button('Baixar historico com parametros', frame.to_csv(index=False).encode(), 'historico_parametros.csv',
                       icon=':material/download:', help='CSV completo das execucoes filtradas, incluindo parametros e referencias.')
    with st.expander('Referencia usada na comparacao'):
        table(frame[['Execucao','Consultas','Candidatos','Melhor atual (Recall@1)','Diferenca Recall@1 vs lider (p.p.)']])
        st.caption('Melhor significa maior Recall@1 no mesmo protocolo, com MRR como desempate. Filtros de loss nao redefinem o lider. As colunas de MRR permanecem nos detalhes.')


def fusion_results():
    outputs = sorted((ARTIFACTS/'fusions').glob('*/results.json'), reverse=True)
    if not outputs:
        st.info('Nenhuma combinacao avaliada.'); return
    output = st.selectbox('Combinacao avaliada', outputs, format_func=lambda p:
        f'{metadata(p.parent)["Data (Sao Paulo)"]} | {metadata(p.parent)["Loss"]} | seed {metadata(p.parent)["Seed"]} | {p.parent.name}')
    results = load(output)
    cfg = load(output.parent/'config.json')
    active = [n for n,w in zip(('Neutra','Sorrindo','Lateral'),cfg['weights']) if w]
    st.caption('Combinacao: '+' + '.join(active))
    mode = st.segmented_control('Detalhe da combinacao', ['Indicadores','Mapas','Parametros'], default='Indicadores')
    if mode == 'Mapas':
        visual_examples(output.parent, 'fusion_visuals'); return
    if mode == 'Parametros':
        table([metadata(output.parent)])
        with st.expander('Configuracao completa'): st.json(cfg)
        return
    rows = []
    base = results.get('frontal', results.get('source_a', results['combinado']))['full']
    for name, result in results.items():
        e = result['full']
        rows.append(dict(Modelo=name, **{f'Recall@{k} (%)':100*e[f'recall_{k}'] for k in (1,3,5,10)},
                         MRR=e['mrr'], **{'Ganho MRR vs referencia':e['mrr']-base['mrr']}))
    table(pd.DataFrame(rows))
    st.bar_chart(pd.DataFrame(rows).set_index('Modelo')[['Recall@1 (%)']], height=260)
    with st.expander('Resultados por pessoa'): table(results['combinado']['rows'])
    st.download_button('Baixar comparacao dos componentes', pd.DataFrame(rows).to_csv(index=False).encode(), 'componentes.csv')


def simple_results(run, result):
    full = result['full']
    n = full['n']
    a, b, c = st.columns(3)
    a.metric('Correto em primeiro', f'{round(full["recall_1"]*n)} / {n}')
    b.metric('Recall@5', f'{full["recall_5"]:.1%}', help='Correto entre os cinco primeiros; de 0% a 100%.')
    c.metric('MRR', f'{full["mrr"]:.3f}', help=HELP['MRR'])
    st.caption(f'{n} cranios por consulta | {full["failures"]} falhas de processamento')
    with st.expander('Todos os indicadores'):
        table([{'Metrica':f'Recall@{k}', 'Valor (%)':100*full[f'recall_{k}']} for k in (1,3,5,10)])
        st.write('Posicao mediana do correto:',full.get('median_rank'))
        st.caption(f'Gallery-{result["gallery_size"]}: Recall@1 {result["gallery"]["recall_1"]:.1%}. Galeria menor: nao comparar diretamente com a completa.')
    a, b = st.columns(2)
    category = a.selectbox('Casos', ['Todos','Acertou em primeiro','Nao acertou em primeiro','Sem ranking'], key='simple_cases')
    cases = [r for r in result['rows'] if category == 'Todos' or
             (category == 'Acertou em primeiro' and r['rank'] == 1) or
             (category == 'Nao acertou em primeiro' and r['rank'] is not None and r['rank'] != 1) or
             (category == 'Sem ranking' and r['rank'] is None)]
    if not cases:
        st.info('Nenhum caso neste filtro.'); return
    pid = b.selectbox('Pessoa avaliada', [r['person'] for r in cases], key='simple_person')
    row = next(r for r in cases if r['person'] == pid)
    st.caption(f'Posicao do cranio correto: {row["rank"]}' if row['rank'] else 'Consulta sem ranking valido')
    people = {p['person']:p for p in load(run/'cohort.json')['people']}
    views = EXPERIMENTS[load(run/'config.json')['experiment']]
    a,b,c = st.columns(3)
    with a: picture(people[pid]['views'].get(views[0][0]), 'Face consultada')
    with b: picture(people[pid]['views'].get(views[1][0]), 'Cranio correto')
    with c: picture(people.get(row['top_candidate'],{}).get('views',{}).get(views[1][0]), 'Primeiro do ranking')
    with st.expander('Exemplos com mapas de sensibilidade'):
        visual_examples(run, 'simple_visuals')


def technical_results(runs, run, result):
    st.caption('IC95% MRR: '+str(result['full']['mrr_ci95'])+' | Incerteza sobre a media, nao sobre a identidade de um par.')
    if (run/'history.json').exists():
        st.line_chart(pd.DataFrame(load(run/'history.json')).set_index('epoch')[['loss','val_mrr']], height=260)
    with st.expander('Rankings por pessoa e resultados por ano'):
        table(result['rows'])
        st.json(result['by_year'], expanded=False)
    a,b = st.columns(2)
    a.download_button('Relatorio HTML', (run/'report.html').read_bytes(), run.name+'.html')
    b.download_button('Rankings CSV', (run/'rankings.csv').read_bytes(), run.name+'.csv')
    with st.expander('Comparacao pareada com outra execucao'):
        others = [p for p in runs if p != run]
        if others:
            other = st.selectbox('Comparar com', others, format_func=run_label)
            try:
                comparison = paired_comparison(other,run)
                st.write(comparison['conclusion'])
                st.write('Diferenca MRR:',comparison['delta_mrr'],' | IC95%:',comparison['ci95'])
                table(comparison['rows'])
            except ValueError as exc: st.info(str(exc))
    evaluations = sorted((ARTIFACTS/'evaluations').glob('*/config.json'),reverse=True)
    if evaluations:
        with st.expander('Teste congelado e outras avaliacoes'):
            selected = st.selectbox('Avaliacao',evaluations,format_func=lambda p:p.parent.name)
            st.json(load(selected.parent/'evaluation.json')['full'])


def upload_row(file):
    token = digest(file.getvalue().hex())
    path = ARTIFACTS/'uploads'/(token+Path(file.name).suffix.lower())
    if not path.exists(): atomic(path, file.getvalue())
    return dict(path=str(path), hash=sha(path))


def fusion_matching():
    outputs = sorted((ARTIFACTS/'fusions').glob('*/results.json'), reverse=True)
    with st.expander('Matching por scores separados'):
        if not outputs:
            st.info('Treine os componentes e avalie uma combinacao na aba Treinamentos.'); return
        run = st.selectbox('Modelo combinado', [p.parent for p in outputs], format_func=lambda p: p.name)
        cols = st.columns(3)
        enabled = [col.checkbox(label, value=True, key='fusion_use_'+name) for col, label, name in
                   zip(cols, ('Neutra', 'Sorrindo', 'Lateral'), ('frontal', 'sorriso', 'lateral'))]
        weights = [int(v) for v in enabled]
        st.caption('Pesos iguais entre os modelos marcados. Modelos desmarcados nao entram no score nem sao executados.')
        if not any(enabled):
            st.warning('Selecione pelo menos um modelo.')
        people = [p for p in load(run/'cohort.json')['people'] if p['split'] == 'validacao']
        count = st.number_input('Cranios na consulta combinada', 2, len(people), min(10, len(people)))
        if st.button('Sortear consulta combinada'):
            previous = st.session_state.get('fusion_gallery', {})
            seed = secrets.randbelow(1000000)
            selected = sample_gallery(people, count, seed)
            while selected['query']['person'] == previous.get('query', {}).get('person'):
                seed = (seed+1) % 1000000
                selected = sample_gallery(people, count, seed)
            st.session_state['fusion_gallery'] = dict(run=str(run), **selected)
        sample = st.session_state.get('fusion_gallery')
        if sample and sample['run'] == str(run):
            picture(sample['query']['views']['FFN'], 'Face consultada')
            if st.button('Comparar com scores separados', disabled=worker_busy() or not any(enabled)):
                launch('compare_fusion', dict(sample, weights=weights))
            matches = [p for p in sorted((ARTIFACTS/'fusion_comparisons').glob('*/result.json'), reverse=True)
                       if Path(load(p)['model']) == Path(run) and load(p)['query'] == sample['query'] and load(p)['candidates'] == sample['candidates']
                       and load(p).get('weights', load(run/'config.json')['weights']) == weights]
            if matches:
                result = load(matches[0])
                st.dataframe(result['ranking'], hide_index=True)
                st.caption('Frontal, sorriso e lateral: cossenos individuais. Contribuicoes: scores padronizados e ponderados; sua soma e o score combinado. Nao sao porcentagens. Vista necessaria ausente implica abstencao.')
                person = st.selectbox('Candidato da combinacao', [r['person'] for r in result['ranking']])
                candidate = next(p for p in sample['candidates'] if p['person'] == person)
                cols = st.columns(3)
                for col, view in zip(cols, ('CF', 'CD')):
                    with col: picture(candidate['views'].get(view), person+' '+view)


def matching_viewer(result, run, config, weights, use_fusion, output):
    lookup = {r['person']: r for r in result['ranking']}
    person = st.selectbox('Cranio no ranking', list(lookup),
        format_func=lambda p: f'{lookup[p]["rank"] or "Sem score"} | {p}',
        key='rank_candidate_'+digest([str(output), weights])[:12])
    candidate = next(c for c in result['candidates'] if c['person'] == person)
    ranked = lookup[person]
    a, b = st.columns(2)
    fv = a.selectbox('Foto da pessoa', [v for v in ('FFN','FFS','FL') if v in result['query']['views']],
                    format_func=lambda v: {'FFN':'Frontal neutra','FFS':'Sorrindo','FL':'Lateral'}[v])
    order = ('CD','CF') if fv == 'FL' else ('CF','CD')
    sv = b.selectbox('Vista do candidato', [v for v in order if v in candidate['views']], key='candidate_view_'+str(fv))
    a, b = st.columns(2)
    show_heat = a.checkbox('Mostrar quadrados de sensibilidade', value=True, key='viewer_heat', help='Testes de oclusao apos o treino; nao landmarks anatomicos.')
    overlap = b.checkbox('Sobrepor face e cranio', value=False, key='viewer_overlap', help='Alinhamento manual das vistas exibidas; nao altera o score.')
    explanation_run = run
    if use_fusion:
        if config.get('rank_fusion_members'):
            st.caption('Ranking: mistura de modelos. O mapa abaixo mostra apenas um componente representativo da fonte B; nao explica o efeito total da mistura.')
        if config.get('ensemble_members'):
            st.caption('Ranking: media das seeds. O mapa abaixo detalha somente um modelo da primeira seed, nao o efeito total do ensemble.')
        active = [n for n,w in zip(('frontal','sorriso','lateral'),weights) if w]
        preferred = {'FFN':'frontal','FFS':'sorriso','FL':'lateral'}.get(fv)
        component = st.selectbox('Modelo do mapa', active, index=active.index(preferred) if preferred in active else 0,
                                  key='viewer_model_'+str(fv)+'_'+''.join(str(w) for w in weights))
        explanation_run = Path(config['children'][component]['run'])
    inspection = dict(run=str(explanation_run), query=result['query'], candidate=candidate)
    def matches_inspection(path):
        saved=load(path.parent/'inputs.json')
        return Path(saved.get('run','')) == explanation_run and all(saved.get(k)==inspection[k] for k in ('query','candidate'))
    paths = [p for p in sorted((ARTIFACTS/'explanations').glob('*/result.json'), reverse=True)
             if (p.parent/'inputs.json').exists() and
             matches_inspection(p)]
    maps = load(paths[0])['maps'] if paths else []
    if show_heat:
        st.caption('Vermelho: ocultar reduziu o score. Azul: ocultar aumentou. Sem cor: efeito pequeno.')
        with st.popover('Por que essas cores?', icon=':material/help:'):
            st.write('A grade divide a imagem em blocos. Cada bloco e ocultado separadamente e o score e recalculado. Vermelho nao significa erro; azul nao significa acerto.')
            st.write('Um mapa todo vermelho significa que todas as regioes testadas aumentavam o score quando presentes. As cores variam com o modelo, a vista e o candidato.')
            st.write('Intensidade relativa a cada imagem. Nao indica pontos anatomicos, probabilidades ou quanto uma regiao participou do treinamento.')
            for m in maps:
                if (m['modality'], m['view']) in [('face',fv), ('skull',sv)]:
                    values = np.array(m['values'])
                    st.caption(f'{m["view"]}: {values.shape[0]} x {values.shape[1]} | {int((values>0).sum())} positivos, {int((values<0).sum())} negativos, {int((values==0).sum())} nulos')
        if not paths: st.info('Mapa ainda nao calculado para este candidato e modelo. As imagens originais continuam visiveis.')
    images = []
    for modality, item, view in [('face', result['query'], fv), ('skull', candidate, sv)]:
        source = item['views'].get(view)
        try:
            im = read_image(source['path']) if source else None
            heat = next((m for m in maps if m['modality'] == modality and m['view'] == view), None)
            if im is not None and show_heat and heat:
                im = sensitivity_overlay(im, heat['values'])
            elif show_heat and paths and heat is None:
                st.caption(f'{view}: sem mapa nesta analise; exibindo original.')
            images.append(im)
        except (OSError, ValueError) as exc:
            st.warning(str(exc)); images.append(None)
    score = f'{ranked["score"]:.4f}' if ranked['score'] is not None else 'indisponivel'
    st.caption(f'Posicao: {ranked["rank"]} | Candidato: {person} | Score: {score}')
    if overlap and all(im is not None for im in images):
        if (fv == 'FL') != (sv == 'CD'):
            st.warning('Vistas com orientacoes diferentes; confira o perfil antes de alinhar.')
        a, b = st.columns(2)
        alpha = a.slider('Opacidade do cranio', 0., 1., .5, .05, key='overlay_alpha')
        scale = b.slider('Escala do cranio', .25, 2., 1., .01, key='overlay_scale')
        a, b, c = st.columns(3)
        angle = a.slider('Rotacao (graus)', -45., 45., 0., .5, key='overlay_angle')
        dx = b.slider('Deslocamento horizontal', -320, 320, 0, key='overlay_x')
        dy = c.slider('Deslocamento vertical', -320, 320, 0, key='overlay_y')
        rendered = visual_overlay(images[0], images[1], alpha, scale, angle, dx, dy)
        st.image(rendered, width=640, caption='Sobreposicao manual; nao altera pontos, modelos ou ranking.')
        import io
        buf = io.BytesIO(); rendered.save(buf, format='PNG')
        st.download_button('Baixar sobreposicao', buf.getvalue(), 'sobreposicao.png', 'image/png')
    else:
        cols = st.columns(2)
        for col, im, caption in zip(cols, images, (result['query']['person']+' | '+str(fv), person+' | '+str(sv))):
            with col:
                if im is not None: st.image(letterbox(im, 560)[0], caption=caption, width='stretch')
                else: st.info('Imagem indisponivel')
    if st.button('Analisar vistas e mapa de sensibilidade', icon=':material/heat:', disabled=worker_busy()):
        launch('explain', inspection)
    if show_heat and paths:
        with st.expander('Valores dos quadrados'):
            st.caption('Score original menos score com o bloco ocultado. Os efeitos nao somam 100%.')
            for m in maps:
                if (m['modality'],m['view']) in [('face',fv),('skull',sv)]:
                    st.write(m['modality']+' '+m['view']); st.dataframe(pd.DataFrame(m['values']))
    return person, candidate, ranked


def candidate_sheet(candidates, preferred, key):
    pages = max(1, (len(candidates)+5)//6)
    page = st.selectbox('Pagina dos candidatos', list(range(1,pages+1)), key=key) if pages > 1 else 1
    shown = candidates[(page-1)*6:page*6]
    canvas = Image.new('RGB',(540,190*((len(shown)+2)//3)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,candidate in enumerate(shown):
        x,y = (i%3)*180,(i//3)*190
        view = next((v for v in preferred if v in candidate['views']), None)
        if view:
            try: canvas.paste(letterbox(read_image(candidate['views'][view]['path']),160)[0],(x+10,y))
            except (OSError,ValueError): draw.text((x+10,y+60),'Imagem indisponivel',fill='black')
        else: draw.text((x+10,y+60),'Vista ausente',fill='black')
        draw.text((x+8,y+164),candidate['person'][:25],fill='black')
        draw.text((x+8,y+178),view or '',fill='black')
    st.image(canvas,width='stretch',caption=f'{len(candidates)} candidatos | pagina {page}/{pages}')


def matching_page():
    metrics_help()
    fusion_runs = sorted((ARTIFACTS/'fusions').glob('*/results.json'), reverse=True)
    fusion_runs = [p for p in fusion_runs if not uses_ce(p.parent)]
    st.caption('Protocolo sem CE: neutra/sorrindo com CF; lateral facial com CD. Modelos antigos CE+CD ficam apenas no historico de resultados.')
    with st.expander('Modelo e vistas da comparacao', expanded=True):
        selectors = st.columns([1,2,1])
        use_fusion = bool(fusion_runs) and selectors[0].selectbox('Tipo de modelo', ['Modelos por vista', 'Experimentos anteriores'],key='matching_model_type') == 'Modelos por vista'
        weights = None
        all_subsets = False
        if use_fusion:
            latest_cohort = load(fusion_runs[0].parent/'config.json')['cohort']
            eligible = [p.parent for p in fusion_runs if load(p.parent/'config.json')['cohort'] == latest_cohort]
            leader = max(eligible, key=lambda p: (load(p/'results.json')['combinado']['full']['recall_1'], load(p/'results.json')['combinado']['full']['mrr']))
            def model_label(path):
                info, cfg = metadata(path), load(path/'config.json')
                label = f'{info["Extrator"]} | {info["Data (Sao Paulo)"]} | {info["Loss"]} | seed {info["Seed"]} | {path.name}'
                for flag, text in [('rank_fusion_members','Mistura RRF'),('ensemble_members','ensemble de seeds'),('weight_search','pesos ajustados')]:
                    if cfg.get(flag): label += ' | '+text
                return label + (' | melhor Recall@1' if path == leader else '')
            run = selectors[1].selectbox('Execucao dos modelos', [p.parent for p in fusion_runs],
                index=[p.parent for p in fusion_runs].index(leader),
                format_func=model_label,key='fusion_model_choice')
            selected_config = load(run/'config.json')
            fixed_blend = bool(selected_config.get('rank_fusion_members'))
            if fixed_blend:
                st.caption('Execucao selecionada: Mistura RRF de modelos')
                st.info('Mistura de dois modelos (RRF). As vistas e pesos das fontes ficam fixos para reproduzir a avaliacao; detalhes em Parametros do treino e protocolo.')
                st.caption(f'Score RRF: soma de peso / (constante + posicao em cada fonte). Vai de 0 ate {1/(selected_config["rank_constant"]+1):.4f}; maior e melhor, sem equivaler a certeza de identidade. Muda com a galeria de candidatos.')
            if st.session_state.get('weights_model_seen_v2') != str(run):
                st.session_state['weights_model_seen_v2'] = str(run)
                st.session_state['matching_combination_mode'] = 'Selecionar modelos'
                for name, weight in zip(('frontal','sorriso','lateral'), selected_config['weights']):
                    st.session_state['matching_use_'+name] = weight > 0
                    st.session_state['match_weight_'+name] = float(weight)
            from Treinamento_DINOv2_face_cranio.profile_ui import view_controls
            weights, all_subsets = view_controls(selected_config, selectors[2])
            if not any(weights):
                st.info('Selecione pelo menos uma vista facial.'); return
            if all_subsets and any(w <= 0 for w in weights):
                st.info('Para avaliar as sete combinacoes, informe um peso positivo para cada vista.'); return
            face_views_used = [v for v, w in zip(('FFN', 'FFS', 'FL'), weights) if w]
            skull_views_used = (['CF'] if any(weights[:2]) else []) + (['CD'] if weights[2] else [])
        else:
            run = choose_run('matching_run')
        if run is None: return
        config = load(run/'config.json')
        if not use_fusion:
            face_views_used, skull_views_used = EXPERIMENTS[config['experiment']]
        if config.get('smoke'):
            st.warning('Modelo de teste tecnico: apenas verifica o funcionamento do fluxo.')
    st.caption('Modelos selecionados: '+', '.join({'FFN':'Neutra','FFS':'Sorrindo','FL':'Lateral'}.get(v,v) for v in face_views_used)+' | '+str(config.get('loss', metadata(run).get('Loss', ''))))
    from Treinamento_DINOv2_face_cranio.profile_ui import profile
    profile(run, weights if use_fusion else None)
    mode = st.segmented_control('Consulta', ['Teste conhecido', 'Enviar imagens'], default='Teste conhecido')
    request = None
    if mode == 'Teste conhecido':
        people = [p for p in load(run/'cohort.json')['people'] if p['split'] == 'validacao']
        if len(people) < 2: return
        a, b = st.columns(2)
        count = a.number_input('Quantidade de cranios', 2, len(people), min(10, len(people)))
        with b.expander('Repetir um sorteio'):
            seed = st.number_input('Seed do sorteio', 0, 999999, 42, help=HELP['Seed'])
            fixed_seed = st.checkbox('Repetir amostra usando a seed informada', value=False, key='fixed_gallery_seed')
        if st.button('Sortear amostra', icon=':material/shuffle:', help='Escolhe uma nova face e candidatos distintos, incluindo o correto uma vez. Nao executa o matching.'):
            previous = st.session_state.get('gallery', {})
            actual_seed = seed if fixed_seed else secrets.randbelow(1000000)
            selection = sample_gallery(people, count, actual_seed)
            # A fresh draw must also change the reference face, not only the candidate order.
            while not fixed_seed and selection['query']['person'] == previous.get('query', {}).get('person'):
                actual_seed = (actual_seed+1) % 1000000
                selection = sample_gallery(people, count, actual_seed)
            st.session_state['gallery'] = dict(run=str(run), seed=actual_seed, **selection)
        saved = st.session_state.get('gallery')
        if saved and saved['run'] == str(run):
            request = dict(run=str(run), query=saved['query'], candidates=saved['candidates'])
            st.caption(f'Amostra selecionada: {saved["query"]["person"]} | Seed utilizada: {saved.get("seed", "nao registrada")}')
    else:
        query = dict(person='consulta', views={})
        for view in face_views_used:
            file = st.file_uploader('Face '+view, type=['jpg', 'jpeg', 'png', 'tif'], key='face_upload_'+view)
            if file: query['views'][view] = upload_row(file)
        count = st.number_input('Candidatos', 1, 100, 2)
        candidates = []
        for i in range(count):
            candidate = dict(person=f'candidato_{i+1:03}', views={})
            with st.expander(f'Candidato {i+1}', expanded=i == 0):
                for view in skull_views_used:
                    file = st.file_uploader(view, type=['jpg', 'jpeg', 'png', 'tif'], key=f'upload_{i}_{view}')
                    if file: candidate['views'][view] = upload_row(file)
            candidates.append(candidate)
        if all(v in query['views'] for v in face_views_used) and all(any(v in c['views'] for v in skull_views_used) for c in candidates):
            request = dict(run=str(run), query=query, candidates=candidates)
    if request:
        from Treinamento_DINOv2_face_cranio.sex_controls import matching_controls
        matching_controls(request['query'])
        request_token = digest(request)
        reference = query_summary(dict(request, ranking=[]))
        st.info(f'Pessoa de referencia: {reference["Pessoa de referencia"]} | Ano: {reference["Ano"]} | {len(request["candidates"])} cranios concorrentes')
        with st.expander('Pessoa consultada e candidatos', expanded=st.session_state.get('compared_request') != request_token):
            a,b = st.columns([1,2])
            with a: picture(request['query']['views'].get(face_views_used[0]), request['query']['person'],240)
            with b: candidate_sheet(request['candidates'],skull_views_used,'draft_candidates')
        if st.button('Comparar', type='primary', icon=':material/compare:', disabled=worker_busy(), help='Calcula o ranking desta amostra com os modelos selecionados. Nao treina novamente.'):
            st.session_state['compared_request'] = request_token
            launch('compare_fusion' if use_fusion else 'compare', dict(request, weights=weights, all_subsets=all_subsets) if use_fusion else request)
    with st.expander('Acompanhamento e logs', expanded=True): progress()
    st.subheader('Historico e resultados')
    browse_history = st.checkbox('Ver historico completo deste modelo', help='Inclui outros sorteios e selecoes de vistas deste modelo. Nao altera a amostra preparada acima.')
    outputs = sorted((ARTIFACTS/('fusion_comparisons' if use_fusion else 'comparisons')).glob('*/result.json'), reverse=True)
    outputs = [p for p in outputs if Path(load(p)['model']) == Path(run)]
    if use_fusion and not browse_history:
        outputs = [p for p in outputs if load(p).get('weights') == weights and load(p).get('all_subsets', False) == all_subsets]
    if request and not browse_history:
        outputs = [p for p in outputs if load(p)['query'] == request['query'] and load(p)['candidates'] == request['candidates']]
        if not outputs:
            st.info('Esta amostra ainda nao tem resultado. Clique em Comparar para calcular o ranking.')
    inspection = None
    if outputs:
        output = st.selectbox('Resultado do matching', outputs, format_func=history_label)
        result = load(output)
        reference = query_summary(result)
        st.info(f'Resultado consultado: {reference["Pessoa de referencia"]} | Ano: {reference["Ano"]} | {len(result["candidates"])} cranios')
        st.caption('Identificador da pessoa conforme o banco; nao e um nome completo inferido. Historico em horario de Sao Paulo.')
        if use_fusion:
            weights = result['weights']
            face_views_used = [v for v,w in zip(('FFN','FFS','FL'),weights) if w]
            skull_views_used = (['CF'] if any(weights[:2]) else []) + (['CD'] if weights[2] else [])
        if use_fusion:
            if not result.get('combinations'):
                from Treinamento_DINOv2_face_cranio.matching_breakdown import show as breakdown_table
                breakdown_table(output.parent,launch,worker_busy)
            if result.get('combinations'):
                def combination_label(bits):
                    return ' + '.join(n for n,b in zip(('Neutra','Sorrindo','Lateral'),bits) if b == '1')
                summaries = []
                for bits, entry in result['combinations'].items():
                    summaries.append({'Combinacao': combination_label(bits), **query_summary(result, entry['ranking'])})
                with st.expander('Comparar as sete combinacoes', expanded=True):
                    table(summaries)
                    st.caption('Top-K: o cranio correto entrou nas primeiras K posicoes desta consulta. Nao e Recall agregado nem probabilidade de identidade. Sem score conta como falha quando a referencia e conhecida.')
                    chart = pd.DataFrame(summaries).set_index('Combinacao')[['Posicao do correto']]
                    if chart['Posicao do correto'].notna().any():
                        st.bar_chart(chart, height=200)
                        st.caption('Posicao do correto: menor e melhor; 1 significa primeiro colocado. Sem score nao tem barra.')
                    st.caption('Mesma amostra. Nao escolha uma combinacao por ter score numericamente maior: as escalas nao sao calibradas entre combinacoes.')
                bits = st.selectbox('Ranking da combinacao', list(result['combinations']), format_func=combination_label,
                                    index=list(result['combinations']).index('111'))
                weights = result['combinations'][bits]['weights']
                result['ranking'] = result['combinations'][bits]['ranking']
                face_views_used = [v for v,w in zip(('FFN','FFS','FL'),weights) if w]
                skull_views_used = (['CF'] if any(weights[:2]) else []) + (['CD'] if weights[2] else [])
            result.setdefault('failures', [])
            for row in result['ranking']:
                row['available_views'] = sorted({v for name in ('frontal', 'sorriso', 'lateral') for v in row[name+'_vistas']})
            with st.expander('Scores detalhados de todos os candidatos'):
                st.dataframe(result['ranking'], hide_index=True)
        valid = [r for r in result['ranking'] if r['score'] is not None]
        st.subheader('Resumo do ranking')
        if use_fusion and not np.allclose(weights, [st.session_state.get('match_weight_'+n,0) * st.session_state.get('matching_use_'+n,False) for n in ('frontal','sorriso','lateral')]) and not config.get('rank_fusion_members'):
            st.caption('O historico abaixo usou outra selecao de pesos. Metricas desta consulta salva:')
            profile(run, weights, key='saved')
        if valid:
            best = valid[0]
            a, b, c = st.columns(3)
            a.caption('Primeiro candidato')
            a.write(best['person'])
            b.metric('Score RRF' if config.get('rank_fusion_members') else ('Score combinado' if use_fusion else 'Similaridade cosseno'), f'{best["score"]:.4f}', help='Fusao das posicoes nas duas fontes; nao e cosseno nem probabilidade.' if config.get('rank_fusion_members') else HELP['Score combinado' if use_fusion else 'Cosseno'])
            gap = best['score']-valid[1]['score'] if len(valid) > 1 else None
            c.metric('Margem para o segundo', f'{gap:.4f}' if gap is not None else 'Sem comparacao', help=HELP['Margem'])
            if len(valid) > 1 and np.isclose(gap, 0, atol=1e-7, rtol=0):
                st.warning('Empate ou margem numericamente desprezivel: a ordem nao demonstra preferencia confiavel; empates exatos usam o identificador para ordenar.')
        st.caption('Similaridade nao e probabilidade de identidade. Nao ha porcentagem de certeza calibrada para este par.')
        ranking_tab, visual, contributions_tab, details_tab = st.tabs(['Ranking', 'Comparador', 'Contribuicoes', 'Detalhes'])
        with visual:
            person, candidate, ranked = matching_viewer(result, run, config, weights, use_fusion, output)
        with ranking_tab:
            from Treinamento_DINOv2_face_cranio.sex_controls import matching_filter
            shown_ranking = matching_filter(result)
            display = [{'Posicao':r['rank'], 'Candidato':r['person'],
                        'Score combinado' if use_fusion else 'Cosseno':r['score'],
                        'Posicao original':r.get('posicao_original',r['rank']),
                        'Vistas':', '.join(r['available_views'])} for r in shown_ranking]
            table(display)
            known = next((r for r in shown_ranking if r['person'] == result['query']['person']), None)
            if known: st.metric('Posicao do cranio correto', known['rank'] or 'Sem score')
            if st.checkbox('Mostrar miniaturas do ranking'):
                ordered = [next(c for c in result['candidates'] if c['person']==r['person']) for r in shown_ranking]
                candidate_sheet(ordered, skull_views_used, 'rank_thumbnails')
        with contributions_tab:
            st.caption('Candidato selecionado no comparador: '+person)
            if use_fusion and config.get('rank_fusion_members'):
                table([dict(Fonte=i+1, Peso=m['weight'], Posicao_na_fonte=ranked[f'source_{i+1}_rank'],
                    Contribuicao=ranked[f'source_{i+1}_contribuicao']) for i,m in enumerate(config['rank_fusion_members'])])
                st.caption(f'RRF: soma de peso / ({config["rank_constant"]} + posicao). Maior e melhor; nao e cosseno nem probabilidade. A contribuicao e por modelo, nao por ponto anatomico.')
            elif use_fusion:
                contributions = []
                for name,label,weight in zip(('frontal','sorriso','lateral'),('Neutra','Sorrindo','Lateral'),weights):
                    contributions.append({'Modelo':label,'Usado':bool(weight),'Peso':weight/sum(weights),
                        'Cosseno':ranked[name],'Contribuicao ao score':ranked[name+'_contribuicao']})
                table(contributions)
                st.bar_chart(pd.DataFrame(contributions).set_index('Modelo')[['Contribuicao ao score']],height=250)
                st.caption(HELP['Contribuicao ao score'])
            else:
                st.info('Modelo individual: nao ha combinacao de scores separados nesta execucao.')
        with details_tab:
            st.write('Execucao:',run.name)
            st.write('Vistas faciais usadas:',', '.join(face_views_used))
            st.write('Vistas cranianas disponiveis para o candidato:',', '.join(ranked['available_views']) or 'Nenhuma')
            st.caption('Imagens completas, sem landmarks obrigatorios. Nao estima espessura de tecido nem correspondencia anatomica.')
            with st.expander('Parametros e modelos'): st.json(config,expanded=False)
            with st.expander('Falhas de entrada'): st.json(result['failures'])


def open_area(area):
    st.session_state['main_area'] = area


def home_page(manifest_path):
    st.markdown('''<div class="home-intro"><h2>Seu laboratório, em um só lugar.</h2>
    <p>Confira as imagens, treine seus modelos e compare os resultados.
    Escolha abaixo o que você quer fazer agora.</p></div>''', unsafe_allow_html=True)
    people = load(manifest_path).get('people', []) if manifest_path else []
    a, b, c = st.columns(3)
    a.metric('Pessoas no inventário', len(people))
    b.metric('Modelos de matching avaliados', len(all_runs()))
    c.metric('Processamento', 'Em andamento' if worker_busy() else 'Disponível')
    cards = [
        ('Matching', 'Comparar face e crânio', 'Consulte uma pessoa ou envie imagens para ver o ranking e o comparador visual.'),
        ('Treinamentos', 'Treinar matching', 'Escolha as vistas e inicie um novo treinamento. Os ajustes avançados são opcionais.'),
        ('Sexo registrado', 'Sexo: referências e modelos', 'Prepare a referência, treine o modelo craniano e consulte os resultados de faces e crânios.'),
        ('Banco', 'Explorar as imagens', 'Veja originais, entradas normalizadas e problemas do inventário.'),
    ]
    for start in (0, 2):
        for column, (area, title, description) in zip(st.columns(2), cards[start:start+2]):
            with column, st.container(border=True):
                st.subheader(title)
                st.write(description)
                st.button('Abrir', key='home_'+area, on_click=open_area, args=(area,), width='stretch')
    st.caption('Os resultados e gráficos dos treinamentos ficam em Resultados, no menu acima.')


def sex_page():
    st.subheader('Sexo: referências e modelos')
    section = st.segmented_control('O que você quer fazer?',
        ['Modelo craniano', 'Resultados de faces', 'Resultados de crânios', 'Sobre os métodos'],
        default='Modelo craniano', key='sex_section')
    st.caption('A origem da referência acompanha cada execução. Curadoria e rótulos automáticos não equivalem a sexo documentado.')
    if section == 'Modelo craniano':
        from Treinamento_DINOv2_face_cranio.sex_controls import cranial_controls
        cranial_controls(launch, busy=worker_busy())
        with st.expander('Acompanhamento e logs', expanded=worker_busy()): progress()
    elif section in ('Resultados de faces', 'Resultados de crânios'):
        from Laboratorio_visual_face_cranio.registered_ui import results
        results('face' if section == 'Resultados de faces' else 'cranio')
    elif section == 'Sobre os métodos':
        st.dataframe([
            {'Área':'Faces · histórico', 'Método':'FairFace e classificadores sobre medidas faciais', 'No painel':'Consultar resultados e procedência'},
            {'Área':'Crânios · histórico', 'Método':'Silhuetas, regressão logística e variantes VGG16', 'No painel':'Consultar resultados existentes'},
            {'Área':'Crânios · novo treino', 'Método':'Silhuetas 16 × 16 + regressão logística, CF/CE/CD', 'No painel':'Preparar referência, treinar e aplicar'},
            {'Área':'Matching DINO', 'Método':'Embeddings visuais e projeções por modalidade', 'No painel':'Treinar, avaliar e comparar'},
        ], hide_index=True, width='stretch')
        st.info('O novo treino craniano é um modelo independente. Ele não executa o VGG16 antigo. O filtro de sexo no matching é opcional e não altera o score visual.')


st.title('DINO · Face e crânio')
inventories = sorted((ARTIFACTS/'inventories').glob('*/manifest.json'), reverse=True)
with st.sidebar:
    manifest_path = st.selectbox('Inventario', inventories, format_func=lambda p: p.parent.name) if inventories else None
    st.caption(str(REPO/'Banco'))
area = st.radio('Area', ['Início', 'Banco', 'Treinamentos', 'Resultados', 'Matching', 'Sexo registrado'], horizontal=True,
                key='main_area', label_visibility='collapsed')
if area == 'Sexo registrado':
    sex_page()
elif area == 'Início': home_page(manifest_path)
elif area == 'Banco': bank_page(manifest_path)
elif area == 'Treinamentos': train_page(manifest_path)
elif area == 'Resultados': results_page()
else: matching_page()
