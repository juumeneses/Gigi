"""Late score fusion. Normalization uses training impostor pairs only."""
from .portable import Path
from itertools import product

import numpy as np

from . import engine
from .backbone import Extractor
from .core import check_stop, digest, load, new_run, np_save, save, sha
from .data import cohort

COMPONENTS = {'frontal': 'E1', 'sorriso': 'S1', 'lateral': 'L2'}


def normalization(scores):
    values = scores[~np.eye(len(scores), dtype=bool) & np.isfinite(scores)]
    if len(values) < 2 or values.std() < 1e-8:
        raise ValueError('Scores de treino insuficientes para normalizacao.')
    return dict(mean=float(values.mean()), std=float(values.std()), n=len(values))


def combine(matrices, stats, weights):
    w = np.asarray(weights, float)
    if len(w) != len(matrices) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Pesos devem ser finitos, nao negativos e somar mais que zero.')
    w = w / w.sum()
    output = np.zeros_like(matrices[0], dtype=float)
    valid = np.ones_like(output, dtype=bool)
    for matrix, stat, weight in zip(matrices, stats, w):
        if weight == 0:
            continue
        if matrix.shape != output.shape or not np.isfinite(stat['std']) or stat['std'] <= 0:
            raise ValueError('Matriz ou normalizacao invalida.')
        valid &= np.isfinite(matrix)
        output += weight*(np.where(np.isfinite(matrix), matrix, stat['mean'])-stat['mean'])/stat['std']
    output[~valid] = -np.inf
    return output


def train_components(manifest, run_dir=None, component='todos', stop=None, log=print, **params):
    if component not in ('todos', *COMPONENTS):
        raise ValueError('Componente invalido')
    run = Path(run_dir) if run_dir else new_run('fusion_training')
    run.mkdir(parents=True, exist_ok=True)
    config = dict(manifest=str(manifest), params=params,
                  manifest_hash=sha(manifest), source_hash=sha(__file__), view_policy='without_CE_v1')
    if (run/'request.json').exists():
        old = load(run/'request.json')
        if {k: v for k, v in old.items() if k != 'source_hash'} != {k: v for k, v in config.items() if k != 'source_hash'}:
            raise ValueError('Retomada incompativel; crie uma nova sequencia.')
        # engine.train separately checks training-code compatibility on resume.
    else:
        save(run/'request.json', config)
    save(run/'status.json', dict(state='RODANDO'))
    c = cohort(manifest, 'E6D', 'matched', params.get('ids'))
    save(run/'cohort.json', c)
    for name, experiment in COMPONENTS.items():
        if component not in ('todos', name):
            continue
        check_stop(stop)
        child = run/name
        if (child/'status.json').exists() and load(child/'status.json')['state'] == 'CONCLUIDO':
            log('Componente ja concluido: '+name, flush=True)
            continue
        log('Treinando componente: '+name, flush=True)
        engine.train(manifest, experiment=experiment, cohort_experiment='E6D', mode='matched',
                     run_dir=str(child), resume=(child/'config.json').exists(), stop=stop, log=log, **params)
        if load(child/'status.json')['state'] != 'CONCLUIDO':
            save(run/'status.json', dict(state=load(child/'status.json')['state']))
            return run
    save(run/'status.json', dict(state='CONCLUIDO'))
    return run


def fuse(bundle, weights=(1., 1., 1.), stop=None, log=print):
    bundle = Path(bundle)
    c = load(bundle/'cohort.json')
    train = [p for p in c['people'] if p['split'] == 'treino']
    val = [p for p in c['people'] if p['split'] == 'validacao']
    if len(train) < 2 or len(val) < 2:
        raise ValueError('Treino e validacao insuficientes.')
    matrices, stats, children = [], [], {}
    for name, experiment in COMPONENTS.items():
        check_stop(stop)
        child = bundle/name
        config, model = engine.load_run(child)
        if config['cohort'] != c['fingerprint'] or config['experiment'] != experiment:
            raise ValueError('Componentes nao compartilham o protocolo esperado.')
        ext = engine.configured_extractor(config)
        if ext.fingerprint != config['extractor']:
            raise ValueError('Extrator alterado')
        data, failures = engine.features(train, experiment, ext, stop, log)
        stat = normalization(engine.score_matrix(model, data, experiment))
        stat['failed_inputs'] = failures
        stats.append(stat)
        data, failures = engine.features(val, experiment, ext, stop, log)
        matrices.append(engine.score_matrix(model, data, experiment))
        children[name] = dict(run=str(child.resolve()), checkpoint_hash=sha(child/'best.pt'), failed_inputs=failures)
    fused = combine(matrices, stats, weights)
    output = new_run('fusions')
    save(output/'config.json', dict(bundle=str(bundle.resolve()), weights=list(weights), normalization=stats,
         children=children, cohort=c['fingerprint'], source_hash=sha(__file__), split='validacao',
         policy='fixed_weights_training_impostor_zscore_strict_missing', smoke=config['smoke']))
    save(output/'cohort.json', c)
    results = {name: engine.evaluation(matrix, val) for name, matrix in zip(COMPONENTS, matrices)}
    results['combinado'] = engine.evaluation(fused, val)
    save(output/'results.json', results)
    np_save(output/'scores.npz', **dict(zip(COMPONENTS, matrices)), combinado=fused)
    engine.export_report(output, results['combinado'], val)
    save(output/'status.json', dict(state='CONCLUIDO'))
    return output


def all_combinations(bundle, stop=None, log=print):
    """Evaluate seven fixed subsets, reusing component score matrices."""
    bundle = Path(bundle)
    ledger_path = bundle/'combinations.json'
    ledger = load(ledger_path) if ledger_path.exists() else {}
    signatures = {name: sha(bundle/name/'best.pt') for name in COMPONENTS}
    if ledger and ledger.get('checkpoints') != signatures:
        raise ValueError('Checkpoints alterados; crie nova sequencia.')
    ledger.setdefault('checkpoints', signatures)
    outputs = ledger.setdefault('outputs', {})
    if '111' not in outputs:
        outputs['111'] = str(fuse(bundle, stop=stop, log=log))
        save(ledger_path, ledger)
    base = Path(outputs['111'])
    config = load(base/'config.json')
    c = load(base/'cohort.json')
    people = [p for p in c['people'] if p['split'] == 'validacao']
    with np.load(base/'scores.npz') as arrays:
        matrices = [arrays[name].copy() for name in COMPONENTS]
    base_results = load(base/'results.json')
    for weights in product((0, 1), repeat=3):
        if not any(weights):
            continue
        check_stop(stop)
        key = ''.join(map(str, weights))
        if key in outputs:
            continue
        log('Avaliando combinacao: '+key, flush=True)
        fused = combine(matrices, config['normalization'], weights)
        output = new_run('fusions')
        save(output/'config.json', dict(config, weights=list(weights)))
        save(output/'cohort.json', c)
        results = dict(base_results, combinado=engine.evaluation(fused, people))
        save(output/'results.json', results)
        np_save(output/'scores.npz', **dict(zip(COMPONENTS, matrices)), combinado=fused)
        engine.export_report(output, results['combinado'], people)
        save(output/'status.json', dict(state='CONCLUIDO'))
        outputs[key] = str(output)
        save(ledger_path, ledger)
    return bundle


def reweight_saved(source, weights, stop=None, log=print):
    source = Path(source)
    config = load(source/'config.json')
    if config.get('split') != 'validacao' or config.get('ensemble_members') or config.get('rank_fusion_members'):
        raise ValueError('Use scores de validacao de uma seed individual')
    for child in config['children'].values():
        if sha(Path(child['run'])/'best.pt') != child['checkpoint_hash']:
            raise ValueError('Checkpoint alterado')
    check_stop(stop)
    c = load(source/'cohort.json')
    people = [p for p in c['people'] if p['split']=='validacao']
    with np.load(source/'scores.npz',allow_pickle=False) as arrays:
        matrices = [arrays[n].copy() for n in COMPONENTS]
    combined = combine(matrices,config['normalization'],weights)
    output = new_run('fusions')
    cfg = dict(config,weights=list(weights))
    cfg.pop('weight_search',None)
    save(output/'config.json',cfg)
    save(output/'cohort.json',c)
    results = dict(load(source/'results.json'),combinado=engine.evaluation(combined,people))
    save(output/'results.json',results)
    np_save(output/'scores.npz',**dict(zip(COMPONENTS,matrices)),combinado=combined)
    engine.export_report(output,results['combinado'],people)
    save(output/'status.json',dict(state='CONCLUIDO'))
    return output


def train_combinations(manifest, run_dir=None, stop=None, log=print, **params):
    bundle = train_components(manifest, run_dir=run_dir, stop=stop, log=log, **params)
    if load(bundle/'status.json')['state'] != 'CONCLUIDO':
        return bundle
    save(bundle/'status.json', dict(state='RODANDO', phase='combinations'))
    try:
        all_combinations(bundle, stop, log)
        save(bundle/'status.json', dict(state='CONCLUIDO'))
    except InterruptedError:
        save(bundle/'status.json', dict(state='INTERROMPIDO'))
    return bundle


def rank_components(matrices, stats, candidates, component_rows, selected_weights):
    fused = combine(matrices, stats, selected_weights)[0]
    weights = np.array(selected_weights)/sum(selected_weights)
    rows = []
    for i, candidate in enumerate(candidates):
        row = dict(person=candidate['person'], score=float(fused[i]) if np.isfinite(fused[i]) else None)
        for j, name in enumerate(COMPONENTS):
            value = matrices[j][0, i]
            row[name] = float(value) if np.isfinite(value) else None
            row[name+'_contribuicao'] = (float(weights[j]*(value-stats[j]['mean'])/stats[j]['std'])
                                       if np.isfinite(value) else None) if weights[j] else 0.
            row[name+'_vistas'] = component_rows[name][candidate['person']]['available_views'] if weights[j] else []
        rows.append(row)
    rows.sort(key=lambda r: (-r['score'] if r['score'] is not None else float('inf'), r['person']))
    for i, row in enumerate(rows):
        row['rank'] = i+1 if row['score'] is not None else None
    return rows


def compare_fusion(run, query, candidates, stop=None, log=print, weights=None, all_subsets=False):
    config = load(Path(run)/'config.json')
    if config.get('rank_fusion_members'):
        if all_subsets or (weights is not None and not np.allclose(weights, config['weights'])):
            raise ValueError('Mistura de modelos usa vistas fixas; crie outra mistura para mudar as vistas.')
        from .rank_blend import compare
        return compare(run, query, candidates, stop, log)
    selected_weights = list(([1, 1, 1] if all_subsets else config['weights']) if weights is None else weights)
    if all_subsets and any(w <= 0 for w in selected_weights):
        raise ValueError('Para avaliar as sete combinacoes, todos os pesos devem ser positivos.')
    # Validate before loading any models or writing an output.
    combine([np.zeros((1, len(candidates)))]*3, config['normalization'], selected_weights)
    if config.get('ensemble_members'):
        from .ensemble import compare
        return compare(run, query, candidates, selected_weights, all_subsets, stop, log)
    matrices, component_rows, failures = [], {}, []
    for j, name in enumerate(COMPONENTS):
        if selected_weights[j] == 0:
            matrices.append(np.full((1, len(candidates)), -np.inf))
            component_rows[name] = {c['person']: dict(available_views=[]) for c in candidates}
            continue
        child = config['children'][name]
        if sha(Path(child['run'])/'best.pt') != child['checkpoint_hash']:
            raise ValueError('Checkpoint foi alterado')
        result = engine.compare(child['run'], query, candidates, stop, log)
        failures.extend(dict(f, component=name) for f in result.get('failures', []))
        lookup = {r['person']: r for r in result['ranking']}
        matrices.append(np.array([[lookup[p['person']]['score'] if lookup[p['person']]['score'] is not None else -np.inf for p in candidates]]))
        component_rows[name] = lookup
    rows = rank_components(matrices, config['normalization'], candidates, component_rows, selected_weights)
    combinations = {}
    if all_subsets:
        for bits in ('100', '010', '001', '110', '101', '011', '111'):
            subset = [int(b)*w for b,w in zip(bits,selected_weights)]
            combinations[bits] = dict(weights=subset, ranking=rank_components(
                matrices, config['normalization'], candidates, component_rows, subset))
    output = new_run('fusion_comparisons')
    save(output/'result.json', dict(model=str(run), query=query, candidates=candidates, ranking=rows,
         weights=selected_weights, failures=failures, all_subsets=all_subsets, combinations=combinations))
    return output
