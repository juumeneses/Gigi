"""Average training-standardized scores across independent seeds."""
from .portable import Path

import numpy as np

from . import engine, fusion
from .core import check_stop, load, new_run, np_save, save, sha
from .weight_search import read_sources


def build(sources, weights, stop=None, log=print, allow_smoke=False):
    datasets = read_sources(sources,allow_smoke=allow_smoke)
    check_stop(stop)
    matrices = [np.mean([(d['matrices'][j]-d['stats'][j]['mean'])/d['stats'][j]['std']
                        for d in datasets], axis=0) for j in range(3)]
    stats = [dict(mean=0., std=1., origin='training_standardized_seed_mean') for _ in range(3)]
    combined = fusion.combine(matrices, stats, weights)
    output = new_run('fusions')
    config = dict(datasets[0]['config'], weights=list(weights), normalization=stats,
                  ensemble_members=[dict(run=str(d['source'].resolve()), config_hash=sha(d['source']/'config.json')) for d in datasets],
                  policy='mean_seed_training_zscores_strict_missing', component_score_space='training_zscore_seed_mean')
    config['ensemble_code_hash'] = sha(__file__)
    config.pop('weight_search', None)
    save(output/'config.json', config)
    save(output/'cohort.json', datasets[0]['cohort'])
    results = {name: engine.evaluation(matrix, datasets[0]['people']) for name,matrix in zip(fusion.COMPONENTS,matrices)}
    results['combinado'] = engine.evaluation(combined, datasets[0]['people'])
    save(output/'results.json', results)
    np_save(output/'scores.npz', **dict(zip(fusion.COMPONENTS,matrices)), combinado=combined)
    engine.export_report(output, results['combinado'], datasets[0]['people'])
    save(output/'status.json', dict(state='CONCLUIDO'))
    log('Ensemble das seeds concluido: '+str(output), flush=True)
    return output


def compare(run, query, candidates, weights, all_subsets=False, stop=None, log=print):
    config = load(Path(run)/'config.json')
    matrices_by_seed, rows, failures = [], {}, []
    for member in config['ensemble_members']:
        check_stop(stop)
        path = Path(member['run'])
        if sha(path/'config.json') != member['config_hash']:
            raise ValueError('Configuracao de uma seed foi alterada')
        cfg = load(path/'config.json')
        matrices = []
        for j,name in enumerate(fusion.COMPONENTS):
            if not weights[j]:
                matrices.append(np.full((1,len(candidates)), -np.inf))
                rows[name] = {c['person']: dict(available_views=[]) for c in candidates}
                continue
            child = cfg['children'][name]
            if sha(Path(child['run'])/'best.pt') != child['checkpoint_hash']:
                raise ValueError('Checkpoint de uma seed foi alterado')
            result = engine.compare(child['run'], query, candidates, stop, log)
            lookup = {r['person']:r for r in result['ranking']}
            matrix = np.array([[lookup[c['person']]['score'] if lookup[c['person']]['score'] is not None else -np.inf for c in candidates]])
            stat = cfg['normalization'][j]
            matrices.append((matrix-stat['mean'])/stat['std'])
            rows[name] = lookup
            failures.extend(dict(f, member=str(path), component=name) for f in result['failures'])
        matrices_by_seed.append(matrices)
    matrices = [np.mean([m[j] for m in matrices_by_seed], axis=0) for j in range(3)]
    ranking = fusion.rank_components(matrices, config['normalization'], candidates, rows, weights)
    combinations = {}
    if all_subsets:
        for bits in ('100','010','001','110','101','011','111'):
            subset = [int(b)*w for b,w in zip(bits,weights)]
            combinations[bits] = dict(weights=subset, ranking=fusion.rank_components(matrices,config['normalization'],candidates,rows,subset))
    output = new_run('fusion_comparisons')
    save(output/'result.json', dict(model=str(run), query=query, candidates=candidates, ranking=ranking,
        weights=weights, failures=failures, all_subsets=all_subsets, combinations=combinations,
        score_space='training_zscore_seed_mean'))
    return output
