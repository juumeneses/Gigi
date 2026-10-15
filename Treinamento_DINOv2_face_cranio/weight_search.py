"""Validation-only score fusion search over completed, compatible seed runs."""
from .portable import Path
import html

import numpy as np

from . import engine, fusion
from .core import ARTIFACTS, atomic, check_stop, csv_write, digest, load, new_run, np_save, save, sha


def grid_weights(step=.05):
    if not np.isfinite(step) or step <= 0:
        raise ValueError('O passo deve ser positivo e finito.')
    divisions = round(1 / step)
    if step <= 0 or divisions < 1 or divisions > 100 or not np.isclose(divisions * step, 1):
        raise ValueError('O passo deve dividir 1 exatamente, entre .01 e 1.')
    values = [(a/divisions, b/divisions, (divisions-a-b)/divisions)
              for a in range(divisions+1) for b in range(divisions+1-a)]
    equal = (1/3, 1/3, 1/3)
    if not any(np.allclose(w, equal) for w in values):
        values.append(equal)
    return values


def rank_metrics(scores, ids):
    """Match engine.evaluation's ID tie break and missing-query denominator."""
    if scores.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Matriz ou identidades invalidas')
    diagonal = np.diag(scores)
    ids = np.asarray(ids)
    ahead = (scores > diagonal[:, None]) | ((scores == diagonal[:, None]) & (ids[None, :] < ids[:, None]))
    ranks = 1 + ahead.sum(axis=1)
    valid = np.isfinite(diagonal)
    rr = np.where(valid, 1/ranks, 0.)
    metrics = dict(mrr=float(rr.mean()), failures=int((~valid).sum()), n=len(ids),
                   **{f'recall_{k}': float((valid & (ranks <= k)).mean()) for k in (1, 3, 5, 10)})
    return metrics


def source_groups():
    """Latest complete equal-weight source per seed, grouped by training protocol."""
    groups = {}
    for path in sorted((ARTIFACTS/'fusions').glob('*/config.json')):
        cfg = load(path)
        if cfg.get('rank_fusion_members') or cfg.get('weight_search') or cfg.get('ensemble_members') or cfg.get('split') != 'validacao' or cfg.get('smoke'):
            continue
        if not np.allclose(cfg['weights'], [1, 1, 1]):
            continue
        if not (path.parent/'scores.npz').exists() or not (path.parent/'results.json').exists():
            continue
        children = [load(Path(cfg['children'][name]['run'])/'config.json') for name in fusion.COMPONENTS]
        if [c['experiment'] for c in children] != list(fusion.COMPONENTS.values()):
            continue
        signatures = [{k: v for k, v in c.items() if k not in ('seed',)} for c in children]
        token = digest(signatures)
        entry = groups.setdefault(token, dict(backbone=children[0].get('backbone', 'dinov2_vits14'),
                     loss=children[0]['loss'], sources={}, protocol=token))
        entry['sources'][children[0]['seed']] = str(path.parent)
    return list(groups.values())


def read_sources(sources, allow_smoke=False):
    datasets, protocol = [], None
    for source in sources:
        source = Path(source)
        cfg = load(source/'config.json')
        if cfg.get('split') != 'validacao' or (cfg.get('smoke') and not allow_smoke):
            raise ValueError('Use somente execucoes completas de validacao, sem teste tecnico.')
        c = load(source/'cohort.json')
        people = [p for p in c['people'] if p['split'] == 'validacao']
        if len(people) < 2 or c['fingerprint'] != cfg['cohort']:
            raise ValueError('Coorte de validacao incompativel')
        ids = [p['person'] for p in people]
        configs = []
        for name, experiment in fusion.COMPONENTS.items():
            child = cfg['children'][name]
            child_path = Path(child['run'])
            cc = load(child_path/'config.json')
            if sha(child_path/'best.pt') != child['checkpoint_hash']:
                raise ValueError('Checkpoint alterado')
            if cc['experiment'] != experiment or cc['cohort'] != cfg['cohort']:
                raise ValueError('Vista ou protocolo incompativel')
            configs.append(cc)
        if len({cc['seed'] for cc in configs}) != 1:
            raise ValueError('Componentes com seeds diferentes')
        signature = digest([[{k: v for k, v in cc.items() if k != 'seed'} for cc in configs], ids])
        if protocol is not None and signature != protocol:
            raise ValueError('Execucoes diferem em pessoas, parametros ou versao do codigo.')
        protocol = signature
        with np.load(source/'scores.npz', allow_pickle=False) as arrays:
            matrices = [arrays[name].copy() for name in fusion.COMPONENTS]
        if any(m.shape != (len(ids), len(ids)) for m in matrices):
            raise ValueError('Dimensoes dos scores incompativeis com a validacao')
        evaluation = load(source/'results.json')['combinado']
        if [r['person'] for r in evaluation['rows']] != ids:
            raise ValueError('Ordem das identidades nao corresponde aos scores')
        stats = cfg['normalization']
        fusion.combine(matrices, stats, [1, 1, 1])
        datasets.append(dict(source=source, config=cfg, cohort=c, people=people, ids=ids,
                             matrices=matrices, stats=stats, seed=configs[0]['seed']))
    if len(datasets) < 3 or len({d['seed'] for d in datasets}) != len(datasets):
        raise ValueError('Selecione ao menos tres seeds distintas do mesmo protocolo.')
    return datasets


def optimize(datasets, step=.05, stop=None, log=print, objective='recall_1'):
    if objective not in ('recall_1', 'mrr'):
        raise ValueError('Objetivo deve ser recall_1 ou mrr')
    rows, per_seed = [], []
    grid = grid_weights(step)
    for i, weights in enumerate(grid):
        check_stop(stop)
        metrics = [rank_metrics(fusion.combine(d['matrices'], d['stats'], weights), d['ids']) for d in datasets]
        row = dict(frontal=weights[0], sorriso=weights[1], lateral=weights[2])
        for metric in ('mrr', 'recall_1', 'recall_3', 'recall_5', 'recall_10'):
            row[metric] = float(np.mean([m[metric] for m in metrics]))
            row[metric+'_sd'] = float(np.std([m[metric] for m in metrics], ddof=1))
        row['failures'] = sum(m['failures'] for m in metrics)
        rows.append(row)
        per_seed.append(metrics)
        if i % 20 == 0:
            log(f'Pesos: {i+1}/{len(grid)}', flush=True)
    def choose(indices):
        return max(range(len(rows)), key=lambda j: (
            round(float(np.mean([per_seed[j][i][objective] for i in indices])), 12),
            round(float(np.mean([per_seed[j][i]['mrr' if objective == 'recall_1' else 'recall_1'] for i in indices])), 12),
            -sum((rows[j][k]-1/3)**2 for k in fusion.COMPONENTS)))
    selected = choose(range(len(datasets)))
    held_out = []
    for i, dataset in enumerate(datasets):
        j = choose([k for k in range(len(datasets)) if k != i])
        held_out.append(dict(seed=dataset['seed'], weights=[rows[j][k] for k in fusion.COMPONENTS],
                             **per_seed[j][i]))
    return rows, selected, held_out


def search_weights(sources, step=.05, stop=None, log=print, objective='recall_1'):
    if objective not in ('recall_1', 'mrr'):
        raise ValueError('Objetivo deve ser recall_1 ou mrr')
    datasets = read_sources(sources)
    output = new_run('weight_searches')
    save(output/'status.json', dict(state='RODANDO'))
    save(output/'config.json', dict(sources=[str(d['source'].resolve()) for d in datasets], step=step,
         objective='mean_full_validation_'+objective, split='validacao', source_hash=sha(__file__),
         input_hashes=[dict(scores=sha(d['source']/'scores.npz'), config=sha(d['source']/'config.json')) for d in datasets]))
    try:
        rows, selected, held_out = optimize(datasets, step, stop, log, objective)
        best = rows[selected]
        weights = [best[k] for k in fusion.COMPONENTS]
        baseline = next(r for r in rows if np.allclose([r[k] for k in fusion.COMPONENTS], [1/3]*3))
        save(output/'grid.json', rows)
        csv_write(output/'grid.csv', rows)
        selected_outputs, seed_rows = [], []
        for dataset in datasets:
            check_stop(stop)
            log(f'Exportando pesos escolhidos: seed {dataset["seed"]}', flush=True)
            scores = fusion.combine(dataset['matrices'], dataset['stats'], weights)
            evaluation = engine.evaluation(scores, dataset['people'])
            equal = engine.evaluation(fusion.combine(dataset['matrices'], dataset['stats'], [1,1,1]), dataset['people'])
            target = new_run('fusions')
            save(target/'config.json', dict(dataset['config'], weights=weights, weight_search=str(output.resolve()),
                 weights_selected_on='validacao', weight_objective=objective, selection_seeds=[d['seed'] for d in datasets]))
            save(target/'cohort.json', dataset['cohort'])
            results = load(dataset['source']/'results.json')
            save(target/'results.json', dict(results, combinado=evaluation))
            np_save(target/'scores.npz', **dict(zip(fusion.COMPONENTS, dataset['matrices'])), combinado=scores)
            engine.export_report(target, evaluation, dataset['people'])
            save(target/'status.json', dict(state='CONCLUIDO'))
            selected_outputs.append(str(target.resolve()))
            seed_rows.append(dict(seed=dataset['seed'], output=str(target.resolve()),
                baseline=equal['full'], selected=evaluation['full'], gallery=evaluation['gallery']))
        summary = dict(weights=weights, best=best, equal_weights=baseline, objective=objective,
            delta_recall_1=best['recall_1']-baseline['recall_1'],
            delta_mrr=best['mrr']-baseline['mrr'], seeds=seed_rows, held_out_seed=held_out,
            outputs=selected_outputs, split='validacao',
            interpretation='Pesos escolhidos nesta validacao; ganho descritivo sujeito a selecao. Teste final nao utilizado. Seeds excluidas da escolha usam as mesmas identidades de validacao.')
        save(output/'summary.json', summary)
        csv_write(output/'seeds.csv', [dict(seed=r['seed'], **r['selected'], baseline_mrr=r['baseline']['mrr']) for r in seed_rows])
        csv_write(output/'held_out_seed.csv', held_out)
        report = '<!doctype html><meta charset="utf-8"><title>Pesos das vistas</title><h1>Busca de pesos na validacao</h1>'
        report += '<p>'+html.escape(summary['interpretation'])+'</p>'
        report += '<p>Pesos neutra / sorriso / lateral: '+html.escape(str(weights))+'</p>'
        report += '<p>Criterio: '+objective+'; Recall@1 medio: '+str(round(100*best['recall_1'],2))+'%</p>'
        report += '<p>MRR medio: '+str(round(best['mrr'],4))+'; pesos iguais: '+str(round(baseline['mrr'],4))+'</p>'
        report += '<ul>'+''.join('<li><a href="'+html.escape(str(Path(p)/'report.html'), quote=True)+'">Exemplos e rankings seed '+str(r['seed'])+'</a></li>' for p,r in zip(selected_outputs, seed_rows))+'</ul>'
        atomic(output/'report.html', report.encode())
        save(output/'status.json', dict(state='CONCLUIDO'))
        log(f'Concluido. Pesos {weights}; Recall@1 {best["recall_1"]:.1%}; iguais {baseline["recall_1"]:.1%}; MRR {best["mrr"]:.4f}', flush=True)
        return output
    except InterruptedError:
        save(output/'status.json', dict(state='INTERROMPIDO'))
        return output
    except Exception as exc:
        save(output/'status.json', dict(state='ERRO', error=str(exc)))
        raise
