"""Run provenance and descriptive comparisons within identical protocols."""
from datetime import datetime, timezone
from .portable import Path
from itertools import groupby
from zoneinfo import ZoneInfo

from .core import digest, load, uses_ce


def started(run):
    path = Path(run)
    for part in (path.name, *(parent.name for parent in list(path.parents)[:8])):
        try:
            return datetime.strptime(part, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def newest_first(paths):
    """Order by the run timestamp, including runs nested under trial/seed folders."""
    return sorted(paths, key=lambda p: (started(p) or datetime.min.replace(tzinfo=timezone.utc), str(p)), reverse=True)


def metadata(run):
    run = Path(run)
    cfg = load(run/'config.json')
    date = started(run)
    configs = [cfg]
    if 'children' in cfg:
        configs = [load(Path(child['run'])/'config.json') for child in cfg['children'].values()]
    if cfg.get('ensemble_members'):
        configs = [load(Path(child['run'])/'config.json') for member in cfg['ensemble_members']
                   for child in load(Path(member['run'])/'config.json')['children'].values()]
    if cfg.get('rank_fusion_members'):
        from .model_profile import leaves
        configs = [load(Path(row['Execucao'])/'config.json') for row in leaves(run)]
    fields = {'Extrator': 'backbone', 'Loss': 'loss', 'Seed': 'seed', 'Epocas limite': 'epochs', 'Learning rate': 'lr',
              'Batch': 'batch', 'Paciencia': 'patience', 'Weight decay': 'weight_decay',
              'Resolucao': 'size', 'Temperatura': 'temperature', 'Margem triplet': 'margin',
              'Enquadramento': 'preprocessing', 'Candidatos de treino': 'negative_sampling'}
    result = {'Data (Sao Paulo)': date.astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M:%S') if date else 'Nao registrada'}
    result['Agregacao'] = 'Mistura RRF de modelos' if cfg.get('rank_fusion_members') else ('Ensemble das seeds' if cfg.get('ensemble_members') else 'Individual')
    if 'weights' in cfg:
        result['Pesos neutra / sorriso / lateral'] = str(cfg['weights'])
        result['Escolha dos pesos'] = 'Busca na validacao' if cfg.get('weight_search') else 'Fixos'
    result['Vistas cranianas (protocolo)'] = 'LEGADO com CE' if uses_ce(run) else 'Sem CE'
    for label, field in fields.items():
        values = list(dict.fromkeys(c.get(field, 'dinov2_vits14' if field == 'backbone' else None) for c in configs))
        result[label] = values[0] if len(values) == 1 else 'Misto: '+str(values)
    criteria = list(dict.fromkeys(c.get('selection_metric', 'mrr') for c in configs))
    result['Criterio do checkpoint'] = criteria[0] if len(criteria) == 1 else str(criteria)
    if (run/'history.json').exists():
        history = load(run/'history.json')
        result['Epocas executadas'] = len(history)
        result['Melhor epoca (MRR)'] = max(history, key=lambda r: r['val_mrr'])['epoch'] if history else None
        if history and configs[0].get('selection_metric') == 'recall_1':
            result['Melhor epoca (Recall@1)'] = max(history, key=lambda r: (r['val_recall_1'], r['val_mrr']))['epoch']
        if history and configs[0].get('selection_metric') == 'recall_10':
            result['Melhor epoca (Recall@10)'] = max(history, key=lambda r: (r['val_recall_10'], r['val_mrr']))['epoch']
        if history and configs[0].get('selection_metric') == 'gallery_recall_1':
            result['Melhor epoca (Top-1 em 10)'] = max(history, key=lambda r: (r['gallery_recall_1'], r['gallery_mrr'], r['gallery_recall_3'], r['gallery_recall_5']))['epoch']
    else:
        result['Epocas executadas'] = None
        result['Melhor epoca (MRR)'] = None
    return result


def comparison_rows(runs, fusion_runs=(), gallery='full', include_smoke=False):
    records = []
    for run in [*runs, *fusion_runs]:
        run = Path(run)
        cfg = load(run/'config.json')
        if cfg.get('smoke') and not include_smoke:
            continue
        combined = (run/'results.json').exists()
        evaluation = load(run/'results.json')['combinado'] if combined else load(run/'evaluation.json')
        metric = evaluation[gallery]
        protocol = digest([cfg['cohort'], sorted(r['person'] for r in evaluation['rows']),
                           gallery, evaluation.get('galleries') if gallery == 'gallery' else None,
                           bool(cfg.get('smoke'))])
        label = (' + '.join(name for name, w in zip(('Neutra','Sorrindo','Lateral'), cfg['weights']) if w)
                 if combined else cfg['experiment'])
        if cfg.get('weight_search'):
            label += ' | pesos ajustados na validacao'
        if cfg.get('ensemble_members'):
            label += ' | ensemble das seeds'
        if cfg.get('rank_fusion_members'):
            label = 'Mistura RRF de modelos'
        ident = run.parent.name+'/'+run.name
        row = dict(metadata(run), Execucao=ident, Modelo=label, Tipo='Combinacao' if combined else 'Individual',
                   Protocolo=protocol[:12], Consultas=metric['n'],
                   Candidatos=metric['n'] if gallery=='full' else evaluation['gallery_size'],
                   MRR=metric['mrr'], **{f'Recall@{k} (%)':100*metric[f'recall_{k}'] for k in (1,3,5,10)})
        records.append((started(run) or datetime.min.replace(tzinfo=timezone.utc), protocol, row))
    records.sort(key=lambda r: (r[0], r[2]['Execucao']))
    best, previous, best_top1, previous_top1 = {}, {}, {}, {}
    for _, group, row in records:
        if group not in best or row['MRR'] > best[group]['MRR']:
            best[group] = row
        if group not in best_top1 or (row['Recall@1 (%)'], row['MRR']) > (best_top1[group]['Recall@1 (%)'], best_top1[group]['MRR']):
            best_top1[group] = row
    for _, batch in groupby(records, key=lambda record: record[0]):
        batch = list(batch)
        for _, group, row in batch:
            top = best_top1[group]
            row['Melhor atual (Recall@1)'] = top['Execucao']
            row['Diferenca Recall@1 vs lider (p.p.)'] = row['Recall@1 (%)']-top['Recall@1 (%)']
            prior_top = previous_top1.get(group)
            top_delta = row['Recall@1 (%)']-prior_top['Recall@1 (%)'] if prior_top else None
            row['Ganho Recall@1 vs melhor anterior (p.p.)'] = top_delta
            row['Evolucao Top-1'] = ('Primeira referencia' if top_delta is None else
                                   'Melhorou' if top_delta > 1e-10 else 'Piorou' if top_delta < -1e-10 else 'Empatou')
            reference = best[group]
            row['Melhor atual (MRR)'] = reference['Execucao']
            row['Diferenca MRR vs melhor atual'] = row['MRR']-reference['MRR']
            for k in (1,3,5,10):
                row[f'Diferenca Recall@{k} vs melhor atual (p.p.)'] = row[f'Recall@{k} (%)']-reference[f'Recall@{k} (%)']
            prior = previous.get(group)
            row['Melhor anterior'] = prior['Execucao'] if prior else None
            delta = row['MRR']-prior['MRR'] if prior else None
            row['Ganho MRR vs melhor anterior'] = delta
            row['Evolucao'] = ('Primeira referencia' if delta is None else
                               'Melhorou' if delta > 1e-10 else 'Piorou' if delta < -1e-10 else 'Empatou')
        for _, group, row in batch:
            if group not in previous or row['MRR'] > previous[group]['MRR']:
                previous[group] = row
            if group not in previous_top1 or (row['Recall@1 (%)'], row['MRR']) > (previous_top1[group]['Recall@1 (%)'], previous_top1[group]['MRR']):
                previous_top1[group] = row
    return [row for _, _, row in reversed(records)]
