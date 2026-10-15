"""Run provenance and descriptive comparisons within identical protocols."""
from datetime import datetime, timezone
from pathlib import Path
from itertools import groupby
from zoneinfo import ZoneInfo

from .core import digest, load, uses_ce


def started(run):
    for part in (Path(run).name, Path(run).parent.name):
        try:
            return datetime.strptime(part, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def metadata(run):
    run = Path(run)
    cfg = load(run/'config.json')
    date = started(run)
    configs = [cfg]
    if 'children' in cfg:
        configs = [load(Path(child['run'])/'config.json') for child in cfg['children'].values()]
    fields = {'Extrator': 'backbone', 'Loss': 'loss', 'Seed': 'seed', 'Epocas limite': 'epochs', 'Learning rate': 'lr',
              'Batch': 'batch', 'Paciencia': 'patience', 'Weight decay': 'weight_decay',
              'Resolucao': 'size', 'Temperatura': 'temperature', 'Margem triplet': 'margin'}
    result = {'Data (Sao Paulo)': date.astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M:%S') if date else 'Nao registrada'}
    result['Vistas cranianas (protocolo)'] = 'LEGADO com CE' if uses_ce(run) else 'Sem CE'
    for label, field in fields.items():
        values = list(dict.fromkeys(c.get(field, 'dinov2_vits14' if field == 'backbone' else None) for c in configs))
        result[label] = values[0] if len(values) == 1 else 'Misto: '+str(values)
    if (run/'history.json').exists():
        history = load(run/'history.json')
        result['Epocas executadas'] = len(history)
        result['Melhor epoca (MRR)'] = max(history, key=lambda r: r['val_mrr'])['epoch'] if history else None
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
        ident = run.parent.name+'/'+run.name
        row = dict(metadata(run), Execucao=ident, Modelo=label, Tipo='Combinacao' if combined else 'Individual',
                   Protocolo=protocol[:12], Consultas=metric['n'],
                   Candidatos=metric['n'] if gallery=='full' else evaluation['gallery_size'],
                   MRR=metric['mrr'], **{f'Recall@{k} (%)':100*metric[f'recall_{k}'] for k in (1,3,5,10)})
        records.append((started(run) or datetime.min.replace(tzinfo=timezone.utc), protocol, row))
    records.sort(key=lambda r: (r[0], r[2]['Execucao']))
    best, previous = {}, {}
    for _, group, row in records:
        if group not in best or row['MRR'] > best[group]['MRR']:
            best[group] = row
    for _, batch in groupby(records, key=lambda record: record[0]):
        batch = list(batch)
        for _, group, row in batch:
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
    return [row for _, _, row in reversed(records)]
