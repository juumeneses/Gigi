"""Display metadata for individual matching queries, not population estimates."""
from zoneinfo import ZoneInfo

from .core import load
from .reporting import started


def query_summary(result, ranking=None):
    ranking = result['ranking'] if ranking is None else ranking
    valid = [r for r in ranking if r.get('score') is not None]
    correct = next((r for r in valid if r['person'] == result['query']['person']), None)
    known = any(c['person'] == result['query']['person'] for c in result['candidates'])
    position = correct.get('rank') if correct else None
    query = result['query']
    identifier = query['person']
    year = query.get('year') or query.get('ano')
    if not year:
        prefix = identifier.split('_')[0]
        year = prefix if len(prefix) == 4 and prefix.isdigit() else 'Nao informado'
    return {'Pessoa de referencia': identifier, 'Ano': str(year),
            'Primeiro colocado': valid[0]['person'] if valid else 'Sem score',
            'Score do primeiro': valid[0]['score'] if valid else None,
            'Posicao do correto': position,
            'Referencia conhecida': known,
            **{f'Top-{k}': ('Sim' if position and position <= k else 'Nao') if known else 'Nao avaliavel'
               for k in (1, 3, 5, 10)},
            'Candidatos com score': len(valid)}


def history_label(path):
    result = load(path)
    summary = query_summary(result)
    date = started(path.parent)
    when = date.astimezone(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y %H:%M:%S') if date else path.parent.name
    position = summary['Posicao do correto'] or 'sem score'
    outcome = f'correto em {position} | Top-1: {summary["Top-1"]}' if summary['Referencia conhecida'] else 'sem referencia conhecida'
    return f'{when} | {summary["Pessoa de referencia"]} | {len(result["candidates"])} cranios | {outcome}'
