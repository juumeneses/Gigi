"""Versioned proposals and explicit per-point review, separate from training labels."""
from pathlib import Path
import csv
import random
import copy
import math

from .core import LEGACY, coordinates, digest, image, load, new_run, regions_for, save, sha
from .data import inventory, validate_references


def splits():
    with (LEGACY/'dataset/splits/split_identidades.csv').open(encoding='utf-8-sig') as f:
        return {r['id_pessoa']: r['split'] for r in csv.DictReader(f)}


def paired(rows, ledger=None):
    ledger = ledger if ledger is not None else splits()
    result = {}
    for row in rows:
        if row['view'] not in ('FFN', 'FFS', 'FL', 'CF', 'CE', 'CD'):
            continue
        item = result.setdefault(row['person'], dict(person=row['person'], year=row['year'],
                                                     split=ledger.get(row['person']), views={}))
        item['views'].setdefault(row['view'], row)
    return sorted([r for r in result.values() if 'FFN' in r['views'] and 'CF' in r['views']],
                  key=lambda r: r['person'])


def select_batch(rows, known=(), count=50, seed=42, ledger=None):
    eligible = [r for r in paired(rows, ledger) if r['split'] in ('treino', 'validacao')]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    eligible.sort(key=lambda r: r['person'] in known)
    # Round-robin years keeps the first batch from being dominated by one year.
    years = sorted({r['year'] for r in eligible})
    pools = {y: [r for r in eligible if r['year'] == y] for y in years}
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for year in years:
            if pools[year] and len(chosen) < count:
                chosen.append(pools[year].pop(0))
    return chosen


def expand(reference, face_run, skull_run, count=50, seed=42, resume=None, stop=None, log=print):
    from .learning import predictor, predict
    base = validate_references(reference)
    import torch
    torch.set_num_threads(2)
    batch = select_batch(inventory(), {r['person'] for r in base['samples']}, count, seed)
    if len(batch) < count:
        raise ValueError(f'Apenas {len(batch)} pares elegiveis; solicitado {count}')
    models = {'face': predictor(face_run), 'cranio': predictor(skull_run)}
    for modality, (_, state) in models.items():
        expected = 'FFN' if modality == 'face' else 'CF'
        if state['config']['modality'] != modality or state['config']['view'] != expected:
            raise ValueError('Localizador incompativel com a vista frontal')
    config = dict(reference=str(reference), fingerprint=base['fingerprint'], count=count, seed=seed,
                  face_run=str(face_run), skull_run=str(skull_run),
                  inputs_hash=digest([{v: {'path': r['path'], 'hash': sha(r['path'])} for v, r in p['views'].items()} for p in batch]),
                  checkpoints={m: sha(Path(p)/'best.pt') for m, p in [('face', face_run), ('cranio', skull_run)]},
                  people=[r['person'] for r in batch])
    run = Path(resume) if resume else new_run('annotation_batches')
    if resume and load(run/'config.json') != config:
        raise ValueError('Lote ou modelos mudaram; crie outra execucao')
    save(run/'config.json', config)
    save(run/'inputs.json', batch)
    for item in batch:
        for modality, view in [('face', 'FFN'), ('cranio', 'CF')]:
            if stop and Path(stop).exists():
                save(run/'status.json', dict(state='INTERROMPIDO'))
                return run
            row = item['views'][view]
            target = run/row['key']/'proposal.json'
            input_hash = sha(row['path'])
            integrity = target.parent/'proposal.sha.json'
            if resume and target.exists() and integrity.exists() and sha(target) == load(integrity)['sha256'] and load(target).get('hash') == input_hash:
                continue
            try:
                im = image(row['path'])
                model, state = models[modality]
                points = predict(im, model, state)
                record = dict(**row, hash=input_hash, split=item['split'], points=points,
                              width=im.shape[1], height=im.shape[0], status='PROPOSTA',
                              regions=regions_for(points, im.shape[1], im.shape[0]),
                              bank_keys=[row['key']], origin='PROPOSTA_MODELO', anatomical_validation=False)
            except Exception as exc:
                record = dict(**row, hash=input_hash, status='ERRO', error=str(exc))
            save(target, record)
            save(integrity, dict(sha256=sha(target)))
            log(f'{item["person"]} {view}: {record["status"]}')
    save(run/'status.json', dict(state='CONCLUIDO'))
    return run


def confirm_image_points(points):
    """One explicit image approval confirms unchanged finite proposals too."""
    confirmed = copy.deepcopy(points)
    for point in confirmed:
        if point['visibility'] != 'NAO_AVALIADO':
            continue
        xy = [point.get('x'), point.get('y')]
        if all(v is not None and math.isfinite(v) for v in xy):
            point['visibility'] = 'VISIVEL'
        else:
            point.update(x=None, y=None, visibility='INDETERMINADO',
                         reason='Sem coordenada ao confirmar a imagem; ponto omitido do treinamento')
    return confirmed


def current_reviews(batch):
    """Only reviews tied to current proposal and original bytes count."""
    batch = Path(batch).resolve()
    latest = {}
    for path in sorted((batch.parents[1]/'point_reviews').glob('*/review.json')):
        data = load(path)
        proposal = Path(data['proposal'])
        if proposal.parent.parent.resolve() != batch or not proposal.is_file():
            continue
        row = load(proposal)
        if data['proposal_hash'] == sha(proposal) and data['image_hash'] == row['hash'] and sha(row['path']) == row['hash']:
            latest[str(proposal)] = (path, data)
    return latest


def review(proposal_path, points, author, reason=''):
    import numpy as np
    proposal_path = Path(proposal_path)
    row = load(proposal_path)
    if not author.strip() or row['hash'] != sha(row['path']):
        raise ValueError('Informe autor e use a imagem original vinculada')
    if sorted(p['numero'] for p in points) != list(range(1, 12)):
        raise ValueError('Informe os 11 pontos uma unica vez')
    for p in points:
        if p['visibility'] not in ('VISIVEL', 'OCULTO', 'CORTADO', 'INDETERMINADO'):
            raise ValueError('Revise a visibilidade de todos os pontos antes de aprovar')
        if p['visibility'] != 'VISIVEL':
            p.update(x=None, y=None, estado=p['visibility'])
        elif p['x'] is None or p['y'] is None or not np.isfinite([p['x'], p['y']]).all() or not (0 <= p['x'] < row['width'] and 0 <= p['y'] < row['height']):
            raise ValueError(f'Coordenada invalida no ponto {p["numero"]}')
        else:
            p['estado'] = 'REVISADO_USUARIO'
    target = new_run('point_reviews')/'review.json'
    save(target, dict(proposal=str(proposal_path), proposal_hash=sha(proposal_path),
                      image_hash=row['hash'], points=points, author=author.strip(), reason=reason))
    return target


def promote(reference, batch, log=print):
    base = validate_references(reference)
    batch = Path(batch)
    reviews = sorted((batch.parents[1]/'point_reviews').glob('*/review.json'))
    latest = {load(p)['proposal']: (p, load(p)) for p in reviews}
    samples = {r['key']: r for r in base['samples']}
    accepted = 0
    for proposal in sorted(batch.glob('*/proposal.json')):
        if str(proposal) not in latest:
            continue
        review_path, decision = latest[str(proposal)]
        row = load(proposal)
        if row['status'] != 'PROPOSTA' or decision['proposal_hash'] != sha(proposal) or decision['image_hash'] != sha(row['path']):
            raise ValueError('Revisao antiga ou imagem alterada: ' + str(proposal))
        if row['split'] not in ('treino', 'validacao'):
            raise ValueError('Teste final nao pode ser promovido neste lote')
        row.update(points=decision['points'], annotation_origin='REVISADO_USUARIO',
                   source=str(review_path), source_hash=sha(review_path), protocol='propostas_ia_11_v1',
                   regions=regions_for(decision['points'], row['width'], row['height']))
        for key in list(samples):
            if samples[key]['person'] == row['person'] and samples[key]['view'] == row['view']:
                del samples[key]
        samples[row['key']] = row
        accepted += 1
    if not accepted:
        raise ValueError('Nenhuma imagem com revisao de pontos completa neste lote')
    result = {k: v for k, v in base.items() if k not in ('fingerprint', 'samples')}
    result.update(samples=list(samples.values()), parent_reference=str(reference), batch=str(batch))
    result['fingerprint'] = digest(result)
    run = new_run('references')
    save(run/'references.json', result)
    validate_references(run/'references.json')
    log(f'{accepted} imagens revisadas incorporadas em {run}')
    return run
