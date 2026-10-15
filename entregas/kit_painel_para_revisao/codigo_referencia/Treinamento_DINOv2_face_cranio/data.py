from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from Laboratorio_visual_face_cranio.data import inventory as scan_bank
from .core import (REPO, VIEWS, VERSION, EXPERIMENTS, check_stop, csv_write, digest,
                   load, new_run, read_image, save, sha)

SPLITS = REPO / 'Experimentos_hibridos_cranio_face/dataset/splits/split_identidades.csv'


def inventory(root=None, resolutions=None, stop=None, log=print):
    root = Path(root or REPO/'Banco').resolve()
    resolutions = {key: str(Path(value).resolve()) for key, value in (resolutions or {}).items()}
    with SPLITS.open(encoding='utf-8-sig') as f:
        ledger = {}
        for row in csv.DictReader(f):
            if row['id_pessoa'] in ledger and ledger[row['id_pessoa']] != row['split']:
                raise ValueError('Split conflitante: ' + row['id_pessoa'])
            ledger[row['id_pessoa']] = row['split']
    rows = [r for r in scan_bank(root) if r['view'] in VIEWS and r['view'] != 'CE']
    run = new_run('inventories')
    groups, hashes = defaultdict(list), defaultdict(set)
    for i, row in enumerate(rows):
        check_stop(stop)
        # Banco also contains view-organized copies (_separados/FFN, CF).
        parent_id = re.fullmatch(r'(\d{4})[ _]+(\d+)[ _]+(.+)', row['person'])
        file_id = re.fullmatch(r'(\d{4})[ _]+(\d+)[ _]+(.+)_(?:FFN|FFS|FL|CF|CE|CD)', Path(row['filename']).stem, re.I)
        identity = parent_id or file_id
        row['identity_origin'] = 'parent' if parent_id else 'filename' if file_id else 'unresolved'
        if identity:
            row['person'] = '_'.join(identity.groups())
        row['split'] = ledger.get(row['person'], 'pendente')
        row['status'], row['error'] = 'OK', ''
        try:
            row['hash'] = sha(row['path'])
            im = read_image(row['path'])
            row['width'], row['height'] = im.size
            hashes[row['hash']].add(row['person'])
        except Exception as exc:
            row['status'], row['error'] = 'ERRO', str(exc)
        groups[(row['person'], row['view'])].append(row)
        if i % 100 == 0:
            log(f'Inventario {i+1}/{len(rows)}', flush=True)
    people = {}
    for (person, view), candidates in groups.items():
        item = people.setdefault(person, dict(person=person, year=candidates[0]['year'],
                                             split=ledger.get(person, 'pendente'), views={}, conflicts={}))
        key = person + '/' + view
        selected = next((r for r in candidates if r['path'] == resolutions.get(key)), None)
        identical = len({r.get('hash') for r in candidates}) == 1 and candidates[0].get('hash')
        if selected is None and (len(candidates) == 1 or identical):
            selected = candidates[0]
        if selected is None:
            item['conflicts'][view] = [r['path'] for r in candidates]
        else:
            item['views'][view] = dict(selected, aliases=[r['path'] for r in candidates],
                                       selection='explicit' if key in resolutions else
                                       'identical_hash' if len(candidates) > 1 else 'unique')
    collisions = {h: sorted(ids) for h, ids in hashes.items() if len(ids) > 1}
    result = dict(version=VERSION, root=str(root), split_hash=sha(SPLITS), rows=rows,
                  people=sorted(people.values(), key=lambda p: p['person']),
                  resolutions=resolutions, identity_hash_collisions=collisions)
    result['fingerprint'] = digest(result)
    save(run/'manifest.json', result)
    csv_write(run/'inventory.csv', rows)
    log(f'Inventario salvo: {run}; {len(rows)} imagens; {len(people)} pessoas', flush=True)
    return run


def manifest(path):
    result = load(path)
    if result['fingerprint'] != digest({k: v for k, v in result.items() if k != 'fingerprint'}):
        raise ValueError('Manifesto alterado')
    return result


def cohort(path, experiment='E1', mode='matched', ids=None):
    source = manifest(path)
    collision_ids = set(p for ids in source['identity_hash_collisions'].values() for p in ids)
    wanted = set(sum(EXPERIMENTS[experiment], [])) if mode == 'matched' else {'FFN', 'CF'}
    people, excluded = [], []
    for person in source['people']:
        if ids is not None and person['person'] not in ids:
            continue
        missing = sorted(wanted - person['views'].keys())
        if person['split'] not in ('treino', 'validacao', 'teste') or missing or person['person'] in collision_ids:
            excluded.append(dict(person=person['person'], split=person['split'], missing=missing,
                                 conflicts=person['conflicts'], reason='IMAGEM_ENTRE_IDENTIDADES' if person['person'] in collision_ids
                                 else 'SPLIT_PENDENTE' if person['split'] not in ('treino', 'validacao', 'teste') else 'VISTA_AUSENTE_OU_AMBIGUA'))
        else:
            # Unreadable files remain in the cohort and evaluation denominators.
            people.append(person)
    signature = digest(dict(people=people, mode=mode, split_hash=source['split_hash']))
    return dict(manifest=str(Path(path).resolve()), manifest_hash=sha(path), people=people,
                excluded=excluded, fingerprint=signature, mode=mode, experiment=experiment)
