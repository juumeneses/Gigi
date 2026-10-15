from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .core import (ARTIFACTS, LEGACY, REPO, VERSION, coordinates, digest, image,
                   load, new_run, save, sha)

VIEWS = {'FFN': 'face', 'FFS': 'face', 'FL': 'face', 'CF': 'cranio',
         'CE': 'cranio', 'CD': 'cranio', 'PF': 'superficie',
         'PE': 'superficie', 'PD': 'superficie'}


def inventory(root=None):
    root = Path(root or REPO / 'Banco')
    rows = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            continue
        match = re.search(r'_(FFN|FFS|FL|CF|CE|CD|PF|PE|PD)$', p.stem, re.I)
        view = match.group(1).upper() if match else 'INDEFINIDA'
        relative = str(p.relative_to(root))
        parts = p.relative_to(root).parts
        rows.append(dict(key=digest(relative)[:20], path=str(p.resolve()), filename=p.name,
                         person=p.parent.name, year=parts[0], view=view,
                         modality=VIEWS.get(view, 'indefinida')))
    return rows


def legacy_sources():
    return sorted((LEGACY / 'resultados/piloto_regioes').glob('*/entradas.json'))


def prepare(source, log=print):
    source = Path(source).resolve()
    entries = load(source)
    bank = inventory()
    by_person = {}
    for row in bank:
        by_person.setdefault((row['person'], row['view']), []).append(row)
    run = new_run('references')
    samples, failures = [], []
    split_path = LEGACY / 'dataset/splits/split_identidades.csv'
    with split_path.open(encoding='utf-8-sig') as f:
        splits = {r['id_pessoa']: r['split'] for r in csv.DictReader(f)}
    for i, entry in enumerate(entries):
        person, modality = entry['id'], entry['modalidade']
        view = 'FFN' if modality == 'face' else 'CF'
        record = dict(person=person, modality=modality, view=view, year=str(entry['ano']))
        try:
            expected = entry['expected_hash']
            candidates = by_person.get((person, view), [])
            matches = [r for r in candidates if sha(r['path']) == expected]
            original = Path(entry['path'])
            if not original.is_absolute():
                original = REPO / original
            if matches:
                path = Path(matches[0]['path'])
                origin = 'Banco: hash identico'
                aliases = [r['key'] for r in matches]
            elif original.is_file() and sha(original) == expected:
                path, origin, aliases = original, 'Referencia legada: sem copia identica no Banco', []
            else:
                raise ValueError('Original da anotacao ausente ou hash divergente')
            im = image(path)
            points = entry['proposals']
            numbers = [int(p['numero']) for p in points]
            if sorted(numbers) != list(range(1, 12)):
                raise ValueError('Protocolo esperado: 11 numeros unicos')
            xy = coordinates(points)
            mask = np.isfinite(xy).all(1)
            h, w = im.shape[:2]
            if ((xy[mask] < 0) | (xy[mask] >= [w, h])).any():
                raise ValueError('Coordenadas fora do original; transformacao precisa ser revisada')
            split = splits.get(person)
            if split not in ('treino', 'validacao', 'teste') or split != entry.get('split'):
                raise ValueError('Split ausente ou divergente da fonte congelada')
            legacy_regions = source.parent / (person + '__' + modality) / 'regioes/data.json'
            regions = load(legacy_regions).get('regions', {}) if legacy_regions.exists() else {}
            samples.append(dict(**record, key=digest([person, modality, view, expected])[:20],
                                path=str(path), hash=expected, origin=origin, bank_keys=aliases,
                                width=w, height=h, points=points, split=split,
                                regions=regions, protocol='propostas_ia_11_v1',
                                annotation_origin='PROPOSTA_IA',
                                source=str(source), source_hash=sha(source),
                                anatomical_validation=False))
            log(f'[{i+1}/{len(entries)}] {person} {modality}: {mask.sum()}/11 pontos; {origin}')
        except Exception as exc:
            failures.append(dict(**record, error=str(exc)))
            log(f'[{i+1}/{len(entries)}] ERRO {person} {modality}: {exc}')
    # Preserve exclusions in the imported reference manifest.
    result = dict(version=VERSION, source=str(source), source_hash=sha(source),
                  split_hash=sha(split_path), samples=samples, failures=failures)
    result['fingerprint'] = digest(result)
    save(run / 'references.json', result)
    log(f'Referencias salvas: {run}')
    if not samples:
        raise ValueError('Nenhuma referencia integra; consulte references.json')
    return run


def reference_runs():
    return sorted((ARTIFACTS / 'references').glob('*/references.json'), reverse=True)


def validate_references(path):
    data = load(path)
    token = data.pop('fingerprint')
    if token != digest(data):
        raise ValueError('Manifesto de referencias foi alterado')
    data['fingerprint'] = token
    seen = {}
    hashes = {}
    for row in data['samples']:
        if row['person'] in seen and seen[row['person']] != row['split']:
            raise ValueError('Pessoa em mais de um split')
        seen[row['person']] = row['split']
        if row['hash'] in hashes and hashes[row['hash']] != row['split']:
            raise ValueError('Mesmo arquivo em mais de um split')
        hashes[row['hash']] = row['split']
        if sha(row['path']) != row['hash']:
            raise ValueError('Original alterado: ' + row['path'])
    return data


def training_samples(reference, modality, view):
    data = validate_references(reference)
    rows = [r for r in data['samples'] if r['modality'] == modality and r['view'] == view]
    train = [r for r in rows if r['split'] == 'treino']
    val = [r for r in rows if r['split'] == 'validacao']
    if not train or not val:
        raise ValueError(f'{modality}/{view}: faltam referencias de treino e/ou validacao desta vista')
    masks = [np.isfinite(coordinates(r['points'])).all(1) for r in train]
    supported = np.any(masks, axis=0)
    if not supported.any():
        raise ValueError('Nenhum ponto disponivel no treino')
    if not any((np.isfinite(coordinates(r['points'])).all(1) & supported).any() for r in val):
        raise ValueError('Validacao sem pontos avaliaveis')
    return data, train, val, supported
