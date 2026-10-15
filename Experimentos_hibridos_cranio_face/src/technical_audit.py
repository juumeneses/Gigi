"""Read-only integrity checks; never scores identity or anatomical accuracy."""
import json

import cv2
import numpy as np

from .common import REPO_ROOT, sha256_file
from .region_pilot import image_input, key, verified


def missing_requirements(extraction, box):
    """Report missing inputs, without inferring visibility or coordinates."""
    skull = extraction.get('item', {}).get('modalidade') == 'cranio'
    points = extraction.get('proposals' if skull else 'points', [])
    field = 'numero' if skull else 'idx'
    available = set()
    for point in points:
        try:
            xy = np.asarray([point.get('x'), point.get('y')], dtype=float)
            if np.isfinite(xy).all():
                available.add(point[field])
        except (ValueError, TypeError, KeyError):
            continue
    return [idx for idx in box.get('required', []) if idx not in available]


def inspect_geometry(extraction, regions, shape, folder):
    height, width = shape[:2]
    rows = []

    def add(region, check, status, reason):
        rows.append(dict(regiao=region, verificacao=check, status=status, motivo=reason))

    for group in ('points', 'selected', 'proposals'):
        if extraction.get('item', {}).get('modalidade') == 'cranio' and group != 'proposals':
            continue
        points = extraction.get(group, [])
        absent = invalid = outside = 0
        for p in points:
            if p.get('x') is None or p.get('y') is None:
                absent += 1
                continue
            try:
                xy = np.asarray([p['x'], p['y']], dtype=float)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if xy.shape != (2,) or not np.isfinite(xy).all():
                invalid += 1
            elif not (0 <= xy[0] < width and 0 <= xy[1] < height):
                outside += 1
        add(group, 'coordenadas', 'ERRO' if invalid else 'ATENCAO' if absent or outside or not points else 'OK_TECNICO',
            f'{len(points)} registros; {absent} ausentes; {invalid} invalidos; {outside} fora da imagem')

    for name, box in regions.get('regions', {}).items():
        if 'crop_bounds' not in box:
            add(name, 'recorte', 'ATENCAO', box.get('reason', 'REGIAO_INDISPONIVEL'))
            rows[-1]['pontos_ausentes'] = ', '.join(map(str, missing_requirements(extraction, box)))
            rows[-1]['proxima_acao'] = 'Revisar visibilidade no original; nao preencher coordenadas por suposicao'
            continue
        try:
            bounds = np.asarray(box['crop_bounds'], dtype=float)
            assert bounds.shape == (4,) and np.isfinite(bounds).all()
            l, t, r, b = bounds
            assert 0 <= l < r <= width and 0 <= t < b <= height
            tr = box['transform']
            scale = float(tr['scale'])
            offset = np.asarray(tr['offset'], dtype=float)
            assert np.isfinite(scale) and scale > 0
            assert offset.shape == (2,) and np.isfinite(offset).all()
            assert np.allclose(tr['input_size'], [r-l, b-t])
            saved = cv2.imread(str(folder / box['arquivo']))
            assert saved is not None
            sh, sw = saved.shape[:2]
            size = np.asarray(tr['size'], dtype=float)
            assert size.shape == (2,) and np.isfinite(size).all() and (size > 0).all()
            assert np.all(np.abs(size - np.array([r-l, b-t])*scale) <= 1)
            assert (offset >= 0).all() and np.all(offset+size <= [sw, sh])
            xy = np.array([[l, t], [r, b]])
            mapped = (xy-[l, t])*scale+offset
            error = float(np.max(np.abs((mapped-offset)/scale+[l, t]-xy)))
            assert error < 1e-6
            add(name, 'recorte_transformacao', 'OK_TECNICO', f'Ida/volta: {error:.3g} px; arquivo {sw}x{sh}')
        except (AssertionError, KeyError, ValueError, TypeError, OSError):
            add(name, 'recorte_transformacao', 'ERRO', 'Limites, arquivo ou transformacao inconsistente')
    return rows


def audit_run(run):
    rows = []
    for item in json.loads((run/'entradas.json').read_text()):
        prefix = dict(id=item['id'], modalidade=item['modalidade'])
        folder = run/key(item)
        path = REPO_ROOT/item['path']
        image, status = image_input(path)
        reason = status
        if image is not None and sha256_file(path) != item['expected_hash']:
            reason = 'ORIGINAL_MODIFICADO'
        rows.append(dict(**prefix, regiao='original', verificacao='integridade',
                         status='OK_TECNICO' if reason == 'OK_TECNICO' else 'ERRO', motivo=reason))
        good = True
        for stage in ('extracao', 'regioes'):
            valid = verified(folder/stage/'data.json')
            good &= valid
            rows.append(dict(**prefix, regiao=stage, verificacao='hash_saida',
                             status='OK_TECNICO' if valid else 'ERRO', motivo='Hash confere' if valid else 'Saida ausente ou alterada'))
        if good and image is not None:
            ex = json.loads((folder/'extracao/data.json').read_text())
            reg = json.loads((folder/'regioes/data.json').read_text())
            rows.extend(dict(**prefix, **row) for row in inspect_geometry(ex, reg, image.shape, folder/'regioes'))
    return rows
