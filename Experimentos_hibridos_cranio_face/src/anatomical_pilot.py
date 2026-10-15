"""Unvalidated contour proposals, never training labels or anatomical truth."""
import cv2
import numpy as np

from .cranios import (foreground_mask, _anatomical_bbox, _symmetry_axis,
                      _multi_threshold_candidates, _best_orbit_pair)

VERSION = 'bordas_cf_piloto_v2_nested'
GUIDE = {
    'ec_direito': 'Candidato a ectoconchion: borda lateral da orbita direita no plano transversal medio. Nao e centro orbital.',
    'ec_esquerdo': 'Candidato a ectoconchion: borda lateral da orbita esquerda no plano transversal medio. Nao e centro orbital.',
    'al_direito': 'Candidato a alare osseo: extremo lateral direito da abertura piriforme, nao asa nasal de tecido mole.',
    'al_esquerdo': 'Candidato a alare osseo: extremo lateral esquerdo da abertura piriforme.',
}


def edge_point(candidate, lateral, midline=False):
    contour = candidate['contour'].reshape(-1, 2)
    if midline:
        # Rasterize to include the intersection even on CHAIN_APPROX_SIMPLE edges.
        x, y, w, h = cv2.boundingRect(candidate['contour'])
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [candidate['contour'] - [x, y]], -1, 255, -1)
        row = h // 2
        xs = np.flatnonzero(mask[row])
        if not len(xs):
            return None
        return np.array([x + (xs[0] if lateral == 'left' else xs[-1]), y + row], float)
    value = contour[:, 0].min() if lateral == 'left' else contour[:, 0].max()
    return np.median(contour[contour[:, 0] == value], axis=0)


def propose(image):
    mask = foreground_mask(image)
    # Keep internal cavities in the search ROI; bone-only masking removes them.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    box = _anatomical_bbox(mask)
    axis, _ = _symmetry_axis(image, box) if box else (image.shape[1]/2, 0)
    candidates = _multi_threshold_candidates(image, mask, box, nested=True) if box else []
    pair = _best_orbit_pair(candidates, box, axis) if box else None
    h, w = image.shape[:2]
    box = box or (0, 0, w-1, h-1)
    bw, bh = box[2]-box[0]+1, box[3]-box[1]+1
    points, evidence = [], []
    chosen = {}
    if pair:
        _, right, left = pair
        chosen.update(ec_direito=(right, 'left', True), ec_esquerdo=(left, 'right', True))
        orbit_y = (right['cy']+left['cy'])/2
        pool = [c for c in candidates if abs(c['cx']-axis) < .10*bw
                and orbit_y+.06*bh < c['cy'] < orbit_y+.30*bh
                and .035*bw < c['w'] < .24*bw and .06*bh < c['h'] < .30*bh]
        if pool:
            nasal = max(pool, key=lambda c: c['area'])
            chosen.update(al_direito=(nasal, 'left', False), al_esquerdo=(nasal, 'right', False))
    for name, definition in GUIDE.items():
        row = dict(ponto=name, x_px=None, y_px=None, visivel=False,
                   metodo=VERSION, detector_versao=VERSION, confianca=None,
                   status_ponto='SEM_PROPOSTA', motivo_ausencia='SEM_CONTORNO_ESTAVEL',
                   definicao=definition, referencia='Srisinghasongkram 2026, Tabela I/Figura 1',
                   anotacao_validada=False, lado='anatomico; pressupoe CF nao espelhada')
        if name in chosen:
            c, lateral, midline = chosen[name]
            near = [a for a in candidates if np.hypot(a['cx']-c['cx'], a['cy']-c['cy']) < .035*np.hypot(bw, bh)
                    and .65 < a['w']/c['w'] < 1.5 and .65 < a['h']/c['h'] < 1.5]
            # One vote per threshold prevents duplicate contours inflating stability.
            votes = {}
            for a in sorted(near, key=lambda a: abs(a['area']-c['area']), reverse=True):
                xy = edge_point(a, lateral, midline)
                if xy is not None:
                    votes[a['percentile']] = xy
            if len(votes) >= 3:
                coordinates = np.array(list(votes.values()))
                center = np.median(coordinates, axis=0)
                spread = float(np.max(np.linalg.norm(coordinates-center, axis=1))/np.hypot(bw, bh))
                row.update(limiares_concordantes=len(votes), dispersao_normalizada=spread)
                if spread <= .025:
                    row.update(x_px=float(center[0]), y_px=float(center[1]), visivel=True,
                               x_norm=float(center[0]/w), y_norm=float(center[1]/h),
                               status_ponto='PROPOSTA_NAO_VALIDADA', motivo_ausencia='')
                    evidence.append(c['contour'])
        points.append(row)
    preview = image.copy()
    cv2.drawContours(preview, evidence, -1, (0, 180, 255), 2)
    for point in points:
        if point['visivel']:
            x, y = int(point['x_px']), int(point['y_px'])
            cv2.circle(preview, (x, y), max(5, w//180), (0, 0, 255), -1)
            cv2.putText(preview, point['ponto'], (x+6, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                        max(.5, w/1800), (0, 0, 255), 2, cv2.LINE_AA)
    return points, mask, preview
