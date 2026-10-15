"""Experimental lower-jaw audit. Scores are evidence scores, not probabilities."""
import cv2
import numpy as np

from .cranios import detect_geometry_v3, POINT_COLORS


def mandibular_arc(mask, box, axis):
    """Fit the bilateral lower contour away from a central pedestal.

    A smooth convex arc must be supported on both sides. A fit alone is not
    sufficient: the predicted center must touch the observed foreground.
    """
    x0, y0, x1, y1 = box
    bw, bh = x1-x0+1, y1-y0+1
    samples = []
    for x in range(max(0, x0), min(mask.shape[1], x1+1)):
        dx = (x-axis)/bw
        if not .12 <= abs(dx) <= .30:
            continue
        ys = np.flatnonzero(mask[:, x] > 0)
        ys = ys[(ys >= y0+.70*bh) & (ys <= y1)]
        if len(ys):
            samples.append((dx, (ys[-1]-y0)/bh))
    if len(samples) < 12:
        return None, 'CONTORNO_INFERIOR_INSUFICIENTE'
    data = np.asarray(samples)
    if min((data[:, 0] < 0).sum(), (data[:, 0] > 0).sum()) < 6:
        return None, 'ARCO_UNILATERAL'
    keep = np.ones(len(data), dtype=bool)
    for _ in range(3):
        if keep.sum() < 12:
            return None, 'ARCO_INSTAVEL'
        coefficients = np.polyfit(data[keep, 0], data[keep, 1], 2)
        residual = np.abs(np.polyval(coefficients, data[:, 0])-data[:, 1])
        keep = residual <= max(.012, 2.5*np.median(residual[keep]))
    a, b, c = coefficients
    if not -5 < a < -.12:
        return None, 'SEM_CURVATURA_MANDIBULAR'
    center = -b/(2*a)
    predicted = np.polyval(coefficients, center)
    residual = float(np.median(residual[keep]))
    if abs(center) > .10 or not .78 <= predicted <= 1.01 or residual > .025:
        return None, 'ARCO_INCOMPATIVEL'
    x, y = float(axis+center*bw), float(y0+predicted*bh)
    if y >= mask.shape[0]-4:
        return None, 'BASE_TRUNCADA'
    # Foreground must exist above the proposal; gaps imply an absent jaw.
    xa, xb = max(0, int(x-.025*bw)), min(mask.shape[1], int(x+.025*bw)+1)
    ya, yb = max(0, int(y-.035*bh)), min(mask.shape[0], int(y)+1)
    if not mask[ya:yb, xa:xb].size or np.mean(mask[ya:yb, xa:xb] > 0) < .7:
        return None, 'SEM_EVIDENCIA_NO_MENTO'
    support = float(keep.mean())
    score = float(np.clip(support*(1-residual/.03), 0, 1))
    return {'x': x, 'y': y, 'score': score, 'residuo_arco_norm': residual,
            'suporte_arco': support, 'coefficients': coefficients.tolist()}, ''


def detect_geometry_v4(image, prior=None):
    metrics, points, mask, _ = detect_geometry_v3(image, prior)
    metrics = dict(metrics, detector_versao='v4_mento_experimental', validacao_anatomica='PENDENTE')
    points = [dict(p) for p in points]
    if not points:
        return metrics, points, mask, image.copy()
    box = tuple(int(metrics[k]) for k in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
    x0, y0, bw, bh = box
    arc, reason = mandibular_arc(mask, (x0, y0, x0+bw-1, y0+bh-1), metrics['eixo_simetria_x'])
    for point in points:
        point['detector_versao'] = metrics['detector_versao']
        point['status_ponto'] = 'LEGADO_NAO_VALIDADO'
        if point['ponto'].startswith(('gonio', 'zigio')):
            point.update(confianca=0., status_ponto='APROXIMACAO_NAO_VALIDADA')
        if point['ponto'] != 'mento':
            continue
        point.update(metodo='arco_mandibular_bilateral_v4', visivel=False,
                     x_px=np.nan, y_px=np.nan, x_norm=np.nan, y_norm=np.nan,
                     confianca=0., motivo_ausencia=reason or 'REVISAO_VISUAL_NECESSARIA', status_ponto='INCERTO')
        if arc:
            point.update(x_px=arc['x'], y_px=arc['y'], x_norm=(arc['x']-x0)/bw,
                         y_norm=(arc['y']-y0)/bh, visivel=True, confianca=arc['score'],
                         suporte_arco=arc['suporte_arco'], residuo_arco_norm=arc['residuo_arco_norm'],
                         status_ponto='PROPOSTA_PARA_REVISAO')
    metrics['mento_motivo'] = reason or 'PROPOSTA_PARA_REVISAO'
    metrics['orbita_mento_norm'] = np.nan
    metrics['largura_mandibular_norm'] = np.nan
    metrics['largura_zigomatica_norm'] = np.nan
    metrics['status_cranio_geo'] = 'REVISAO'
    metrics['regioes_centrais_detectadas'] = sum(p['visivel'] and p['confianca'] >= .45 for p in points if p['ponto'] in ['orbita_direita', 'orbita_esquerda', 'nasal', 'mento'])
    preview = image.copy()
    for point in points:
        if point['visivel'] and np.isfinite(point['x_px']):
            xy = tuple(int(round(point[k])) for k in ['x_px', 'y_px'])
            color = POINT_COLORS[point['ponto']] if point['ponto']=='mento' else (150,150,150)
            cv2.circle(preview, xy, max(4, min(image.shape[:2])//130), color, -1)
            cv2.putText(preview, point['ponto'], (xy[0]+5,xy[1]-6), cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return metrics, points, mask, preview
