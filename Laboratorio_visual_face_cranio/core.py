from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from Treinamento_DINOv2_face_cranio.portable import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
LEGACY = REPO / 'Experimentos_hibridos_cranio_face'
VERSION = 'landmarks_lab_v1'
POINT_NAMES = ['glabela proposta', 'base nasal proposta', 'nasal direito',
               'nasal esquerdo', 'ocular lateral direito', 'ocular lateral esquerdo',
               'mento proposto', 'zigomatico direito', 'zigomatico esquerdo',
               'ocular medial direito', 'ocular medial esquerdo']
GROUPS = {'olho_direito': [5, 10], 'olho_esquerdo': [6, 11],
          'nariz': [2, 3, 4], 'central': [1, 2, 3, 4, 5, 6],
          'mandibula_queixo': [7, 8, 9]}
MESH_MAP = [9, 2, 64, 294, 33, 263, 152, 234, 454, 133, 362]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save(path, data):
    atomic(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode())


def load(path):
    return json.loads(Path(path).read_text())


def image(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('ARQUIVO_AUSENTE: ' + str(path))
    with path.open('rb') as f:
        if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('LFS_NAO_BAIXADO: ' + str(path))
    im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('IMAGEM_ILEGIVEL: ' + str(path))
    return im


def write_image(path, im):
    ok, buf = cv2.imencode(Path(path).suffix, im)
    if not ok:
        raise ValueError('Falha ao codificar imagem')
    atomic(path, buf.tobytes())


def new_run(kind):
    path = ARTIFACTS / kind / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path.mkdir(parents=True)
    return path


def letterbox(im, size=600):
    h, w = im.shape[:2]
    scale = min(size / w, size / h)
    # Warp with one uniform scale, including subpixel offsets, for exact point mapping.
    matrix = np.array([[scale, 0, (size - w * scale) / 2],
                       [0, scale, (size - h * scale) / 2]], dtype=float)
    return cv2.warpAffine(im, matrix, (size, size)), matrix


def transform(points, matrix):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.c_[xy, np.ones(len(xy))] @ np.asarray(matrix).T


def coordinates(points):
    xy = np.full((11, 2), np.nan)
    for p in points:
        n = int(p['numero'])
        if 1 <= n <= 11 and p.get('x') is not None and p.get('y') is not None:
            xy[n - 1] = [p['x'], p['y']]
    return xy


def regions_for(points, width, height):
    xy = coordinates(points)
    result = {}
    for name, required in GROUPS.items():
        p = xy[np.array(required) - 1]
        if not np.isfinite(p).all():
            result[name] = dict(status='REGIAO_INDISPONIVEL', required=required)
            continue
        low, high = p.min(0), p.max(0)
        margin = np.maximum((high - low) * .2, 4)
        raw = np.r_[low - margin, high + margin]
        box = np.clip(raw, [0, 0, 0, 0], [width, height, width, height])
        if (box[2:] <= box[:2]).any():
            result[name] = dict(status='REGIAO_INDISPONIVEL', required=required)
            continue
        result[name] = dict(status='PROPOSTA', required=required, box=box.tolist(),
                            margin=margin.tolist(),
                            outside_fraction=float(1 - np.prod(box[2:] - box[:2]) /
                                                   np.prod(raw[2:] - raw[:2])))
    return result


def iou(a, b):
    a, b = np.array(a), np.array(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - inter
    return float(inter / union) if union else 0.


def overlay(im, points=(), regions=None, contours=(), names=False, point_color=(0, 240, 240)):
    out = im.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(w, h) / 500))
    for contour in contours:
        cv2.polylines(out, [np.asarray(contour, np.int32)], False, (0, 180, 255), thickness)
    for name, region in (regions or {}).items():
        if 'box' not in region:
            continue
        x1, y1, x2, y2 = map(round, region['box'])
        cv2.rectangle(out, (x1, y1), (x2, y2), (210, 100, 220), thickness)
    for p in points:
        if p.get('x') is None or p.get('y') is None:
            continue
        x, y = round(p['x']), round(p['y'])
        cv2.circle(out, (x, y), thickness * 3, point_color, thickness)
        label = str(p['numero'])
        if names:
            label += ' ' + POINT_NAMES[int(p['numero']) - 1]
        cv2.putText(out, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.35, thickness * .35), point_color, thickness)
    return out


def view_frame(im, region=None):
    if region and 'box' in region:
        x1, y1, x2, y2 = region['box']
        # Square context is only a viewport; analytical boxes stay unchanged.
        side = max(x2 - x1, y2 - y1, 16) * 1.3
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        h, w = im.shape[:2]
        im = im[max(0, int(cy-side/2)):min(h, int(cy+side/2)+1),
                max(0, int(cx-side/2)):min(w, int(cx+side/2)+1)]
    return letterbox(im)[0]
