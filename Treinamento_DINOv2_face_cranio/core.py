from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from .portable import Path

import numpy as np
from PIL import Image, ImageOps

from Laboratorio_visual_face_cranio.core import atomic, digest, load, save, sha

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
VERSION = 'dinov2_matching_v1'
VIEWS = ('FFN', 'FFS', 'FL', 'CF', 'CE', 'CD')
# Keep embedding indices stable for historical checkpoints; CE is inactive.
ACTIVE_VIEWS = ('FFN', 'FFS', 'FL', 'CF', 'CD')


def uses_ce(run):
    config = load(Path(run)/'config.json')
    if config.get('rank_fusion_members'):
        return any(uses_ce(m['run']) for m in config['rank_fusion_members'] if m['weight'] > 0)
    if 'children' in config:
        return any(uses_ce(child['run']) for child in config['children'].values())
    return 'CE' in EXPERIMENTS[config['experiment']][1]

EXPERIMENTS = {
    'E0': (['FFN'], ['CF']), 'E1': (['FFN'], ['CF']),
    'E2': (['FFN', 'FFS'], ['CF']), 'E3': (['FFN', 'FL'], ['CF']),
    'E4': (['FFN'], ['CF', 'CE', 'CD']),
    'E5': (['FFN', 'FL'], ['CF', 'CE', 'CD']),
    'E6': (['FFN', 'FFS', 'FL'], ['CF', 'CE', 'CD']),
    'S1': (['FFS'], ['CF']),
    'L1': (['FL'], ['CE', 'CD']),
    'L2': (['FL'], ['CD']),
    'E6D': (['FFN', 'FFS', 'FL'], ['CF', 'CD']),
}
LABELS = {'E0': 'DINOv2 direto', 'E1': 'Frontal neutra', 'E2': 'Adicionar sorriso',
          'E3': 'Adicionar lateral facial', 'E4': 'Adicionar laterais cranianas',
          'E5': 'Laterais face e cranio', 'E6': 'Todas as vistas',
          'S1': 'Sorriso isolado', 'L1': 'Legado: laterais CE+CD',
          'L2': 'Lateral direita FL x CD', 'E6D': 'Todas as vistas sem CE'}


def new_run(kind):
    path = ARTIFACTS / kind / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_stop(stop):
    if stop and Path(stop).exists():
        raise InterruptedError('Parada solicitada; a execucao pode ser retomada.')


def read_image(path):
    path = Path(path)
    with path.open('rb') as f:
        if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('PONTEIRO_LFS')
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB').copy()


def letterbox(im, size=224):
    scale = min(size / im.width, size / im.height)
    w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    canvas = Image.new('RGB', (size, size), (124, 116, 104))
    x, y = (size - w) // 2, (size - h) // 2
    canvas.paste(im.resize((w, h), Image.Resampling.BICUBIC), (x, y))
    return canvas, dict(scale_x=w/im.width, scale_y=h/im.height, offset_x=x, offset_y=y,
                        original_width=im.width, original_height=im.height, size=size)


def csv_write(path, rows):
    out = io.StringIO()
    if rows:
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    atomic(path, out.getvalue().encode('utf-8-sig'))


def np_save(path, **values):
    out = io.BytesIO()
    np.savez_compressed(out, **values)
    atomic(path, out.getvalue())


def checkpoint(path, data):
    import torch
    out = io.BytesIO()
    torch.save(data, out)
    atomic(path, out.getvalue())


def visual_overlay(face, skull, opacity=.5, scale=1., angle=0., dx=0., dy=0., size=640):
    """Inspection-only transform in canvas pixels; never used as a model input."""
    import cv2
    base = letterbox(face, size)[0].convert('RGBA')
    factor = min(size/skull.width, size/skull.height)
    w, h = max(1, round(skull.width*factor)), max(1, round(skull.height*factor))
    layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    layer.paste(skull.resize((w, h), Image.Resampling.BICUBIC).convert('RGBA'), ((size-w)//2, (size-h)//2))
    matrix = cv2.getRotationMatrix2D((size/2, size/2), angle, scale)
    matrix[:, 2] += [dx, dy]
    warped = cv2.warpAffine(np.array(layer), matrix, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    warped[..., 3] = (warped[..., 3].astype(float)*opacity).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(warped)).convert('RGB')
