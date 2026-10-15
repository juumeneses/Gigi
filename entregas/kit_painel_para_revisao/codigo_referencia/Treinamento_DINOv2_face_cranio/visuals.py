"""Cached, pair-specific occlusion maps; never anatomical annotations."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import engine
from .core import ARTIFACTS, check_stop, digest, load, new_run, save, sha


def signature(run, query, candidate):
    run = Path(run)
    return digest(dict(run=str(run), checkpoint=sha(run/'best.pt'),
                       config=sha(run/'config.json'), algorithm=sha(engine.__file__),
                       inputs=[{v: sha(r['path']) for v, r in p['views'].items()}
                               for p in (query, candidate)], query=query, candidate=candidate))


def cached_map(run, query, candidate, stop=None):
    key = signature(run, query, candidate)
    for path in sorted((ARTIFACTS/'explanations').glob('*/inputs.json'), reverse=True):
        if load(path).get('signature') == key and (path.parent/'result.json').exists():
            return path.parent
    target = new_run('explanations')
    save(target/'inputs.json', dict(run=str(run), query=query, candidate=candidate, signature=key))
    save(target/'result.json', engine.explain(run, query, candidate, stop=stop))
    return target


def overlay(image, values):
    """Keep zero-effect blocks unchanged; label every block by row/column."""
    heat = np.asarray(values, dtype=float)
    if heat.ndim != 2 or not np.isfinite(heat).all():
        raise ValueError('Mapa de sensibilidade invalido')
    canvas = image.convert('RGB').copy()
    layer = Image.new('RGBA', canvas.size)
    draw = ImageDraw.Draw(layer)
    scale = max(float(np.abs(heat).max()), 1e-8)
    for y, row in enumerate(heat):
        for x, value in enumerate(row):
            box = (x*canvas.width//heat.shape[1], y*canvas.height//heat.shape[0],
                   (x+1)*canvas.width//heat.shape[1]-1, (y+1)*canvas.height//heat.shape[0]-1)
            color = (220, 45, 45) if value > 0 else (35, 110, 235)
            draw.rectangle(box, fill=(*color, round(150*abs(value)/scale)), outline=(255,255,255,160))
            draw.text((box[0]+3, box[1]+3), f'{y+1}.{x+1}', fill='white', stroke_width=1, stroke_fill='black')
    return Image.alpha_composite(canvas.convert('RGBA'), layer).convert('RGB')


def visual_report(run, stop=None):
    """Two validation examples per model: first hit and first miss when present."""
    run = Path(run)
    if (run/'results.json').exists():
        children = [Path(v['run']) for v in load(run/'config.json')['children'].values()]
    elif (run/'evaluation.json').exists():
        children = [run]
    else:
        children = sorted(p.parent for p in run.glob('*/evaluation.json'))
    records = []
    for child in children:
        rows = load(child/'evaluation.json')['rows']
        people = {p['person']: p for p in load(child/'cohort.json')['people']}
        examples = [next((r for r in rows if r['rank'] == 1), None),
                    next((r for r in rows if r['rank'] is not None and r['rank'] != 1), None)]
        for row in (r for r in examples if r):
            check_stop(stop)
            try:
                print('Exemplo visual: '+child.name+' '+row['person'], flush=True)
                path = cached_map(child, people[row['person']], people[row['top_candidate']], stop)
                records.append(dict(model=str(child), person=row['person'], correct=row['rank']==1, path=str(path)))
            except InterruptedError:
                raise
            except (ValueError, OSError, RuntimeError) as exc:
                records.append(dict(model=str(child), person=row['person'], error=str(exc)))
            save(child/'visual_examples.json', [r for r in records if r['model'] == str(child)])
            save(run/'visual_examples.json', records)
    save(run/'visual_examples.json', records)
    return run


def matching_maps(output, combined=False, stop=None):
    result = load(Path(output)/'result.json')
    best = next((r for r in result['ranking'] if r['score'] is not None), None)
    if best is None:
        return
    candidate = next(c for c in result['candidates'] if c['person'] == best['person'])
    run = Path(result['model'])
    runs = [run]
    if combined:
        cfg = load(run/'config.json')
        runs = [Path(cfg['children'][name]['run']) for name, weight in
                zip(('frontal','sorriso','lateral'), result['weights']) if weight > 0]
    for child in runs:
        cached_map(child, result['query'], candidate, stop)
