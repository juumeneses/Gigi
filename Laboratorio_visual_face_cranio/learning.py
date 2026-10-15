from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch import nn

from .core import (VERSION, atomic, coordinates, digest, image, iou, letterbox, load,
                   new_run, overlay, regions_for, save, sha, transform, write_image)
from .data import training_samples


def network():
    return nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                         nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 64),
                         nn.ReLU(), nn.Linear(64, 22), nn.Sigmoid())


def masked_loss(pred, target, mask):
    if not mask.any():
        raise ValueError('Lote sem pontos supervisionados')
    # Index before arithmetic so NaNs in missing labels never enter the loss.
    return (pred[mask] - target[mask]).square().sum(-1).mean()


def tensor_image(im, size):
    canvas, matrix = letterbox(im, size)
    x = torch.from_numpy(canvas[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255
    return x, matrix


def dataset(rows, size, supported):
    xs, ys, masks = [], [], []
    for row in rows:
        x, matrix = tensor_image(image(row['path']), size)
        points = coordinates(row['points'])
        mask = np.isfinite(points).all(1) & supported
        target = transform(points, matrix) / size
        target[~mask] = 0
        xs.append(x)
        ys.append(torch.tensor(target, dtype=torch.float32))
        masks.append(torch.tensor(mask))
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


def checkpoint_save(path, state):
    buf = io.BytesIO()
    torch.save(state, buf)
    atomic(path, buf.getvalue())
    save(str(path) + '.sha.json', {'sha256': sha(path)})


def checkpoint_load(path):
    path = Path(path)
    if sha(path) != load(str(path) + '.sha.json')['sha256']:
        raise ValueError('Checkpoint corrompido')
    return torch.load(path, map_location='cpu', weights_only=True)


def train(reference, modality, view, epochs=100, lr=.001, batch_size=8, size=128,
          seed=42, resume=None, stop=None, log=print):
    if not (1 <= epochs <= 10000 and 32 <= size <= 512 and batch_size > 0 and lr > 0):
        raise ValueError('Parametros de treino invalidos')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    refs, train_rows, val_rows, supported = training_samples(reference, modality, view)
    config = dict(version=VERSION, architecture='cnn_coordinates_11_v1',
                  modality=modality, view=view, lr=lr, batch_size=batch_size, size=size,
                  seed=seed, reference=str(Path(reference).resolve()),
                  reference_fingerprint=refs['fingerprint'], supported=supported.tolist(),
                  train_ids=[r['person'] for r in train_rows],
                  validation_ids=[r['person'] for r in val_rows],
                  torch_version=str(torch.__version__),
                  selection='menor perda na validacao; referencias IA',
                  anatomical_validation=False)
    run = Path(resume) if resume else new_run('training')
    sink = log
    def log(message):
        with (run / 'execution.log').open('a') as stream:
            stream.write(message + '\n')
        sink(message)
    token = digest(config)
    model = network()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start, best, history = 0, float('inf'), []
    if resume:
        if load(run / 'config.json')['fingerprint'] != token:
            raise ValueError('Retomada exige os mesmos dados, versao e hiperparametros')
        state = checkpoint_load(run / 'last.pt')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, history = state['epoch'], state['best_loss'], state['history']
        if epochs <= start:
            raise ValueError(f'Escolha total de epocas maior que {start}')
    else:
        save(run / 'config.json', dict(config, fingerprint=token))
    xt, yt, mt = dataset(train_rows, size, supported)
    xv, yv, mv = dataset(val_rows, size, supported)
    baseline = (yt * mt[..., None]).sum(0) / mt.sum(0).clamp_min(1)[..., None]
    save(run / 'status.json', dict(state='RODANDO', epoch=start, target_epochs=epochs))
    log(f'RUN {run}; {len(train_rows)} treino, {len(val_rows)} validacao; referencias IA')
    for epoch in range(start, epochs):
        if stop and Path(stop).exists():
            break
        model.train()
        order = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed + epoch))
        total, count = 0., 0
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            if not mt[indices].any():
                continue
            optimizer.zero_grad()
            pred = model(xt[indices]).reshape(-1, 11, 2)
            loss = masked_loss(pred, yt[indices], mt[indices])
            if not torch.isfinite(loss):
                raise ValueError('Perda nao finita')
            loss.backward()
            optimizer.step()
            n = int(mt[indices].sum())
            total += loss.item() * n
            count += n
        model.eval()
        with torch.no_grad():
            val_loss = masked_loss(model(xv).reshape(-1, 11, 2), yv, mv).item()
        improved = val_loss < best
        if improved:
            best = val_loss
        history.append(dict(epoch=epoch + 1, train_loss=total / max(count, 1),
                            val_loss=val_loss))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch + 1, best_loss=best, history=history,
                     config=config, baseline=baseline)
        checkpoint_save(run / 'last.pt', state)
        if improved:
            checkpoint_save(run / 'best.pt', state)
        save(run / 'history.json', history)
        save(run / 'status.json', dict(state='RODANDO', epoch=epoch + 1, target_epochs=epochs))
        log(f'Epoca {epoch+1}/{epochs}: treino={total/max(count,1):.6f}; validacao={val_loss:.6f}')
    completed = bool(history) and history[-1]['epoch'] >= epochs
    save(run / 'status.json', dict(state='CONCLUIDO' if completed else 'INTERROMPIDO',
                                   epoch=history[-1]['epoch'] if history else 0, target_epochs=epochs))
    if (run / 'best.pt').exists():
        evaluate(run, log=log)
    return run


def predictor(run):
    state = checkpoint_load(Path(run) / 'best.pt')
    model = network()
    model.load_state_dict(state['model'])
    model.eval()
    return model, state


def predict(im, model, state):
    config = state['config']
    x, matrix = tensor_image(im, config['size'])
    with torch.no_grad():
        xy = model(x[None]).reshape(11, 2).numpy() * config['size']
    xy = transform(xy, cv2.invertAffineTransform(matrix))
    h, w = im.shape[:2]
    points = []
    for i, (x, y) in enumerate(xy):
        available = config['supported'][i] and 0 <= x < w and 0 <= y < h
        points.append(dict(numero=i+1, x=float(x) if available else None,
                           y=float(y) if available else None,
                           estado='PROPOSTA_MODELO' if available else 'INDISPONIVEL',
                           visibility='NAO_AVALIADO',
                           reason='' if available else 'SEM_SUPERVISAO_OU_FORA_DA_IMAGEM'))
    return points


def evaluate(run, log=print):
    run = Path(run)
    model, state = predictor(run)
    config = state['config']
    refs, _, rows, supported = training_samples(config['reference'], config['modality'], config['view'])
    if refs['fingerprint'] != config['reference_fingerprint']:
        raise ValueError('Referencias diferentes das usadas neste checkpoint')
    per_point, cases, all_errors, baseline_errors, region_scores = [], [], [], [], []
    for row in rows:
        im = image(row['path'])
        points = predict(im, model, state)
        pred, ref = coordinates(points), coordinates(row['points'])
        mask = np.isfinite(ref).all(1) & supported
        _, matrix = letterbox(im, config['size'])
        base = transform(state['baseline'].numpy() * config['size'],
                         cv2.invertAffineTransform(matrix))
        diagonal = float(np.hypot(*im.shape[:2]))
        for i in range(11):
            evaluated = bool(mask[i] and np.isfinite(pred[i]).all())
            error = float(np.linalg.norm(pred[i] - ref[i])) if evaluated else None
            baseline_error = float(np.linalg.norm(base[i] - ref[i])) if mask[i] else None
            per_point.append(dict(person=row['person'], point=i+1, evaluated=evaluated,
                                  error_px=error, error_diagonal=error/diagonal if evaluated else None,
                                  baseline_px=baseline_error))
            if evaluated:
                all_errors.append(error/diagonal)
            if baseline_error is not None:
                baseline_errors.append(baseline_error/diagonal)
        h, w = im.shape[:2]
        regions = regions_for(points, w, h)
        reference_regions = regions_for(row['points'], w, h)
        for name, r in reference_regions.items():
            if 'box' in r:
                score = iou(r['box'], regions[name]['box']) if 'box' in regions[name] else 0.
                region_scores.append(dict(person=row['person'], region=name, iou=score))
        target = run / 'evaluation' / (row['key'] + '.png')
        write_image(target, letterbox(overlay(im, points, regions))[0])
        cases.append(dict(key=row['key'], person=row['person'], path=row['path'],
                          points=points, reference_points=row['points'], preview=str(target),
                          regions=regions, unavailable=int((~np.isfinite(pred).all(1)).sum())))
    result = dict(checkpoint_epoch=state['epoch'], best_val_loss=state['best_loss'],
                  n_validation=len(rows), n_reference_points=sum(p['baseline_px'] is not None for p in per_point),
                  n_evaluated=len(all_errors), failures=sum(p['baseline_px'] is not None and not p['evaluated'] for p in per_point),
                  median_diagonal=float(np.median(all_errors)) if all_errors else None,
                  p95_diagonal=float(np.percentile(all_errors, 95)) if all_errors else None,
                  baseline_median_diagonal=float(np.median(baseline_errors)) if baseline_errors else None,
                  points=per_point, cases=cases, regions=region_scores,
                  interpretation='Discordancia em relacao a pseudorrotulos IA; nao precisao anatomica independente')
    save(run / 'evaluation.json', result)
    log(f'Avaliacao: {len(rows)} pessoa(s), {len(all_errors)} pontos avaliados; epoca {state["epoch"]}')
    return result
