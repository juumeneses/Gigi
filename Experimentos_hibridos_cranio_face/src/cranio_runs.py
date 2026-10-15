"""Versioned, image-level checkpoints for the cranial audit."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .common import LAB_ROOT, abs_repo, read_image, save_csv, sha256_file, write_image, bool_series
from .cranios import detect_geometry_v2, detect_geometry_v3, _orbital_prior


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temporary, path)


def image_error(path):
    if path is None or not path.is_file():
        return 'IMAGEM_AUSENTE'
    with path.open('rb') as stream:
        if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            return 'GIT_LFS_NAO_BAIXADO'
    return ''


def run_resumable(manifest, config, force=False):
    root = LAB_ROOT / 'resultados/cranios'
    root.mkdir(parents=True, exist_ok=True)
    split_path = LAB_ROOT / 'dataset/splits/split_identidades.csv'
    splits = pd.read_csv(split_path)
    train_ids = set(splits.loc[splits.split.eq('treino'), 'id_pessoa'].astype(str))
    if not train_ids:
        raise RuntimeError('Split de treino vazio: prior orbital nao pode usar validacao/teste.')
    inputs = []
    for record in manifest.to_dict('records'):
        path = abs_repo(record.get('imagem_cf'))
        error = image_error(path)
        inputs.append((record, path, error, sha256_file(path) if path and path.is_file() else 'ausente'))
    signature = hashlib.sha256(json.dumps({
        'inputs': [(str(r['id_pessoa']), digest) for r, _, _, digest in inputs],
        'config': config, 'splits': sha256_file(split_path),
        'detector': sha256_file(Path(__file__).with_name('cranios.py')),
        'runner': sha256_file(Path(__file__)),
    }, sort_keys=True).encode()).hexdigest()
    pointer = root / 'execucao_atual.json'
    previous = json.loads(pointer.read_text()) if pointer.exists() else {}
    run_id = previous.get('run_id') if not force and previous.get('signature') == signature else None
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = root / 'runs' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {'run_id': run_id, 'signature': signature, 'config': config, 'status': 'RODANDO'}
    atomic_json(pointer, metadata)
    atomic_json(folder / 'execucao.json', metadata)
    prior_path = folder / 'prior.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
    else:
        source = []
        for record, path, error, _ in inputs:
            if str(record['id_pessoa']) not in train_ids or error:
                continue
            image = read_image(path)
            if image is not None:
                metrics, points, _, _ = detect_geometry_v2(image)
                source.append((metrics, points))
            print(f"[PRIOR TREINO] {record['id_pessoa']}", flush=True)
        prior = _orbital_prior(source)
        atomic_json(prior_path, prior)
    rows, points = [], []
    try:
        for index, (record, path, error, digest) in enumerate(inputs, 1):
            identity = str(record['id_pessoa'])
            key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            item = folder / f'{key}.json'
            preview, mask_path = folder / f'{key}.jpg', folder / f'{key}.png'
            cached = json.loads(item.read_text()) if item.exists() else None
            reusable = cached and cached.get('input_hash') == digest and cached['metrics'].get('status_cranio_geo') != 'FALHA'
            reusable = reusable and read_image(preview) is not None and read_image(mask_path) is not None
            if not reusable:
                image = None if error else read_image(path)
                base = {'ano': record['ano'], 'id_pessoa': identity, 'imagem_cf': record.get('imagem_cf', '')}
                if image is None:
                    metrics, detected = {'status_cranio_geo': 'FALHA', 'motivo_cranio_geo': error or 'IMAGEM_ILEGIVEL', 'regioes_centrais_detectadas': 0}, []
                else:
                    metrics, detected, mask, annotated = detect_geometry_v3(image, prior)
                    if not write_image(preview, annotated) or not write_image(mask_path, mask):
                        raise IOError(f'Falha ao salvar saidas: {identity}')
                cached = {'input_hash': digest, 'metrics': {**base, **metrics,
                          'preview_path': str(preview.relative_to(LAB_ROOT)) if image is not None else '',
                          'mask_path': str(mask_path.relative_to(LAB_ROOT)) if image is not None else '',
                          'run_id': run_id},
                          'points': [{**base, **p} for p in detected]}
                atomic_json(item, cached)
            rows.append(cached['metrics']); points.extend(cached['points'])
            print(f"[{index}/{len(inputs)}] {identity}: {'RETOMADO' if reusable else cached['metrics']['status_cranio_geo']}", flush=True)
        mdf, pdf = pd.DataFrame(rows), pd.DataFrame(points)
        for destination in (folder, root):
            save_csv(mdf, destination / 'metricas_cranios_cf.csv')
            save_csv(pdf, destination / 'pontos_geometrico_anatomicos.csv')
        eligible = manifest.loc[bool_series(manifest.usar_matching), ['id_pessoa']].merge(mdf, on='id_pessoa', how='left', validate='one_to_one')
        coverage = float(eligible.regioes_centrais_detectadas.ge(4).mean()) if len(eligible) else 0.
        save_csv(pd.DataFrame([{'elegiveis': len(eligible), 'cobertura_core': coverage,
                 'limite': config['cranios']['gate_cobertura'], 'ativar_anatomia_matching': False,
                 'motivo': 'VALIDACAO_VISUAL_PENDENTE', 'run_id': run_id}]), root / 'gate_viabilidade.csv')
        metadata.update(status='CONCLUIDO', imagens=len(rows), falhas=int(mdf.status_cranio_geo.eq('FALHA').sum()))
        return mdf, pdf
    except BaseException:
        metadata['status'] = 'INTERROMPIDO'
        raise
    finally:
        atomic_json(folder / 'execucao.json', metadata)
        atomic_json(pointer, metadata)
