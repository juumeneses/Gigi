"""Development-only preprocessing ablation; never exports training labels."""
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, write_image, save_csv, sha256_file
from src.cranio_runs import atomic_json, image_error
from src.anatomical_pilot import propose, VERSION
from src.validation import review_sample


def variants(image):
    yield 'original', image, {'scale_x': 1., 'scale_y': 1.}
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
    yield 'clahe', cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), {'scale_x': 1., 'scale_y': 1., 'clip_limit': 2., 'tiles': [8, 8]}
    yield 'bilateral', cv2.bilateralFilter(image, 7, 25, 25), {'scale_x': 1., 'scale_y': 1., 'd': 7, 'sigma_color': 25, 'sigma_space': 25}
    yield 'brilho_110', np.clip(image.astype(np.float32)*1.1, 0, 255).astype(np.uint8), {'scale_x': 1., 'scale_y': 1., 'gain': 1.1}
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (max(1, round(w*.85)), max(1, round(h*.85))), interpolation=cv2.INTER_AREA)
    yield 'escala_085', scaled, {'scale_x': scaled.shape[1]/w, 'scale_y': scaled.shape[0]/h}


def original_coordinates(point, transform):
    mapped = dict(point)
    if point['visivel']:
        mapped['x_px'] = point['x_px']/transform['scale_x']
        mapped['y_px'] = point['y_px']/transform['scale_y']
    return mapped


def main():
    started = time.monotonic()
    manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    splits = pd.read_csv(LAB_ROOT/'dataset/splits/split_identidades.csv')
    sample = review_sample(manifest, splits)
    sample = sample[sample.grupo_revisao.eq('desenvolvimento')].groupby('ano', group_keys=False).head(2)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_preprocessamento'
    folder = LAB_ROOT/'resultados/cranios/runs'/run_id
    folder.mkdir(parents=True)
    metadata = dict(run_id=run_id, status='RODANDO', detector=VERSION, treinamento_autorizado=False,
                    script_hash=sha256_file(Path(__file__)), detector_hash=sha256_file(LAB_ROOT/'src/anatomical_pilot.py'),
                    base_hash=sha256_file(LAB_ROOT/'src/cranios.py'), escopo='10 CF desenvolvimento; 5 variantes; sem treino',
                    criterio_estabilidade='todos os 5 detectam; desvio maximo <= 0.02 diagonal original; nao e acuracia')
    atomic_json(folder/'execucao.json', metadata)
    save_csv(sample, folder/'amostra.csv')
    rows, stability = [], []
    try:
        for index, rec in enumerate(sample.to_dict('records'), 1):
            identity = str(rec['id_pessoa'])
            path = REPO_ROOT/'Treinamento_cranios/dados_originais/cranios'/str(rec['ano'])/'CF'/f'{rec["id_cf"]}.jpg'
            digest = sha256_file(path) if path.is_file() else 'ausente'
            error = image_error(path)
            image = None if error else read_image(path)
            key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            metrics = dict(id_pessoa=identity, ano=rec['ano'], imagem_cf=str(path.relative_to(REPO_ROOT)), run_id=run_id)
            if image is None:
                metrics.update(status_cranio_geo='FALHA', motivo_cranio_geo=error or 'IMAGEM_ILEGIVEL')
                rows.append(dict(id_pessoa=identity, variante='indisponivel', status='ERRO', motivo=metrics['motivo_cranio_geo']))
                atomic_json(folder/(key+'.json'), dict(input_hash=digest, metrics=metrics, points=[]))
                continue
            mapped, tiles = {}, []
            for name, derived, transform in variants(image):
                points, mask, preview = propose(derived)
                mapped[name] = [original_coordinates(p, transform) for p in points]
                subfolder = folder/'variantes'/key
                for suffix, output in [('.jpg', derived), ('_pontos.jpg', preview), ('_mascara.png', mask)]:
                    if not write_image(subfolder/(name+suffix), output):
                        raise IOError('Falha ao gravar derivadas')
                atomic_json(subfolder/(name+'.json'), dict(transform=transform, input_hash=digest, points_original=mapped[name]))
                for point in mapped[name]:
                    rows.append(dict(id_pessoa=identity, ano=rec['ano'], variante=name, status='PROPOSTA' if point['visivel'] else 'ABSTENCAO', **point))
                tile = np.full((320, 420, 3), 245, np.uint8)
                factor = min(420/preview.shape[1], 285/preview.shape[0])
                small = cv2.resize(preview, (round(preview.shape[1]*factor), round(preview.shape[0]*factor)))
                tile[35:35+small.shape[0], :small.shape[1]] = small
                cv2.putText(tile, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,20), 1)
                tiles.append(tile)
                print(f'[{index}/10] {identity} / {name}: {sum(p["visivel"] for p in points)}/4 propostas', flush=True)
            baseline = mapped['original']
            for point in baseline:
                observations = [p for pp in mapped.values() for p in pp if p['ponto']==point['ponto'] and p['visivel']]
                spread = None
                if len(observations) >= 2:
                    coords = np.array([[p['x_px'], p['y_px']] for p in observations])
                    spread = float(np.max(np.linalg.norm(coords-np.median(coords, axis=0), axis=1))/np.hypot(*image.shape[:2]))
                stable = len(observations)==5 and spread is not None and spread <= .02
                stability.append(dict(id_pessoa=identity, ponto=point['ponto'], variantes_com_proposta=len(observations),
                                      desvio_max_diagonal=spread, estavel=stable, validado=False))
                point.update(estavel_5_variantes=stable, variantes_com_proposta=len(observations), desvio_max_diagonal=spread)
            sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:]+[np.full_like(tiles[0],245)])])
            if not write_image(folder/(key+'_comparacao.jpg'), sheet):
                raise IOError('Falha ao salvar comparacao')
            metrics.update(status_cranio_geo='REVISAO', preview_path=str((folder/'variantes'/key/'original_pontos.jpg').relative_to(LAB_ROOT)),
                           mask_path=str((folder/'variantes'/key/'original_mascara.png').relative_to(LAB_ROOT)),
                           comparacao_path=str((folder/(key+'_comparacao.jpg')).relative_to(LAB_ROOT)))
            atomic_json(folder/(key+'.json'), dict(input_hash=digest, metrics=metrics, points=baseline))
            if sha256_file(path) != digest:
                raise RuntimeError('Original mudou durante processamento')
            save_csv(pd.DataFrame(rows), folder/'propostas_variantes.csv')
            save_csv(pd.DataFrame(stability), folder/'estabilidade.csv')
        frame = pd.DataFrame(rows)
        save_csv(frame, folder/'propostas_variantes.csv')
        summary = frame.groupby(['variante','status']).size().unstack(fill_value=0)
        save_csv(summary.reset_index(), folder/'resumo.csv')
        metadata.update(status='CONCLUIDO', duracao_s=time.monotonic()-started, pontos_estaveis=sum(p['estavel'] for p in stability))
    except BaseException:
        metadata['status'] = 'INTERROMPIDO'
        raise
    finally:
        atomic_json(folder/'execucao.json', metadata)
    print(f'RESULTADOS: {folder}', flush=True)


if __name__ == '__main__':
    main()
