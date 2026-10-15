"""Small development-only comparison on locally available original CF images."""
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, write_image, save_csv, sha256_file
from src.cranio_runs import atomic_json, image_error
from src.cranios import detect_geometry_v3
from src.cranio_v4 import detect_geometry_v4
from src.validation import review_sample


def main():
    manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    splits = pd.read_csv(LAB_ROOT/'dataset/splits/split_identidades.csv')
    sample = review_sample(manifest, splits)
    sample = sample[sample.grupo_revisao.eq('desenvolvimento')].groupby('ano', group_keys=False).head(2)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_piloto_mento'
    folder = LAB_ROOT/'resultados/cranios/runs'/run_id
    folder.mkdir(parents=True)
    metadata = {'run_id': run_id, 'status': 'RODANDO', 'escopo': '10 imagens de desenvolvimento; originais CF',
                'detector': 'v4_mento_experimental', 'prior': 'fixo V3, sem ajuste sobre teste',
                'detector_hash': sha256_file(LAB_ROOT/'src/cranio_v4.py')}
    atomic_json(folder/'execucao.json', metadata)
    rows = []
    try:
        for i, rec in enumerate(sample.to_dict('records'), 1):
            identity = str(rec['id_pessoa'])
            original = REPO_ROOT/'Treinamento_cranios/dados_originais/cranios'/str(rec['ano'])/'CF'/f'{rec["id_cf"]}.jpg'
            error = image_error(original)
            image = None if error else read_image(original)
            base = {'id_pessoa': identity, 'ano': rec['ano'], 'imagem_cf_manifesto': rec['imagem_cf'],
                    'imagem_cf': str(original.relative_to(REPO_ROOT)), 'run_id': run_id}
            key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            if image is None:
                metrics, points = {**base, 'status_cranio_geo': 'FALHA', 'motivo_cranio_geo': error or 'IMAGEM_ILEGIVEL'}, []
            else:
                _, old_points, _, old_preview = detect_geometry_v3(image)
                metrics, points, mask, preview = detect_geometry_v4(image)
                for suffix, output in [('.jpg', preview), ('.png', mask), ('_v3.jpg', old_preview)]:
                    if not write_image(folder/(key+suffix), output):
                        raise IOError('Falha ao gravar preview')
                old = next(p for p in old_points if p['ponto']=='mento')
                new = next(p for p in points if p['ponto']=='mento')
                metrics = {**base, **metrics, 'mento_v3_y': old['y_px'], 'mento_v4_y': new['y_px'],
                           'preview_path': str((folder/(key+'.jpg')).relative_to(LAB_ROOT)),
                           'preview_v3_path': str((folder/(key+'_v3.jpg')).relative_to(LAB_ROOT)),
                           'mask_path': str((folder/(key+'.png')).relative_to(LAB_ROOT))}
            atomic_json(folder/(key+'.json'), {'input_hash': sha256_file(original) if original.is_file() else 'ausente',
                                              'metrics': metrics, 'points': points})
            rows.append(metrics)
            print(f'[{i}/{len(sample)}] {identity}: {metrics.get("mento_motivo", metrics.get("motivo_cranio_geo"))}', flush=True)
        save_csv(pd.DataFrame(rows), folder/'metricas_cranios_cf.csv')
        metadata['status'] = 'CONCLUIDO'
    except BaseException:
        metadata['status'] = 'INTERROMPIDO'
        raise
    finally:
        atomic_json(folder/'execucao.json', metadata)
    print(f'RESULTADOS: {folder}', flush=True)


if __name__ == '__main__':
    main()
