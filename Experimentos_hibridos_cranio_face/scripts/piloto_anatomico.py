"""Build a small development-only reference-review batch; does not train."""
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, IMAGE_EXTS, read_image, write_image, save_csv, sha256_file
from src.cranio_runs import atomic_json, image_error
from src.validation import review_sample
from src.anatomical_pilot import propose, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--por-ano', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.por_ano <= 10:
        parser.error('--por-ano deve estar entre 1 e 10')
    manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    splits = pd.read_csv(LAB_ROOT/'dataset/splits/split_identidades.csv')
    sample = review_sample(manifest, splits)
    sample = sample[sample.grupo_revisao.eq('desenvolvimento')].groupby('ano', group_keys=False).head(args.por_ano)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_piloto_anatomico'
    folder = LAB_ROOT/'resultados/cranios/runs'/run_id
    folder.mkdir(parents=True)
    metadata = dict(run_id=run_id, status='RODANDO', detector=VERSION, treinamento_autorizado=False,
                    escopo='Apenas desenvolvimento; propostas nao sao anotacoes de referencia',
                    detector_hash=sha256_file(LAB_ROOT/'src/anatomical_pilot.py'),
                    base_hash=sha256_file(LAB_ROOT/'src/cranios.py'), parametros=vars(args))
    atomic_json(folder/'execucao.json', metadata)
    save_csv(sample, folder/'amostra.csv')
    rows, all_points = [], []
    try:
        for i, rec in enumerate(sample.to_dict('records'), 1):
            identity = str(rec['id_pessoa'])
            base = REPO_ROOT/'Treinamento_cranios/dados_originais/cranios'/str(rec['ano'])/'CF'
            paths = sorted(p for p in base.glob(str(rec['id_cf'])+'.*') if p.suffix.lower() in IMAGE_EXTS)
            path = paths[0] if len(paths) == 1 else base/(str(rec['id_cf'])+'.jpg')
            error = 'ORIGINAL_AMBIGUO' if len(paths) > 1 else image_error(path)
            image = None if error else read_image(path)
            key = hashlib.sha256(identity.encode()).hexdigest()[:24]
            metrics = dict(id_pessoa=identity, ano=rec['ano'], run_id=run_id,
                           imagem_cf=str(path.relative_to(REPO_ROOT)), status_cranio_geo='REVISAO')
            points = []
            if image is None:
                metrics.update(status_cranio_geo='FALHA', motivo_cranio_geo=error or 'IMAGEM_ILEGIVEL')
            else:
                points, mask, preview = propose(image)
                for field, suffix, output in [('preview_path', '.jpg', preview), ('mask_path', '.png', mask)]:
                    target = folder/(key+suffix)
                    if not write_image(target, output):
                        raise IOError(f'Falha ao salvar {target}')
                    metrics[field] = str(target.relative_to(LAB_ROOT))
                metrics['propostas'] = sum(p['visivel'] for p in points)
            atomic_json(folder/(key+'.json'), dict(input_hash=sha256_file(path) if path.is_file() else 'ausente',
                                                  metrics=metrics, points=points))
            rows.append(metrics)
            all_points.extend(dict(id_pessoa=identity, ano=rec['ano'], **p) for p in points)
            save_csv(pd.DataFrame(rows), folder/'metricas_cranios_cf.csv')
            save_csv(pd.DataFrame(all_points), folder/'propostas_nao_validadas.csv')
            print(f'[{i}/{len(sample)}] {identity}: {metrics.get("propostas", 0)}/4 propostas; {metrics["status_cranio_geo"]}', flush=True)
        metadata.update(status='CONCLUIDO', imagens=len(rows), propostas=sum(r.get('propostas', 0) for r in rows))
    except BaseException:
        metadata['status'] = 'INTERROMPIDO'
        raise
    finally:
        atomic_json(folder/'execucao.json', metadata)
    print(f'RESULTADOS: {folder}', flush=True)


if __name__ == '__main__':
    main()
