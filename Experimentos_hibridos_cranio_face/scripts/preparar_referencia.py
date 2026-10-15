"""Prepare 30 development images and the article's 17-point annotation schema."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, save_csv, sha256_file
from src.cranio_runs import atomic_json
from src.reference_batch import point_schema, select_batch, resolve_original, PROTOCOL, SOURCE


def main():
    manifest_path = LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv'
    splits_path = LAB_ROOT/'dataset/splits/split_identidades.csv'
    selected = select_batch(pd.read_csv(manifest_path), pd.read_csv(splits_path))
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_referencia_artigo_30'
    folder = LAB_ROOT/'dataset/referencia_anatomica'/batch
    folder.mkdir(parents=True)
    schema = point_schema()
    save_csv(schema, folder/'pontos_artigo.csv')
    images, annotations = [], []
    for record in selected.to_dict('records'):
        row = dict(id_pessoa=record['id_pessoa'], ano=record['ano'], split=record['split'], grupo_revisao=record['grupo_revisao'],
                   score_qualidade_cranio=record.get('score_qualidade_cranio'))
        for view in ['CF', 'CE', 'CD']:
            path, status = resolve_original(REPO_ROOT, record, view)
            row['imagem_'+view] = str(path.relative_to(REPO_ROOT)) if path else ''
            row['status_'+view] = status
            row['hash_'+view] = sha256_file(path) if path else ''
            if view == 'CF' and status == 'OK':
                h, w = read_image(path).shape[:2]
                row.update(largura=w, altura=h)
        images.append(row)
        for point in schema.to_dict('records'):
            annotations.append(dict(id_pessoa=row['id_pessoa'], ano=row['ano'], ponto=point['ponto'],
                                    conjunto=point['conjunto'], x_px=None, y_px=None, estado='PENDENTE',
                                    observacao='', responsavel='', validado=False, protocolo=PROTOCOL, input_hash=row['hash_CF']))
    save_csv(pd.DataFrame(images), folder/'imagens.csv')
    save_csv(pd.DataFrame(annotations), folder/'anotacoes_pendentes.csv')
    atomic_json(folder/'lote.json', dict(lote=batch, protocolo=PROTOCOL, fonte=SOURCE,
                status='AGUARDANDO_ANOTACAO_E_VERIFICACAO', treinamento_autorizado=False,
                imagens=len(images), cf_ok=sum(r['status_CF']=='OK' for r in images), pontos_por_imagem=len(schema),
                manifest_hash=sha256_file(manifest_path), splits_hash=sha256_file(splits_path),
                selecao='6 por ano, desenvolvimento somente, espacadas pelo score de qualidade; nao e amostra de acuracia'))
    print(folder)
    print(pd.DataFrame(images).groupby(['ano','status_CF']).size().to_string())


if __name__ == '__main__':
    main()
