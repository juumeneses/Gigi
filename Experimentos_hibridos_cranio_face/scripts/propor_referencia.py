"""Seed untouched drafts with explicitly unvalidated automatic proposals."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, save_csv
from src.reference_batch import point_schema, save_draft
from src.anatomical_pilot import propose, VERSION


def main():
    folders = sorted((LAB_ROOT/'dataset/referencia_anatomica').glob('*/lote.json'))
    if not folders:
        raise RuntimeError('Prepare o lote primeiro.')
    folder = folders[-1].parent
    schema = point_schema()
    rows = []
    for rec in pd.read_csv(folder/'imagens.csv').to_dict('records'):
        identity = rec['id_pessoa']
        if list((folder/'rascunhos'/identity).glob('*.csv')):
            rows.append(dict(id_pessoa=identity, status='RASCUNHO_PRESERVADO', propostas=0))
            continue
        path = REPO_ROOT/rec['imagem_CF']
        image = read_image(path)
        if image is None:
            rows.append(dict(id_pessoa=identity, status='IMAGEM_INDISPONIVEL', propostas=0))
            continue
        points, _, _ = propose(image)
        draft = pd.DataFrame(dict(ponto=schema.ponto, estado='PENDENTE', x_px=float('nan'), y_px=float('nan'), observacao=''))
        count = 0
        for point in points:
            if not point['visivel']:
                continue
            mask = draft.ponto.eq(point['ponto'])
            if not mask.any():
                continue
            draft.loc[mask, ['x_px','y_px','estado','observacao']] = [point['x_px'], point['y_px'], 'PROPOSTA',
                f'Proposta automatica {VERSION}; sem verificacao anatomica. Nao usar como referencia de treino.']
            count += 1
        if count:
            save_draft(folder, identity, draft, schema, path, rec['hash_CF'], 'Detector automatico; nao e revisor anatomico')
        rows.append(dict(id_pessoa=identity, status='PROPOSTAS_NAO_VALIDADAS' if count else 'SEM_PROPOSTA', propostas=count))
        print(f'{identity}: {count} propostas', flush=True)
    save_csv(pd.DataFrame(rows), folder/'resumo_propostas.csv')
    print(f'TOTAL: {sum(r["propostas"] for r in rows)}; lote {folder.name}')


if __name__ == '__main__':
    main()
