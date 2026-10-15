"""Reference preparation only: drafts are never exported as training labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import IMAGE_EXTS, read_image, sha256_file, save_csv
from .validation import review_sample

PROTOCOL = 'artigo_thai_cf_v1_rascunho'
SOURCE = 'Srisinghasongkram et al. 2026, Tabela I p.122, Figura 1 p.123'


def point_schema():
    definitions = [
        ('g', 'Glabella', 'Ponto mediano entre relevos supraorbitais.', False, 9, 'Relevo ambiguo na frontal.'),
        ('n', 'Nasion', 'Ponto mediano da sutura frontonasal.', False, 9, 'Nao estimar sutura invisivel.'),
        ('ns', 'Nasospinale', 'Mediana no nivel das margens inferiores da abertura nasal.', False, 9, 'Nao e centro nasal nem ponta arbitraria da espinha.'),
        ('pr', 'Prosthion', 'Ponto alveolar mediano entre incisivos centrais superiores.', False, 9, 'Nao marcar coroa do dente.'),
        ('id', 'Infradentale', 'Ponto superior mediano do septo alveolar entre incisivos inferiores.', False, 9, 'Pode ser invisivel ou estar danificado.'),
        ('zy', 'Zygion', 'Ponto lateral do arco zigomatico.', True, 9, 'Nao equivale ao extremo da mascara.'),
        ('al', 'Alare osseo', 'Extremo lateral da abertura nasal ossea na Figura 1.', True, 9, 'Tabela I usa asa nasal: interpretacao ossea requer ratificacao especializada.'),
        ('pg', 'Pogonion', 'Ponto anterior mediano da eminencia mentoniana.', False, 17, 'Anterioridade nao confirmavel apenas pela CF.'),
        ('gn', 'Gnathion', 'Ponto mediano entre pogonion e menton conforme Tabela I.', False, 17, 'Nao confundir com mento do detector legado.'),
        ('ec', 'Ectoconchion', 'Borda lateral orbital no plano transversal medio definido no artigo.', True, 17, 'Nao e centro orbital; fixar plano antes de anotar.'),
        ('mo', 'Medial orbit', 'Crista lacrimal anterior no mesmo plano transversal de ec.', True, 17, 'Nao e canto interno de tecido mole.'),
        ('go', 'Gonion', 'Angulo da mandibula, transicao entre ramo e corpo.', True, 17, 'Laterais auxiliam; nao transferir pixel CE/CD para CF.'),
    ]
    rows = []
    for short, name, definition, bilateral, group, caution in definitions:
        for side in (['direito', 'esquerdo'] if bilateral else ['mediano']):
            rows.append(dict(ponto=f'{short}_{side}' if bilateral else short, sigla=short,
                             nome=name, lado=side, conjunto=group, definicao=definition,
                             cuidado=caution, fonte=SOURCE, protocolo=PROTOCOL))
    return pd.DataFrame(rows)


def select_batch(manifest, splits):
    sample = review_sample(manifest, splits)
    sample = sample[sample.grupo_revisao.eq('desenvolvimento')].copy()
    # Spread across the existing quality score; it is not anatomical accuracy.
    sample['_quality'] = pd.to_numeric(sample.get('score_qualidade_cranio', pd.Series(index=sample.index, dtype=float)), errors='coerce')
    groups = []
    for _, group in sample.groupby('ano'):
        group = group.sort_values(['_quality', 'id_pessoa'], na_position='last')
        if len(group) < 6:
            raise ValueError('Menos de seis identidades de desenvolvimento em um ano.')
        groups.append(group.iloc[np.linspace(0, len(group)-1, 6).round().astype(int)])
    if len(groups) != 5:
        raise ValueError('Esperados cinco anos para o lote de 30 imagens.')
    return pd.concat(groups).drop(columns='_quality').reset_index(drop=True)


def resolve_original(repo, record, view):
    identity = str(record[f'id_{view.lower()}'])
    folder = repo/'Treinamento_cranios/dados_originais/cranios'/str(record['ano'])/view
    matches = sorted(p for p in folder.glob(identity+'.*') if p.suffix.lower() in IMAGE_EXTS)
    if len(matches) != 1:
        return None, 'AUSENTE' if not matches else 'AMBIGUA'
    path = matches[0]
    with path.open('rb') as stream:
        if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            return path, 'LFS_NAO_BAIXADO'
    return path, 'OK' if read_image(path) is not None else 'ILEGIVEL'


def validate_draft(frame, schema, width, height):
    if frame.ponto.tolist() != schema.ponto.tolist():
        raise ValueError('Nomes ou ordem dos pontos alterados.')
    for row in frame.to_dict('records'):
        if row['estado'] not in ['PENDENTE', 'PROPOSTA', 'LOCALIZADO', 'NAO_VISIVEL', 'DUVIDA']:
            raise ValueError('Estado invalido.')
        x, y = pd.to_numeric(row['x_px'], errors='coerce'), pd.to_numeric(row['y_px'], errors='coerce')
        if row['estado'] in ['LOCALIZADO', 'PROPOSTA']:
            if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError(f"{row['ponto']}: coordenadas ausentes ou fora da imagem.")
        elif not (pd.isna(x) and pd.isna(y)):
            raise ValueError(f"{row['ponto']}: limpe coordenadas de ponto nao localizado.")


def save_draft(folder, identity, frame, schema, image_path, expected_hash, reviewer):
    image = read_image(image_path)
    if image is None or sha256_file(image_path) != expected_hash:
        raise ValueError('Imagem ausente ou modificada desde a preparacao.')
    if not reviewer.strip():
        raise ValueError('Informe o responsavel pelo rascunho.')
    validate_draft(frame, schema, image.shape[1], image.shape[0])
    output = frame.copy()
    output['id_pessoa'] = identity
    output['input_hash'] = expected_hash
    output['responsavel'] = reviewer.strip()
    protocols = schema.protocolo.dropna().unique() if 'protocolo' in schema else [PROTOCOL]
    if len(protocols) != 1:
        raise ValueError('O esquema deve declarar um unico protocolo.')
    output['protocolo'] = protocols[0]
    output['validado'] = False
    output['salvo_em'] = pd.Timestamp.now(tz='UTC').isoformat()
    # Append immutable revisions; no draft silently becomes a reference.
    destination = folder/'rascunhos'/identity/(pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'.csv')
    save_csv(output, destination)
    return destination
