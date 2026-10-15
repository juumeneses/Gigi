"""One visually inspected skull; experimental 2D proxies, not training labels."""
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, write_image, save_csv, sha256_file
from src.reference_batch import save_draft


def main():
    source = LAB_ROOT/'dataset/referencia_anatomica/20260926T193433785591Z_referencia_artigo_30'
    identity = '2017_30_EC'
    images = pd.read_csv(source/'imagens.csv')
    images = images[images.id_pessoa.eq(identity)].copy()
    record = images.iloc[0]
    original = REPO_ROOT/record.imagem_CF
    image = read_image(original)
    assert image is not None and sha256_file(original) == record.hash_CF
    previous = pd.read_csv(source/'rascunhos'/identity/'20260926T204156219600Z.csv')
    schema_source = pd.read_csv(source/'pontos_artigo.csv')
    keep = previous[previous.estado.eq('PROPOSTA')].ponto.tolist()
    schema = schema_source.set_index('ponto').loc[keep].reset_index()
    draft = previous[previous.ponto.isin(keep)][['ponto','x_px','y_px','estado','observacao']].reset_index(drop=True)
    # Coordinates on the 1952 x 1273 displayed original, individually inspected.
    additions = [
        ('queixo_inferior_2d', 'Queixo inferior projetado', 'mediano', (1003, 1157),
         'Borda inferior mediana aparente da mandibula, acima do fragmento/base inferior.',
         'Referencia de contorno 2D; nao e pogonion nem gnathion confirmado.'),
        ('malar_direito_2d', 'Regiao malar direita', 'direito', (749, 810),
         'Regiao do corpo zigomatico abaixo e lateral ao rebordo orbital.',
         'Centro visual aproximado, nao e zygion nem ponto de maxima projecao 3D.'),
        ('malar_esquerdo_2d', 'Regiao malar esquerda', 'esquerdo', (1240, 803),
         'Regiao do corpo zigomatico abaixo e lateral ao rebordo orbital.',
         'Centro visual aproximado, nao e zygion nem ponto de maxima projecao 3D.'),
        ('orbita_medial_direita_2d', 'Borda orbital medial direita', 'direito', (946, 702),
         'Margem medial aparente da orbita na altura aproximada do ec ipsilateral.',
         'Proxy visual; crista lacrimal anterior nao confirmada. Nao e canto do olho nem mo validado.'),
        ('orbita_medial_esquerda_2d', 'Borda orbital medial esquerda', 'esquerdo', (1044, 691),
         'Margem medial aparente da orbita na altura aproximada do ec ipsilateral.',
         'Proxy visual; crista lacrimal anterior nao confirmada. Nao e canto do olho nem mo validado.'),
    ]
    protocol = 'piloto_cf_11_propostas_visuais_2d_v1'
    h, w = image.shape[:2]
    for point, name, side, (x, y), definition, caution in additions:
        schema = pd.concat([schema, pd.DataFrame([dict(ponto=point, sigla=point, nome=name,
            lado=side, conjunto=11, definicao=definition, cuidado=caution,
            fonte='Proposta operacional IA: nao definida como landmark do artigo', protocolo=protocol)])], ignore_index=True)
        draft.loc[len(draft)] = [point, round(x*w/1952), round(y*h/1273), 'PROPOSTA',
            'IA visual nao validada. '+definition+' '+caution+' Lados pressupostos nao espelhados.']
    schema['protocolo'] = protocol
    schema['conjunto'] = 11
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    folder = source.parent/(stamp+'_piloto_reduzido_11_uma_imagem')
    folder.mkdir(parents=True)
    output = save_draft(folder, identity, draft, schema, original, record.hash_CF,
                        'Codex - proposta visual IA; sem validacao anatomica')
    save_csv(images, folder/'imagens.csv')
    save_csv(schema, folder/'pontos_artigo.csv')
    info = dict(lote=folder.name, protocolo=protocol, imagens=1, pontos_por_imagem=11,
                treinamento_autorizado=False, origem=str(source.relative_to(LAB_ROOT)),
                status='PROPOSTAS_IA_NAO_VALIDADAS', novos_pontos_sao_proxies_2d=True)
    (folder/'lote.json').write_text(json.dumps(info, indent=2))
    (folder/'protocolo.md').write_text(
        '# Piloto reduzido: uma imagem, 11 propostas\n\n'
        'Seis propostas anteriores preservadas; cinco referencias visuais 2D adicionadas. '
        'Os novos pontos nao sao equivalentes anatomicos validados de zy, pg, gn ou mo. '
        'Malar representa uma regiao, ainda sem regra de repetibilidade validada. '
        'Todas as coordenadas sao propostas de IA, nao verdade de treino. '
        'Revisao visual favoravel nao comprova precisao anatomica. '
        'Lados seguem o cranio, pressupondo imagem nao espelhada.\n\n'
        'No preview: amarelo = proposta anterior, ciano = proposta nova; numeros seguem a tabela.\n')
    overlay = image.copy()
    legend = []
    for number, row in enumerate(draft.to_dict('records'), 1):
        color = (0, 255, 255) if number <= 6 else (255, 255, 0)
        xy = (int(row['x_px']), int(row['y_px']))
        cv2.circle(overlay, xy, 10, color, 3)
        cv2.putText(overlay, str(number), (xy[0]+15, xy[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
        cv2.putText(overlay, str(number), (xy[0]+15, xy[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        legend.append(dict(numero=number, ponto=row['ponto'], estado='PROPOSTA', validado=False))
    target = folder/'preview_11_pontos.jpg'
    assert write_image(target, overlay)
    save_csv(pd.DataFrame(legend), folder/'legenda.csv')
    assert sha256_file(original) == record.hash_CF
    assert len(draft) == 11 and draft.ponto.is_unique
    assert not pd.read_csv(output).validado.any()
    print(target, flush=True)


if __name__ == '__main__':
    main()
