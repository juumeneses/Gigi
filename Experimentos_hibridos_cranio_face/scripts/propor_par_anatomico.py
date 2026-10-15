"""Visually proposed correspondences for one catalogued pair, not ground truth."""
import base64
import hashlib
import html
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT


def main():
    identity = '2017_30_EC'
    manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    match = manifest[manifest.id_pessoa.eq(identity)]
    assert len(match) == 1
    face_path = REPO_ROOT/'Treinamento_faces/dados_originais/faces/2017/images/2017_30_EC_FFN.jpg'
    skull_path = REPO_ROOT/'Treinamento_cranios/dados_originais/cranios/2017/CF/2017_30_EC_CF.jpg'
    assert str(face_path.relative_to(REPO_ROOT)) in match.iloc[0].astype(str).tolist()
    assert '2017_30_EC_CF' in match.iloc[0].astype(str).tolist()
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    out = LAB_ROOT/'dataset/referencia_anatomica'/('par_'+stamp+'_'+identity)
    out.mkdir(parents=True)
    # Skull coordinates refer to the 1952x1273 inspected rendering; face to 519x733.
    points = [
        ('Glabella (g)', 'Glabella cutanea (g\u2032)', (996,622), (251,322), 'Relevo anterior nao confirmavel pela frontal.'),
        ('Nasospinale (ns)', 'Subnasale (sn)', (1000,849), (249,467), 'Relacao de nivel/posicao; nao sao pontos identicos.'),
        ('Alare osseo direito', 'Alare cutaneo direito', (937,791), (207,447), 'Abertura ossea deve ser interpretada dentro do nariz, nao sobreposta as asas.'),
        ('Alare osseo esquerdo', 'Alare cutaneo esquerdo', (1067,796), (291,445), 'Largura ossea e externa diferentes; tecido mole.'),
        ('Ectoconchion direito', 'Exocanthion direito', (736,702), (116,364), 'Borda orbital versus comissura palpebral; sem coincidencia obrigatoria.'),
        ('Ectoconchion esquerdo', 'Exocanthion esquerdo', (1250,691), (382,350), 'Borda orbital versus comissura palpebral; plano precisa de revisao.'),
        ('Menton projetado', 'Menton cutaneo projetado', (1003,1157), (261,644), 'Limite inferior mediano aparente; pose altera projecao, nao e pogonion.'),
        ('Zygion direito proposto', 'Zygion cutaneo direito proposto', (684,806), (67,405), 'Extremo lateral aparente do arco; na face a projecao exata do arco nao e visivel.'),
        ('Zygion esquerdo proposto', 'Zygion cutaneo esquerdo proposto', (1314,795), (449,399), 'Extremo lateral aparente do arco; nao confundir com maior largura arbitraria da face.'),
        ('Margem orbital medial direita (proxy)', 'Endocanthion direito', (946,702), (204,363), 'Crista lacrimal nao confirmada: nao rotular como mo anatomico validado.'),
        ('Margem orbital medial esquerda (proxy)', 'Endocanthion esquerdo', (1044,691), (304,356), 'Margem ossea versus comissura palpebral; apenas relacao regional.'),
    ]
    images = [Image.open(skull_path).convert('RGB'), Image.open(face_path).convert('RGB')]
    hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [skull_path,face_path]}
    rows = []
    for i, (bone, skin, skull_xy, face_xy, caution) in enumerate(points, 1):
        coords = []
        for im, xy, inspected in zip(images,[skull_xy,face_xy],[(1952,1273),(519,733)]):
            coords.append((round(xy[0]*im.width/inspected[0]), round(xy[1]*im.height/inspected[1])))
        rows.append(dict(numero=i, cranio=bone, face=skin, cranio_x=coords[0][0], cranio_y=coords[0][1],
                         face_x=coords[1][0], face_y=coords[1][1], estado='PROPOSTA_IA', validado=False, cuidado=caution))
    pd.DataFrame(rows).to_csv(out/'correspondencias.csv',index=False)
    panels = []
    for modality, im, prefix in zip(['CRANIO','FACE'],images,['cranio','face']):
        scale = min(760/im.width, 860/im.height)
        resized = im.resize((round(im.width*scale), round(im.height*scale)))
        panel = Image.new('RGB',(800,930),'#eeeeee')
        left, top = (800-resized.width)//2, 45
        panel.paste(resized,(left,top))
        draw = ImageDraw.Draw(panel)
        draw.text((20,12),f'{modality} - {identity} - PROPOSTAS NAO VALIDADAS',fill='black')
        for row in rows:
            x,y = left+row[prefix+'_x']*scale, top+row[prefix+'_y']*scale
            draw.ellipse((x-5,y-5,x+5,y+5),outline='#00ffff',width=2)
            draw.rectangle((x+7,y-16,x+27,y-1),fill='black')
            draw.text((x+9,y-15),str(row['numero']),fill='#00ffff')
        panels.append(panel)
    pair = Image.new('RGB',(1600,930),'white')
    for i,panel in enumerate(panels):
        pair.paste(panel,(800*i,0))
    pair.save(out/'par_marcado.jpg',quality=95)
    metadata = dict(id_pessoa=identity, origem_pareamento='ID no manifesto_mestre.csv, nao reconhecimento visual',
                    original_hashes=hashes, treinamento_autorizado=False, validado=False,
                    nota='Escalas de exibicao independentes. Nao e sobreposicao, ajuste geometrico ou prova de identidade.')
    (out/'proveniencia.json').write_text(json.dumps(metadata,indent=2))
    b64=base64.b64encode((out/'par_marcado.jpg').read_bytes()).decode()
    document = '<!doctype html><meta charset="utf-8"><title>Par anatomico experimental</title>'
    document += '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
    document += '<h1>2017_30_EC: correspondencias propostas</h1><p>Propostas de IA, sem validacao anatomica. '
    document += 'Pareamento pelo ID do dataset. Lados pressupostos nao espelhados. Escalas independentes: nao e sobreposicao.</p>'
    document += f'<img src="data:image/jpeg;base64,{b64}" alt="Cranio e face com pontos numerados">'
    document += pd.DataFrame(rows)[['numero','cranio','face','cuidado']].to_html(index=False,escape=True)
    document += '<p>8 e 9 substituem os centros malares anteriores por extremos laterais propostos. '
    document += '10 e 11 permanecem proxies, nao marcos lacrimais confirmados. Nenhuma coincidencia pixel a pixel e exigida.</p>'
    document += '<p>Referencia: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10772000/">Estudo de superimposicao craniofacial</a>; '
    document += '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11874240/">Definicoes de pontos osseos e cutaneos</a>.</p>'
    (out/'relatorio.html').write_text(document)
    assert all(hashlib.sha256((REPO_ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert len(rows)==11 and len({r['numero'] for r in rows})==11
    print(out,flush=True)


if __name__ == '__main__':
    main()
