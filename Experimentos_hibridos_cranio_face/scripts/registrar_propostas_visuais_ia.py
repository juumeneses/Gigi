"""Register three individually inspected images, not an automatic detector.

Coordinates were proposed visually by the assistant using the article's figure.
They are approximate, unvalidated, and prohibited as training ground truth.
"""
import sys
from pathlib import Path
import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT, read_image, write_image, save_csv
from src.reference_batch import point_schema, save_draft

# Coordinates on the actual image displayed during inspection, before rescaling.
PROPOSALS = {
    '2017_30_EC': ((1952, 1273), {'g': (996,622), 'ec_direito': (736,702), 'ec_esquerdo': (1250,691),
                                    'al_direito': (937,791), 'al_esquerdo': (1067,796), 'ns': (1000,849)}),
    '2017_41_GP': ((1848,1206), {'g': (935,480), 'ec_direito': (654,552), 'ec_esquerdo': (1216,541),
                                  'al_direito': (856,698), 'al_esquerdo': (1012,700)}),
    '2017_73_MMPCC': ((1907,1244), {'g': (916,552), 'ec_direito': (663,617), 'ec_esquerdo': (1200,615),
                                     'al_direito': (836,763), 'al_esquerdo': (997,773), 'ns': (914,824)}),
}
REASONS = {
    'g': 'Regiao mediana entre relevos supraorbitais; projecao exata incerta na CF.',
    'ec': 'Rebordo lateral visivel da orbita, aproximadamente a meia altura; plano transversal precisa de revisao.',
    'al': 'Margem lateral aparente da abertura nasal; artefatos da renderizacao tornam a coordenada aproximada.',
    'ns': 'Mediana no nivel inferior aparente da abertura nasal; distinguir espinha de margem inferior em revisao.',
}
DOUBTS = {
    'n': 'Sutura frontonasal nao distinguivel com seguranca nesta renderizacao.',
    'pr': 'Limite alveolar superior/coroa nao separavel com seguranca.',
    'id': 'Septo alveolar inferior nao identificavel com seguranca.',
    'zy': 'Extremo do arco zigomatico em profundidade nao confirmado pela frontal.',
    'pg': 'Anterioridade da eminencia mentoniana nao determinada em CF.',
    'gn': 'Depende de pogonion/menton e definicao ratificada; nao usar borda inferior arbitraria.',
    'mo': 'Crista lacrimal anterior nao discriminada com seguranca.',
    'go': 'Transicao ramo/corpo e angulo mandibular ambiguos na projecao frontal.',
    'ns': 'Margem inferior nasal fragmentada/artefatual; nao ha coordenada segura.',
}


def main():
    folder = LAB_ROOT/'dataset/referencia_anatomica/20260926T193433785591Z_referencia_artigo_30'
    images = pd.read_csv(folder/'imagens.csv')
    schema = point_schema()
    summary = []
    for identity, (display_size, coordinates) in PROPOSALS.items():
        # Never supersede an existing human or automatic draft silently.
        if list((folder/'rascunhos'/identity).glob('*.csv')):
            print(f'{identity}: rascunho existente preservado')
            continue
        record = images[images.id_pessoa.eq(identity)].iloc[0]
        path = REPO_ROOT/record.imagem_CF
        image = read_image(path)
        h,w = image.shape[:2]
        rows = []
        preview = image.copy()
        for p in schema.ponto:
            short = p.split('_')[0]
            if p in coordinates:
                dx,dy = coordinates[p]
                x,y = round(dx*w/display_size[0]), round(dy*h/display_size[1])
                state, note = 'PROPOSTA', REASONS[short]
                cv2.circle(preview,(x,y),max(7,w//200),(0,255,255),2)
                cv2.putText(preview,p,(x+10,y-8),cv2.FONT_HERSHEY_SIMPLEX,max(.6,w/2200),(0,255,255),2,cv2.LINE_AA)
            else:
                x,y,state,note = None,None,'DUVIDA',DOUBTS.get(short,'Localizacao incerta na frontal.')
            rows.append(dict(ponto=p,x_px=x,y_px=y,estado=state,
                observacao='IA visual, nao especialista, nao validada. '+note+' Lados pressupostos em imagem nao espelhada.'))
        draft = pd.DataFrame(rows)
        output = save_draft(folder,identity,draft,schema,path,record.hash_CF,'Codex - proposta visual IA; sem validacao anatomica')
        target = output.with_suffix('.jpg')
        if not write_image(target,preview):
            raise IOError('Preview nao gravado')
        summary.append(dict(id_pessoa=identity,propostas=len(coordinates),duvidas=17-len(coordinates),
                            rascunho=str(output.relative_to(LAB_ROOT)),preview=str(target.relative_to(LAB_ROOT)),validado=False))
        print(identity, len(coordinates), target, flush=True)
    if summary:
        save_csv(pd.DataFrame(summary),folder/'propostas_visuais_ia.csv')


if __name__ == '__main__':
    main()
