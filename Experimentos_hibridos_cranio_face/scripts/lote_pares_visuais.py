"""Record individually inspected AI proposals; never training labels."""
import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import LAB_ROOT, REPO_ROOT

ROOT = LAB_ROOT/'dataset/pares_visuais/lote_10_20260926'
SOURCE = LAB_ROOT/'dataset/referencia_anatomica/20260926T193433785591Z_referencia_artigo_30'

# Full inspection-sheet pixel coordinates (including header and face offset).
# These are individual visual estimates, NOT a detector or copied template.
PROPOSALS = {
    '2017_41_GP': {
        'cranio': [(290,265),None,(254,358),(333,358),(153,286),(440,280),(310,592),(115,371),(486,362),(269,286),(310,280)],
        'face': [(893,256),(901,390),(881,376),(924,376),(790,299),(1001,287),(901,542),(742,343),(1056,335),(850,298),(941,293)]},
    '2017_73_MMPCC': {
        'cranio': [(272,277),(272,406),(232,375),(316,380),(139,310),(421,308),(277,590),(119,386),(477,386),(251,310),(299,308)],
        'face': [(893,257),(893,364),(855,350),(931,350),(806,286),(988,284),(887,511),(751,334),(1034,333),(853,286),(939,285)]},
    '2017_76_MVXP': {
        'cranio': [(283,261),None,(250,377),(320,378),(157,296),(417,292),(279,587),(125,359),(453,356),(265,296),(304,292)],
        'face': [(895,279),(898,407),(856,391),(939,390),(788,310),(1000,298),None,(752,352),(1045,342),(850,311),(944,303)]},
    '2017_92_SRAK': {
        'cranio': [(307,277),(305,414),(272,379),(338,379),(165,322),(440,326),(302,593),(127,371),(477,372),(287,322),(328,326)],
        'face': [(895,321),(895,428),(863,412),(931,411),(801,336),(995,335),(890,558),(759,379),(1035,371),(855,338),(938,338)]},
    '2017_99_VDSR': {
        'cranio': [(303,253),None,(264,354),(339,354),(165,281),(433,276),(305,557),(117,358),(477,357),(276,281),(327,276)],
        'face': [(900,284),(907,406),(870,390),(946,392),(807,313),(1004,308),(911,544),(745,354),(1062,350),(863,317),(950,313)]},
    '2019_115_KCCSF': {
        'cranio': [(289,300),(289,426),(255,386),(323,389),(164,323),(421,328),(291,609),(140,394),(455,398),(271,323),(309,328)],
        'face': [(902,286),(910,382),(879,369),(934,369),(828,303),(986,303),(908,494),(790,344),(1016,340),(875,304),(936,304)]},
    '2019_156_MFR': {
        'cranio': [(304,324),(304,448),(270,412),(340,412),(173,341),(438,340),(303,602),(141,397),(468,397),(283,341),(331,340)],
        'face': [(902,256),(904,359),(868,347),(941,346),(812,273),(990,266),(902,473),(772,315),(1029,306),(873,278),(939,275)]},
    '2019_233_SFG': {
        'cranio': [(293,223),(294,378),(259,341),(330,339),(145,254),(440,253),(290,579),(95,328),(500,322),(273,254),(317,253)],
        'face': [(895,281),(900,381),(869,367),(929,366),(813,301),(982,297),(900,490),(779,340),(1018,335),(863,304),(934,302)]},
    '2019_234_SFLL': {
        'cranio': [(278,268),(278,382),(251,347),(307,348),(157,294),(401,295),(281,550),(119,345),(441,342),(257,294),(301,295)],
        'face': [(895,263),(901,371),(870,355),(933,355),(813,286),(991,285),(895,502),(758,330),(1030,325),(865,291),(940,291)]},
    '2019_260_WRV': {
        'cranio': [(302,301),(301,431),(272,400),(332,400),(182,328),(427,332),(299,603),(148,379),(468,375),(281,328),(323,332)],
        'face': [(911,282),(911,384),(876,369),(948,370),(829,302),(993,302),None,(785,341),(1032,340),(881,306),(942,307)]},
}


def inspect():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(SOURCE/'imagens.csv').iloc[1:11]
    manifest = pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    data = []
    for row in records.to_dict('records'):
        identity = row['id_pessoa']
        match = manifest[manifest.id_pessoa.eq(identity)]
        assert len(match) == 1
        face = next(v for v in match.iloc[0].astype(str) if v.startswith('Treinamento_faces/dados_originais/') and '/images/' in v and v.endswith('.jpg'))
        info = dict(id_pessoa=identity, ano=row['ano'], split=row['split'], views={})
        canvas = Image.new('RGB',(1200,630),'#eeeeee')
        for j,(kind,relative) in enumerate([('cranio',row['imagem_CF']),('face',face)]):
            path = REPO_ROOT/relative
            im = Image.open(path).convert('RGB')
            # Remove only the black screenshot margin for skull inspection.
            box = ImageChops.difference(im,Image.new('RGB',im.size)).point(lambda p: 255 if p>35 else 0).getbbox() if kind=='cranio' else (0,0,*im.size)
            cropped = im.crop(box)
            scale = min(580/cropped.width,580/cropped.height)
            size = (round(cropped.width*scale),round(cropped.height*scale))
            offset = ((600-size[0])//2,(600-size[1])//2)
            canvas.paste(cropped.resize(size),(600*j+offset[0],30+offset[1]))
            ImageDraw.Draw(canvas).text((600*j+12,8),identity+' '+kind,fill='black')
            info['views'][kind] = dict(path=relative,hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                box=box,size=size,offset=offset,original_size=im.size)
        canvas.save(ROOT/(identity+'_inspecao.jpg'),quality=95)
        data.append(info)
    (ROOT/'entradas.json').write_text(json.dumps(data,indent=2))
    print('\n'.join(r['id_pessoa'] for r in data))


def original_xy(view, xy, modality):
    if xy is None:
        return None
    x,y = xy[0]-(600 if modality=='face' else 0), xy[1]-30
    left,top,right,bottom = view['box']
    ox,oy = view['offset']
    sw,sh = view['size']
    assert ox <= x < ox+sw and oy <= y < oy+sh
    return (round(left+(x-ox)*(right-left)/sw),round(top+(y-oy)*(bottom-top)/sh))


def annotate():
    entries = json.loads((ROOT/'entradas.json').read_text())
    definitions = pd.read_csv(LAB_ROOT/'dataset/referencia_anatomica/par_20260926T210854690712Z_2017_30_EC/correspondencias.csv')
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'runs'/stamp
    output.mkdir(parents=True)
    summary, all_rows = [], []
    for entry in entries:
        identity = entry['id_pessoa']
        assert identity in PROPOSALS
        pair_folder = output/identity
        pair_folder.mkdir()
        frames = {}
        rows = []
        for kind in ['cranio','face']:
            v = entry['views'][kind]
            path = REPO_ROOT/v['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==v['hash']
            image = Image.open(path).convert('RGB')
            marked = image.copy()
            draw = ImageDraw.Draw(marked)
            for i,xy in enumerate(PROPOSALS[identity][kind],1):
                coord = original_xy(v,xy,kind)
                definition = definitions[definitions.numero.eq(i)].iloc[0]
                reason = str(definition.cuidado)
                if coord is None:
                    reason = 'Limite inferior nasal artefatual, nao localizado com seguranca.' if kind=='cranio' else 'Limite cutaneo inferior do queixo oculto/ambiguo pela barba.'
                rows.append(dict(id_pessoa=identity,ano=entry['ano'],split=entry['split'],numero=i,
                    modalidade=kind,ponto=definition[kind],x_px=coord[0] if coord else None,
                    y_px=coord[1] if coord else None,estado='PROPOSTA_IA' if coord else 'DUVIDA',
                    validado=False,observacao=reason,origem='Inspecao visual individual Codex',
                    protocolo='correspondencias_regionais_11_v1',input_hash=v['hash'],imagem=v['path']))
                if coord:
                    x,y=coord
                    radius=max(3,round(min(image.size)*.006))
                    draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline='#00ffff',width=max(2,radius//3))
                    draw.text((x+radius+2,y-radius-10),str(i),fill='#00ffff',stroke_width=1,stroke_fill='black')
            marked.save(pair_folder/(kind+'_marcado.jpg'),quality=95)
            frames[kind]=marked
        # Render equal-sized panels with numbered marks at the inspected scale.
        sheet = Image.open(ROOT/(identity+'_inspecao.jpg')).convert('RGB')
        draw = ImageDraw.Draw(sheet)
        for kind in ['cranio','face']:
            for i,xy in enumerate(PROPOSALS[identity][kind],1):
                if xy is None:
                    continue
                x,y=xy
                draw.ellipse((x-4,y-4,x+4,y+4),outline='#00ffff',width=2)
                draw.rectangle((x+5,y-15,x+23,y),fill='black')
                draw.text((x+7,y-14),str(i),fill='#00ffff')
        sheet.save(pair_folder/'par_marcado.jpg',quality=95)
        pd.DataFrame(rows).to_csv(pair_folder/'pontos.csv',index=False)
        all_rows.extend(rows)
        summary.append(dict(id_pessoa=identity,ano=entry['ano'],split=entry['split'],
            propostas=sum(r['estado']=='PROPOSTA_IA' for r in rows),duvidas=sum(r['estado']=='DUVIDA' for r in rows)))
        print(identity,summary[-1]['propostas'],'propostas',flush=True)
    pd.DataFrame(summary).to_csv(output/'resumo.csv',index=False)
    pd.DataFrame(all_rows).to_csv(output/'pontos.csv',index=False)
    metadata=dict(entradas=entries,treinamento_autorizado=False,validado=False,
        autoria='Propostas visuais individuais de IA, sem validacao anatomica',
        pareamento='IDs do manifesto, nao reconhecimento visual',
        observacao='Lados pressupostos nao espelhados. Correspondencias regionais, nao coincidencia osso-pele. Escalas independentes.')
    (output/'lote.json').write_text(json.dumps(metadata,indent=2))
    assert len(all_rows)==220 and len(summary)==10
    assert not pd.DataFrame(all_rows).duplicated(['id_pessoa','modalidade','numero']).any()
    print(output,flush=True)


if __name__=='__main__':
    if '--annotate' in sys.argv:
        annotate()
    else:
        inspect()
