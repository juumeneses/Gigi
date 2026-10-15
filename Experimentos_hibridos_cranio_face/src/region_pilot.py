"""Isolated regional audit. No training, anatomical approval or legacy writes."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

from .common import LAB_ROOT, REPO_ROOT, read_image, sha256_file

VERSION = 'regional_audit_v1'
RUNS = LAB_ROOT/'resultados/piloto_regioes'
STAGES = ['extracao', 'regioes', 'estabilidade', 'relatorio']
FACE_GROUPS = {
    'olho_direito': [33,133,159,145], 'olho_esquerdo': [263,362,386,374],
    'nariz': [64,294,168,2], 'central': [33,263,64,294,168,2],
    'mandibula_queixo': [172,397,152,234,454],
}
SKULL_GROUPS = {'olho_direito':[5,10], 'olho_esquerdo':[6,11],
                'nariz':[2,3,4], 'central':[1,2,3,4,5,6],
                'mandibula_queixo':[7,8,9]}
FACE_EQUIVALENTS = {1:9,2:2,3:64,4:294,5:33,6:263,7:152,8:234,9:454,10:133,11:362}
VISIBILITY = ['NAO_AVALIADO','VISIVEL','OCULTO','CORTADO','INDETERMINADO']
VARIANTS = ['original','brilho_090','brilho_110','escala_090','escala_110','rotacao_-3','rotacao_3']


def digest(value):
    normalized=json.loads(json.dumps(value,allow_nan=False))
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic(path, content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def json_save(path, data):
    atomic(path,json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False).encode())


def image_save(path, im):
    ok,data=cv2.imencode(Path(path).suffix,im)
    if not ok: raise ValueError('Falha ao codificar imagem')
    atomic(path,data.tobytes())


def image_input(path):
    path=Path(path)
    if not path.exists(): return None,'AUSENTE'
    with path.open('rb') as f:
        if f.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            return None,'LFS_NAO_BAIXADO'
    im=read_image(path)
    return (im,'OK_TECNICO') if im is not None else (None,'ILEGIVEL')


def source_manifest():
    batch=LAB_ROOT/'dataset/pares_visuais/lote_10_20260926/runs/20260926T211414700889Z'
    single=LAB_ROOT/'dataset/referencia_anatomica/par_20260926T210854690712Z_2017_30_EC'
    entries=json.loads((batch/'lote.json').read_text())['entradas']
    split=pd.read_csv(LAB_ROOT/'dataset/splits/split_identidades.csv').set_index('id_pessoa')
    manifest=pd.read_csv(LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv')
    previous=pd.read_csv(single/'correspondencias.csv')
    provenance=json.loads((single/'proveniencia.json').read_text())
    items=[]
    for entry in entries:
        frame=pd.read_csv(batch/entry['id_pessoa']/'pontos.csv')
        for kind in ['face','cranio']:
            rows=frame[frame.modalidade.eq(kind)]
            proposals=[dict(numero=int(r.numero),x=float(r.x_px) if pd.notna(r.x_px) else None,
                            y=float(r.y_px) if pd.notna(r.y_px) else None,
                            estado=str(r.estado),observacao=str(r.observacao)) for r in rows.itertuples()]
            view=entry['views'][kind]
            items.append(dict(id=entry['id_pessoa'],modalidade=kind,path=view['path'],
                              expected_hash=view['hash'],proposals=proposals))
    for kind in ['face','cranio']:
        path=next(p for p in provenance['original_hashes'] if ('Treinamento_faces/' in p)==(kind=='face'))
        proposals=[dict(numero=int(r['numero']),x=float(r[kind+'_x']),y=float(r[kind+'_y']),
                        estado='PROPOSTA_IA',observacao=str(r['cuidado'])) for r in previous.to_dict('records')]
        items.append(dict(id='2017_30_EC',modalidade=kind,path=path,
                          expected_hash=provenance['original_hashes'][path],proposals=proposals))
    assert len(items)==22 and len({(i['id'],i['modalidade']) for i in items})==22
    for item in items:
        row=split.loc[item['id']]
        if row['split'] not in ['treino','validacao']: raise ValueError('Teste final nao permitido')
        record=manifest[manifest.id_pessoa.eq(item['id'])]
        if len(record)!=1: raise ValueError('ID ausente ou ambiguo no manifesto')
        values=record.iloc[0].astype(str).tolist()
        if item['modalidade']=='face' and item['path'] not in values: raise ValueError('Face divergente do manifesto')
        if item['modalidade']=='cranio' and Path(item['path']).stem not in values: raise ValueError('CF divergente do manifesto')
        item.update(ano=int(row['ano']),split=str(row['split']))
    return sorted(items,key=lambda i:(i['id'],i['modalidade']))


def box_for(points, required, width, height):
    if any(k not in points for k in required):
        return dict(status='REGIAO_INDISPONIVEL',reason='PONTOS_NECESSARIOS_AUSENTES')
    xy=np.array([points[k] for k in required],dtype=float)
    if not np.isfinite(xy).all(): return dict(status='REGIAO_INDISPONIVEL',reason='COORDENADAS_INVALIDAS')
    low=xy.min(0); high=xy.max(0); margin=np.maximum((high-low)*.20,4)
    raw=np.r_[low-margin,high+margin]
    clipped=np.array([max(0,raw[0]),max(0,raw[1]),min(width,raw[2]),min(height,raw[3])])
    area=float(np.prod(raw[2:]-raw[:2])); valid=np.maximum(clipped[2:]-clipped[:2],0)
    if np.any(valid<=0): return dict(status='REGIAO_INDISPONIVEL',reason='FORA_DA_IMAGEM')
    fraction=max(0.,1-float(np.prod(valid))/area)
    return dict(status='ATENCAO' if fraction>0 else 'OK_TECNICO',raw=raw.tolist(),
                box=clipped.tolist(),margin=margin.tolist(),outside_fraction=fraction,
                visibilidade='NAO_AVALIADO',reason='BOX_PARCIAL' if fraction>0 else '')


def letterbox(im,size=256):
    h,w=im.shape[:2]; scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    x,y=(size-nw)//2,(size-nh)//2
    result=np.zeros((size,size,3),np.uint8)
    result[y:y+nh,x:x+nw]=cv2.resize(im,(nw,nh))
    return result,dict(scale=scale,offset=[x,y],size=[nw,nh],input_size=[w,h])


def transform_points(points, matrix):
    p=np.asarray(points,dtype=float).reshape(-1,2)
    return np.c_[p,np.ones(len(p))]@np.asarray(matrix).T


def variant(im,name):
    h,w=im.shape[:2]; matrix=np.array([[1.,0,0],[0,1.,0]])
    if name.startswith('brilho'):
        factor=.9 if name.endswith('090') else 1.1
        return np.clip(im.astype(float)*factor,0,255).astype(np.uint8),matrix
    if name.startswith('escala') or name.startswith('rotacao'):
        scale=(.9 if name.endswith('090') else 1.1) if name.startswith('escala') else 1.
        angle=float(name.split('_')[1]) if name.startswith('rotacao') else 0.
        matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),angle,scale)
        return cv2.warpAffine(im,matrix,(w,h)),matrix
    return im.copy(),matrix


def clipped_by_transform(box,matrix,w,h):
    x1,y1,x2,y2=box
    corners=transform_points([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],matrix)
    return bool((corners[:,0]<0).any() or (corners[:,1]<0).any() or
                (corners[:,0]>w).any() or (corners[:,1]>h).any())


def iou(a,b):
    a=np.asarray(a); b=np.asarray(b)
    intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersection
    return float(intersection/union) if union else None


def segment(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    touch=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    return mask,contours,touch


class Mesh:
    def __init__(self):
        import mediapipe as mp
        self.mp=mp
    def detect(self,im):
        # Static mode and a fresh graph prevent variant-order tracking effects.
        with self.mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,
                refine_landmarks=True,min_detection_confidence=.5) as detector:
            result=detector.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks: return []
        h,w=im.shape[:2]
        return [dict(idx=i,x=float(p.x*w),y=float(p.y*h),z=float(p.z),detectado=True,
                     visibilidade='NAO_AVALIADO') for i,p in enumerate(result.multi_face_landmarks[0].landmark)]


def key(item): return item['id']+'__'+item['modalidade']


def fingerprint(item,config):
    path=REPO_ROOT/item['path']
    return digest(dict(item=item,config=config,current_hash=sha256_file(path) if path.exists() else 'AUSENTE'))


def verified(path,token=None):
    try:
        data=json.loads(path.read_text())
        checksum=data.pop('payload_hash')
        if digest(data)!=checksum: return False
        if token is not None and data['fingerprint']!=token: return False
        for filename,expected in data['assets'].items():
            asset=path.parent/filename
            if not asset.exists() or sha256_file(asset)!=expected: return False
        return True
    except (OSError,ValueError,KeyError): return False


def seal(path,data,assets,token):
    data.update(fingerprint=token,assets={str(p.relative_to(path.parent)):sha256_file(p) for p in assets})
    data['payload_hash']=digest(data)
    json_save(path,data)


def extract(folder,item,mesh,config):
    im,status=image_input(REPO_ROOT/item['path']); assets=[]
    data=dict(item=item,status=status,reason='',points=[],selected=[],contours=[],proposals=item['proposals'])
    if im is None:
        data.update(status='ERRO',reason=status)
        return data,assets
    if sha256_file(REPO_ROOT/item['path'])!=item['expected_hash']:
        data.update(status='ERRO',reason='ORIGINAL_MODIFICADO'); return data,assets
    h,w=im.shape[:2]; data.update(width=w,height=h,medidas_globais_liberadas=False)
    data['status']='OK_TECNICO'
    if item['modalidade']=='face':
        try: points=mesh.detect(im)
        except Exception as error:
            data.update(status='ERRO',reason=f'MEDIAPIPE: {type(error).__name__}: {error}')
            return data,assets
        data['points']=points; data['quantidade_pontos']=len(points)
        if not points: data.update(status='ATENCAO',reason='FACE_NAO_DETECTADA')
        lookup={p['idx']:p for p in points}
        for name,spec in config['mapping']['landmarks'].items():
            if all(i in lookup for i in spec['indices']):
                xy=np.mean([[lookup[i]['x'],lookup[i]['y']] for i in spec['indices']],axis=0)
                data['selected'].append(dict(name=name,indices=spec['indices'],x=float(xy[0]),y=float(xy[1]),
                                             detectado=True,visibilidade='NAO_AVALIADO'))
        connections=mesh.mp.solutions.face_mesh
        for name,edges in [('oval',connections.FACEMESH_FACE_OVAL),('olho_direito',connections.FACEMESH_RIGHT_EYE),
                           ('olho_esquerdo',connections.FACEMESH_LEFT_EYE),('labios',connections.FACEMESH_LIPS)]:
            data['contours'].append(dict(name=name,edges=[[int(a),int(b)] for a,b in sorted(edges)],estimado=True))
        data['discordancia']=[]
        for p in item['proposals']:
            idx=FACE_EQUIVALENTS[p['numero']]
            if idx in lookup and p['x'] is not None and p['y'] is not None:
                distance=float(np.hypot(lookup[idx]['x']-p['x'],lookup[idx]['y']-p['y']))
                data['discordancia'].append(dict(numero=p['numero'],indice=idx,deslocamento_px=distance,
                    deslocamento_diagonal=distance/np.hypot(w,h),tipo='DISCORDANCIA_ENTRE_METODOS'))
    else:
        mask,contours,touch=segment(im)
        mask_path=folder/'mask.png'; image_save(mask_path,mask); assets.append(mask_path)
        data.update(contours=[c.reshape(-1,2).tolist() for c in contours],contato_borda=touch,
                    foreground_fraction=float((mask>0).mean()),status='ATENCAO',
                    reason='SEGMENTACAO_NAO_REVISADA'+(';CONTATO_BORDA' if touch else ''),
                    medidas_globais_liberadas=False)
        if not contours: data['reason']='MASCARA_VAZIA'
    return data,assets


def make_regions(folder,item,extraction):
    im,_=image_input(REPO_ROOT/item['path'])
    if im is None or extraction['status']=='ERRO': return dict(status='ERRO',regions={},reason='EXTRACAO_FALHOU'),[]
    h,w=im.shape[:2]
    if item['modalidade']=='face':
        points={p['idx']:[p['x'],p['y']] for p in extraction['points']}; groups=FACE_GROUPS
    else:
        points={p['numero']:[p['x'],p['y']] for p in item['proposals'] if p['x'] is not None and p['y'] is not None}; groups=SKULL_GROUPS
    regions={}; assets=[]
    for name,required in groups.items():
        b=box_for(points,required,w,h)
        b['origem']='MEDIAPIPE' if item['modalidade']=='face' else 'PROPOSTAS_IA_ANTERIORES'
        b['required']=required
        if 'box' in b:
            x1,y1,x2,y2=b['box']; bounds=[int(np.floor(x1)),int(np.floor(y1)),int(np.ceil(x2)),int(np.ceil(y2))]
            l,t,r,d=bounds; crop,transform=letterbox(im[t:d,l:r])
            p=folder/(name+'.jpg'); image_save(p,crop); assets.append(p)
            b.update(crop_bounds=bounds,transform=transform,arquivo=p.name)
        regions[name]=b
    return dict(status='ATENCAO' if any(b['status']!='OK_TECNICO' for b in regions.values()) else 'OK_TECNICO',regions=regions),assets


def stability(folder,item,extraction,regions,mesh,log):
    im,_=image_input(REPO_ROOT/item['path'])
    if im is None or extraction['status']=='ERRO':
        return dict(status='ERRO',rows=[dict(variante=v,status='ERRO',reason='EXTRACAO_FALHOU') for v in VARIANTS]),[]
    h,w=im.shape[:2]; rows=[]; assets=[]
    initial={p['idx']:[p['x'],p['y']] for p in extraction['points']}
    original_mask=segment(im)[0] if item['modalidade']=='cranio' else None
    for name in VARIANTS:
        log(f"{key(item)} variante {name}")
        altered,matrix=variant(im,name); inverse=cv2.invertAffineTransform(matrix)
        p=folder/(name+'.jpg'); image_save(p,altered); assets.append(p)
        cut={r:clipped_by_transform(b['box'],matrix,w,h) for r,b in regions['regions'].items() if 'box' in b}
        row=dict(variante=name,matrix=matrix.tolist(),inverse=inverse.tolist(),regioes_cortadas=cut,
                 status='OK_TECNICO',reason='',detectado=False,mediana_px=None,p95_px=None,boxes_iou={})
        if item['modalidade']=='face':
            try: found=extraction['points'] if name=='original' else mesh.detect(altered)
            except Exception as error:
                row.update(status='ERRO',reason=f'MEDIAPIPE: {type(error).__name__}: {error}')
                rows.append(row); continue
            back=transform_points([[p['x'],p['y']] for p in found],inverse) if found else []
            restored={p['idx']:xy.tolist() for p,xy in zip(found,back)}
            row.update(detectado=bool(found),quantidade_pontos=len(found),points_restored=restored)
            errors=[]
            for idx,xy in initial.items():
                dest=transform_points([xy],matrix)[0]
                if idx in restored and 0<=dest[0]<w and 0<=dest[1]<h:
                    errors.append(float(np.linalg.norm(np.array(restored[idx])-xy)))
            if errors:
                row.update(mediana_px=float(np.median(errors)),p95_px=float(np.percentile(errors,95)),
                           mediana_diagonal=float(np.median(errors)/np.hypot(w,h)))
            for region,required in FACE_GROUPS.items():
                baseline=regions['regions'][region]
                # Transform the same regional point IDs back before boxing.
                b=box_for(restored,required,w,h)
                row['boxes_iou'][region]=iou(baseline['box'],b['box']) if 'box' in baseline and 'box' in b and not cut.get(region,True) else None
            if any(cut.values()):
                row.update(mediana_px=None,p95_px=None,mediana_diagonal=None)
            if not found or not initial: row.update(status='ATENCAO',reason='FACE_NAO_DETECTADA_OU_SEM_BASELINE')
        else:
            mask,_,_=segment(altered)
            restored=cv2.warpAffine(mask,inverse,(w,h),flags=cv2.INTER_NEAREST)
            valid=cv2.warpAffine(np.full((h,w),255,np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST)>0
            union=((restored>0)|(original_mask>0))&valid
            row.update(detectado=bool(mask.any()),mask_iou=float((((restored>0)&(original_mask>0))&valid).sum()/union.sum()) if union.any() else None,
                       valid_fraction=float(valid.mean()),status='ATENCAO',reason='ESTABILIDADE_SEGMENTACAO_NAO_DETECTOR_DE_PONTOS')
        if any(cut.values()): row.update(status='ATENCAO',reason=row['reason']+';REGIAO_CORTADA_NA_VARIANTE')
        rows.append(row)
    return dict(status='ATENCAO' if any(r['status']!='OK_TECNICO' for r in rows) else 'OK_TECNICO',rows=rows),assets


def render_overlay(im,extraction,regions,layers):
    out=im.copy(); size=max(2,min(im.shape[:2])//180)
    lookup={p['idx']:(p['x'],p['y']) for p in extraction['points']}
    if 'Malha completa' in layers:
        for x,y in lookup.values(): cv2.circle(out,(round(x),round(y)),size,(0,190,0),-1)
    if 'Seleção regional' in layers:
        for p in extraction['selected']: cv2.circle(out,(round(p['x']),round(p['y'])),size*2,(0,255,255),2)
    if 'Propostas anteriores' in layers:
        for p in extraction['proposals']:
            if p['x'] is not None and p['y'] is not None:
                xy=(round(p['x']),round(p['y'])); cv2.circle(out,xy,size*2,(255,255,0),2)
                cv2.putText(out,str(p['numero']),(xy[0]+4,xy[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
    if 'Contornos' in layers:
        for contour in extraction['contours']:
            if isinstance(contour,dict):
                for a,b in contour['edges']:
                    if a in lookup and b in lookup:
                        cv2.line(out,tuple(map(round,lookup[a])),tuple(map(round,lookup[b])),(255,150,0),2)
            else: cv2.polylines(out,[np.array(contour,np.int32)],True,(255,150,0),2)
    if 'Boxes' in layers:
        for name,b in regions.get('regions',{}).items():
            if 'box' in b:
                x1,y1,x2,y2=map(round,b['box']); cv2.rectangle(out,(x1,y1),(x2,y2),(190,0,190),2)
                cv2.putText(out,name,(x1,max(15,y1)),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,0,190),1)
    return out


def report(run,items,config):
    points_rows=[]; boxes_rows=[]; stability_rows=[]; disagreement_rows=[]
    rows=[]; parts=['<!doctype html><meta charset="utf-8"><title>Piloto de regioes</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:48%;vertical-align:top}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>Pontos, contornos e regioes face-cranio</h1><p>11 pares de desenvolvimento; pseudorrotulos. Sem validacao anatomica independente. Sem treino ou matching.</p>']
    assets=[]
    for item in items:
        folder=run/key(item)
        ex=json.loads((folder/'extracao/data.json').read_text()); reg=json.loads((folder/'regioes/data.json').read_text())
        stab=json.loads((folder/'estabilidade/data.json').read_text())
        for point in ex['points']:
            points_rows.append(dict(id=item['id'],modalidade=item['modalidade'],**point))
        for region,box in reg.get('regions',{}).items():
            boxes_rows.append(dict(id=item['id'],modalidade=item['modalidade'],regiao=region,**box))
        for value in ex.get('discordancia',[]): disagreement_rows.append(dict(id=item['id'],**value))
        for value in stab['rows']:
            stability_rows.append(dict(id=item['id'],modalidade=item['modalidade'],**{
                k:v for k,v in value.items() if k not in ['points_restored','matrix','inverse']}))
        row=dict(id=item['id'],modalidade=item['modalidade'],ano=item['ano'],split=item['split'],
                 status=ex['status'],motivo=ex.get('reason',''),pontos=len(ex['points']),
                 regioes_disponiveis=sum('box' in b for b in reg.get('regions',{}).values()),
                 variantes_total=len(stab['rows']),variantes_detectadas=sum(r.get('detectado',False) for r in stab['rows']))
        rows.append(row)
        parts.append('<h2>'+html.escape(key(item))+'</h2><p>'+html.escape(str(row))+'</p>')
        im,_=image_input(REPO_ROOT/item['path'])
        if im is not None and ex['status']!='ERRO':
            over=render_overlay(im,ex,reg,['Seleção regional','Propostas anteriores','Contornos','Boxes'])
            for name,image in [('original',im),('saida',over)]:
                thumb=cv2.resize(image,(max(1,round(image.shape[1]*min(1,700/image.shape[0]))),min(700,image.shape[0])))
                p=run/'previews'/(key(item)+'_'+name+'.jpg'); image_save(p,thumb); assets.append(p)
                parts.append('<img alt="'+name+'" src="data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()+'">')
        parts.append('<p>Comparacao visual nao significa aprovacao anatomica.</p>')
    summary=pd.DataFrame(rows); csv=run/'resumo.csv'; atomic(csv,summary.to_csv(index=False).encode('utf-8-sig')); assets.append(csv)
    parts.append(summary.to_html(index=False,escape=True))
    parts.append('<h2>Estabilidade tecnica</h2><p>Mediana/P95 por variante, em pixels originais. Celulas vazias incluem falha ou corte; nao equivalem a zero. IoU craniano mede mascara, nao ponto anatomico.</p>')
    metrics=pd.DataFrame(stability_rows)
    columns=[c for c in ['id','modalidade','variante','status','reason','detectado','mediana_px','p95_px','mediana_diagonal','mask_iou'] if c in metrics]
    parts.append(metrics[columns].to_html(index=False,escape=True,na_rep='NA'))
    parts.append('<h2>Discordancia entre metodos</h2><p>Propostas anteriores versus indices candidatos MediaPipe; nao e erro anatomico.</p>')
    parts.append(pd.DataFrame(disagreement_rows).to_html(index=False,escape=True))
    for name,values in [('pontos',points_rows),('boxes',boxes_rows),('estabilidade',stability_rows),('discordancia',disagreement_rows)]:
        p=run/(name+'.csv'); atomic(p,pd.DataFrame(values).to_csv(index=False).encode('utf-8-sig')); assets.append(p)
    parts.append('<h2>Disponibilidade regional</h2>')
    for modality in ['face','cranio']:
        for region in FACE_GROUPS:
            values=[b for b in boxes_rows if b['modalidade']==modality and b['regiao']==region]
            available=sum('box' in b for b in values)
            parts.append(f'<p>{modality} / {region}: {available}/11 boxes disponiveis. '+
                         ('Manter para revisao visual.' if available==11 else 'Manter apenas casos com pontos suficientes; nao imputar ausencias.')+'</p>')
    parts.append('<h2>Recomendacao</h2><p>Priorizar auditoria das regioes orbitais e nasal. Manter mandibula e zigios como exploratorios; nenhuma regiao foi validada. Boxes cranianas dependem das propostas anteriores. Estabilidade de segmentacao nao e estabilidade de um detector anatomico.</p>')
    target=run/'relatorio.html'; atomic(target,'\n'.join(parts).encode()); assets.append(target)
    return assets


def _run_audit(run=None,stage='all',resume=False,force=False,limit=None):
    import mediapipe as mp
    if stage not in ['all']+STAGES: raise ValueError('Etapa invalida')
    if limit is not None and limit<1: raise ValueError('Limit deve ser positivo')
    if force and resume: raise ValueError('Escolha force OU resume')
    mapping=json.loads((LAB_ROOT/'configs/landmarks_face.yaml').read_text())
    config=dict(version=VERSION,code_hash=sha256_file(Path(__file__)),mediapipe=mp.__version__,opencv=cv2.__version__,
                numpy=np.__version__,python=__import__('platform').python_version(),mapping=mapping,
                static_image_mode=True,refine_landmarks=True,max_num_faces=1,min_detection_confidence=.5,
                variants=VARIANTS,box_margin=.2,minimum_margin=4,matching_liberado=False)
    run=Path(run).resolve() if run else RUNS/pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    if force: run=run.parent/(run.name+'_refazer_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ'))
    if (run/'config.json').exists():
        if not resume: raise ValueError('Execucao existe: use --resume ou --force')
        if json.loads((run/'config.json').read_text())!=config: raise ValueError('Codigo/configuracao mudou: crie nova execucao')
        items=json.loads((run/'entradas.json').read_text())
    else:
        if stage not in ['all','extracao']: raise ValueError('Pre-requisito ausente: execute extracao primeiro')
        items=source_manifest(); run.mkdir(parents=True,exist_ok=True)
        json_save(run/'config.json',config); json_save(run/'entradas.json',items)
    def log(message):
        print(message,flush=True)
        with (run/'execucao.log').open('a') as f: f.write(message+'\n')
    log('RUN_DIR='+str(run)); started=time.time()
    mesh=Mesh(); stages=STAGES if stage=='all' else [stage]
    processed=0
    for step in stages:
        if step=='relatorio':
            for item in items:
                token=fingerprint(item,config)
                for needed in STAGES[:3]:
                    if not verified(run/key(item)/needed/'data.json',token): raise ValueError(f'Pre-requisito invalido: {key(item)} {needed}')
            assets=report(run,items,config)
            seal(run/'relatorio.json',dict(status='OK_TECNICO'),assets,digest(config))
            continue
        for index,item in enumerate(items,1):
            folder=run/key(item)/step; folder.mkdir(parents=True,exist_ok=True)
            target=folder/'data.json'; token=fingerprint(item,config)
            for needed in STAGES[:STAGES.index(step)]:
                if not verified(run/key(item)/needed/'data.json',token): raise ValueError(f'Pre-requisito invalido: {key(item)} {needed}')
            if resume and verified(target,token):
                log(f'{step} {index}/{len(items)} {key(item)} reutilizado'); continue
            log(f'{step} {index}/{len(items)} {key(item)} iniciando')
            if step=='extracao': data,assets=extract(folder,item,mesh,config)
            else:
                ex=json.loads((run/key(item)/'extracao/data.json').read_text())
                if step=='regioes': data,assets=make_regions(folder,item,ex)
                else:
                    reg=json.loads((run/key(item)/'regioes/data.json').read_text())
                    data,assets=stability(folder,item,ex,reg,mesh,log)
            seal(target,data,assets,token)
            processed+=1
            json_save(run/'progresso.json',dict(etapa=step,item=key(item),indice=index,total=len(items),duracao_s=time.time()-started,status='RODANDO'))
            log(f'{step} {key(item)} {data["status"]}')
            if limit and processed>=limit:
                json_save(run/'progresso.json',dict(status='INTERROMPIDO_CONTROLADO',duracao_s=time.time()-started))
                return run
    json_save(run/'progresso.json',dict(status='CONCLUIDO',etapa=stage,duracao_s=time.time()-started))
    log('Concluido '+str(run)); return run


def run_audit(run=None,stage='all',resume=False,force=False,limit=None):
    # The advisory lock is released by the OS even after an interrupted process.
    import fcntl
    RUNS.mkdir(parents=True,exist_ok=True)
    with (RUNS/'.execucao.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Outro piloto esta em execucao')
        return _run_audit(run,stage,resume,force,limit)


def save_review(run,item,region,session,reviewer,state,reason,flags):
    if session not in ['A','B'] or state not in VISIBILITY or not reviewer.strip() or not reason.strip():
        raise ValueError('Sessao, autor, estado e motivo sao obrigatorios')
    if region not in FACE_GROUPS: raise ValueError('Regiao invalida')
    path=run/key(item)/'regioes/data.json'
    if not verified(path): raise ValueError('Saida de regiao ausente ou corrompida')
    source=REPO_ROOT/item['path']
    if not source.exists() or sha256_file(source)!=item['expected_hash']: raise ValueError('Original mudou')
    row=dict(id=item['id'],modalidade=item['modalidade'],regiao=region,sessao=session,
             autor=reviewer.strip(),estado=state,motivo=reason.strip(),flags=flags,
             input_hash=item['expected_hash'],output_hash=sha256_file(path),version=VERSION,
             validado=False,data=pd.Timestamp.now(tz='UTC').isoformat())
    dest=run/'revisoes'/ (pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    json_save(dest,row); return dest
