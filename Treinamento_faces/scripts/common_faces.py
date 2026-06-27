# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json, shutil
import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
GENERATED_IMAGE_MARKERS=('_preview','_crop','_corrigida','_antes','_depois')
FACE_OVAL_ORDER=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
MAIN_LANDMARKS=[10,152,234,454,33,263,61,291,1,4,13,14,127,356,58,288,172,397]
LEFT_EYE_OUTER=33; RIGHT_EYE_OUTER=263; CHIN=152; FOREHEAD=10; LEFT_FACE=234; RIGHT_FACE=454; MOUTH_LEFT=61; MOUTH_RIGHT=291
METRIC_RULES={
 'score_original':{'ok':(70,200),'warn':(55,70),'desc':'Score geral da imagem original. >=70 bom; 55–70 aceitável visualmente; <55 revisar.'},
 'score_sugerido':{'ok':(70,200),'warn':(55,70),'desc':'Score da melhor sugestão/correção.'},
 'ganho_score':{'ok':(3,999),'warn':(0.5,3),'desc':'Diferença sugestão-original. >=3 indica melhora relevante.'},
 'pontos_fora_imagem':{'ok':(0,0.01),'warn':(0.01,0.05),'desc':'Proporção de pontos fora da imagem. Ideal 0.'},
 'original_pontos_fora_imagem':{'ok':(0,0.01),'warn':(0.01,0.05),'desc':'Pontos fora na original.'},
 'sugerido_pontos_fora_imagem':{'ok':(0,0.01),'warn':(0.01,0.05),'desc':'Pontos fora na sugestão.'},
 'area_face_normalizada':{'ok':(0.08,0.80),'warn':(0.06,0.90),'desc':'Área relativa da face na imagem.'},
 'original_area_face_normalizada':{'ok':(0.08,0.80),'warn':(0.06,0.90),'desc':'Área relativa da face original.'},
 'sugerido_area_face_normalizada':{'ok':(0.08,0.80),'warn':(0.06,0.90),'desc':'Área relativa da sugestão.'},
 'margem_borda':{'ok':(0.02,1.0),'warn':(0.0,0.02),'desc':'Margem mínima da face até a borda.'},
 'original_margem_borda':{'ok':(0.02,1.0),'warn':(0.0,0.02),'desc':'Margem da original.'},
 'sugerido_margem_borda':{'ok':(0.02,1.0),'warn':(0.0,0.02),'desc':'Margem da sugestão.'},
 'inclinacao_olhos_graus':{'ok':(0,10),'warn':(10,18),'desc':'Inclinação entre olhos. Ideal <10°.'},
 'original_inclinacao_olhos_graus':{'ok':(0,10),'warn':(10,18),'desc':'Inclinação dos olhos na original.'},
 'sugerido_inclinacao_olhos_graus':{'ok':(0,10),'warn':(10,18),'desc':'Inclinação dos olhos na sugestão.'},
 'razao_largura_altura_face':{'ok':(0.50,1.05),'warn':(0.38,1.20),'desc':'Largura/altura da face.'},
 'razao_olhos_largura_face':{'ok':(0.48,0.78),'warn':(0.42,0.85),'desc':'Distância olhos/largura da face.'},
 'razao_boca_largura_face':{'ok':(0.22,0.55),'warn':(0.16,0.65),'desc':'Largura boca/largura da face.'},
 'razao_bbox_contorno':{'ok':(0.45,1.10),'warn':(0.35,1.25),'desc':'Largura/altura da caixa do contorno.'},
 'score_outlier_pca':{'ok':(0,3),'warn':(3,6),'desc':'Distância no PCA. Alto = revisar visualmente, não erro automático.'},
 'prob_masculino_fairface':{'ok':(0.80,1.0),'warn':(0.65,0.80),'desc':'Probabilidade masculino pelo FairFace.'},
 'prob_feminino_fairface':{'ok':(0.80,1.0),'warn':(0.65,0.80),'desc':'Probabilidade feminino pelo FairFace.'},
 'prob_masculino_modelo':{'ok':(0.80,1.0),'warn':(0.65,0.80),'desc':'Probabilidade masculino pelo modelo próprio.'},
 'prob_feminino_modelo':{'ok':(0.80,1.0),'warn':(0.65,0.80),'desc':'Probabilidade feminino pelo modelo próprio.'},
}
def as_root(root='.'): return Path(root).expanduser().resolve()
def faces_root(root): return root/'dados_originais'/'faces'
def resultados_root(root): return root/'resultados'
def year_dir(root,year): return faces_root(root)/str(year)
def images_dir(root,year): return year_dir(root,year)/'images'
def marks_dir(root,year): return year_dir(root,year)/'marcacoes_landmarks'
def previews_dir(root,year): return marks_dir(root,year)/'previews'
def corrected_dir(root,year): return year_dir(root,year)/'images_melhoradas_nitidez'
def backup_dir(root,year): return year_dir(root,year)/'backup_originais_nitidez'
def genero_dir(root): return resultados_root(root)/'genero_faces'
def matching_features_dir(root): return resultados_root(root)/'features_matching_faces'
def ensure_base_dirs(root):
    faces_root(root).mkdir(parents=True,exist_ok=True)
    for sub in ['checks','falhas','analises_faces','relatorios','logs','curadoria_matching','revisao_manual_faces','genero_faces','features_matching_faces']:
        (resultados_root(root)/sub).mkdir(parents=True,exist_ok=True)
def discover_years(root,years=None):
    if years: return sorted([str(y) for y in years])
    fr=faces_root(root)
    if not fr.exists(): return []
    return sorted([p.name for p in fr.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name)==4])
def prepare_year(root,year):
    images_dir(root,year).mkdir(parents=True,exist_ok=True)
    for sub in ['landmarks_csv','landmarks_json','contorno_rosto_csv','contorno_cranio_like_csv','previews']:
        (marks_dir(root,year)/sub).mkdir(parents=True,exist_ok=True)
def read_csv_safe(path):
    path=Path(path)
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()
def safe_to_csv(df,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False,encoding='utf-8-sig')
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def read_image(path):
    try:
        data=np.fromfile(str(path),dtype=np.uint8); return cv2.imdecode(data,cv2.IMREAD_COLOR)
    except Exception: return cv2.imread(str(path))
def write_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True); ext=path.suffix.lower(); ext=ext if ext in ['.jpg','.jpeg','.png'] else '.jpg'
    ok,enc=cv2.imencode(ext,image)
    if not ok: return False
    enc.tofile(str(path)); return True
def list_images_for_year(root,year):
    d=images_dir(root,year)
    if not d.exists(): return []
    return sorted([p for p in d.iterdir() if is_original_face_image(p)])
def is_generated_image_file(path):
    p=Path(path); stem=p.stem.lower()
    return any(m in stem for m in GENERATED_IMAGE_MARKERS)
def is_original_face_image(path):
    p=Path(path)
    if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS: return False
    stem=p.stem
    if is_generated_image_file(p): return False
    return stem.upper().endswith('_FFN')
def cleanup_generated_outputs(root,years=None,keep_logs=False):
    root=as_root(root); years=discover_years(root,years); removed=[]
    res=resultados_root(root)
    if res.exists():
        for child in res.iterdir():
            if keep_logs and child.name=='logs':
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(str(child))
    for y in years:
        md=marks_dir(root,y)
        if md.exists():
            shutil.rmtree(md); removed.append(str(md))
        legacy=[year_dir(root,y)/'images_corrigidas_auto',year_dir(root,y)/'backup_originais_auto']
        for d in [corrected_dir(root,y), backup_dir(root,y), *legacy]:
            if d.exists():
                shutil.rmtree(d); removed.append(str(d))
        imgdir=images_dir(root,y)
        if imgdir.exists():
            for p in list(imgdir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS and is_generated_image_file(p):
                    p.unlink(); removed.append(str(p))
    ensure_base_dirs(root)
    for y in years:
        prepare_year(root,y)
    return removed
def find_image_path(root,year,stem):
    for d in [images_dir(root,year), year_dir(root,year)]:
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem==stem: return p
    yd=year_dir(root,year)
    if yd.exists():
        for p in yd.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem==stem: return p
    return None
def preview_path(root,year,stem): return previews_dir(root,year)/f'{stem}_preview.jpg'
def draw_landmarks_preview(image,landmarks,face_oval_order=FACE_OVAL_ORDER):
    out=image.copy(); h,w=out.shape[:2]; gr=max(2,int(min(h,w)/380)); cr=max(4,int(min(h,w)/180)); th=max(2,int(min(h,w)/280))
    for lm in landmarks:
        x=int(lm.x*w); y=int(lm.y*h)
        if 0<=x<w and 0<=y<h: cv2.circle(out,(x,y),gr,(0,230,0),-1)
    pts=[]
    for idx in face_oval_order:
        if idx>=len(landmarks): continue
        lm=landmarks[idx]; x=int(lm.x*w); y=int(lm.y*h)
        if 0<=x<w and 0<=y<h: pts.append((x,y))
    if len(pts)>=2:
        arr=np.array(pts,dtype=np.int32); cv2.polylines(out,[arr],True,(255,0,0),th)
        for x,y in pts: cv2.circle(out,(x,y),cr,(0,0,255),-1)
    return out
def landmark_xy(lms,idx,w,h):
    lm=lms[idx]; return float(lm.x*w),float(lm.y*h)
def score_landmarks(image,lms):
    h,w=image.shape[:2]; xs=np.array([lm.x for lm in lms],float); ys=np.array([lm.y for lm in lms],float)
    out_ratio=float(((xs<0)|(xs>1)|(ys<0)|(ys>1)).mean()); x_min,x_max=xs.min(),xs.max(); y_min,y_max=ys.min(),ys.max()
    face_area=float((x_max-x_min)*(y_max-y_min)); border_margin=float(min(x_min,y_min,1-x_max,1-y_max))
    le=landmark_xy(lms,LEFT_EYE_OUTER,w,h); re=landmark_xy(lms,RIGHT_EYE_OUTER,w,h); eye_dx=re[0]-le[0]; eye_dy=re[1]-le[1]
    eye_dist=float(np.hypot(eye_dx,eye_dy)); eye_angle=float(abs(np.degrees(np.arctan2(eye_dy,eye_dx))))
    chin=landmark_xy(lms,CHIN,w,h); forehead=landmark_xy(lms,FOREHEAD,w,h); face_h=float(abs(chin[1]-forehead[1]))
    lf=landmark_xy(lms,LEFT_FACE,w,h); rf=landmark_xy(lms,RIGHT_FACE,w,h); face_w=float(abs(rf[0]-lf[0]))
    ml=landmark_xy(lms,MOUTH_LEFT,w,h); mr=landmark_xy(lms,MOUTH_RIGHT,w,h); mouth_w=float(abs(mr[0]-ml[0]))
    eye_ratio=eye_dist/max(face_w,1.0); mouth_ratio=mouth_w/max(face_w,1.0); wh=face_w/max(face_h,1.0)
    score=100.0
    if face_area<0.08: score-=25
    elif face_area<0.14: score-=10
    if face_area>0.80: score-=15
    score-=out_ratio*80.0
    if eye_angle>10: score-=min(30,(eye_angle-10)*2.5)
    if border_margin<0.005: score-=20
    elif border_margin<0.02: score-=10
    if not (0.50<=wh<=1.05): score-=12
    if not (0.48<=eye_ratio<=0.78): score-=10
    if not (0.22<=mouth_ratio<=0.55): score-=8
    return score, {'score':round(float(score),3),'pontos_fora_imagem':round(out_ratio,4),'area_face_normalizada':round(face_area,4),'margem_borda':round(border_margin,4),'inclinacao_olhos_graus':round(eye_angle,3),'razao_olhos_largura_face':round(eye_ratio,4),'razao_boca_largura_face':round(mouth_ratio,4),'razao_largura_altura_face':round(wh,4)}
def crop_face_by_landmarks(image,landmarks,margin=0.32):
    h,w=image.shape[:2]; xs=np.array([lm.x*w for lm in landmarks]); ys=np.array([lm.y*h for lm in landmarks])
    x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max(); bw=x2-x1; bh=y2-y1
    x1=int(max(0,x1-margin*bw)); x2=int(min(w,x2+margin*bw)); y1=int(max(0,y1-margin*bh)); y2=int(min(h,y2+margin*bh))
    if x2<=x1 or y2<=y1: return image.copy()
    crop=image[y1:y2,x1:x2].copy(); return crop if crop.size else image.copy()
def load_manual_status_map(root):
    p=resultados_root(root)/'revisao_manual_faces'/'revisao_manual_faces.csv'; df=read_csv_safe(p)
    if df.empty or not {'ano','id_imagem','status_manual'}.issubset(df.columns): return {}
    if 'timestamp' in df.columns: df=df.sort_values('timestamp')
    return {(str(r.get('ano','')),str(r.get('id_imagem',''))):{'status_manual':str(r.get('status_manual','')),'observacao_manual':str(r.get('observacao',''))} for _,r in df.iterrows()}
def save_manual_status(root,year,stem,status,obs=''):
    p=resultados_root(root)/'revisao_manual_faces'/'revisao_manual_faces.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    df=read_csv_safe(p); row={'timestamp':pd.Timestamp.now().isoformat(),'ano':str(year),'id_imagem':str(stem),'status_manual':str(status),'observacao':str(obs)}
    df=pd.concat([df,pd.DataFrame([row])],ignore_index=True); safe_to_csv(df,p); return p
def confidence_label(prob):
    try: p=float(prob)
    except Exception: return 'ERRO'
    return 'ALTA' if p>=0.80 else ('MEDIA' if p>=0.65 else ('BAIXA' if p>=0.55 else 'INDEFINIDA'))
def class_from_gender_probs(prob_m,prob_f,prefix=''):
    pmax=max(float(prob_m),float(prob_f)); conf=confidence_label(pmax)
    if pmax<0.55: return 'INDEFINIDO',conf
    return ('MASCULINO_PROVAVEL',conf) if prob_m>=prob_f else ('FEMININO_PROVAVEL',conf)
def genero_final_from_manual(status,classe_ia=''):
    s=str(status).upper(); c=str(classe_ia).upper()
    if s in ['CONFIRMAR_IA','ACEITAR_IA']:
        if 'MASCULINO' in c: return 'MASCULINO','FAIRFACE_CONFIRMADO_MANUALMENTE',True
        if 'FEMININO' in c: return 'FEMININO','FAIRFACE_CONFIRMADO_MANUALMENTE',True
        return 'INDEFINIDO','IA_INDEFINIDA_CONFIRMADA',False
    if s in ['MASCULINO','MASCULINO_MANUAL']: return 'MASCULINO','CORRIGIDO_MANUALMENTE',True
    if s in ['FEMININO','FEMININO_MANUAL']: return 'FEMININO','CORRIGIDO_MANUALMENTE',True
    if s in ['INDEFINIDO','INDEFINIDO_MANUAL']: return 'INDEFINIDO','INDEFINIDO_MANUAL',False
    if s in ['NAO_USAR','NÃO_USAR','NAO_USAR_TREINO']: return 'NAO_USAR','NAO_USAR_MANUAL',False
    return '', '', False
def normalizar_status_bool(valor): return str(valor).strip().upper() in ['TRUE','1','SIM','YES']
def color_status_value(v):
    s=str(v).upper()
    if s in ['OK','OK_AUTO','OK_MANUAL','MANTER_ORIGINAL_AUTO','CORRIGIDO_AUTO','TRUE','SIM','MASCULINO','FEMININO','MASCULINO_PROVAVEL','FEMININO_PROVAVEL','ALTA','CONCORDANTE_ALTA']:
        return 'background-color: #d1fae5; color: #065f46; font-weight: 600;'
    if 'REVISAR' in s or 'PENDENTE' in s or 'MEDIA' in s or 'BAIXA' in s or 'INDEFINIDO' in s or 'DIVERGENTE' in s:
        return 'background-color: #fef3c7; color: #92400e; font-weight: 600;'
    if 'FALHA' in s or 'EXCLUIR' in s or 'ERRO' in s or 'NAO_USAR' in s or 'FALSE' in s:
        return 'background-color: #fee2e2; color: #991b1b; font-weight: 600;'
    if 'CORRIGIR' in s or 'APLICAR' in s: return 'background-color: #dbeafe; color: #1e40af; font-weight: 600;'
    return ''
def color_metric_value(col,value):
    try: v=float(value)
    except Exception: return ''
    c=str(col)
    if c not in METRIC_RULES:
        if c.startswith('prob_'):
            return 'background-color: #d1fae5; color: #065f46; font-weight: 600;' if v>=0.80 else ('background-color: #fef3c7; color: #92400e; font-weight: 600;' if v>=0.65 else 'background-color: #fee2e2; color: #991b1b; font-weight: 600;')
        return ''
    ok_min,ok_max=METRIC_RULES[c]['ok']; warn_min,warn_max=METRIC_RULES[c]['warn']
    if ok_min<=v<=ok_max: return 'background-color: #d1fae5; color: #064e3b;'
    if warn_min<=v<=warn_max: return 'background-color: #fef3c7; color: #78350f;'
    return 'background-color: #fee2e2; color: #7f1d1d;'
def style_dataframe(df):
    if not isinstance(df,pd.DataFrame) or df.empty: return df
    def style_col(col):
        name=col.name
        status_cols=['qualidade_landmarks','status_manual','status_final','usar_matching','acao_automatica','classe_fairface','confianca_fairface','status_genero_ia','status_genero_manual','genero_final','origem_genero_final','usar_para_treino_genero','classe_modelo','confianca_modelo','concordancia_fairface_modelo','status_revisao_comparacao']
        if name in status_cols: return [color_status_value(x) for x in col]
        if name in METRIC_RULES or str(name).startswith('prob_'): return [color_metric_value(name,x) for x in col]
        return ['' for _ in col]
    try: return df.style.apply(style_col,axis=0)
    except Exception: return df
def metric_legend_dataframe():
    return pd.DataFrame([{'métrica':k,'verde_ok':f"{v['ok'][0]} a {v['ok'][1]}",'amarelo_atenção':f"{v['warn'][0]} a {v['warn'][1]}",'interpretação':v['desc']} for k,v in METRIC_RULES.items()])
def load_curadoria(root):
    out=resultados_root(root)/'curadoria_matching'
    return read_csv_safe(out/'status_matching_faces.csv'), read_csv_safe(out/'dataset_matching_faces.csv'), read_csv_safe(out/'resumo_curadoria_matching.csv')
def load_genero(root):
    out=genero_dir(root)
    return read_csv_safe(out/'predicoes_genero_fairface.csv'), read_csv_safe(out/'genero_faces_curado.csv'), read_csv_safe(out/'features_genero_faces.csv'), read_csv_safe(out/'predicoes_genero_metricas_faces.csv'), read_csv_safe(out/'comparacao_genero_fairface_modelo.csv')
def normalize_key_columns(df):
    if not isinstance(df,pd.DataFrame) or df.empty: return df
    df=df.copy()
    for c in ['ano','id_imagem']:
        if c in df.columns: df[c]=df[c].astype(str)
    return df
def matching_pending_mask(df):
    if df.empty: return pd.Series([],dtype=bool)
    usar=df.get('usar_matching',pd.Series(False,index=df.index)).astype(str).str.upper().isin(['TRUE','1','SIM','YES'])
    status=df.get('status_final',pd.Series('',index=df.index)).astype(str).str.upper()
    manual=df.get('status_manual',pd.Series('',index=df.index)).astype(str).str.upper()
    action=df.get('acao_automatica',pd.Series('',index=df.index)).astype(str).str.upper()
    needs=status.str.contains('REVISAR|FALHA|PENDENTE|CORRIGIR',regex=True,na=False) | manual.eq('REVISAR_MANUAL') | action.str.contains('PENDENTE',regex=False,na=False)
    return needs & ~usar
def genero_pending_mask(df):
    if df.empty: return pd.Series([],dtype=bool)
    final=df.get('genero_final',pd.Series('',index=df.index)).astype(str).str.upper()
    manual=df.get('status_genero_manual',pd.Series('',index=df.index)).astype(str).str.upper()
    decided=manual.isin(['CONFIRMAR_IA','MASCULINO','FEMININO','INDEFINIDO','NAO_USAR','NÃO_USAR']) | final.isin(['MASCULINO','FEMININO','NAO_USAR','NÃO_USAR'])
    return ~decided
def rebuild_matching_outputs(root,status_df):
    out=resultados_root(root)/'curadoria_matching'
    status_df=normalize_key_columns(status_df)
    safe_to_csv(status_df,out/'status_matching_faces.csv')
    if status_df.empty:
        safe_to_csv(pd.DataFrame(),out/'dataset_matching_faces.csv')
        safe_to_csv(pd.DataFrame(),out/'resumo_curadoria_matching.csv')
        return status_df
    use=status_df.get('usar_matching',pd.Series(False,index=status_df.index)).astype(str).str.upper().isin(['TRUE','1','SIM','YES'])
    safe_to_csv(status_df[use].copy(),out/'dataset_matching_faces.csv')
    cols=[c for c in ['ano','status_final','usar_matching'] if c in status_df.columns]
    resumo=status_df.groupby(cols,dropna=False).size().reset_index(name='qtd') if cols else pd.DataFrame({'qtd':[len(status_df)]})
    safe_to_csv(resumo,out/'resumo_curadoria_matching.csv')
    return status_df
def apply_matching_decision(root,year,stem,status_manual,obs=''):
    out=resultados_root(root)/'curadoria_matching'
    p=out/'status_matching_faces.csv'
    df=read_csv_safe(p)
    if df.empty:
        return False,'Curadoria matching vazia. Rode a curadoria antes.'
    df=normalize_key_columns(df)
    for c in ['qualidade_landmarks','status_manual','observacao_manual','status_final','usar_matching','acao_automatica','decisao_motivo']:
        if c not in df.columns: df[c]=''
    m=df['ano'].eq(str(year)) & df['id_imagem'].eq(str(stem))
    if not m.any():
        return False,'Imagem não encontrada na curadoria matching.'
    s=str(status_manual).strip().upper()
    df.loc[m,'status_manual']=s
    df.loc[m,'observacao_manual']=str(obs)
    if s in ['OK_MANUAL','MANTER_ORIGINAL','MANTER_ORIGINAL_AUTO']:
        df.loc[m,'qualidade_landmarks']='OK'
        df.loc[m,'status_final']='OK_MANUAL'
        df.loc[m,'usar_matching']=True
        df.loc[m,'acao_automatica']='USAR_ORIGINAL'
        df.loc[m,'decisao_motivo']='manual ok'
    elif s in ['EXCLUIR_MANUAL','EXCLUIR','NAO_USAR','NÃO_USAR']:
        df.loc[m,'qualidade_landmarks']='EXCLUIR'
        df.loc[m,'status_final']='EXCLUIR_MANUAL'
        df.loc[m,'usar_matching']=False
        df.loc[m,'acao_automatica']='NAO_USAR'
        df.loc[m,'decisao_motivo']='manual excluir'
    else:
        df.loc[m,'qualidade_landmarks']='REVISAR'
        df.loc[m,'status_final']='REVISAR_MANUAL'
        df.loc[m,'usar_matching']=False
        df.loc[m,'acao_automatica']='NAO_USAR'
        df.loc[m,'decisao_motivo']='manual revisar'
    save_manual_status(root,year,stem,s,obs)
    rebuild_matching_outputs(root,df)
    return True,'Matching atualizado.'
def genero_manual_path(root): return genero_dir(root)/'revisao_manual_genero_faces.csv'
def append_genero_manual(root,year,stem,status,obs=''):
    p=genero_manual_path(root)
    df=read_csv_safe(p)
    row={'timestamp':pd.Timestamp.now().isoformat(),'ano':str(year),'id_imagem':str(stem),'status_genero_manual':str(status),'observacao_genero':str(obs)}
    safe_to_csv(pd.concat([df,pd.DataFrame([row])],ignore_index=True),p)
    return p
def latest_genero_manual_map(root):
    df=read_csv_safe(genero_manual_path(root))
    if df.empty or not {'ano','id_imagem','status_genero_manual'}.issubset(df.columns): return {}
    df=normalize_key_columns(df)
    if 'timestamp' in df.columns: df=df.sort_values('timestamp')
    return {(str(r.get('ano','')),str(r.get('id_imagem',''))):{'status_genero_manual':str(r.get('status_genero_manual','')),'observacao_genero':str(r.get('observacao_genero',''))} for _,r in df.iterrows()}
def rebuild_genero_curado(root,apply_high=False,high=.80):
    out=genero_dir(root)
    pred=normalize_key_columns(read_csv_safe(out/'predicoes_genero_fairface.csv'))
    if pred.empty:
        return False,'Rode FairFace antes.'
    manual=latest_genero_manual_map(root)
    anterior=normalize_key_columns(read_csv_safe(out/'genero_faces_curado.csv'))
    auto_high=set()
    if not anterior.empty and {'ano','id_imagem','origem_genero_final'}.issubset(anterior.columns):
        ah=anterior['origem_genero_final'].astype(str).str.upper().eq('FAIRFACE_ALTA_CONFIANCA')
        auto_high=set(map(tuple,anterior.loc[ah,['ano','id_imagem']].astype(str).values.tolist()))
    rows=[]
    for _,r in pred.iterrows():
        y=str(r.get('ano','')); stem=str(r.get('id_imagem',''))
        cl=str(r.get('classe_fairface','')).upper()
        try:
            pm=float(r.get('prob_masculino_fairface',np.nan)); pf=float(r.get('prob_feminino_fairface',np.nan)); pmax=max(pm,pf)
        except Exception:
            pmax=np.nan
        info=manual.get((y,stem),{})
        ms=str(info.get('status_genero_manual','')).upper()
        obs=str(info.get('observacao_genero',''))
        classe_valida=cl in ['MASCULINO_PROVAVEL','FEMININO_PROVAVEL']
        qualifica_alta=pmax==pmax and pmax>=high and classe_valida
        aplicar_auto=not ms and classe_valida and ((apply_high and qualifica_alta) or (y,stem) in auto_high)
        gf,org,usar=genero_final_from_manual(ms,cl)
        if aplicar_auto:
            gf='MASCULINO' if 'MASCULINO' in cl else 'FEMININO'
            org='FAIRFACE_ALTA_CONFIANCA'
            usar=True
            obs='Definido automaticamente por alta confiança FairFace.'
        if not gf:
            gf='INDEFINIDO'; org='AGUARDANDO_CURADORIA'; usar=False
        row=dict(r)
        row.update({'status_genero_manual':ms,'status_genero_automatico':'CONFIRMADO_ALTA_CONFIANCA' if aplicar_auto else '', 'genero_final':gf,'origem_genero_final':org,'usar_para_treino_genero':bool(usar),'observacao_genero':obs})
        rows.append(row)
    cur=normalize_key_columns(pd.DataFrame(rows))
    safe_to_csv(cur,out/'genero_faces_curado.csv')
    safe_to_csv(cur.groupby(['genero_final','origem_genero_final','usar_para_treino_genero'],dropna=False).size().reset_index(name='qtd'),out/'resumo_genero_faces_curado.csv')
    return True,'Gênero curado atualizado.'
def apply_genero_decision(root,year,stem,status,obs=''):
    append_genero_manual(root,year,stem,status,obs)
    return rebuild_genero_curado(root,apply_high=False)
def overview_counts(root,years):
    ana=load_all_analysis(root,years)
    status,dataset,resumo=load_curadoria(root)
    pred,cur,feat,predm,comp=load_genero(root)
    rows=[]
    for y in years:
        photos=list_images_for_year(root,y)
        sy=status[status.get('ano',pd.Series(dtype=str)).astype(str)==str(y)].copy() if not status.empty and 'ano' in status.columns else pd.DataFrame()
        sf=sy.get('status_final',pd.Series('',index=sy.index)).astype(str).str.upper()
        usar=sy.get('usar_matching',pd.Series(False,index=sy.index)).astype(str).str.upper().isin(['TRUE','1','SIM','YES'])
        ok_manual=usar & sf.str.contains('MANUAL',na=False)
        ok_auto=usar & ~ok_manual
        revisar_matching=matching_pending_mask(sy) if not sy.empty else pd.Series([],dtype=bool)
        revisar_original=sy.get('qualidade_landmarks',pd.Series('',index=sy.index)).astype(str).str.upper().eq('REVISAR')
        status_ids=set(sy.get('id_imagem',pd.Series(dtype=str)).astype(str))
        sem_status=sum(1 for p in photos if p.stem not in status_ids)
        cy=cur[cur.get('ano',pd.Series(dtype=str)).astype(str)==str(y)].copy() if not cur.empty and 'ano' in cur.columns else pd.DataFrame()
        py=pred[pred.get('ano',pd.Series(dtype=str)).astype(str)==str(y)].copy() if not pred.empty and 'ano' in pred.columns else pd.DataFrame()
        genero_src=cy if not cy.empty else py
        revisar_genero=int(genero_pending_mask(genero_src).sum()) if not genero_src.empty else 0
        origem=cy.get('origem_genero_final',pd.Series('',index=cy.index)).astype(str).str.upper()
        rows.append({
            'ano':str(y),
            'fotos':len(photos),
            'previews':len(list(previews_dir(root,y).glob('*_preview.jpg'))) if previews_dir(root,y).exists() else 0,
            'ok_automatico':int(ok_auto.sum()),
            'ok_manual':int(ok_manual.sum()),
            'sem_processar':sem_status,
            'revisar_imagem_original':int(revisar_original.sum()),
            'revisar_matching':int(revisar_matching.sum()),
            'revisar_genero':revisar_genero,
            'genero_alta_confianca':int(origem.eq('FAIRFACE_ALTA_CONFIANCA').sum()),
            'genero_definido_manual':int(origem.str.contains('MANUALMENTE|_MANUAL',regex=True,na=False).sum()),
            'aprovadas_para_genero':int(usar.sum()),
            'fairface_processadas':int((pred.get('ano',pd.Series(dtype=str)).astype(str)==str(y)).sum()) if not pred.empty and 'ano' in pred.columns else 0,
        })
    result=pd.DataFrame(rows)
    if not result.empty:
        total={'ano':'TOTAL'}
        total.update({c:int(pd.to_numeric(result[c],errors='coerce').fillna(0).sum()) for c in result.columns if c!='ano'})
        result=pd.concat([result,pd.DataFrame([total])],ignore_index=True)
    return result
def load_all_analysis(root,years):
    geral=resultados_root(root)/'analises_faces'/'analise_padroes_faces_todos_anos.csv'; df=read_csv_safe(geral)
    if df.empty:
        frames=[]
        for y in years:
            p=marks_dir(root,y)/f'analise_padroes_faces_{y}.csv'; tmp=read_csv_safe(p)
            if not tmp.empty: tmp['ano']=tmp.get('ano',y); frames.append(tmp)
        df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if df.empty: return df
    if 'ano' in df.columns: df['ano']=df['ano'].astype(str)
    if 'id_imagem' in df.columns: df['id_imagem']=df['id_imagem'].astype(str)
    if 'qualidade_landmarks' in df.columns: df['qualidade_landmarks']=df['qualidade_landmarks'].astype(str).str.upper()
    manual=load_manual_status_map(root)
    if manual:
        df=df.copy(); df['status_manual']=''; df['observacao_manual']=''
        for i,row in df.iterrows():
            key=(str(row.get('ano','')),str(row.get('id_imagem',''))); info=manual.get(key)
            if not info: continue
            status=str(info.get('status_manual','')).upper(); df.at[i,'status_manual']=status; df.at[i,'observacao_manual']=str(info.get('observacao_manual',''))
            if status in ['OK_MANUAL','MANTER_ORIGINAL']: df.at[i,'qualidade_landmarks']='OK'; df.at[i,'motivo_qualidade']='aprovada_manual'
            elif status in ['EXCLUIR_MANUAL','EXCLUIR']: df.at[i,'qualidade_landmarks']='EXCLUIR'; df.at[i,'motivo_qualidade']='excluida_manual'
            elif status in ['REVISAR_MANUAL']: df.at[i,'qualidade_landmarks']='REVISAR'; df.at[i,'motivo_qualidade']='revisar_manual'
    return df


# === PATCH V4.1: regras de cores ampliadas e legenda completa ===
EXTRA_METRIC_RULES = {
    "largura_face_px": {"ok": (120.0, 999999999.0), "warn": (90.0, 120.0), "desc": "Largura da face em pixels. Depende da resolução; valores muito baixos indicam rosto pequeno/corte."},
    "altura_face_px": {"ok": (140.0, 999999999.0), "warn": (100.0, 140.0), "desc": "Altura da face em pixels. Depende da resolução; valores muito baixos indicam rosto pequeno/corte."},
    "distancia_olhos_px": {"ok": (60.0, 999999999.0), "warn": (40.0, 60.0), "desc": "Distância entre olhos em pixels. Depende da resolução; baixo pode indicar rosto pequeno."},
    "largura_boca_px": {"ok": (40.0, 999999999.0), "warn": (25.0, 40.0), "desc": "Largura da boca em pixels. Depende da resolução; baixo pode indicar rosto pequeno ou landmarks ruins."},
    "area_contorno_px2": {"ok": (15000.0, 999999999.0), "warn": (8000.0, 15000.0), "desc": "Área do contorno facial em pixels². Depende da resolução; baixo pode indicar rosto pequeno/corte."},
    "perimetro_contorno_px": {"ok": (400.0, 999999999.0), "warn": (250.0, 400.0), "desc": "Perímetro do contorno facial em pixels. Depende da resolução; baixo pode indicar rosto pequeno."},
    "bbox_contorno_w_px": {"ok": (120.0, 999999999.0), "warn": (90.0, 120.0), "desc": "Largura do bounding box do contorno."},
    "bbox_contorno_h_px": {"ok": (140.0, 999999999.0), "warn": (100.0, 140.0), "desc": "Altura do bounding box do contorno."},
}
try:
    METRIC_RULES.update(EXTRA_METRIC_RULES)
except Exception:
    METRIC_RULES = EXTRA_METRIC_RULES.copy()

STATUS_COLOR_COLS = {
    "qualidade_landmarks", "status_manual", "status_final", "usar_matching", "acao_automatica",
    "classe_fairface", "confianca_fairface", "status_genero_ia", "status_genero_manual",
    "genero_final", "origem_genero_final", "usar_para_treino_genero",
    "classe_modelo", "confianca_modelo", "concordancia_fairface_modelo", "status_revisao_comparacao",
}

def _style_status_cell_v41(v):
    s = str(v).strip().upper()
    if s in ["OK", "OK_AUTO", "OK_MANUAL", "MANTER_ORIGINAL_AUTO", "CORRIGIDO_AUTO", "TRUE", "SIM", "YES", "1", "MASCULINO", "FEMININO", "MASCULINO_PROVAVEL", "FEMININO_PROVAVEL", "ALTA", "CONCORDANTE_ALTA", "CONFIRMAR_IA", "ACEITAR_SUGESTAO"]:
        return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
    if any(x in s for x in ["REVISAR", "PENDENTE", "MEDIA", "MÉDIA", "BAIXA", "INDEFINIDO", "DIVERGENTE", "AGUARDANDO"]):
        return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
    if any(x in s for x in ["FALHA", "EXCLUIR", "ERRO", "NAO_USAR", "NÃO_USAR", "FALSE"]):
        return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
    if any(x in s for x in ["CORRIGIR", "APLICAR"]):
        return "background-color: #dbeafe; color: #1e40af; font-weight: 600;"
    return ""

def _style_numeric_cell_v41(col_name, v):
    try:
        fv = float(v)
    except Exception:
        return ""
    name = str(col_name)
    # Probabilidades: colore a confiança da probabilidade da classe em si.
    if name.startswith("prob_"):
        if fv >= 0.80:
            return "background-color: #d1fae5; color: #065f46; font-weight: 600;"
        if fv >= 0.65:
            return "background-color: #fef3c7; color: #92400e; font-weight: 600;"
        if fv >= 0.55:
            return "background-color: #ffedd5; color: #9a3412; font-weight: 600;"
        return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
    rule = METRIC_RULES.get(name)
    if not rule:
        return ""
    ok_min, ok_max = rule["ok"]
    warn_min, warn_max = rule["warn"]
    if ok_min <= fv <= ok_max:
        return "background-color: #d1fae5; color: #065f46;"
    if warn_min <= fv <= warn_max:
        return "background-color: #fef3c7; color: #92400e;"
    return "background-color: #fee2e2; color: #991b1b;"

def style_dataframe(df):
    import pandas as pd
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    def style_col(col):
        name = str(col.name)
        if name in STATUS_COLOR_COLS:
            return [_style_status_cell_v41(x) for x in col]
        return [_style_numeric_cell_v41(name, x) for x in col]
    try:
        return df.style.apply(style_col, axis=0)
    except Exception:
        return df

def metric_legend_dataframe():
    rows = []
    for k, v in METRIC_RULES.items():
        rows.append({"métrica": k, "verde_ok": f"{v['ok'][0]} a {v['ok'][1]}", "amarelo_atenção": f"{v['warn'][0]} a {v['warn'][1]}", "interpretação": v.get("desc", "")})
    return pd.DataFrame(rows)
