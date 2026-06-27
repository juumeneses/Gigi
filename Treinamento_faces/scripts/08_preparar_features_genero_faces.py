# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd
from common_faces import *
def lm_feats(root,y,stem):
    df=read_csv_safe(marks_dir(root,y)/'landmarks_csv'/f'{stem}_landmarks.csv'); feats={}
    if df.empty: return feats
    sub=df[df.idx.isin(MAIN_LANDMARKS)].copy(); cont=df[df.idx.isin(FACE_OVAL_ORDER)].copy(); cont=cont if not cont.empty else df
    cx,cy=cont.x_px.mean(),cont.y_px.mean(); scale=max(cont.x_px.max()-cont.x_px.min(),cont.y_px.max()-cont.y_px.min(),1.0)
    for _,r in sub.iterrows(): feats[f'lm_{int(r.idx)}_x_norm']=(float(r.x_px)-cx)/scale; feats[f'lm_{int(r.idx)}_y_norm']=(float(r.y_px)-cy)/scale
    return feats
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); args=ap.parse_args(); root=as_root(args.root); cur=read_csv_safe(genero_dir(root)/'genero_faces_curado.csv'); ana=read_csv_safe(resultados_root(root)/'analises_faces'/'analise_padroes_faces_todos_anos.csv'); _,dataset,_=load_curadoria(root)
    if cur.empty or ana.empty: print('[ERRO] Rode curadoria de gênero e análise antes'); return
    cur.ano=cur.ano.astype(str); cur.id_imagem=cur.id_imagem.astype(str); ana.ano=ana.ano.astype(str); ana.id_imagem=ana.id_imagem.astype(str); df=cur.merge(ana,on=['ano','id_imagem'],how='left',suffixes=('','_analise'))
    if not dataset.empty:
        dataset.ano=dataset.ano.astype(str); dataset.id_imagem=dataset.id_imagem.astype(str); df=df.merge(dataset[['ano','id_imagem','status_final','usar_matching']].drop_duplicates(),on=['ano','id_imagem'],how='left',suffixes=('','_curadoria'))
    rows=[]
    for _,r in df.iterrows():
        row=dict(r); y=str(r.get('ano','')); stem=str(r.get('id_imagem',''))
        try:
            larg=float(row.get('largura_face_px',np.nan)); alt=float(row.get('altura_face_px',np.nan)); olhos=float(row.get('distancia_olhos_px',np.nan)); boca=float(row.get('largura_boca_px',np.nan)); row['largura_face_norm_altura']=larg/max(alt,1); row['altura_face_norm_largura']=alt/max(larg,1); row['olhos_norm_altura']=olhos/max(alt,1); row['boca_norm_altura']=boca/max(alt,1)
        except Exception: pass
        row.update(lm_feats(root,y,stem)); rows.append(row)
    safe_to_csv(pd.DataFrame(rows),genero_dir(root)/'features_genero_faces.csv'); print('[OK] features gênero')
if __name__=='__main__': main()
