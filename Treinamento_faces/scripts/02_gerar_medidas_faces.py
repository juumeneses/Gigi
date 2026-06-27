# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd
from common_faces import *
def dist(a,b): return float(np.hypot(a.x_px-b.x_px,a.y_px-b.y_px))
def area(xs,ys): x=np.array(xs); y=np.array(ys); return float(.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))
def per(xs,ys): pts=np.column_stack([xs,ys]); return float(np.sqrt(((pts-np.roll(pts,-1,axis=0))**2).sum(axis=1)).sum())
def proc(p):
    df=pd.read_csv(p); idx=df.set_index('idx')
    for i in [33,263,152,10,234,454,61,291]:
        if i not in idx.index: raise ValueError(f'landmark {i} ausente')
    larg=dist(idx.loc[234],idx.loc[454]); alt=dist(idx.loc[10],idx.loc[152]); olhos=dist(idx.loc[33],idx.loc[263]); boca=dist(idx.loc[61],idx.loc[291]); cont=idx.loc[[i for i in FACE_OVAL_ORDER if i in idx.index]].copy(); bw=float(cont.x_px.max()-cont.x_px.min()); bh=float(cont.y_px.max()-cont.y_px.min())
    return {'id_imagem':str(df.id_imagem.iloc[0]),'largura_face_px':larg,'altura_face_px':alt,'razao_largura_altura_face':larg/max(alt,1),'distancia_olhos_px':olhos,'razao_olhos_largura_face':olhos/max(larg,1),'largura_boca_px':boca,'razao_boca_largura_face':boca/max(larg,1),'area_contorno_px2':area(cont.x_px,cont.y_px),'perimetro_contorno_px':per(cont.x_px,cont.y_px),'bbox_contorno_w_px':bw,'bbox_contorno_h_px':bh,'razao_bbox_contorno':bw/max(bh,1)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); args=ap.parse_args(); root=as_root(args.root); ensure_base_dirs(root); fall=[]
    for y in discover_years(root,args.years):
        rows=[]; fl=[]
        for p in sorted((marks_dir(root,y)/'landmarks_csv').glob('*_landmarks.csv')):
            try: r=proc(p); r['ano']=y; rows.append(r)
            except Exception as e: row={'ano':y,'id_imagem':p.stem.replace('_landmarks',''),'erro':str(e)}; fl.append(row); fall.append(row)
        safe_to_csv(pd.DataFrame(rows),marks_dir(root,y)/f'medidas_faces_{y}.csv'); safe_to_csv(pd.DataFrame(fl),marks_dir(root,y)/f'falhas_medidas_faces_{y}.csv')
    safe_to_csv(pd.DataFrame(fall),resultados_root(root)/'falhas'/'falhas_medidas_faces_todos_anos.csv'); print('[OK] medidas')
if __name__=='__main__': main()
