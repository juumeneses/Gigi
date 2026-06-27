# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd
from common_faces import *
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--years',nargs='*'); args=ap.parse_args(); root=as_root(args.root); frames=[]
    for y in discover_years(root,args.years):
        p=marks_dir(root,y)/f'medidas_faces_{y}.csv'; df=read_csv_safe(p)
        if not df.empty: df['ano']=y; frames.append(df)
    if not frames: print('[ERRO] sem medidas'); return
    df=pd.concat(frames,ignore_index=True); feat=[c for c in ['razao_largura_altura_face','razao_olhos_largura_face','razao_boca_largura_face','razao_bbox_contorno','largura_face_px','altura_face_px'] if c in df.columns]
    try:
        from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
        x=df[feat].astype(float).replace([np.inf,-np.inf],np.nan); x=x.fillna(x.median(numeric_only=True)); p=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(x)); df['pca_face_1']=p[:,0]; df['pca_face_2']=p[:,1]; d=np.sqrt((p[:,0]-np.median(p[:,0]))**2+(p[:,1]-np.median(p[:,1]))**2); df['score_outlier_pca']=np.round(d,4); lim=np.percentile(d,95) if len(d)>=10 else d.mean()+2*d.std(); df['outlier_pca']=d>lim
    except Exception: df['pca_face_1']=0; df['pca_face_2']=0; df['score_outlier_pca']=0; df['outlier_pca']=False
    labs=[]; mots=[]
    for _,r in df.iterrows():
        m=[]
        for c,lo,hi,n in [('razao_largura_altura_face',.5,1.05,'largura_altura_atipico'),('razao_olhos_largura_face',.48,.78,'olhos_largura_atipico'),('razao_boca_largura_face',.22,.55,'boca_largura_atipico'),('razao_bbox_contorno',.45,1.10,'bbox_contorno_atipico')]:
            try:
                if not (lo<=float(r.get(c,np.nan))<=hi): m.append(n)
            except Exception: pass
        if bool(r.get('outlier_pca',False)): m.append('outlier_pca')
        labs.append('REVISAR' if m else 'OK'); mots.append(';'.join(m) if m else 'metricas_ok')
    df['qualidade_landmarks']=labs; df['motivo_qualidade']=mots
    for y,g in df.groupby('ano'): safe_to_csv(g,marks_dir(root,str(y))/f'analise_padroes_faces_{y}.csv')
    out=resultados_root(root)/'analises_faces'; safe_to_csv(df,out/'analise_padroes_faces_todos_anos.csv'); safe_to_csv(df.groupby(['ano','qualidade_landmarks']).size().reset_index(name='qtd'),out/'resumo_analise_padroes_faces.csv'); print('[OK] analise')
if __name__=='__main__': main()
