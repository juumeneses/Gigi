# -*- coding: utf-8 -*-
import argparse, pandas as pd
from common_faces import *
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); args=ap.parse_args(); root=as_root(args.root); _,dataset,_=load_curadoria(root); comp=read_csv_safe(genero_dir(root)/'comparacao_genero_fairface_modelo.csv'); feat=read_csv_safe(genero_dir(root)/'features_genero_faces.csv')
    if dataset.empty: print('[ERRO] dataset matching vazio'); return
    dataset.ano=dataset.ano.astype(str); dataset.id_imagem=dataset.id_imagem.astype(str); df=dataset.copy()
    if not comp.empty:
        comp.ano=comp.ano.astype(str); comp.id_imagem=comp.id_imagem.astype(str); cols=[c for c in ['ano','id_imagem','genero_final','origem_genero_final','classe_fairface','prob_masculino_fairface','prob_feminino_fairface','confianca_fairface','classe_modelo','prob_masculino_modelo','prob_feminino_modelo','confianca_modelo','concordancia_fairface_modelo','status_revisao_comparacao'] if c in comp.columns]; df=df.merge(comp[cols].drop_duplicates(['ano','id_imagem']),on=['ano','id_imagem'],how='left')
    if not feat.empty:
        feat.ano=feat.ano.astype(str); feat.id_imagem=feat.id_imagem.astype(str); fcols=[c for c in feat.columns if c not in df.columns and (c.startswith('lm_') or c in ['razao_largura_altura_face','razao_olhos_largura_face','razao_boca_largura_face','razao_bbox_contorno','pca_face_1','pca_face_2','score_outlier_pca'])]
        if fcols: df=df.merge(feat[['ano','id_imagem']+fcols].drop_duplicates(['ano','id_imagem']),on=['ano','id_imagem'],how='left')
    out=matching_features_dir(root); out.mkdir(parents=True,exist_ok=True); safe_to_csv(df,out/'features_matching_faces_com_genero.csv'); print('[OK] exportado')
if __name__=='__main__': main()
