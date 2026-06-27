# -*- coding: utf-8 -*-
import argparse, pandas as pd
from common_faces import *
def base(s):
    s=str(s).upper(); return 'MASCULINO' if 'MASCULINO' in s else ('FEMININO' if 'FEMININO' in s else 'INDEFINIDO')
def stc(r):
    ia=base(r.get('classe_fairface','')); md=base(r.get('classe_modelo','')); ci=str(r.get('confianca_fairface','')).upper(); cm=str(r.get('confianca_modelo','')).upper()
    if ia in ['MASCULINO','FEMININO'] and ia==md: return 'CONCORDANTE_ALTA' if ci=='ALTA' and cm=='ALTA' else 'CONCORDANTE_MEDIA'
    if ia=='INDEFINIDO' or md=='INDEFINIDO': return 'INDEFINIDO_REVISAR'
    return 'DIVERGENTE_REVISAR'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); args=ap.parse_args(); root=as_root(args.root); out=genero_dir(root); cur=read_csv_safe(out/'genero_faces_curado.csv'); pred=read_csv_safe(out/'predicoes_genero_metricas_faces.csv')
    if cur.empty or pred.empty: print('[ERRO] Rode curadoria e modelo'); return
    cur.ano=cur.ano.astype(str); cur.id_imagem=cur.id_imagem.astype(str); pred.ano=pred.ano.astype(str); pred.id_imagem=pred.id_imagem.astype(str); df=cur.merge(pred,on=['ano','id_imagem'],how='left')
    df['genero_fairface']=df['classe_fairface'].apply(base)
    df['genero_curadoria']=df['genero_final'].apply(base)
    df['genero_modelo']=df['classe_modelo'].apply(base)
    df['concordancia_fairface_modelo']=df.apply(stc,axis=1)
    df['resultado_modelo_curadoria']=df.apply(lambda r:'ACERTO' if r['genero_modelo'] in ['MASCULINO','FEMININO'] and r['genero_modelo']==r['genero_curadoria'] else ('DIVERGENCIA' if r['genero_modelo'] in ['MASCULINO','FEMININO'] and r['genero_curadoria'] in ['MASCULINO','FEMININO'] else 'SEM_DEFINICAO'),axis=1)
    df['status_revisao_comparacao']=df['resultado_modelo_curadoria'].map({'ACERTO':'OK','DIVERGENCIA':'REVISAR','SEM_DEFINICAO':'REVISAR'})
    safe_to_csv(df,out/'comparacao_genero_fairface_modelo.csv'); print('[OK] comparação')
if __name__=='__main__': main()
