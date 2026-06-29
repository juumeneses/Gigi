# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd
from common_faces import *

def runs_dir(root):
    p=resultados_root(root)/'runs'; p.mkdir(parents=True,exist_ok=True); return p

def append_run_history(root,run_id,params,metrics_rows,best_model):
    rows=[]
    for m in metrics_rows:
        row={'run_id':run_id,'timestamp':pd.Timestamp.now().isoformat(),'tipo':'genero_face','melhor_modelo':best_model,'parametros_json':json.dumps(params,ensure_ascii=False,sort_keys=True)}
        row.update(m)
        for c in ['accuracy','precision_masculino','recall_masculino','f1_masculino']:
            if c in row and pd.notna(row[c]):
                try: row[f'{c}_pct']=round(float(row[c])*100,2)
                except Exception: pass
        rows.append(row)
    path=runs_dir(root)/'historico_treinos_faces.csv'
    old=read_csv_safe(path)
    safe_to_csv(pd.concat([old,pd.DataFrame(rows)],ignore_index=True),path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--models',nargs='+',choices=['logistic_regression','svm_rbf','random_forest'],default=['logistic_regression','svm_rbf','random_forest'])
    ap.add_argument('--test-size',type=float,default=.25)
    ap.add_argument('--random-state',type=int,default=42)
    ap.add_argument('--selection-metric',choices=['f1_masculino','accuracy','precision_masculino','recall_masculino'],default='f1_masculino')
    ap.add_argument('--logistic-max-iter',type=int,default=2000)
    ap.add_argument('--svm-c',type=float,default=1.0)
    ap.add_argument('--svm-gamma',choices=['scale','auto'],default='scale')
    ap.add_argument('--rf-trees',type=int,default=300)
    args=ap.parse_args(); root=as_root(args.root); out=genero_dir(root); df=read_csv_safe(out/'features_genero_faces.csv')
    if df.empty: print('[ERRO] Rode features antes'); return
    mask=df.genero_final.astype(str).str.upper().isin(['MASCULINO','FEMININO']) & df.usar_para_treino_genero.astype(str).str.upper().isin(['TRUE','1','SIM'])
    train=df[mask].copy()
    if train.empty or train.genero_final.nunique()<2 or len(train)<6: print('[ERRO] Poucos dados curados para treinar.'); return
    ignore=('ano','id_imagem','imagem','classe','status','genero','origem','observacao','motivo','crop','modelo','confianca','usar','prob_')
    feat=[]
    for c in train.columns:
        if any(str(c).startswith(p) for p in ignore): continue
        try: pd.to_numeric(train[c],errors='raise'); feat.append(c)
        except Exception: pass
    X=train[feat].replace([np.inf,-np.inf],np.nan).astype(float); X=X.fillna(X.median(numeric_only=True)); y=train.genero_final.astype(str).str.upper().map({'FEMININO':0,'MASCULINO':1}).astype(int)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import joblib
    strat=y if min(y.value_counts())>=2 and len(y)>=10 else None
    test_size=min(.5,max(.05,float(args.test_size)))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=args.random_state,stratify=strat) if len(y)>=10 else (X,X,y,y)
    available={'logistic_regression':Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=args.logistic_max_iter,class_weight='balanced',random_state=args.random_state))]),'svm_rbf':Pipeline([('scaler',StandardScaler()),('clf',SVC(C=args.svm_c,gamma=args.svm_gamma,probability=True,class_weight='balanced',random_state=args.random_state))]),'random_forest':RandomForestClassifier(n_estimators=args.rf_trees,random_state=args.random_state,class_weight='balanced')}
    models={name:available[name] for name in args.models}
    metrics=[]; fitted={}; bestn=None; best=-1; run_id=f"face_genero_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    for n,m in models.items():
        try:
            m.fit(Xtr,ytr); pr=m.predict(Xte)
            values={'accuracy':accuracy_score(yte,pr),'precision_masculino':precision_score(yte,pr,zero_division=0),'recall_masculino':recall_score(yte,pr,zero_division=0),'f1_masculino':f1_score(yte,pr,zero_division=0)}
            metrics.append({'run_id':run_id,'modelo':n,'n_total':len(train),'n_treino':len(Xtr),'n_teste':len(Xte),'test_size':test_size,'random_state':args.random_state,'metrica_selecao':args.selection_metric,**values}); fitted[n]=m
            score=float(values[args.selection_metric])
            if score>best: best=score; bestn=n
        except Exception as e: metrics.append({'modelo':n,'erro':str(e)})
    if bestn is None: print('[ERRO] Nenhum modelo conseguiu treinar.'); return
    model=fitted[bestn]; model.fit(X,y); joblib.dump({'model':model,'feature_cols':feat,'best_name':bestn,'params':vars(args),'run_id':run_id},out/'modelo_genero_faces.pkl'); safe_to_csv(pd.DataFrame(metrics),out/'metricas_modelo_genero_faces.csv')
    cm=confusion_matrix(yte,fitted[bestn].predict(Xte),labels=[0,1]); safe_to_csv(pd.DataFrame(cm,index=['real_FEMININO','real_MASCULINO'],columns=['pred_FEMININO','pred_MASCULINO']).reset_index().rename(columns={'index':'classe_real'}),out/'matriz_confusao_genero_faces.csv')
    allX=df[feat].replace([np.inf,-np.inf],np.nan).astype(float); allX=allX.fillna(X.median(numeric_only=True)); probs=model.predict_proba(allX); classes=list(model.classes_) if hasattr(model,'classes_') else list(model.named_steps['clf'].classes_); im=classes.index(1); iff=classes.index(0)
    rows=[]
    for i,r in df.iterrows():
        pm=float(probs[i,im]); pf=float(probs[i,iff]); cl,conf=class_from_gender_probs(pm,pf); rows.append({'ano':r.get('ano'),'id_imagem':r.get('id_imagem'),'prob_masculino_modelo':round(pm,6),'prob_feminino_modelo':round(pf,6),'classe_modelo':cl,'confianca_modelo':conf,'modelo_metricas':bestn})
    safe_to_csv(pd.DataFrame(rows),out/'predicoes_genero_metricas_faces.csv'); append_run_history(root,run_id,vars(args),metrics,bestn); print('[OK] modelo',bestn)
if __name__=='__main__': main()
