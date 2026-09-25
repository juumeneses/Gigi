#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import resnet18

LAB = Path(__file__).resolve().parents[1]; REPO = LAB.parent; sys.path.insert(0, str(LAB))
from src.common import abs_repo, load_config, read_image, save_csv


class ProjectionTwoTower(nn.Module):
    def __init__(self, face_dim: int, embedding_dim: int = 256):
        super().__init__()
        self.skull = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Dropout(.15), nn.Linear(384, embedding_dim))
        self.face = nn.Sequential(nn.Linear(face_dim, 256), nn.ReLU(), nn.Dropout(.10), nn.Linear(256, embedding_dim))
        self.logit_scale = nn.Parameter(torch.tensor(2.0)); self.bias = nn.Parameter(torch.tensor(0.0))
    def embeddings(self, skull, face):
        return nn.functional.normalize(self.skull(skull), dim=1), nn.functional.normalize(self.face(face), dim=1)
    def forward(self, skull, face):
        a,b=self.embeddings(skull,face); return (a*b).sum(1)*self.logit_scale.exp().clamp(max=30)+self.bias


def image_tensor(path: Path):
    im=read_image(path)
    if im is None: raise ValueError(f"imagem invalida: {path}")
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); h,w=im.shape[:2]; scale=256/min(h,w); im=cv2.resize(im,(int(w*scale),int(h*scale)))
    h,w=im.shape[:2]; y=(h-224)//2; x=(w-224)//2; im=im[y:y+224,x:x+224].astype(np.float32)/255
    im=(im-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(im.transpose(2,0,1))


def extract_skull_embeddings(data: pd.DataFrame, force=False):
    out=LAB/"modelos/resnet18_cf_embeddings.npz"
    if out.exists() and not force:
        z=np.load(out,allow_pickle=True); return {str(i):e for i,e in zip(z["ids"],z["embeddings"])}
    checkpoint=Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if not checkpoint.exists(): raise RuntimeError("pesos ResNet18 ImageNet nao encontrados no cache local")
    model=resnet18(weights=None); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); model.fc=nn.Identity(); model.eval()
    ids=[]; embeddings=[]
    with torch.no_grad():
        for start in range(0,len(data),16):
            batch=data.iloc[start:start+16]; tensors=[]; valid=[]
            for _,r in batch.iterrows():
                try: tensors.append(image_tensor(abs_repo(r["imagem_cf"]))); valid.append(str(r["id_pessoa"]))
                except Exception: pass
            if tensors:
                emb=model(torch.stack(tensors)).cpu().numpy(); ids.extend(valid); embeddings.extend(emb)
            print(f"__EMBED__ {min(start+16,len(data))}/{len(data)}",flush=True)
    np.savez_compressed(out,ids=np.array(ids),embeddings=np.asarray(embeddings,np.float32)); return dict(zip(ids,embeddings))


def metrics(y, prob):
    pred=(prob>=.5).astype(int)
    return {"roc_auc":float(roc_auc_score(y,prob)),"pr_auc":float(average_precision_score(y,prob)),"f1":float(f1_score(y,pred)),"brier":float(brier_score_loss(y,prob)),"accuracy":float((pred==y).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/pipeline.yaml"); ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--force-embeddings",action="store_true"); args=ap.parse_args()
    cfg=load_config(args.config); seed=int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    paired=pd.read_csv(LAB/"dataset/manifestos/dataset_pareado_matching.csv"); pairs=pd.read_csv(LAB/"dataset/manifestos/pares_matching.csv")
    original=pd.read_csv(REPO/"Treinamento_faces/resultados/features_matching_faces/features_matching_faces_final.csv"); pca=[c for c in original if c.startswith("face_pca_")]
    faces=original[["id_pessoa",*pca]].copy(); faces["id_pessoa"]=faces.id_pessoa.astype(str); paired["id_pessoa"]=paired.id_pessoa.astype(str); data=paired.merge(faces,on="id_pessoa",how="left")
    eligible=data[data.usar_matching].copy(); skull_map=extract_skull_embeddings(eligible,args.force_embeddings)
    scaler=StandardScaler().fit(eligible[eligible.split=="treino"][pca].fillna(0)); joblib.dump(scaler,LAB/"modelos/scaler_face_pca.joblib"); face_scaled=scaler.transform(eligible[pca].fillna(0)).astype(np.float32)
    face_map=dict(zip(eligible.id_pessoa,face_scaled)); byid=eligible.set_index("id_pessoa")
    def arrays(split):
        part=pairs[pairs.split==split]; s=[];f=[];y=[];meta=[]
        for _,r in part.iterrows():
            fi,ci=str(r.face_id),str(r.cranio_id)
            if fi in face_map and ci in skull_map: s.append(skull_map[ci]);f.append(face_map[fi]);y.append(float(r.label));meta.append((fi,ci))
        return torch.tensor(np.asarray(s),dtype=torch.float32),torch.tensor(np.asarray(f),dtype=torch.float32),torch.tensor(y,dtype=torch.float32),meta
    tr=arrays("treino"); va=arrays("validacao"); te=arrays("teste"); model=ProjectionTwoTower(len(pca),int(cfg["matching"]["embedding_dim"])); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); lossfn=nn.BCEWithLogitsLoss(); best=-1; patience=0; history=[]; best_path=LAB/"modelos/two_tower_resnet18_pca.pt"
    bs=64
    for epoch in range(1,args.epochs+1):
        model.train(); perm=torch.randperm(len(tr[2])); losses=[]
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; opt.zero_grad(); logits=model(tr[0][idx],tr[1][idx]); loss=lossfn(logits,tr[2][idx]); loss.backward(); opt.step(); losses.append(float(loss))
        model.eval()
        with torch.no_grad(): vp=torch.sigmoid(model(va[0],va[1])).numpy()
        vm=metrics(va[2].numpy(),vp); history.append({"epoch":epoch,"loss":np.mean(losses),**{f"val_{k}":v for k,v in vm.items()}}); print(f"epoch={epoch} loss={np.mean(losses):.4f} val_auc={vm['roc_auc']:.4f}",flush=True)
        if vm["roc_auc"]>best+1e-4: best=vm["roc_auc"];patience=0;torch.save({"model":model.state_dict(),"face_dim":len(pca),"pca_columns":pca,"seed":seed},best_path)
        else: patience+=1
        if patience>=12: break
    pd.DataFrame(history).to_csv(LAB/"resultados/matching/historico_two_tower.csv",index=False)
    saved=torch.load(best_path,map_location="cpu",weights_only=False); model.load_state_dict(saved["model"]);model.eval()
    pair_predictions=[]
    for split,arr in [("treino",tr),("validacao",va),("teste",te)]:
        with torch.no_grad(): prob=torch.sigmoid(model(arr[0],arr[1])).numpy()
        for (fi,ci),label,p in zip(arr[3],arr[2].numpy(),prob): pair_predictions.append({"split":split,"face_id":fi,"cranio_id":ci,"label":int(label),"score_global_two_tower":float(p)})
    pred_df=pd.DataFrame(pair_predictions);save_csv(pred_df,LAB/"resultados/matching/predicoes_two_tower.csv"); testp=pred_df[pred_df.split=="teste"]; result={"backbone":"resnet18_imagenet_frozen","face_features":len(pca),"classification":metrics(testp.label.to_numpy(),testp.score_global_two_tower.to_numpy())}
    ids=eligible[eligible.split=="teste"].id_pessoa.astype(str).tolist(); S=torch.tensor(np.asarray([skull_map[i] for i in ids]),dtype=torch.float32);F=torch.tensor(np.asarray([face_map[i] for i in ids]),dtype=torch.float32)
    with torch.no_grad(): se=model.skull(S);fe=model.face(F);se=nn.functional.normalize(se,dim=1);fe=nn.functional.normalize(fe,dim=1);scores=(fe@se.T).numpy()
    ranks=[]
    for i,q in enumerate(ids):
        order=np.argsort(-scores[i]); rank=int(np.where(order==i)[0][0])+1
        for pos,j in enumerate(order[:20],1): ranks.append({"query_id":q,"candidate_id":ids[j],"rank":pos,"score_global_two_tower":float(scores[i,j]),"correto":i==j})
    save_csv(pd.DataFrame(ranks),LAB/"resultados/matching/ranking_two_tower_teste.csv"); positions=[]
    for i in range(len(ids)): positions.append(int(np.where(np.argsort(-scores[i])==i)[0][0])+1)
    pos=np.asarray(positions);result["ranking"]={"n_queries":len(ids),"recall_at_1":float(np.mean(pos<=1)),"recall_at_5":float(np.mean(pos<=5)),"recall_at_10":float(np.mean(pos<=10)),"recall_at_20":float(np.mean(pos<=20)),"mrr":float(np.mean(1/pos)),"median_rank":float(np.median(pos))}
    (LAB/"resultados/matching/metricas_two_tower.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
