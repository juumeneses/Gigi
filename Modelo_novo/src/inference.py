from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .domain_preprocessing import normalize_skull_image, normalize_uploaded_face
from .model import TwoTower, require_torch, torch
from .paths import PROJECT_DIR


def ensemble_checkpoint_paths(variant: str="domain_normalized_hard_negatives") -> list[Path]:
    summary_path=PROJECT_DIR/"studies/leave_one_year_out/summary.csv"
    if not summary_path.exists(): raise FileNotFoundError("Resumo dos treinamentos LOYO não encontrado")
    summary=pd.read_csv(summary_path,dtype={"holdout_year":str})
    selected=summary[summary.variant.astype(str).eq(variant)]
    years=("2017","2019","2021","2022","2023")
    if selected.holdout_year.duplicated().any(): raise RuntimeError(f"Há treinamentos duplicados para a variante {variant}")
    rows=selected.set_index("holdout_year")
    missing=sorted(set(years)-set(rows.index))
    if missing: raise RuntimeError(f"Faltam modelos da variante {variant}: {missing}")
    paths=[PROJECT_DIR/str(rows.loc[year,"run_dir"])/"best.pt" for year in years]
    absent=[str(path) for path in paths if not path.exists()]
    if absent: raise FileNotFoundError(f"Checkpoints não encontrados: {absent}")
    return paths


def _tensor(image: Image.Image, size: int, device):
    from torchvision import transforms
    transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return transform(image).unsqueeze(0).to(device)


def load_ensemble(variant: str="domain_normalized_hard_negatives") -> tuple[list, object]:
    require_torch()
    device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
    models=[]
    for path in ensemble_checkpoint_paths(variant):
        state=torch.load(path,map_location=device)
        cfg=state["config"]; mc=cfg.get("model",{})
        model=TwoTower(int(mc.get("embedding_dim",64)),False,bool(mc.get("freeze_backbone",True)),float(mc.get("dropout",.15)),str(mc.get("view_fusion","mean")),int(mc.get("max_skull_views",3)),False,128,str(mc.get("face_view_fusion","mean")),int(mc.get("max_face_views",3))).to(device)
        model.load_state_dict(state["model_state"]); model.eval(); models.append((model,cfg))
    return models,device


def rank_uploaded_candidates(face: Image.Image, candidates: list[tuple[str,list[Image.Image]]], ensemble=None) -> pd.DataFrame:
    if len(candidates)!=10: raise ValueError("Envie exatamente 10 candidatos")
    if any(len(views)!=3 for _,views in candidates): raise ValueError("Cada candidato precisa das vistas CF, CE e CD")
    models,device=ensemble or load_ensemble()
    scores=np.zeros(len(candidates),dtype=float)
    with torch.no_grad():
        for model,cfg in models:
            size=int(cfg.get("data",{}).get("image_size",224))
            face_tensor=_tensor(normalize_uploaded_face(face,size),size,device)
            face_embedding=model.face_encoder(face_tensor)
            for index,(_,views) in enumerate(candidates):
                normalized=[normalize_skull_image(view,size)[0] for view in views]
                skull=torch.stack([_tensor(image,size,device).squeeze(0) for image in normalized]).unsqueeze(0)
                batch,view_count,channels,height,width=skull.shape
                encoded=model.skull_encoder(skull.reshape(batch*view_count,channels,height,width))
                skull_embedding=model.fuse_skull_views(encoded,batch,view_count)
                scores[index]+=float((face_embedding@skull_embedding.T).item())
    scores/=len(models)
    result=pd.DataFrame({"candidato":[name for name,_ in candidates],"similaridade_media":scores})
    result=result.sort_values("similaridade_media",ascending=False).reset_index(drop=True)
    result.insert(0,"posição",np.arange(1,len(result)+1))
    result["diferença_para_líder"]=result.similaridade_media.iloc[0]-result.similaridade_media
    return result
