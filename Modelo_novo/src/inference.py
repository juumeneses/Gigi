from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .domain_preprocessing import normalize_skull_image, normalize_uploaded_face
from .model import TwoTower, require_torch, torch
from .paths import PROJECT_DIR


def ensemble_checkpoint_paths(variant: str="domain_normalized_hard_negatives", require_complete: bool = True) -> list[Path]:
    summary_path=PROJECT_DIR/"studies/leave_one_year_out/summary.csv"
    if not summary_path.exists(): raise FileNotFoundError("Resumo dos treinamentos LOYO não encontrado")
    summary=pd.read_csv(summary_path,dtype={"holdout_year":str})
    selected=summary[summary.variant.astype(str).eq(variant)]
    years=("2017","2019","2021","2022","2023")
    if selected.empty:
        raise RuntimeError(f"Variante não encontrada no resumo LOYO: {variant}")
    selected=selected.copy()
    selected["checkpoint_exists"]=selected["run_dir"].map(lambda value: (PROJECT_DIR/str(value)/"best.pt").exists())
    selected["mtime"]=selected["run_dir"].map(lambda value: (PROJECT_DIR/str(value)/"best.pt").stat().st_mtime if (PROJECT_DIR/str(value)/"best.pt").exists() else 0)
    selected=selected.sort_values(["holdout_year","checkpoint_exists","mtime"],ascending=[True,False,False]).drop_duplicates("holdout_year",keep="first")
    rows=selected.set_index("holdout_year")
    missing=sorted(set(years)-set(rows.index))
    if missing and require_complete: raise RuntimeError(f"Faltam modelos da variante {variant}: {missing}")
    paths=[PROJECT_DIR/str(rows.loc[year,"run_dir"])/"best.pt" for year in years if year in rows.index]
    absent=[str(path) for path in paths if not path.exists()]
    if absent and require_complete: raise FileNotFoundError(f"Checkpoints não encontrados: {absent}")
    paths=[path for path in paths if path.exists()]
    if not paths: raise FileNotFoundError(f"Nenhum checkpoint existente para a variante {variant}")
    return paths


def available_checkpoint_sets() -> pd.DataFrame:
    rows=[]
    seen_paths=set()

    def ranking_metrics(run_dir: Path) -> dict:
        ranking=run_dir/"ranking_test.csv"
        if not ranking.exists():
            return {}
        try:
            df=pd.read_csv(ranking)
        except Exception:
            return {}
        if df.empty or not {"query_id","rank","correto"}.issubset(df.columns):
            return {}
        correct=df[df["correto"].fillna(False)].copy()
        ranks=pd.to_numeric(correct["rank"],errors="coerce")
        queries=max(int(df["query_id"].nunique()),1)
        return {
            "top_1":float(ranks.le(1).sum()/queries),
            "top_3":float(ranks.le(3).sum()/queries),
            "top_5":float(ranks.le(5).sum()/queries),
            "mrr":float((1/ranks).mean()) if len(ranks) else 0.0,
        }

    def add_row(row: dict) -> None:
        row["key"]=f"{row['fonte']}::{row['variant']}"
        row["label"]=str(row.get("label") or row["variant"])
        row.setdefault("top_1", np.nan)
        row.setdefault("top_3", np.nan)
        row.setdefault("top_5", np.nan)
        row.setdefault("mrr", np.nan)
        rows.append(row)

    summary_path=PROJECT_DIR/"studies/leave_one_year_out/summary.csv"
    if summary_path.exists():
        summary=pd.read_csv(summary_path,dtype={"holdout_year":str})
        for variant, group in summary.groupby("variant", dropna=False):
            checkpoints=[]
            years=[]
            for _, row in group.iterrows():
                checkpoint=PROJECT_DIR/str(row.get("run_dir"))/"best.pt"
                if checkpoint.exists():
                    checkpoints.append(checkpoint)
                    seen_paths.add(checkpoint.resolve())
                    years.append(str(row.get("holdout_year")))
            if checkpoints:
                existing=group[group["run_dir"].map(lambda value: (PROJECT_DIR/str(value)/"best.pt").exists())]
                add_row({
                    "fonte":"loyo",
                    "variant":str(variant),
                    "label":f"LOYO · {variant}",
                    "checkpoints":checkpoints,
                    "anos":", ".join(sorted(set(years))),
                    "n_checkpoints":len(checkpoints),
                    "completo":len(set(years))>=5,
                    "mtime":max(path.stat().st_mtime for path in checkpoints),
                    "top_1":float(pd.to_numeric(existing.get("top_1"),errors="coerce").mean()) if not existing.empty and "top_1" in existing else np.nan,
                    "top_3":float(pd.to_numeric(existing.get("top_3"),errors="coerce").mean()) if not existing.empty and "top_3" in existing else np.nan,
                    "top_5":float(pd.to_numeric(existing.get("top_5"),errors="coerce").mean()) if not existing.empty and "top_5" in existing else np.nan,
                    "mrr":float(pd.to_numeric(existing.get("mrr"),errors="coerce").mean()) if not existing.empty and "mrr" in existing else np.nan,
                })
                for _, item in existing.iterrows():
                    checkpoint=PROJECT_DIR/str(item.get("run_dir"))/"best.pt"
                    add_row({
                        "fonte":"checkpoint_loyo",
                        "variant":checkpoint.parent.name,
                        "label":f"LOYO {item.get('holdout_year')} · {item.get('variant')} · {checkpoint.parent.name}",
                        "checkpoints":[checkpoint],
                        "anos":str(item.get("holdout_year")),
                        "n_checkpoints":1,
                        "completo":False,
                        "mtime":checkpoint.stat().st_mtime,
                        "top_1":float(item.get("top_1", np.nan)),
                        "top_3":float(item.get("top_3", np.nan)),
                        "top_5":float(item.get("top_5", np.nan)),
                        "mrr":float(item.get("mrr", np.nan)),
                    })
    for checkpoint in sorted((PROJECT_DIR/"runs").glob("*/best.pt")):
        seen_paths.add(checkpoint.resolve())
        metrics=ranking_metrics(checkpoint.parent)
        add_row({
            "fonte":"run_unico",
            "variant":checkpoint.parent.name,
            "label":f"Run único · {checkpoint.parent.name}",
            "checkpoints":[checkpoint],
            "anos":"todos",
            "n_checkpoints":1,
            "completo":False,
            "mtime":checkpoint.stat().st_mtime,
            **metrics,
        })
    for checkpoint in sorted((PROJECT_DIR/"studies").rglob("best.pt")):
        if checkpoint.resolve() in seen_paths:
            continue
        metrics=ranking_metrics(checkpoint.parent)
        add_row({
            "fonte":"checkpoint_avulso",
            "variant":checkpoint.parent.name,
            "label":f"Checkpoint avulso · {checkpoint.parent.name}",
            "checkpoints":[checkpoint],
            "anos":"-",
            "n_checkpoints":1,
            "completo":False,
            "mtime":checkpoint.stat().st_mtime,
            **metrics,
        })
    if not rows:
        return pd.DataFrame()
    df=pd.DataFrame(rows)
    df["mrr_sort"]=pd.to_numeric(df["mrr"],errors="coerce").fillna(-1)
    return df.sort_values(["mrr_sort","completo","n_checkpoints","mtime"],ascending=[False,False,False,False]).drop(columns=["mrr_sort"]).reset_index(drop=True)


def best_available_checkpoint_paths(selection: str | None = None) -> tuple[list[Path], dict]:
    available=available_checkpoint_sets()
    if available.empty:
        raise FileNotFoundError("Nenhum checkpoint best.pt encontrado. Treine o two-tower primeiro.")
    if selection:
        selected=available[
            available["key"].astype(str).eq(str(selection))
            | available["variant"].astype(str).eq(str(selection))
            | available["label"].astype(str).eq(str(selection))
        ]
        if selected.empty:
            raise FileNotFoundError(f"Nenhum checkpoint existente para a seleção {selection}")
        row=selected.iloc[0]
    else:
        row=available.iloc[0]
    info={key: row.get(key) for key in ["key","fonte","variant","label","anos","n_checkpoints","completo","top_1","top_3","top_5","mrr"]}
    return list(row["checkpoints"]), info


def _tensor(image: Image.Image, size: int, device):
    from torchvision import transforms
    transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return transform(image).unsqueeze(0).to(device)


def load_ensemble(selection: str | None = None) -> tuple[list, object]:
    require_torch()
    device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
    models=[]
    paths, info = best_available_checkpoint_paths(selection)
    for path in paths:
        state=torch.load(path,map_location=device)
        cfg=state["config"]; mc=cfg.get("model",{})
        model=TwoTower(int(mc.get("embedding_dim",64)),False,bool(mc.get("freeze_backbone",True)),float(mc.get("dropout",.15)),str(mc.get("view_fusion","mean")),int(mc.get("max_skull_views",3)),False,128,str(mc.get("face_view_fusion","mean")),int(mc.get("max_face_views",3))).to(device)
        model.load_state_dict(state["model_state"]); model.eval(); models.append((model,cfg))
    for _, cfg in models:
        cfg["_checkpoint_info"]=info
    return models,device


def _normalize_sex(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("M"):
        return "MASCULINO"
    if text.startswith("F"):
        return "FEMININO"
    return "INDEFINIDO"


def rank_uploaded_candidates(
    face: Image.Image,
    candidates: list[tuple[str,list[Image.Image]]],
    ensemble=None,
    reference_sex: str = "INDEFINIDO",
    candidate_sexes: list[str] | None = None,
    sex_mode: str = "Desligado",
    sex_weight: float = 0.05,
) -> pd.DataFrame:
    if len(candidates)!=10: raise ValueError("Envie exatamente 10 candidatos")
    if any(not (1 <= len(views) <= 3) for _,views in candidates): raise ValueError("Cada candidato precisa de 1 a 3 vistas do crânio")
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
    reference = _normalize_sex(reference_sex)
    candidate_sexes = candidate_sexes or ["INDEFINIDO"] * len(candidates)
    normalized_candidate_sexes = [_normalize_sex(value) for value in candidate_sexes]
    sex_mode_norm = str(sex_mode or "Desligado")
    sex_scores = []
    final_scores = []
    filtered = []
    for embedding_score, candidate_sex in zip(scores, normalized_candidate_sexes):
        known = reference != "INDEFINIDO" and candidate_sex != "INDEFINIDO"
        match = known and reference == candidate_sex
        sex_scores.append(1.0 if match else (0.0 if known else np.nan))
        if sex_mode_norm.startswith("Filtro") and known and not match:
            final_scores.append(float(embedding_score) - 999.0)
            filtered.append(True)
        elif sex_mode_norm.startswith("Penal"):
            penalty = float(sex_weight) if known and not match else 0.0
            final_scores.append(float(embedding_score) - penalty)
            filtered.append(False)
        else:
            final_scores.append(float(embedding_score))
            filtered.append(False)
    result=pd.DataFrame({
        "candidato":[name for name,_ in candidates],
        "vistas_usadas":[len(views) for _,views in candidates],
        "sexo_referencia":reference,
        "sexo_candidato":normalized_candidate_sexes,
        "score_embedding":scores,
        "score_sexo":sex_scores,
        "score_final":final_scores,
        "filtrado_por_sexo":filtered,
    })
    result=result.sort_values("score_final",ascending=False).reset_index(drop=True)
    result.insert(0,"posição",np.arange(1,len(result)+1))
    result["diferença_para_líder"]=result.score_final.iloc[0]-result.score_final
    result["similaridade_media"]=result["score_embedding"]
    return result
