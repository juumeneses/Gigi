from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from .paths import PROJECT_DIR
from .training import train_two_tower
from .retrieval import retrieval_metrics, sampled_gallery_metrics


VIEW_COMBINATIONS = {
    "CF": ["CF"],
    "CE_CD": ["CE", "CD"],
    "CF_CE": ["CF", "CE"],
    "CF_CD": ["CF", "CD"],
    "CF_CE_CD": ["CF", "CE", "CD"],
}


def _append_summary(path: Path, row: dict) -> pd.DataFrame:
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def run_view_ablation(splits: pd.DataFrame, base_config: dict, combinations: list[str], epochs: int | None = None) -> Path:
    study_id = "view_ablation"
    root = PROJECT_DIR / "studies" / study_id; root.mkdir(parents=True, exist_ok=True)
    summary = root / "summary.csv"
    for name in combinations:
        if name not in VIEW_COMBINATIONS: raise ValueError(f"Combinação desconhecida: {name}")
        cfg = copy.deepcopy(base_config); cfg["data"]["skull_views"] = VIEW_COMBINATIONS[name]
        if epochs: cfg["training"]["epochs"] = epochs
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / f"{name}_{stamp}"
        _, result = train_two_tower(splits, cfg, run_dir=run_dir, evaluation_split="val")
        metrics = result["val"]
        if summary.exists():
            existing = pd.read_csv(summary)
            existing = existing[existing["combination"].astype(str) != name]
            existing.to_csv(summary, index=False)
        _append_summary(summary, {"study_id":study_id,"combination":name,"views":"+".join(VIEW_COMBINATIONS[name]),"best_epoch":result["best_epoch"],**metrics,"run_dir":str(run_dir.relative_to(PROJECT_DIR))})
    table = pd.read_csv(summary).sort_values("mrr", ascending=False)
    table.to_csv(summary, index=False)
    (root / "best.json").write_text(json.dumps(table.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_loyo_fold(fold: pd.DataFrame, year: str, config: dict, epochs: int | None = None) -> Path:
    root = PROJECT_DIR / "studies" / "leave_one_year_out"; root.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    if epochs: cfg["training"]["epochs"] = epochs
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"holdout_{year}_{stamp}"
    _, result = train_two_tower(fold, cfg, run_dir=run_dir, evaluation_split="test")
    metrics=result["test"]
    variant=str(cfg.get("experiment",{}).get("variant") or ("domain_normalized" if bool(cfg.get("data",{}).get("use_domain_normalized",False)) else "original"))
    summary=root / "summary.csv"
    current=pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    if not current.empty:
        if "variant" not in current: current["variant"]="original"
        current=current[~(current.holdout_year.astype(str).eq(str(year)) & current.variant.astype(str).eq(variant))]
    out=pd.concat([current,pd.DataFrame([{"holdout_year":str(year),"variant":variant,"best_epoch":result["best_epoch"],**metrics,"run_dir":str(run_dir.relative_to(PROJECT_DIR))}])],ignore_index=True)
    out.to_csv(summary,index=False)
    return summary


def summarize_multiseed(results: pd.DataFrame) -> pd.DataFrame:
    metrics=["top_1","top_3","top_5","recall_at_10","mrr","median_rank","mean_rank"]
    rows=[]
    for year,group in results.groupby("holdout_year",sort=True):
        row={"holdout_year":str(year),"n_seeds":int(group.seed.nunique()),"n_queries":int(group.n_queries.iloc[0])}
        for metric in metrics:
            row[f"{metric}_mean"]=float(group[metric].mean()); row[f"{metric}_std"]=float(group[metric].std(ddof=1)) if len(group)>1 else 0.0
        rows.append(row)
    per_seed=[]
    for seed,group in results.groupby("seed"):
        total=float(group.n_queries.sum()); row={"seed":int(seed)}
        for metric in metrics:
            row[metric]=float((group[metric]*group.n_queries).sum()/total)
        per_seed.append(row)
    overall=pd.DataFrame(per_seed); row={"holdout_year":"TODOS","n_seeds":int(results.seed.nunique()),"n_queries":int(results.groupby("seed").n_queries.sum().max())}
    for metric in metrics:
        row[f"{metric}_mean"]=float(overall[metric].mean()); row[f"{metric}_std"]=float(overall[metric].std(ddof=1)) if len(overall)>1 else 0.0
    rows.append(row)
    return pd.DataFrame(rows)


def run_multiseed_loyo(base_config: dict, years: list[str], seeds: list[int], epochs: int|None=None) -> tuple[Path,Path]:
    """Repete LOYO sem recriar splits; resultados existentes são retomados."""
    variant=str(base_config.get("experiment",{}).get("variant","multiseed")); root=PROJECT_DIR/"studies"/"multiseed_loyo"/variant; root.mkdir(parents=True,exist_ok=True)
    results_path=root/"results.csv"; results=pd.read_csv(results_path,dtype={"holdout_year":str}) if results_path.exists() else pd.DataFrame()
    split_root=PROJECT_DIR/"data/splits/leave_one_year_out"
    for year in years:
        split_path=split_root/f"holdout_{year}.csv"
        if not split_path.exists(): raise FileNotFoundError(f"Split congelado ausente: {split_path}")
        fold=pd.read_csv(split_path); test_years=set(fold[fold.split.eq("test")].ano.astype(str))
        if test_years!={str(year)}: raise RuntimeError(f"Split holdout {year} inválido: {test_years}")
        for seed in seeds:
            if not results.empty and ((results.holdout_year.astype(str)==str(year)) & (results.seed.astype(int)==int(seed))).any():
                print(f"Holdout {year}, seed {seed}: resultado existente, pulando",flush=True); continue
            cfg=copy.deepcopy(base_config); cfg["seed"]=int(seed)
            if epochs: cfg["training"]["epochs"]=int(epochs)
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_dir=root/"runs"/f"holdout_{year}_seed_{seed}_{stamp}"
            _,result=train_two_tower(fold,cfg,run_dir=run_dir,evaluation_split="test"); metrics=result["test"]
            row={"holdout_year":str(year),"seed":int(seed),"variant":variant,"best_epoch":result["best_epoch"],**metrics,"run_dir":str(run_dir.relative_to(PROJECT_DIR))}
            results=pd.concat([results,pd.DataFrame([row])],ignore_index=True); results.to_csv(results_path,index=False)
            summarize_multiseed(results).to_csv(root/"summary.csv",index=False)
    summary_path=root/"summary.csv"; summarize_multiseed(results).to_csv(summary_path,index=False)
    return results_path,summary_path


def average_similarity_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    if len(matrices)<2: raise ValueError("O ensemble requer pelo menos dois modelos")
    shape=np.asarray(matrices[0]).shape
    if any(np.asarray(matrix).shape!=shape for matrix in matrices): raise ValueError("Matrizes de similaridade incompatíveis")
    return np.mean(np.stack(matrices).astype(np.float64),axis=0)


def build_multiseed_ensemble(source_variant: str="domain_normalized_hard_negatives") -> tuple[Path,Path]:
    source=PROJECT_DIR/"studies/multiseed_loyo"/source_variant/"results.csv"
    if not source.exists(): raise FileNotFoundError(f"Resultados multissemente ausentes: {source}")
    results=pd.read_csv(source,dtype={"holdout_year":str}); expected_years={"2017","2019","2021","2022","2023"}
    if set(results.holdout_year)!=expected_years: raise RuntimeError(f"Anos incompletos no ensemble: {sorted(set(results.holdout_year))}")
    seed_sets={year:tuple(int(value) for value in sorted(group.seed.astype(int).unique())) for year,group in results.groupby("holdout_year")}
    if len(set(seed_sets.values()))!=1 or len(next(iter(seed_sets.values())))<2: raise RuntimeError(f"Sementes inconsistentes: {seed_sets}")
    variant=f"ensemble_{source_variant}"; root=PROJECT_DIR/"studies/ensembles"/variant; root.mkdir(parents=True,exist_ok=True); rows=[]
    for year,group in results.groupby("holdout_year",sort=True):
        matrices=[]; query_ids=None; candidate_ids=None
        for _,record in group.sort_values("seed").iterrows():
            ranking=pd.read_csv(PROJECT_DIR/str(record.run_dir)/"ranking_test.csv")
            pivot=ranking.pivot(index="query_id",columns="candidate_id",values="score").sort_index().sort_index(axis=1)
            current_queries=pivot.index.astype(str).tolist(); current_candidates=pivot.columns.astype(str).tolist()
            if query_ids is not None and (current_queries!=query_ids or current_candidates!=candidate_ids): raise RuntimeError(f"Rankings desalinhados em {year}")
            query_ids=current_queries; candidate_ids=current_candidates; matrices.append(pivot.to_numpy(float))
        similarity=average_similarity_matrices(matrices); metrics,ranking=retrieval_metrics(similarity,query_ids,candidate_ids)
        run_dir=root/f"holdout_{year}"; run_dir.mkdir(parents=True,exist_ok=True); ranking.to_csv(run_dir/"ranking_test.csv",index=False)
        (run_dir/"ensemble.json").write_text(json.dumps({"source_variant":source_variant,"aggregation":"raw_score_mean","seeds":list(seed_sets[year]),"n_models":len(matrices)},ensure_ascii=False,indent=2),encoding="utf-8")
        rows.append({"holdout_year":year,"variant":variant,"aggregation":"raw_score_mean","n_models":len(matrices),"seeds":"+".join(map(str,seed_sets[year])),**metrics,"run_dir":str(run_dir.relative_to(PROJECT_DIR))})
    summary=pd.DataFrame(rows); summary_path=root/"summary.csv"; summary.to_csv(summary_path,index=False)
    return root,summary_path


def run_multiseed_validation(base_config: dict, seeds: list[int], epochs: int|None=None) -> tuple[Path,Path]:
    """Estudo exploratório restrito a treino/validação; o teste não é criado nem carregado."""
    variant=str(base_config.get("experiment",{}).get("variant","validation")); root=PROJECT_DIR/"studies"/"validation_multiseed"/variant; root.mkdir(parents=True,exist_ok=True)
    results_path=root/"results.csv"; results=pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    frozen=pd.read_csv(PROJECT_DIR/"data/splits/all_splits.csv"); study=frozen[frozen.split.isin(["train","val"])].copy()
    if set(study.split)!={"train","val"} or study.pair_key.duplicated().any(): raise RuntimeError("Split de validação inválido")
    for seed in seeds:
        if not results.empty and results.seed.astype(int).eq(int(seed)).any(): print(f"Validação seed {seed}: resultado existente, pulando",flush=True); continue
        cfg=copy.deepcopy(base_config); cfg["seed"]=int(seed)
        if epochs: cfg["training"]["epochs"]=int(epochs)
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); run_dir=root/"runs"/f"seed_{seed}_{stamp}"
        _,result=train_two_tower(study,cfg,run_dir=run_dir,evaluation_split="val"); metrics=result["val"]
        row={"holdout_year":"VALIDACAO","seed":int(seed),"variant":variant,"best_epoch":result["best_epoch"],**metrics,"run_dir":str(run_dir.relative_to(PROJECT_DIR))}
        results=pd.concat([results,pd.DataFrame([row])],ignore_index=True); results.to_csv(results_path,index=False)
    summary=summarize_multiseed(results); summary_path=root/"summary.csv"; summary.to_csv(summary_path,index=False)
    return results_path,summary_path


def _gallery_from_ranking(path: Path, size=10, repeats=50, seed=2026) -> dict:
    ranking=pd.read_csv(path); pivot=ranking.pivot(index="query_id",columns="candidate_id",values="score").sort_index().sort_index(axis=1)
    return sampled_gallery_metrics(pivot.to_numpy(float),size,repeats,seed)


def run_validation_tuning(base_config: dict, seeds: list[int]=(42,123,456), epochs: int|None=None) -> tuple[Path,Path]:
    """Busca em duas fases no split de validação; nunca inclui linhas de teste."""
    trials={
        "baseline":{"temperature":.07,"margin":.20,"hard_negative_weight":.50},
        "temperature_005":{"temperature":.05,"margin":.20,"hard_negative_weight":.50},
        "temperature_010":{"temperature":.10,"margin":.20,"hard_negative_weight":.50},
        "margin_010":{"temperature":.07,"margin":.10,"hard_negative_weight":.50},
        "margin_030":{"temperature":.07,"margin":.30,"hard_negative_weight":.50},
        "weight_025":{"temperature":.07,"margin":.20,"hard_negative_weight":.25},
        "weight_075":{"temperature":.07,"margin":.20,"hard_negative_weight":.75},
    }
    seeds=[int(value) for value in seeds]; screen_seed=seeds[0]; root=PROJECT_DIR/"studies/validation_tuning/gallery10"; root.mkdir(parents=True,exist_ok=True)
    results_path=root/"results.csv"; results=pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    frozen=pd.read_csv(PROJECT_DIR/"data/splits/all_splits.csv"); study=frozen[frozen.split.isin(["train","val"])].copy()
    def execute(trial,seed):
        nonlocal results
        if not results.empty and ((results.trial.astype(str)==trial)&(results.seed.astype(int)==seed)).any(): print(f"Tuning {trial}, seed {seed}: existente, pulando",flush=True); return
        cfg=copy.deepcopy(base_config); cfg["seed"]=seed; cfg.setdefault("experiment",{})["variant"]=f"validation_tuning_{trial}"
        cfg["training"].update(trials[trial]); cfg["training"].update({"checkpoint_metric":"gallery10_mrr","selection_gallery_size":10,"selection_gallery_repeats":50,"selection_gallery_seed":2026})
        if epochs: cfg["training"]["epochs"]=int(epochs)
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"); run_dir=root/"runs"/f"{trial}_seed_{seed}_{stamp}"
        _,result=train_two_tower(study,cfg,run_dir=run_dir,evaluation_split="val"); gallery=_gallery_from_ranking(run_dir/"ranking_val.csv")
        row={"trial":trial,"seed":seed,**trials[trial],"best_epoch":result["best_epoch"],**result["val"],**gallery,"run_dir":str(run_dir.relative_to(PROJECT_DIR))}
        results=pd.concat([results,pd.DataFrame([row])],ignore_index=True); results.to_csv(results_path,index=False)
    for trial in trials: execute(trial,screen_seed)
    screen=results[results.seed.astype(int).eq(screen_seed)].sort_values("gallery10_mrr",ascending=False)
    best_nonbaseline=screen[~screen.trial.eq("baseline")].head(2).trial.tolist(); finalists=["baseline",*best_nonbaseline]
    for trial in finalists:
        for seed in seeds[1:]: execute(trial,seed)
    confirmed=results[results.trial.isin(finalists) & results.seed.astype(int).isin(seeds)]
    metrics=["gallery10_top_1","gallery10_top_3","gallery10_top_5","gallery10_mrr","gallery10_mean_rank","mrr","top_1","top_5","recall_at_10"]
    rows=[]
    for trial,group in confirmed.groupby("trial"):
        row={"trial":trial,"n_seeds":int(group.seed.nunique()),"temperature":group.temperature.iloc[0],"margin":group.margin.iloc[0],"hard_negative_weight":group.hard_negative_weight.iloc[0]}
        for metric in metrics: row[f"{metric}_mean"]=float(group[metric].mean()); row[f"{metric}_std"]=float(group[metric].std(ddof=1)) if len(group)>1 else 0.0
        rows.append(row)
    summary=pd.DataFrame(rows).sort_values("gallery10_mrr_mean",ascending=False); summary_path=root/"summary.csv"; summary.to_csv(summary_path,index=False)
    (root/"selection.json").write_text(json.dumps({"screen_seed":screen_seed,"screen_ranking":screen[["trial","gallery10_mrr"]].to_dict("records"),"finalists":finalists,"confirmation_seeds":seeds},ensure_ascii=False,indent=2),encoding="utf-8")
    return results_path,summary_path
