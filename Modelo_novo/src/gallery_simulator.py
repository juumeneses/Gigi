from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROJECT_DIR

EXPECTED_YEARS = ("2017", "2019", "2021", "2022", "2023")


def _ranking_files_for_variant(variant: str) -> list[Path]:
    """Resolve exatamente um ranking LOYO por ano para a variante solicitada."""
    summary_paths=[PROJECT_DIR/"studies/leave_one_year_out/summary.csv",*sorted((PROJECT_DIR/"studies/ensembles").glob("*/summary.csv"))]
    summaries=[pd.read_csv(path,dtype={"holdout_year":str}) for path in summary_paths if path.exists()]
    if not summaries: raise FileNotFoundError("Nenhum resumo LOYO ou de ensemble encontrado")
    summary=pd.concat(summaries,ignore_index=True)
    required = {"holdout_year", "variant", "run_dir"}
    missing_columns = required.difference(summary.columns)
    if missing_columns:
        raise RuntimeError(f"Colunas ausentes no resumo LOYO: {sorted(missing_columns)}")
    selected = summary[summary.variant.astype(str).eq(variant)].copy()
    duplicates = selected[selected.duplicated("holdout_year", keep=False)].holdout_year.unique()
    if len(duplicates):
        raise RuntimeError(f"Mais de um resultado para a variante '{variant}' nos anos: {sorted(duplicates)}")
    rows = selected.set_index("holdout_year")
    missing_years = sorted(set(EXPECTED_YEARS).difference(rows.index))
    if missing_years:
        available = sorted(summary.variant.astype(str).unique())
        raise RuntimeError(
            f"Variante '{variant}' incompleta; faltam os anos {missing_years}. "
            f"Variantes disponíveis: {available}"
        )
    paths = [PROJECT_DIR / str(rows.loc[year, "run_dir"]) / "ranking_test.csv" for year in EXPECTED_YEARS]
    missing_files = [str(path) for path in paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Rankings registrados mas não encontrados: {missing_files}")
    return paths


def _metrics(ranks: pd.Series, size: int) -> dict:
    values = ranks.to_numpy(float)
    return {"n":int(len(values)),"gallery_size":size,"top_1":float(np.mean(values<=1)),"top_3":float(np.mean(values<=3)),"top_5":float(np.mean(values<=5)),"mrr":float(np.mean(1/values)),"mean_rank":float(np.mean(values)),"median_rank":float(np.median(values))}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    pos=labels==1; n_pos=pos.sum(); n_neg=(~pos).sum()
    return float((ranks[pos].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)) if n_pos and n_neg else float("nan")


def _best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float,float]:
    candidates=np.unique(np.quantile(scores,np.linspace(0.02,0.98,97))); best=(-1.0,float(candidates[0]))
    for threshold in candidates:
        pred=scores>=threshold; tpr=np.mean(pred[labels==1]); tnr=np.mean(~pred[labels==0]); balanced=(tpr+tnr)/2
        if balanced>best[0]: best=(float(balanced),float(threshold))
    return best[1],best[0]


def _cross_validated_rejection(samples: pd.DataFrame, seed: int) -> dict:
    rng=np.random.default_rng(seed); query_ids=samples.query_id.unique().copy(); rng.shuffle(query_ids); folds=np.array_split(query_ids,5); rows=[]
    for i,test_ids in enumerate(folds):
        train=samples[~samples.query_id.isin(test_ids)]; test=samples[samples.query_id.isin(test_ids)]
        threshold,_=_best_threshold(train.present.to_numpy(int),train.top_score.to_numpy(float))
        labels=test.present.to_numpy(int); pred=test.top_score.to_numpy(float)>=threshold
        rows.append({"fold":i+1,"threshold":threshold,"sensitivity_present":float(np.mean(pred[labels==1])),"specificity_absent":float(np.mean(~pred[labels==0])),"balanced_accuracy":float((np.mean(pred[labels==1])+np.mean(~pred[labels==0]))/2)})
    frame=pd.DataFrame(rows)
    return {"auc_top_score":_auc(samples.present.to_numpy(int),samples.top_score.to_numpy(float)),"cv_balanced_accuracy":float(frame.balanced_accuracy.mean()),"cv_sensitivity_present":float(frame.sensitivity_present.mean()),"cv_specificity_absent":float(frame.specificity_absent.mean()),"threshold_mean":float(frame.threshold.mean()),"threshold_std":float(frame.threshold.std()),"folds":rows}


def _sample_gallery(group: pd.DataFrame, query: str, size: int, scenario: str, rng: np.random.Generator, sex_map: dict[str,str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    correct=group[group.candidate_id.eq(query)].iloc[0]
    wrong=group[~group.candidate_id.eq(query)].copy()
    if scenario=="hard": selected=wrong.nlargest(size-1,"score")
    elif scenario=="same_sex":
        same=wrong[wrong.candidate_id.map(sex_map).eq(sex_map.get(query))]
        pool=same if len(same)>=size-1 else wrong
        selected=pool.iloc[rng.choice(len(pool),size-1,replace=False)]
    else: selected=wrong.iloc[rng.choice(len(wrong),size-1,replace=False)]
    present=pd.concat([selected,pd.DataFrame([correct])]).sort_values("score",ascending=False).reset_index(drop=True)
    present["sim_rank"]=np.arange(1,len(present)+1)
    if scenario=="hard": absent=wrong.nlargest(size,"score")
    elif scenario=="same_sex":
        same=wrong[wrong.candidate_id.map(sex_map).eq(sex_map.get(query))]; pool=same if len(same)>=size else wrong
        absent=pool.iloc[rng.choice(len(pool),size,replace=False)].sort_values("score",ascending=False)
    else: absent=wrong.iloc[rng.choice(len(wrong),size,replace=False)].sort_values("score",ascending=False)
    return present,absent


def _sample_fast(group: pd.DataFrame, query: str, size: int, scenario: str, rng: np.random.Generator, sex_map: dict[str,str]) -> tuple[int,float,float,float,float]:
    candidates=group.candidate_id.astype(str).to_numpy(); scores=group.score.to_numpy(float)
    correct_idx=int(np.flatnonzero(candidates==query)[0]); correct_score=float(scores[correct_idx]); wrong_idx=np.flatnonzero(candidates!=query)
    if scenario=="same_sex":
        query_sex=sex_map.get(query); same=np.array([i for i in wrong_idx if sex_map.get(candidates[i])==query_sex],dtype=int)
        pool=same if len(same)>=size else wrong_idx
    else: pool=wrong_idx
    if scenario=="hard": selected=wrong_idx[np.argsort(scores[wrong_idx])[-(size-1):]]; absent_idx=wrong_idx[np.argsort(scores[wrong_idx])[-size:]]
    else: selected=rng.choice(pool,size-1,replace=False); absent_idx=rng.choice(pool,size,replace=False)
    present_scores=np.concatenate(([correct_score],scores[selected])); ordered=np.sort(present_scores)[::-1]
    absent_ordered=np.sort(scores[absent_idx])[::-1]
    rank=1+int(np.sum(scores[selected]>correct_score))
    return rank,float(ordered[0]),float(ordered[0]-ordered[1]),float(absent_ordered[0]),float(absent_ordered[0]-absent_ordered[1])


def simulate_galleries(size: int=10, repeats: int=100, seed: int=42, output_dir: Path|None=None, variant: str="ensemble_domain_normalized_hard_negatives") -> tuple[pd.DataFrame,dict]:
    if size<2: raise ValueError("A galeria precisa ter pelo menos 2 candidatos")
    split=pd.read_csv(PROJECT_DIR/"data/splits/all_splits.csv"); sex_map=dict(zip(split.pair_key.astype(str),split.sexo_face.astype(str)))
    ranking_files=_ranking_files_for_variant(variant)
    rng=np.random.default_rng(seed); rank_rows=[]; detection_rows=[]
    for path in ranking_files:
        ranking=pd.read_csv(path); year=str(ranking.query_id.iloc[0]).split("::",1)[0]
        if ranking.groupby("query_id").size().min()<size: raise ValueError(f"Galeria {size} maior que candidatos de {year}")
        for query,group in ranking.groupby("query_id",sort=False):
            for scenario in ("random","same_sex","hard"):
                scenario_repeats=1 if scenario=="hard" else repeats
                for repeat in range(scenario_repeats):
                    rank,p_top,p_margin,a_top,a_margin=_sample_fast(group,str(query),size,scenario,rng,sex_map)
                    rank_rows.append({"year":year,"query_id":query,"sex":sex_map.get(str(query),"INDEFINIDO"),"scenario":scenario,"repeat":repeat,"rank":rank})
                    detection_rows.append({"year":year,"query_id":query,"scenario":scenario,"repeat":repeat,"present":1,"top_score":p_top,"margin":p_margin})
                    detection_rows.append({"year":year,"query_id":query,"scenario":scenario,"repeat":repeat,"present":0,"top_score":a_top,"margin":a_margin})
    ranks=pd.DataFrame(rank_rows); detection=pd.DataFrame(detection_rows)
    summaries=[]
    for keys,group in ranks.groupby(["scenario","year"],sort=True): summaries.append({"scenario":keys[0],"year":keys[1],**_metrics(group["rank"],size)})
    for scenario,group in ranks.groupby("scenario",sort=True): summaries.append({"scenario":scenario,"year":"TODOS",**_metrics(group["rank"],size)})
    summary=pd.DataFrame(summaries)
    rejection={scenario:_cross_validated_rejection(group,seed) for scenario,group in detection.groupby("scenario")}
    output_dir=output_dir or PROJECT_DIR/"reports"/f"gallery_{size}_{variant}"
    output_dir.mkdir(parents=True,exist_ok=True); summary.to_csv(output_dir/"metrics.csv",index=False); ranks.to_csv(output_dir/"simulated_ranks.csv",index=False); detection.to_csv(output_dir/"presence_samples.csv",index=False)
    report={"gallery_size":size,"variant":variant,"repeats":repeats,"seed":seed,"queries":int(ranks.query_id.nunique()),"ranking_files":[str(path.relative_to(PROJECT_DIR)) for path in ranking_files],"rejection":rejection,"warning":"Scores sao relativos; limiares sao experimentais e nao probabilidades periciais."}
    (output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary,report
