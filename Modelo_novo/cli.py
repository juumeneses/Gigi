from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.baselines import run_baselines
from src.index_builder import build_metadata
from src.preprocess import preprocess_manifest
from src.splits import create_leave_one_year_out, create_splits
from src.training import train_two_tower
from src.studies import VIEW_COMBINATIONS, build_multiseed_ensemble, run_loyo_fold, run_multiseed_loyo, run_multiseed_validation, run_validation_tuning, run_view_ablation
from src.gallery_simulator import simulate_galleries
from src.domain_preprocessing import audit_domains, normalize_domains


def _merge_config(base: dict, override: dict) -> dict:
    merged=dict(base)
    for key,value in override.items():
        if key in merged and isinstance(merged[key],dict) and isinstance(value,dict): merged[key]=_merge_config(merged[key],value)
        else: merged[key]=value
    return merged


def _load_config(target: Path, seen: set[Path]|None=None) -> dict:
    target=target.resolve(); seen=set() if seen is None else seen
    if target in seen: raise ValueError(f"Herança circular de configuração: {target}")
    seen.add(target); current=yaml.safe_load(target.read_text(encoding="utf-8")) or {}; parent=current.pop("extends",None)
    if not parent: return current
    parent_path=Path(parent); parent_path=parent_path if parent_path.is_absolute() else target.parent/parent_path
    return _merge_config(_load_config(parent_path,seen),current)


def config(path: str | None) -> dict:
    target = Path(path) if path else PROJECT_DIR / "configs/default.yaml"
    return _load_config(target)


def metadata() -> pd.DataFrame:
    path = PROJECT_DIR / "data/manifests/metadata_pares.csv"
    return pd.read_csv(path) if path.exists() else build_metadata()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Matching face-cranio reproduzivel")
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-index")
    sub.add_parser("create-splits")
    pre = sub.add_parser("preprocess"); pre.add_argument("--limit", type=int); pre.add_argument("--force", action="store_true")
    sub.add_parser("baselines")
    sub.add_parser("train")
    views = sub.add_parser("ablate-views", help="Compara vistas usando somente treino/validacao")
    views.add_argument("--combination", choices=["all", *VIEW_COMBINATIONS], default="all")
    views.add_argument("--epochs", type=int)
    loyo = sub.add_parser("loyo", help="Treina um holdout anual")
    loyo.add_argument("--year", choices=["2017", "2019", "2021", "2022", "2023"], required=True)
    loyo.add_argument("--epochs", type=int)
    multi=sub.add_parser("multiseed-loyo",help="Repete os holdouts com múltiplas sementes e splits congelados")
    multi.add_argument("--years",nargs="+",choices=["all","2017","2019","2021","2022","2023"],default=["all"])
    multi.add_argument("--seeds",nargs="+",type=int,default=[42,123,456]); multi.add_argument("--epochs",type=int)
    ensemble=sub.add_parser("ensemble-rankings",help="Combina por média os rankings de múltiplas sementes")
    ensemble.add_argument("--source-variant",default="domain_normalized_hard_negatives")
    validation=sub.add_parser("multiseed-validation",help="Avalia múltiplas sementes sem carregar o teste")
    validation.add_argument("--seeds",nargs="+",type=int,default=[42,123,456]); validation.add_argument("--epochs",type=int)
    tuning=sub.add_parser("tune-gallery10",help="Busca parâmetros somente na validação para galerias de dez")
    tuning.add_argument("--seeds",nargs="+",type=int,default=[42,123,456]); tuning.add_argument("--epochs",type=int)
    gallery=sub.add_parser("simulate-gallery",help="Simula galerias menores usando rankings LOYO")
    gallery.add_argument("--size",type=int,default=10); gallery.add_argument("--repeats",type=int,default=100); gallery.add_argument("--seed",type=int,default=42)
    gallery.add_argument("--variant",default="ensemble_domain_normalized_hard_negatives",help="Nome da variante registrada no resumo LOYO")
    audit=sub.add_parser("audit-domains",help="Audita diferenças visuais entre anos"); audit.add_argument("--samples",type=int,default=10)
    norm=sub.add_parser("normalize-images",help="Cria imagens alinhadas e padronizadas"); norm.add_argument("--size",type=int,default=224); norm.add_argument("--limit",type=int); norm.add_argument("--force",action="store_true")
    args = parser.parse_args(); cfg = config(args.config)
    if args.command == "build-index":
        _, report = build_metadata(); print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "create-splits":
        df = create_splits(metadata(), cfg["seed"], (cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"])); create_leave_one_year_out(df); print(df.split.value_counts())
    elif args.command == "preprocess":
        result = preprocess_manifest(metadata(), cfg["data"]["image_size"], args.force, args.limit); print(result.status.value_counts())
    elif args.command == "baselines":
        path = PROJECT_DIR / "data/splits/test.csv"
        if not path.exists(): create_splits(metadata(), cfg["seed"])
        print(run_baselines(pd.read_csv(path)).to_string(index=False))
    elif args.command == "train":
        path = PROJECT_DIR / "data/splits/all_splits.csv"
        if not path.exists(): create_splits(metadata(), cfg["seed"])
        print(train_two_tower(pd.read_csv(path), cfg)[1])
    elif args.command == "ablate-views":
        path = PROJECT_DIR / "data/splits/all_splits.csv"
        if not path.exists(): create_splits(metadata(), cfg["seed"])
        combinations = list(VIEW_COMBINATIONS) if args.combination == "all" else [args.combination]
        print(f"Ablação sem consultar o teste: {', '.join(combinations)}")
        print(run_view_ablation(pd.read_csv(path), cfg, combinations, args.epochs))
    elif args.command == "loyo":
        folder = PROJECT_DIR / "data/splits/leave_one_year_out"
        path = folder / f"holdout_{args.year}.csv"
        if not path.exists(): create_leave_one_year_out(metadata(), seed=cfg["seed"])
        fold = pd.read_csv(path)
        test_years = set(fold[fold.split.eq("test")].ano.astype(str))
        if test_years != {args.year}: raise RuntimeError(f"Split holdout inválido: {test_years}")
        print(f"Holdout {args.year}: {fold.split.value_counts().to_dict()}")
        print(run_loyo_fold(fold, args.year, cfg, args.epochs))
    elif args.command == "multiseed-loyo":
        years=["2017","2019","2021","2022","2023"] if "all" in args.years else args.years
        results,summary=run_multiseed_loyo(cfg,years,args.seeds,args.epochs); print(f"Resultados: {results}\nResumo: {summary}")
    elif args.command == "ensemble-rankings":
        root,summary=build_multiseed_ensemble(args.source_variant); print(f"Ensemble: {root}\nResumo: {summary}")
    elif args.command == "multiseed-validation":
        results,summary=run_multiseed_validation(cfg,args.seeds,args.epochs); print(f"Resultados de validação: {results}\nResumo: {summary}")
    elif args.command == "tune-gallery10":
        results,summary=run_validation_tuning(cfg,args.seeds,args.epochs); print(f"Busca: {results}\nResumo: {summary}")
    elif args.command == "simulate-gallery":
        summary,report=simulate_galleries(args.size,args.repeats,args.seed,variant=args.variant); print(summary.to_string(index=False)); print(json.dumps(report,ensure_ascii=False,indent=2))
    elif args.command == "audit-domains":
        details,summary=audit_domains(metadata(),args.samples,cfg.get("seed",42)); print(f"Imagens auditadas: {len(details)}"); print(summary.to_string(index=False,max_cols=12))
    elif args.command == "normalize-images":
        dc=cfg.get("data",{})
        result=normalize_domains(metadata(),args.size,args.force,args.limit,dc.get("face_views",["FFN"]),dc.get("face_source_root")); print(result.status.value_counts().to_string())


if __name__ == "__main__":
    main()
