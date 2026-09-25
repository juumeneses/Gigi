#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import LAB_ROOT, ensure_dirs, load_config, write_run_metadata
from src.cranios import run_cranios
from src.dataset import build_audit, build_locked_splits, build_master_manifest, create_link_tree
from src.faces import run_faces
from src.hybrid import build_component_scores, evaluate_hybrid
from src.matching import build_pair_table, build_paired_dataset, run_anatomical_ranking
from src.reports import generate_reports


STAGES = ["dataset", "auditoria", "cranios", "faces", "matching", "relatorios"]


def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline do laboratorio hibrido cranio-face")
    ap.add_argument("--stage", choices=["all", *STAGES], default="all")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--years", nargs="*", type=int)
    return ap.parse_args()


def main() -> int:
    args = parse_args(); ensure_dirs(); cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 20260915)); years = args.years or cfg.get("years")
    selected = STAGES if args.stage == "all" else STAGES[:STAGES.index(args.stage)+1]
    try:
        manifest = build_master_manifest(years, force=args.force and "dataset" in selected)
        splits = build_locked_splits(manifest, seed, force=False)
        if "dataset" in selected:
            create_link_tree(manifest, force=args.force); write_run_metadata("dataset", cfg, seed, {"identidades": len(manifest)})
            print(f"[OK] dataset: {len(manifest)} identidades")
        if "auditoria" in selected:
            build_audit(manifest, splits); write_run_metadata("auditoria", cfg, seed); print("[OK] auditoria")
        skull, skull_points = run_cranios(manifest, cfg, force=args.force and "cranios" in selected)
        if "cranios" in selected:
            write_run_metadata("cranios", cfg, seed, {"imagens": len(skull), "pontos": len(skull_points)}); print(f"[OK] cranios: {len(skull)}")
        face, face_points = run_faces(manifest, force=args.force and "faces" in selected)
        if "faces" in selected:
            write_run_metadata("faces", cfg, seed, {"imagens": len(face), "pontos": len(face_points)}); print(f"[OK] faces: {len(face)}")
        paired = build_paired_dataset(manifest, splits, skull, face); pairs = build_pair_table(paired, seed)
        ranking, metrics = run_anatomical_ranking(paired)
        gate_path = LAB_ROOT / "resultados/cranios/gate_viabilidade.csv"
        import pandas as pd
        gate = pd.read_csv(gate_path); anatomy_enabled = bool(gate.iloc[0]["ativar_anatomia_matching"])
        hybrid = evaluate_hybrid(build_component_scores(paired, pairs), anatomy_enabled)
        if "matching" in selected:
            write_run_metadata("matching", cfg, seed, {"pares": len(pairs), **metrics, "hibrido": hybrid}); print(f"[OK] matching: {metrics}; hibrido={hybrid['status']}")
        if "relatorios" in selected:
            report = generate_reports(manifest, skull, face, ranking, metrics, int(cfg["relatorios"]["max_cards_por_pagina"]))
            write_run_metadata("relatorios", cfg, seed, {"index": str(report)}); print(f"[OK] relatorio: {report}")
        return 0
    except Exception as exc:
        log = LAB_ROOT / "logs" / "ultima_falha.log"; log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[ERRO] {exc}\nLog: {log}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
