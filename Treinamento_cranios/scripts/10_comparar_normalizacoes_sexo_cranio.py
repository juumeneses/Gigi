# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from common_cranios import *


def run_step(cmd: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(cwd))
    if rc:
        raise SystemExit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--normalization-kinds", nargs="+", default=[NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS])
    ap.add_argument("--force-silhouettes", action="store_true")
    ap.add_argument("--backend", choices=["sklearn", "vgg16", "vgg16_finetune", "article_vgg16"], default="vgg16_finetune")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    args = ap.parse_args()

    root = as_root(args.root)
    ensure_base_dirs(root)
    allowed = {NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS}
    kinds = [k for k in args.normalization_kinds if k in allowed]
    if not kinds:
        raise SystemExit("nenhuma normalizacao valida informada")

    rows = []
    for kind in kinds:
        print(f"\n[ETAPA] normalizacao={kind}", flush=True)
        sil_cmd = [
            sys.executable,
            str(root / "scripts" / "05_preparar_silhuetas_sexo_cranio.py"),
            "--root",
            str(root),
            "--normalization-kind",
            kind,
        ]
        if args.years:
            sil_cmd += ["--years"] + [str(y) for y in args.years]
        if args.force_silhouettes:
            sil_cmd.append("--force")
        run_step(sil_cmd, root)

        train_cmd = [
            sys.executable,
            str(root / "scripts" / "07_treinar_sexo_cranio.py"),
            "--root",
            str(root),
            "--backend",
            args.backend,
            "--epochs",
            str(int(args.epochs)),
            "--image-size",
            str(int(args.image_size)),
            "--batch-size",
            str(int(args.batch_size)),
            "--weights",
            args.weights,
            "--learning-rate",
            str(float(args.learning_rate)),
            "--normalization-kind",
            kind,
        ]
        run_step(train_cmd, root)

        metrics = read_csv_safe(sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
        comp = read_csv_safe(sexo_dir(root) / "comparacao_genero_face_cranio.csv")
        if metrics.empty:
            rows.append({"normalizacao": kind, "status": "sem_metricas"})
            continue
        row = metrics.iloc[-1].to_dict()
        row["normalizacao"] = kind
        row["status"] = "OK"
        if not comp.empty and "resultado_genero" in comp.columns:
            res = comp["resultado_genero"].astype(str).str.upper()
            total = max(len(comp), 1)
            row["comparados"] = len(comp)
            row["acertos"] = int(res.eq("ACERTO").sum())
            row["divergencias"] = int(res.eq("DIVERGENCIA").sum())
            row["indefinidos"] = int(res.eq("INDEFINIDO").sum())
            row["acertos_pct"] = round(row["acertos"] / total * 100, 2)
            row["divergencias_pct"] = round(row["divergencias"] / total * 100, 2)
            row["indefinidos_pct"] = round(row["indefinidos"] / total * 100, 2)
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["balanced_accuracy_pct", "f1_pct", "acertos_pct"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)
        out["melhor_normalizacao"] = False
        out.loc[0, "melhor_normalizacao"] = True
    safe_to_csv(out, sexo_dir(root) / "comparativo_normalizacoes.csv")
    print(f"[OK] comparativo salvo: {sexo_dir(root) / 'comparativo_normalizacoes.csv'}", flush=True)


if __name__ == "__main__":
    main()
