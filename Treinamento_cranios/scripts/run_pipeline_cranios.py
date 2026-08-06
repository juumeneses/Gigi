# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from common_cranios import NORMALIZATION_KINDS, NORMALIZATION_ORIGINAL, as_root, cleanup_generated_outputs, discover_years, ensure_base_dirs


BASE_STEPS = [
    "00_checar_dataset_cranios.py",
    "01_extrair_contorno_cranios.py",
    "02_gerar_shape_cranios_64pts.py",
    "03_gerar_medidas_cranios.py",
    "04_analisar_qualidade_cranios.py",
    "08_exportar_features_cranios_final.py",
]
SEXO_STEPS = [
    "05_preparar_silhuetas_sexo_cranio.py",
    "06_juntar_genero_faces_cranios.py",
    "07_treinar_sexo_cranio.py",
    "08_exportar_features_cranios_final.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-sexo", action="store_true")
    ap.add_argument("--faces-root", default="../Treinamento_faces")
    ap.add_argument("--use-normalized", action="store_true")
    ap.add_argument("--normalization-kind", default=NORMALIZATION_ORIGINAL, choices=[NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS])
    args = ap.parse_args()
    if args.use_normalized:
        args.normalization_kind = "atual"
    if args.normalization_kind != NORMALIZATION_ORIGINAL:
        os.environ["CRANIOS_NORMALIZATION_KIND"] = args.normalization_kind
    root = as_root(args.root)
    ensure_base_dirs(root)
    years = discover_years(root, args.years)
    if args.force:
        removed = cleanup_generated_outputs(root, years, keep_logs=True)
        print(f"[FORCE] removidos {len(removed)} outputs gerados")
    steps = list(BASE_STEPS)
    if args.with_sexo:
        steps = BASE_STEPS[:-1] + SEXO_STEPS
    for name in steps:
        cmd = [sys.executable, str(root / "scripts" / name), "--root", str(root)]
        if args.years and name in ["00_checar_dataset_cranios.py", "01_extrair_contorno_cranios.py", "02_gerar_shape_cranios_64pts.py", "03_gerar_medidas_cranios.py", "05_preparar_silhuetas_sexo_cranio.py"]:
            cmd += ["--years"] + [str(y) for y in args.years]
        if args.force and name in ["01_extrair_contorno_cranios.py", "02_gerar_shape_cranios_64pts.py", "05_preparar_silhuetas_sexo_cranio.py"]:
            cmd.append("--force")
        if name in ["01_extrair_contorno_cranios.py", "05_preparar_silhuetas_sexo_cranio.py"]:
            cmd += ["--normalization-kind", args.normalization_kind]
        if name == "06_juntar_genero_faces_cranios.py":
            cmd += ["--faces-root", args.faces_root]
        print("\n[ETAPA]", name)
        print("[CMD]", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(root))
        if rc:
            raise SystemExit(rc)
    print("[OK] pipeline cranios completo")


if __name__ == "__main__":
    main()
