# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

import pandas as pd

from common_cranios import *


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--n-points", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    all_rows, flat_rows, fails = [], [], []

    for y in discover_years(root, args.years):
        for p in sorted((contornos_dir(root) / str(y)).glob("*_contorno.csv")):
            stem = p.stem.replace("_contorno", "")
            out_csv = landmarks_dir(root) / str(y) / f"{stem}_pseudolandmarks_{args.n_points}.csv"
            if out_csv.exists() and not args.force:
                df_old = read_csv_safe(out_csv)
                all_rows.extend(df_old.to_dict("records"))
                continue
            df = read_csv_safe(p)
            if df.empty:
                fails.append({"ano": y, "id_imagem": stem, "erro": "contorno vazio"})
                continue
            try:
                pts = df[["x_px", "y_px"]].to_numpy(float)
                xy, norm = resample_closed_contour(pts, args.n_points, align_pca=True)
            except Exception as e:
                fails.append({"ano": y, "id_imagem": stem, "erro": str(e)})
                continue
            rows = []
            for i, ((x, ypx), (xn, yn)) in enumerate(zip(xy, norm)):
                rows.append({"ano": str(y), "id_imagem": stem, "id_pessoa": person_id_from_stem(stem), "idx": i, "x_px": x, "y_px": ypx, "x_norm": xn, "y_norm": yn, "tipo_ponto": "PSEUDO_CONTOUR"})
            safe_to_csv(pd.DataFrame(rows), out_csv)
            all_rows.extend(rows)
            frow = {"ano": str(y), "id_imagem": stem, "id_pessoa": person_id_from_stem(stem)}
            for i, (_, (xn, yn)) in enumerate(zip(xy, norm)):
                frow[f"cranio_shape_{i:03d}_x"] = float(xn)
                frow[f"cranio_shape_{i:03d}_y"] = float(yn)
            flat_rows.append(frow)

            img_path = df.get("imagem_cranio", pd.Series([""])).iloc[0]
            img = read_image(Path(img_path)) if img_path else None
            if img is not None:
                contour = pts.astype(np.int32).reshape(-1, 1, 2)
                write_image(previews_dir(root) / str(y) / f"{stem}_shape64_preview.jpg", draw_contour_preview(img, contour, xy))

    if not flat_rows and all_rows:
        tmp = pd.DataFrame(all_rows)
        for (ano, stem), g in tmp.groupby(["ano", "id_imagem"]):
            frow = {"ano": str(ano), "id_imagem": stem, "id_pessoa": person_id_from_stem(stem)}
            for _, r in g.sort_values("idx").iterrows():
                i = int(r["idx"])
                frow[f"cranio_shape_{i:03d}_x"] = r["x_norm"]
                frow[f"cranio_shape_{i:03d}_y"] = r["y_norm"]
            flat_rows.append(frow)
    safe_to_csv(pd.DataFrame(flat_rows), features_dir(root) / "shape_cranios_64pts.csv")
    safe_to_csv(pd.DataFrame(fails), landmarks_dir(root) / "falhas_shape_cranios_64pts.csv")
    print(f"[OK] shape cranios: {len(flat_rows)}")


if __name__ == "__main__":
    main()
