from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import LAB_ROOT, abs_repo, read_image, write_image


FACE_ANCHORS = ("centro_olho_direito", "centro_olho_esquerdo", "mento")
SKULL_ANCHORS = ("orbita_direita", "orbita_esquerda", "mento")


def _points_by_id(frame: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for _, row in frame.iterrows():
        if not bool(row.get("visivel", False)) or not np.isfinite(row.get("x_px", np.nan)):
            continue
        result.setdefault(str(row["id_pessoa"]), {})[str(row["ponto"])] = (
            float(row["x_px"]), float(row["y_px"])
        )
    return result


def _fit_canvas(image: np.ndarray, size: int = 640):
    h, w = image.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros((size, size, 3), np.uint8)
    x, y = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    transform = np.array([[scale, 0, x], [0, scale, y]], np.float32)
    return canvas, transform


def _transform_points(points, matrix):
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, matrix).reshape(-1, 2)


def create_matching_overlays(manifest: pd.DataFrame, ranking: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Cria sobreposicoes exploratorias alinhadas por olhos/orbitas e mento."""
    if ranking.empty:
        return ranking
    face_points = pd.read_csv(LAB_ROOT / "resultados/faces/pontos_anatomicos_faces.csv")
    skull_points = pd.read_csv(LAB_ROOT / "resultados/cranios/pontos_geometrico_anatomicos.csv")
    fp, sp = _points_by_id(face_points), _points_by_id(skull_points)
    meta = manifest.copy(); meta["id_pessoa"] = meta["id_pessoa"].astype(str)
    meta = meta.set_index("id_pessoa")
    output = LAB_ROOT / "resultados/matching/sobreposicoes"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, rank in ranking[ranking["rank"] <= top_k].iterrows():
        query, candidate = str(rank["query_id"]), str(rank["candidate_id"])
        row = rank.to_dict(); row["overlay_status"] = "INDISPONIVEL"; row["overlay_path"] = ""
        if query not in meta.index or candidate not in meta.index:
            rows.append(row); continue
        face = read_image(abs_repo(meta.loc[query, "imagem_face"]))
        skull = read_image(abs_repo(meta.loc[candidate, "imagem_cf"]))
        if face is None or skull is None:
            rows.append(row); continue
        skull_canvas, skull_to_canvas = _fit_canvas(skull)
        src, dst = [], []
        for face_name, skull_name in zip(FACE_ANCHORS, SKULL_ANCHORS):
            if face_name in fp.get(query, {}) and skull_name in sp.get(candidate, {}):
                src.append(fp[query][face_name]); dst.append(sp[candidate][skull_name])
        status = "ALINHADO_3_PONTOS" if len(src) == 3 else "APROXIMADO"
        if len(src) >= 2:
            dst_canvas = _transform_points(dst, skull_to_canvas)
            face_to_canvas, _ = cv2.estimateAffinePartial2D(
                np.asarray(src, np.float32), dst_canvas, method=cv2.LMEDS
            )
        else:
            face_canvas, face_to_canvas = _fit_canvas(face)
        if face_to_canvas is None:
            face_canvas, face_to_canvas = _fit_canvas(face); status = "APROXIMADO"
        else:
            face_canvas = cv2.warpAffine(face, face_to_canvas, (640, 640))
        all_face = list(fp.get(query, {}).values())
        mask = np.zeros(face.shape[:2], np.uint8)
        if len(all_face) >= 3:
            hull = cv2.convexHull(np.asarray(all_face, np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.dilate(mask, np.ones((31, 31), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (41, 41), 0)
            warped_mask = cv2.warpAffine(mask, face_to_canvas, (640, 640)).astype(np.float32) / 255
        else:
            warped_mask = np.full((640, 640), .38, np.float32)
        alpha = (warped_mask * .42)[..., None]
        overlay = (skull_canvas * (1 - alpha) + face_canvas * alpha).astype(np.uint8)
        for p in _transform_points(src, face_to_canvas) if src else []:
            cv2.circle(overlay, tuple(np.round(p).astype(int)), 6, (0, 255, 255), -1)
        cv2.putText(overlay, status.replace("_", " "), (12, 625), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 255, 255), 1, cv2.LINE_AA)
        target = output / query / f"{int(rank['rank']):02d}_{candidate}.jpg"
        write_image(target, overlay)
        row["overlay_status"] = status
        row["overlay_path"] = str(target.relative_to(LAB_ROOT))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(LAB_ROOT / "resultados/matching/ranking_visual.csv", index=False)
    return result
