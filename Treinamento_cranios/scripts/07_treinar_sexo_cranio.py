# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common_cranios import *


EPOCH_PROGRESS_PREFIX = "__EPOCH_PROGRESS__"
EPOCH_PROGRESS_CURRENT_PREFIX = "__EPOCH_PROGRESS_CURRENT__"
EPOCH_PROGRESS_DONE_PREFIX = "__EPOCH_PROGRESS_DONE__"


def stop_flag_path(root: Path) -> Path:
    return sexo_dir(root) / "parar_treino_sexo_cranio.flag"


def stop_requested(root: Path) -> bool:
    return stop_flag_path(root).exists()


def _metric_text(logs: dict | None) -> str:
    if not logs:
        return ""
    names = ["loss", "accuracy", "val_loss", "val_accuracy"]
    parts = []
    for name in names:
        if name in logs:
            try:
                parts.append(f"{name}={float(logs[name]):.4f}")
            except Exception:
                parts.append(f"{name}={logs[name]}")
    return " - " + " · ".join(parts) if parts else ""


def epoch_progress_callback(tf, total_epochs: int, root: Path, steps_total: int | None = None):
    class EpochProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_started_at = 0.0
            self.last_batch_update = 0.0

        def _steps_total(self):
            if steps_total:
                return int(steps_total)
            try:
                return int(self.params.get("steps") or 0)
            except Exception:
                return 0

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_number = epoch + 1
            self.epoch_started_at = time.time()
            self.last_batch_update = 0.0
            steps = self._steps_total()
            detail = f"0.0% (0/{steps} batches)" if steps else "0.0%"
            print(f"{EPOCH_PROGRESS_CURRENT_PREFIX} Época {epoch + 1}/{total_epochs} em andamento - {detail}", flush=True)

        def on_train_batch_end(self, batch, logs=None):
            if stop_requested(root):
                print(f"{EPOCH_PROGRESS_DONE_PREFIX} Parada solicitada; encerrando treino após o batch atual", flush=True)
                self.model.stop_training = True
                return
            now = time.time()
            if now - self.last_batch_update < 0.8:
                return
            self.last_batch_update = now
            steps = self._steps_total()
            done = int(batch) + 1
            pct = (done / steps * 100.0) if steps else 0.0
            # Keras does not expose the current epoch in params on every backend.
            # The panel mainly needs a stable line that updates in place.
            epoch_txt = f"{getattr(self, '_epoch_number', '?')}/{total_epochs}"
            if steps:
                print(f"{EPOCH_PROGRESS_CURRENT_PREFIX} Época {epoch_txt} em andamento - {pct:.1f}% ({done}/{steps} batches)", flush=True)
            else:
                print(f"{EPOCH_PROGRESS_CURRENT_PREFIX} Época {epoch_txt} em andamento - batch {done}", flush=True)

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.time() - self.epoch_started_at if self.epoch_started_at else 0.0
            print(f"{EPOCH_PROGRESS_DONE_PREFIX} Época {epoch + 1}/{total_epochs} concluída em {elapsed:.1f}s{_metric_text(logs)}", flush=True)
            if stop_requested(root):
                print(f"{EPOCH_PROGRESS_DONE_PREFIX} Treino interrompido pelo usuário", flush=True)
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            print(f"{EPOCH_PROGRESS_DONE_PREFIX} Treino finalizado{_metric_text(logs)}", flush=True)

    return EpochProgressCallback()


def runs_dir(root: Path) -> Path:
    p = resultados_root(root) / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_run_history(root: Path, run_id: str, tipo: str, params: dict, metrics: dict, comparacao: pd.DataFrame | None = None) -> None:
    row = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "tipo": tipo,
        "parametros_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
    }
    row.update(metrics)
    if comparacao is not None and not comparacao.empty and "resultado_genero" in comparacao.columns:
        total = max(len(comparacao), 1)
        res = comparacao["resultado_genero"].astype(str).str.upper()
        row["comparados"] = len(comparacao)
        row["acertos"] = int(res.eq("ACERTO").sum())
        row["divergencias"] = int(res.eq("DIVERGENCIA").sum())
        row["indefinidos"] = int(res.eq("INDEFINIDO").sum())
        row["acertos_pct"] = round(row["acertos"] / total * 100, 2)
        row["divergencias_pct"] = round(row["divergencias"] / total * 100, 2)
        row["indefinidos_pct"] = round(row["indefinidos"] / total * 100, 2)
    path = runs_dir(root) / "historico_treinos_cranios.csv"
    old = read_csv_safe(path)
    safe_to_csv(pd.concat([old, pd.DataFrame([row])], ignore_index=True), path)


def keras_history_summary(history) -> dict:
    hist = getattr(history, "history", {}) or {}

    def last_value(name: str):
        values = hist.get(name, [])
        if not values:
            return np.nan
        return float(values[-1])

    def best_min(name: str):
        values = hist.get(name, [])
        return float(np.nanmin(values)) if values else np.nan

    def best_max(name: str):
        values = hist.get(name, [])
        return float(np.nanmax(values)) if values else np.nan

    out = {
        "train_loss": round(last_value("loss"), 6),
        "train_accuracy": round(last_value("accuracy"), 6),
        "val_loss": round(last_value("val_loss"), 6),
        "val_accuracy": round(last_value("val_accuracy"), 6),
        "best_train_loss": round(best_min("loss"), 6),
        "best_train_accuracy": round(best_max("accuracy"), 6),
        "best_val_loss": round(best_min("val_loss"), 6),
        "best_val_accuracy": round(best_max("val_accuracy"), 6),
        "train_accuracy_pct": round(best_max("accuracy") * 100, 2) if np.isfinite(best_max("accuracy")) else np.nan,
        "val_accuracy_pct": round(best_max("val_accuracy") * 100, 2) if np.isfinite(best_max("val_accuracy")) else np.nan,
    }
    return out


def fallback_predictions(root: Path, reason: str) -> None:
    dataset = read_csv_safe(sexo_dir(root) / "dataset_treino_sexo_cranio.csv")
    rows = []
    if not dataset.empty:
        for _, r in dataset.drop_duplicates(["ano", "id_pessoa"]).iterrows():
            rows.append({
                "ano": r.get("ano"),
                "id_pessoa": r.get("id_pessoa"),
                "id_imagem_cranio": r.get("id_imagem_cranio", r.get("id_imagem", "")),
                "prob_masculino_cranio": 0.5,
                "prob_feminino_cranio": 0.5,
                "classe_cranio": "INDEFINIDO",
                "confianca_cranio": "INDEFINIDO",
                "modelo_sexo_cranio": "nao_treinado",
                "motivo": reason,
            })
    safe_to_csv(pd.DataFrame(rows), sexo_dir(root) / "predicoes_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame([{"modelo": "nao_treinado", "erro": reason}]), sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")


def acquire_lock(root: Path) -> Path | None:
    lock = sexo_dir(root) / "treino_sexo_cranio.lock"
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        pid_text = lock.read_text(encoding="utf-8", errors="ignore").strip()
        pid = None
        try:
            pid = int(pid_text)
        except Exception:
            pass

        pid_alive = False
        if pid:
            try:
                os.kill(pid, 0)
                pid_alive = True
            except OSError:
                pid_alive = False

        if pid_alive and age < 60 * 60:
            print(
                f"[AVISO] treino ja esta em execucao ha {age/60:.1f} min "
                f"(pid={pid}): {lock}",
                flush=True,
            )
            print("[AVISO] se estiver travado, use o botao 'Parar treino' no painel e tente novamente.", flush=True)
            return None

        if pid_alive:
            print(
                f"[AVISO] lock antigo encontrado ha {age/60:.1f} min com pid ainda ativo ({pid}); "
                "substituindo por novo treino.",
                flush=True,
            )
        else:
            print(
                f"[AVISO] lock antigo/stale removido; nenhum processo ativo foi encontrado "
                f"(pid={pid_text or 'vazio'}, idade={age/60:.1f} min).",
                flush=True,
            )
        lock.unlink()
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def load_training_rows(root: Path):
    dataset = read_csv_safe(sexo_dir(root) / "dataset_treino_sexo_cranio.csv")
    sil = read_csv_safe(sexo_dir(root) / "silhuetas_sexo_cranio.csv")
    if dataset.empty or sil.empty:
        return pd.DataFrame(), "dataset_treino_ou_silhuetas_vazio"
    if len(dataset) < 20 or dataset["genero_final"].astype(str).str.upper().nunique() < 2:
        return pd.DataFrame(), "dados_insuficientes_para_treino"
    for frame in [dataset, sil]:
        for c in ["ano", "id_pessoa"]:
            if c in frame.columns:
                frame[c] = frame[c].astype(str)
    labels = dataset[["ano", "id_pessoa", "genero_final"]].drop_duplicates()
    sil_ok = sil[sil["status"].astype(str).str.upper().isin(["OK", "JA_EXISTIA"])].copy()
    rows = sil_ok.merge(labels, on=["ano", "id_pessoa"], how="inner")
    if rows.empty:
        return pd.DataFrame(), "silhuetas_sem_rotulo_das_faces"
    return rows, ""


def save_comparison(root: Path, people: pd.DataFrame, pred: pd.DataFrame, normalization_kind: str = NORMALIZATION_ORIGINAL) -> pd.DataFrame:
    comp = people[["ano", "id_pessoa", "genero_final"]].drop_duplicates().merge(pred, on=["ano", "id_pessoa"], how="left")
    comp = comp.rename(columns={"genero_final": "genero_definido_face"})
    comp["normalizacao"] = normalization_kind
    comp["genero_previsto_cranio"] = comp.get("classe_cranio", pd.Series("", index=comp.index)).astype(str).str.upper().map(
        lambda x: "MASCULINO" if "MASCULINO" in x else ("FEMININO" if "FEMININO" in x else "INDEFINIDO")
    )
    comp["resultado_genero"] = np.where(
        comp["genero_previsto_cranio"].eq("INDEFINIDO"),
        "INDEFINIDO",
        np.where(comp["genero_definido_face"].astype(str).str.upper().eq(comp["genero_previsto_cranio"]), "ACERTO", "DIVERGENCIA"),
    )
    if "prob_masculino_cranio" in comp.columns:
        comp["prob_masculino_cranio_pct"] = (comp["prob_masculino_cranio"].astype(float) * 100).round(2)
    if "prob_feminino_cranio" in comp.columns:
        comp["prob_feminino_cranio_pct"] = (comp["prob_feminino_cranio"].astype(float) * 100).round(2)
    cols = [
        "ano",
        "id_pessoa",
        "normalizacao",
        "genero_definido_face",
        "genero_previsto_cranio",
        "resultado_genero",
        "classe_cranio",
        "confianca_cranio",
        "threshold_masculino",
        "estrategia_voto",
        "prob_masculino_cranio_pct",
        "prob_feminino_cranio_pct",
        "prob_masculino_cranio",
        "prob_feminino_cranio",
        "modelo_sexo_cranio",
    ]
    comp = comp[[c for c in cols if c in comp.columns]].copy()
    safe_to_csv(comp, sexo_dir(root) / "comparacao_genero_face_cranio.csv")
    return comp


def binary_class_weights(y: np.ndarray) -> dict[int, float]:
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2).astype(float)
    total = float(counts.sum())
    weights = {}
    for cls, count in enumerate(counts):
        weights[cls] = float(total / (2.0 * count)) if count > 0 else 1.0
    return weights


def class_from_masculino_threshold(prob_m: float, threshold: float) -> tuple[str, str]:
    pm = float(prob_m)
    pf = 1.0 - pm
    conf = confidence_label(max(pm, pf))
    return ("MASCULINO_PROVAVEL", conf) if pm >= float(threshold) else ("FEMININO_PROVAVEL", conf)


def split_people_train_val_test(people: pd.DataFrame, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    people = people.copy().reset_index(drop=True)
    people_y = people["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    train_val, test = train_test_split(
        people,
        test_size=0.25,
        random_state=random_state,
        stratify=people_y if people_y.value_counts().min() >= 2 else None,
    )
    train_val_y = train_val["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    train, val = train_test_split(
        train_val,
        test_size=0.2,
        random_state=random_state,
        stratify=train_val_y if train_val_y.value_counts().min() >= 2 else None,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def key_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, df[["ano", "id_pessoa"]].astype(str).values.tolist()))


def mask_by_people(meta_df: pd.DataFrame, keys: set[tuple[str, str]]) -> np.ndarray:
    return meta_df.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in keys, axis=1).to_numpy()


def aggregate_multiview_predictions(
    pv: pd.DataFrame,
    model_name: str,
    threshold: float,
    strategy: str,
) -> pd.DataFrame:
    pred_rows = []
    for (ano, pid), g0 in pv.groupby(["ano", "id_pessoa"]):
        g = g0.copy()
        if strategy == "laterais":
            side_g = g[g["lado"].astype(str).str.upper().isin(["CE", "CD"])].copy()
            if not side_g.empty:
                g = side_g
            weights = np.ones(len(g), dtype=float)
        elif strategy == "ponderada":
            weights = g["lado"].astype(str).str.upper().map({"CF": 0.4, "CE": 1.0, "CD": 1.0}).fillna(1.0).astype(float).to_numpy()
        else:
            weights = np.ones(len(g), dtype=float)
        probs = g["prob_masculino_vista"].astype(float).to_numpy()
        pm = float(np.average(probs, weights=weights)) if len(probs) else 0.5
        cl, conf = class_from_masculino_threshold(pm, threshold)
        pred_rows.append({
            "ano": ano,
            "id_pessoa": pid,
            "id_imagem_cranio": "",
            "prob_masculino_cranio": round(pm, 6),
            "prob_feminino_cranio": round(float(1.0 - pm), 6),
            "threshold_masculino": round(float(threshold), 4),
            "estrategia_voto": strategy,
            "classe_cranio": cl,
            "confianca_cranio": conf,
            "modelo_sexo_cranio": model_name,
        })
    return pd.DataFrame(pred_rows)


def evaluate_people_predictions(people: pd.DataFrame, pred: pd.DataFrame, keys: set[tuple[str, str]]) -> tuple[dict, pd.DataFrame]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    eval_df = pred[pred.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in keys, axis=1)].merge(
        people.rename(columns={"genero_final": "genero_real"}),
        on=["ano", "id_pessoa"],
        how="left",
    )
    y_true = eval_df["genero_real"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    y_pred = eval_df["classe_cranio"].astype(str).str.upper().map(lambda x: 1 if "MASCULINO" in x else 0).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "f1_masculino": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision_masculino": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_masculino": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy_pct": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "balanced_accuracy_pct": round(float(balanced_accuracy_score(y_true, y_pred)) * 100, 2),
        "f1_pct": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "precision_pct": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "recall_pct": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["real_FEMININO", "real_MASCULINO"], columns=["pred_FEMININO", "pred_MASCULINO"]).reset_index().rename(columns={"index": "classe_real"})
    return metrics, cm_df


def grid_search_threshold_strategy(people: pd.DataFrame, pv: pd.DataFrame, model_name: str, val_keys: set[tuple[str, str]]) -> tuple[dict, pd.DataFrame]:
    rows = []
    thresholds = np.round(np.arange(0.30, 0.5001, 0.05), 2)
    for strategy in ["laterais", "ponderada"]:
        for threshold in thresholds:
            pred = aggregate_multiview_predictions(pv, model_name, float(threshold), strategy)
            metrics, _ = evaluate_people_predictions(people, pred, val_keys)
            rows.append({
                "estrategia_voto": strategy,
                "threshold_masculino": float(threshold),
                **metrics,
            })
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(["balanced_accuracy", "f1_masculino", "recall_masculino", "accuracy"], ascending=False).reset_index(drop=True)
    return grid.iloc[0].to_dict(), grid


def image_features(path: Path, side: str, size: int = 64) -> np.ndarray | None:
    img = read_image(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    side_vec = np.array([side == "CF", side == "CE", side == "CD"], dtype=np.float32)
    return np.concatenate([small.reshape(-1), side_vec])


def train_sklearn(root: Path, min_samples: int, normalization_kind: str = NORMALIZATION_ORIGINAL) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows, reason = load_training_rows(root)
    if rows.empty:
        fallback_predictions(root, reason)
        print(f"[AVISO] sexo cranio nao treinado: {reason}", flush=True)
        return

    print(f"[INFO] preparando features das silhuetas: {len(rows)} vistas", flush=True)
    feats, y, meta = [], [], []
    for _, r in rows.iterrows():
        feat = image_features(Path(r["silhueta"]), str(r.get("lado", "")))
        if feat is None:
            continue
        feats.append(feat)
        y.append(1 if str(r["genero_final"]).upper() == "MASCULINO" else 0)
        meta.append(r)
    if len(feats) < min_samples or len(set(y)) < 2:
        fallback_predictions(root, "silhuetas_validas_insuficientes")
        print("[AVISO] sexo cranio nao treinado: silhuetas validas insuficientes", flush=True)
        return

    X = np.vstack(feats)
    yarr = np.asarray(y, dtype=int)
    meta_df = pd.DataFrame(meta).reset_index(drop=True)
    people = meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates().reset_index(drop=True)
    people_y = people["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    train_people, test_people = train_test_split(
        people,
        test_size=0.25,
        random_state=42,
        stratify=people_y if people_y.value_counts().min() >= 2 else None,
    )
    train_keys = set(map(tuple, train_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    is_train = meta_df.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in train_keys, axis=1).to_numpy()

    print(f"[INFO] treinando modelo rapido: {int(is_train.sum())} vistas treino, {int((~is_train).sum())} vistas teste", flush=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(X[is_train], yarr[is_train])
    prob = model.predict_proba(X)[:, 1]
    joblib.dump(model, sexo_dir(root) / "modelo_sexo_cranio_sklearn.pkl")

    prob_rows = []
    for r, pm in zip(meta, prob):
        prob_rows.append({
            "ano": r["ano"],
            "id_pessoa": r["id_pessoa"],
            "lado": r["lado"],
            "prob_masculino_vista": round(float(pm), 6),
            "prob_feminino_vista": round(float(1 - pm), 6),
        })
    pv = pd.DataFrame(prob_rows)
    pred_rows = []
    for (ano, pid), g in pv.groupby(["ano", "id_pessoa"]):
        pm = float(g["prob_masculino_vista"].mean())
        pf = 1.0 - pm
        cl, conf = class_from_probs(pm, pf)
        pred_rows.append({
            "ano": ano,
            "id_pessoa": pid,
            "id_imagem_cranio": "",
            "prob_masculino_cranio": round(pm, 6),
            "prob_feminino_cranio": round(pf, 6),
            "classe_cranio": cl,
            "confianca_cranio": conf,
            "modelo_sexo_cranio": "sklearn_logistic_silhueta_64_media_vistas",
        })
    pred = pd.DataFrame(pred_rows)

    test_keys = set(map(tuple, test_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    eval_df = pred[pred.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in test_keys, axis=1)].merge(
        people.rename(columns={"genero_final": "genero_real"}),
        on=["ano", "id_pessoa"],
        how="left",
    )
    y_true = eval_df["genero_real"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    y_pred = eval_df["prob_masculino_cranio"].astype(float).ge(0.5).astype(int)
    run_id = f"cranio_genero_sklearn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = [{
        "run_id": run_id,
        "normalizacao": normalization_kind,
        "modelo": "sklearn_logistic_silhueta_64_media_vistas",
        "n_individuos": int(len(people)),
        "n_silhuetas": int(len(X)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_masculino": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    safe_to_csv(pd.DataFrame(metrics), sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(cm, index=["real_FEMININO", "real_MASCULINO"], columns=["pred_FEMININO", "pred_MASCULINO"]).reset_index().rename(columns={"index": "classe_real"}), sexo_dir(root) / "matriz_confusao_sexo_cranio.csv")
    safe_to_csv(pv, sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv")
    safe_to_csv(pred, sexo_dir(root) / "predicoes_sexo_cranio.csv")
    comp = save_comparison(root, people, pred, normalization_kind)
    append_run_history(root, run_id, "genero_cranio", {"backend": "sklearn", "min_samples": min_samples, "image_size": 64, "normalizacao": normalization_kind}, metrics[0], comp)
    print("[OK] modelo sexo cranio rapido treinado", flush=True)


def train_vgg16(root: Path, min_samples: int, epochs: int, image_size: int, batch_size: int, weights: str, normalization_kind: str = NORMALIZATION_ORIGINAL) -> None:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    rows, reason = load_training_rows(root)
    if rows.empty:
        fallback_predictions(root, reason)
        print(f"[AVISO] sexo cranio nao treinado: {reason}", flush=True)
        return

    print(f"[INFO] preparando imagens para VGG16: {len(rows)} vistas, size={image_size}", flush=True)
    imgs, y, meta = [], [], []
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        img = read_image(Path(r["silhueta"]))
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)
        imgs.append(img)
        y.append(1 if str(r["genero_final"]).upper() == "MASCULINO" else 0)
        meta.append(r)
        if i % 250 == 0:
            print(f"[INFO] imagens carregadas: {i}/{len(rows)}", flush=True)

    if len(imgs) < min_samples or len(set(y)) < 2:
        fallback_predictions(root, "silhuetas_validas_insuficientes")
        print("[AVISO] sexo cranio nao treinado: silhuetas validas insuficientes", flush=True)
        return

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    except Exception as e:
        fallback_predictions(root, f"tensorflow_indisponivel: {e}")
        print(f"[AVISO] TensorFlow indisponivel: {e}", flush=True)
        return

    X = np.stack(imgs).astype(np.float32)
    yarr = np.asarray(y, dtype=np.float32)
    meta_df = pd.DataFrame(meta).reset_index(drop=True)
    people = meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates().reset_index(drop=True)
    people_y = people["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    train_people, test_people = train_test_split(
        people,
        test_size=0.25,
        random_state=42,
        stratify=people_y if people_y.value_counts().min() >= 2 else None,
    )
    train_keys = set(map(tuple, train_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    is_train = meta_df.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in train_keys, axis=1).to_numpy()

    cache_name = f"embeddings_vgg16_{weights}_{image_size}.npz"
    cache_path = sexo_dir(root) / cache_name
    if cache_path.exists():
        print(f"[INFO] usando embeddings em cache: {cache_path.name}", flush=True)
        data = np.load(cache_path)
        emb = data["emb"]
    else:
        print(f"[INFO] carregando VGG16 weights={weights}", flush=True)
        try:
            base = VGG16(weights=None if weights == "none" else "imagenet", include_top=False, input_shape=(image_size, image_size, 3))
        except Exception as e:
            if weights != "none":
                print(f"[AVISO] nao foi possivel carregar pesos imagenet ({e}); usando weights=none", flush=True)
                weights = "none"
                base = VGG16(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
            else:
                raise
        base.trainable = False
        print("[INFO] extraindo embeddings VGG16 congelados", flush=True)
        emb = base.predict(preprocess_input(X.copy()), batch_size=batch_size, verbose=0)
        emb = emb.reshape((emb.shape[0], -1)).astype(np.float32)
        np.savez_compressed(cache_path, emb=emb)
        print(f"[INFO] embeddings salvos: {cache_path}", flush=True)

    print(f"[INFO] treinando cabeca neural VGG16: epochs={epochs}", flush=True)
    tf.keras.utils.set_random_seed(42)
    clf = models.Sequential([
        layers.Input(shape=(emb.shape[1],)),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(1, activation="sigmoid"),
    ])
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    train_steps = max(1, int(np.ceil((len(emb[is_train]) * 0.8) / max(int(batch_size), 1))))
    callbacks = [
        epoch_progress_callback(tf, epochs, root, train_steps),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]
    clf.fit(
        emb[is_train],
        yarr[is_train],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    if stop_requested(root):
        print("[AVISO] treino VGG16 embeddings interrompido; modelo e métricas não foram atualizados", flush=True)
        return
    print("[INFO] calculando predições VGG16 embeddings", flush=True)
    prob = clf.predict(emb, batch_size=batch_size, verbose=0).reshape(-1)
    clf.save(sexo_dir(root) / "modelo_sexo_cranio_vgg16_head.keras")

    prob_rows = []
    for r, pm in zip(meta, prob):
        prob_rows.append({
            "ano": r["ano"],
            "id_pessoa": r["id_pessoa"],
            "lado": r["lado"],
            "prob_masculino_vista": round(float(pm), 6),
            "prob_feminino_vista": round(float(1 - pm), 6),
        })
    pv = pd.DataFrame(prob_rows)
    pred_rows = []
    for (ano, pid), g in pv.groupby(["ano", "id_pessoa"]):
        pm = float(g["prob_masculino_vista"].mean())
        pf = 1.0 - pm
        cl, conf = class_from_probs(pm, pf)
        pred_rows.append({
            "ano": ano,
            "id_pessoa": pid,
            "id_imagem_cranio": "",
            "prob_masculino_cranio": round(pm, 6),
            "prob_feminino_cranio": round(pf, 6),
            "classe_cranio": cl,
            "confianca_cranio": conf,
            "modelo_sexo_cranio": f"vgg16_{weights}_silhueta_{image_size}_head_media_vistas",
        })
    pred = pd.DataFrame(pred_rows)
    test_keys = set(map(tuple, test_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    eval_df = pred[pred.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in test_keys, axis=1)].merge(
        people.rename(columns={"genero_final": "genero_real"}),
        on=["ano", "id_pessoa"],
        how="left",
    )
    y_true = eval_df["genero_real"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    y_pred = eval_df["prob_masculino_cranio"].astype(float).ge(0.5).astype(int)
    run_id = f"cranio_genero_vgg16_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = [{
        "run_id": run_id,
        "normalizacao": normalization_kind,
        "modelo": f"vgg16_{weights}_silhueta_{image_size}_head_media_vistas",
        "n_individuos": int(len(people)),
        "n_silhuetas": int(len(X)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_masculino": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision_masculino": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_masculino": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy_pct": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "f1_pct": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "precision_pct": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "recall_pct": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
    }]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    safe_to_csv(pd.DataFrame(metrics), sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(cm, index=["real_FEMININO", "real_MASCULINO"], columns=["pred_FEMININO", "pred_MASCULINO"]).reset_index().rename(columns={"index": "classe_real"}), sexo_dir(root) / "matriz_confusao_sexo_cranio.csv")
    safe_to_csv(pv, sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv")
    safe_to_csv(pred, sexo_dir(root) / "predicoes_sexo_cranio.csv")
    comp = save_comparison(root, people, pred, normalization_kind)
    append_run_history(
        root,
        run_id,
        "genero_cranio",
        {"backend": "vgg16", "epochs": epochs, "image_size": image_size, "batch_size": batch_size, "weights": weights, "min_samples": min_samples, "normalizacao": normalization_kind},
        metrics[0],
        comp,
    )
    print("[OK] modelo sexo cranio VGG16 treinado", flush=True)


def load_article_arrays(rows: pd.DataFrame, image_size: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    imgs, labels, meta = [], [], []
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        img = read_image(Path(r["silhueta"]))
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        imgs.append(img.astype(np.float32))
        labels.append(1 if str(r["genero_final"]).upper() == "MASCULINO" else 0)
        meta.append(r.to_dict())
        if i % 300 == 0:
            print(f"[INFO] silhuetas carregadas: {i}/{len(rows)}", flush=True)
    return np.stack(imgs).astype(np.float32), np.asarray(labels, dtype=np.float32), pd.DataFrame(meta)


def augment_article_images(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Augmentation do artigo: original, flip horizontal, rotacao -5 e +5 graus."""
    out_x, out_y = [], []
    h, w = X.shape[1:3]
    center = (w / 2.0, h / 2.0)
    for img, label in zip(X, y):
        out_x.append(img)
        out_y.append(label)
        out_x.append(cv2.flip(img, 1))
        out_y.append(label)
        for angle in (-5, 5):
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rot = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            out_x.append(rot)
            out_y.append(label)
    return np.stack(out_x).astype(np.float32), np.asarray(out_y, dtype=np.float32)


def build_article_model(image_size: int, weights: str, learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.vgg16 import VGG16

    try:
        base = VGG16(weights=None if weights == "none" else "imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    except Exception as e:
        if weights != "none":
            print(f"[AVISO] pesos imagenet indisponiveis ({e}); usando weights=none", flush=True)
            base = VGG16(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
        else:
            raise
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x)),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def imagenet_vgg16_notop_weights() -> str:
    """Return a local VGG16 ImageNet weights file when Keras cache lookup cannot use it."""
    expected_md5 = "6d6bbae143d832006294945121d1f1fc"
    candidates = [
        Path.home() / ".keras" / "models" / "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        Path("/Users/giovanni/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    ]
    try:
        import hashlib

        for path in candidates:
            if path.exists():
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest == expected_md5:
                    return str(path)
    except Exception:
        pass
    return "imagenet"


def build_vgg16_finetune_model(image_size: int, weights: str, learning_rate: float):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.vgg16 import VGG16

    resolved_weights = None if weights == "none" else imagenet_vgg16_notop_weights()
    if resolved_weights not in {None, "imagenet"}:
        print(f"[INFO] usando pesos ImageNet VGG16 locais: {resolved_weights}", flush=True)
    try:
        base = VGG16(weights=resolved_weights, include_top=False, input_shape=(image_size, image_size, 3))
    except Exception as e:
        if weights != "none":
            print(f"[AVISO] pesos imagenet indisponiveis ({e}); usando weights=none", flush=True)
            base = VGG16(weights=None, include_top=False, input_shape=(image_size, image_size, 3))
        else:
            raise
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5_")
    trainable = [layer.name for layer in base.layers if layer.trainable]
    print(f"[INFO] fine-tuning parcial VGG16: camadas treinaveis={trainable}", flush=True)
    model = models.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x)),
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model, trainable


def train_vgg16_finetune(root: Path, min_samples: int, epochs: int, image_size: int, batch_size: int, weights: str, learning_rate: float, normalization_kind: str = NORMALIZATION_ORIGINAL) -> None:
    import tensorflow as tf

    rows, reason = load_training_rows(root)
    if rows.empty:
        fallback_predictions(root, reason)
        print(f"[AVISO] sexo cranio VGG16 fine-tuning nao treinado: {reason}", flush=True)
        return
    X, yarr, meta_df = load_article_arrays(rows, image_size)
    if len(X) < min_samples or len(set(yarr.astype(int))) < 2:
        fallback_predictions(root, "silhuetas_validas_insuficientes")
        return

    people = meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates().reset_index(drop=True)
    train_people, val_people, test_people = split_people_train_val_test(people)
    train_keys = key_set(train_people)
    val_keys = key_set(val_people)
    test_keys = key_set(test_people)
    is_train = mask_by_people(meta_df, train_keys)
    is_val = mask_by_people(meta_df, val_keys)

    class_weight = binary_class_weights(yarr[is_train])
    print(
        "[INFO] VGG16 fine-tuning: "
        f"treino_individuos={len(train_people)}, validacao_individuos={len(val_people)}, teste_individuos={len(test_people)}, "
        f"class_weight={class_weight}, image_size={image_size}, lr={learning_rate}",
        flush=True,
    )

    tf.keras.utils.set_random_seed(42)
    model, trainable_layers = build_vgg16_finetune_model(image_size, weights, learning_rate)
    train_steps = max(1, int(np.ceil(len(X[is_train]) / max(int(batch_size), 1))))
    callbacks = [
        epoch_progress_callback(tf, epochs, root, train_steps),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    history = model.fit(
        X[is_train],
        yarr[is_train],
        validation_data=(X[is_val], yarr[is_val]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
    )
    if stop_requested(root):
        print("[AVISO] treino VGG16 fine-tuning interrompido; modelo e métricas não foram atualizados", flush=True)
        return

    print("[INFO] calculando predições VGG16 fine-tuning", flush=True)
    prob_m = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    run_id = f"cranio_genero_vgg16_finetune_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    model_name = f"vgg16_finetune_block5_{weights}_silhueta_{image_size}_threshold_multivista"
    out_model = sexo_dir(root) / "modelo_sexo_cranio_vgg16_finetune.keras"
    model.save(out_model)

    pv_rows = []
    for (_, r), pm in zip(meta_df.iterrows(), prob_m):
        pv_rows.append({
            "ano": str(r["ano"]),
            "id_pessoa": str(r["id_pessoa"]),
            "lado": str(r["lado"]),
            "prob_masculino_vista": round(float(pm), 6),
            "prob_feminino_vista": round(float(1 - pm), 6),
            "pred_vista": "MASCULINO" if float(pm) >= 0.5 else "FEMININO",
            "modelo_sexo_cranio": model_name,
        })
    pv = pd.DataFrame(pv_rows)

    best, grid = grid_search_threshold_strategy(people, pv, model_name, val_keys)
    best_threshold = float(best["threshold_masculino"])
    best_strategy = str(best["estrategia_voto"])
    print(
        "[INFO] melhor threshold/estratégia na validação: "
        f"estrategia={best_strategy}, threshold={best_threshold:.2f}, "
        f"balanced_accuracy={float(best['balanced_accuracy_pct']):.2f}%, f1_masculino={float(best['f1_pct']):.2f}%",
        flush=True,
    )

    pred = aggregate_multiview_predictions(pv, model_name, best_threshold, best_strategy)
    metrics, cm_df = evaluate_people_predictions(people, pred, test_keys)
    test_grid_rows = []
    for strategy in ["laterais", "ponderada"]:
        for threshold in np.round(np.arange(0.30, 0.5001, 0.05), 2):
            ptest = aggregate_multiview_predictions(pv, model_name, float(threshold), strategy)
            mtest, _ = evaluate_people_predictions(people, ptest, test_keys)
            test_grid_rows.append({"estrategia_voto": strategy, "threshold_masculino": float(threshold), **mtest})
    test_grid = pd.DataFrame(test_grid_rows).sort_values(["balanced_accuracy", "f1_masculino", "recall_masculino"], ascending=False)

    metrics.update({
        "run_id": run_id,
        "normalizacao": normalization_kind,
        "modelo": model_name,
        "n_individuos": int(len(people)),
        "n_silhuetas": int(len(X)),
        "n_treino_individuos": int(len(train_people)),
        "n_validacao_individuos": int(len(val_people)),
        "n_teste_individuos": int(len(test_people)),
        "threshold_masculino": round(best_threshold, 4),
        "estrategia_voto": best_strategy,
        "class_weight_feminino": round(float(class_weight.get(0, 1.0)), 6),
        "class_weight_masculino": round(float(class_weight.get(1, 1.0)), 6),
        "learning_rate": float(learning_rate),
        "fine_tuning_layers": ",".join(trainable_layers),
        "melhor_val_accuracy_pct": round(float(max(history.history.get("val_accuracy", [0]))) * 100, 2),
        **keras_history_summary(history),
    })

    view_metrics = []
    by_view = pv.merge(meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates(), on=["ano", "id_pessoa"], how="left")
    by_view["split"] = by_view.apply(lambda r: "teste" if (str(r["ano"]), str(r["id_pessoa"])) in test_keys else ("validacao" if (str(r["ano"]), str(r["id_pessoa"])) in val_keys else "treino"), axis=1)
    test_view = by_view[by_view["split"].eq("teste")].copy()
    for side, g in test_view.groupby("lado"):
        side_pred = g[["ano", "id_pessoa", "lado", "prob_masculino_vista", "prob_feminino_vista", "modelo_sexo_cranio"]].copy()
        side_pred["classe_cranio"] = side_pred["prob_masculino_vista"].astype(float).map(lambda pm: class_from_masculino_threshold(pm, best_threshold)[0])
        side_pred["confianca_cranio"] = side_pred["prob_masculino_vista"].astype(float).map(lambda pm: class_from_masculino_threshold(pm, best_threshold)[1])
        side_pred = side_pred.rename(columns={"prob_masculino_vista": "prob_masculino_cranio", "prob_feminino_vista": "prob_feminino_cranio"})
        side_metrics, _ = evaluate_people_predictions(people, side_pred, test_keys)
        view_metrics.append({
            "run_id": run_id,
            "modelo": model_name,
            "lado": side,
            "n_vistas_teste": int(len(g)),
            **side_metrics,
        })

    safe_to_csv(pd.DataFrame([metrics]), sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(view_metrics), sexo_dir(root) / "metricas_modelo_sexo_cranio_por_vista.csv")
    safe_to_csv(cm_df, sexo_dir(root) / "matriz_confusao_sexo_cranio.csv")
    safe_to_csv(grid, sexo_dir(root) / "grid_threshold_validacao_sexo_cranio.csv")
    safe_to_csv(test_grid, sexo_dir(root) / "grid_threshold_teste_sexo_cranio.csv")
    safe_to_csv(pv, sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv")
    safe_to_csv(pred, sexo_dir(root) / "predicoes_sexo_cranio.csv")
    comp = save_comparison(root, people, pred, normalization_kind)
    append_run_history(
        root,
        run_id,
        "genero_cranio_vgg16_finetune",
        {
            "backend": "vgg16_finetune",
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "weights": weights,
            "learning_rate": learning_rate,
            "class_weight": class_weight,
            "threshold_grid": "0.30:0.50:0.05",
            "voto_opcoes": ["laterais", "ponderada"],
            "voto_escolhido": best_strategy,
            "threshold_escolhido": best_threshold,
            "fine_tuning": "block5_*",
            "normalizacao": normalization_kind,
        },
        metrics,
        comp,
    )
    print(
        "[OK] VGG16 fine-tuning treinado: "
        f"balanced_accuracy={metrics['balanced_accuracy_pct']}%, f1_masculino={metrics['f1_pct']}%, "
        f"threshold={best_threshold:.2f}, estrategia={best_strategy}",
        flush=True,
    )


def evaluate_article_predictions(meta_df: pd.DataFrame, prob_m: np.ndarray, model_name: str, test_keys: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pv_rows = []
    for (_, r), pm in zip(meta_df.iterrows(), prob_m):
        pv_rows.append({
            "ano": str(r["ano"]),
            "id_pessoa": str(r["id_pessoa"]),
            "lado": str(r["lado"]),
            "prob_masculino_vista": round(float(pm), 6),
            "prob_feminino_vista": round(float(1 - pm), 6),
            "pred_vista": "MASCULINO" if float(pm) >= 0.5 else "FEMININO",
            "modelo_sexo_cranio": model_name,
        })
    pv = pd.DataFrame(pv_rows)
    pred_rows = []
    for (ano, pid), g in pv.groupby(["ano", "id_pessoa"]):
        pm_mean = float(g["prob_masculino_vista"].mean())
        pf_mean = 1.0 - pm_mean
        votes_m = int(g["prob_masculino_vista"].astype(float).ge(0.5).sum())
        votes_f = int(len(g) - votes_m)
        if votes_m == votes_f:
            pm = pm_mean
        else:
            pm = votes_m / max(len(g), 1)
        pf = 1.0 - pm
        cl, conf = class_from_probs(pm, pf)
        pred_rows.append({
            "ano": ano,
            "id_pessoa": pid,
            "id_imagem_cranio": "",
            "prob_masculino_cranio": round(pm_mean, 6),
            "prob_feminino_cranio": round(pf_mean, 6),
            "votos_masculino": votes_m,
            "votos_feminino": votes_f,
            "classe_cranio": cl,
            "confianca_cranio": conf,
            "modelo_sexo_cranio": model_name,
        })
    pred = pd.DataFrame(pred_rows)
    by_view = pv.merge(meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates(), on=["ano", "id_pessoa"], how="left")
    by_view["split"] = by_view.apply(lambda r: "teste" if (str(r["ano"]), str(r["id_pessoa"])) in test_keys else "treino", axis=1)
    return pv, pred, by_view


def train_article_vgg16(root: Path, min_samples: int, epochs: int, image_size: int, batch_size: int, weights: str, learning_rate: float, normalization_kind: str = NORMALIZATION_ORIGINAL) -> None:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    rows, reason = load_training_rows(root)
    if rows.empty:
        fallback_predictions(root, reason)
        print(f"[AVISO] sexo cranio artigo nao treinado: {reason}", flush=True)
        return
    if image_size < 128:
        print("[AVISO] modo artigo recomenda 256; mantendo valor informado.", flush=True)
    X, yarr, meta_df = load_article_arrays(rows, image_size)
    if len(X) < min_samples or len(set(yarr.astype(int))) < 2:
        fallback_predictions(root, "silhuetas_validas_insuficientes")
        return
    people = meta_df[["ano", "id_pessoa", "genero_final"]].drop_duplicates().reset_index(drop=True)
    people_y = people["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    train_people, test_people = train_test_split(
        people,
        test_size=0.25,
        random_state=42,
        stratify=people_y if people_y.value_counts().min() >= 2 else None,
    )
    train_keys = set(map(tuple, train_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    test_keys = set(map(tuple, test_people[["ano", "id_pessoa"]].astype(str).values.tolist()))
    is_train = meta_df.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in train_keys, axis=1).to_numpy()

    X_train_aug, y_train_aug = augment_article_images(X[is_train], yarr[is_train])
    print(f"[INFO] modo artigo VGG16: treino={len(X_train_aug)} imagens aumentadas, teste={int((~is_train).sum())} vistas originais", flush=True)
    tf.keras.utils.set_random_seed(42)
    model = build_article_model(image_size, weights, learning_rate)
    train_steps = max(1, int(np.ceil((len(X_train_aug) * 0.8) / max(int(batch_size), 1))))
    callbacks = [
        epoch_progress_callback(tf, epochs, root, train_steps),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    if stop_requested(root):
        print("[AVISO] treino VGG16 modo artigo interrompido; modelo e métricas não foram atualizados", flush=True)
        return
    print("[INFO] calculando predições VGG16 modo artigo", flush=True)
    prob = model.predict(X, batch_size=batch_size, verbose=0)[:, 1]
    run_id = f"cranio_genero_article_vgg16_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    model_name = f"article_vgg16_{weights}_silhueta_{image_size}_aug_flip_rot5_voto_3vistas"
    out_model = sexo_dir(root) / "modelo_sexo_cranio_article_vgg16.keras"
    model.save(out_model)
    pv, pred, by_view = evaluate_article_predictions(meta_df, prob, model_name, test_keys)

    eval_people = pred[pred.apply(lambda r: (str(r["ano"]), str(r["id_pessoa"])) in test_keys, axis=1)].merge(
        people.rename(columns={"genero_final": "genero_real"}), on=["ano", "id_pessoa"], how="left"
    )
    y_true = eval_people["genero_real"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
    y_pred = eval_people["classe_cranio"].astype(str).str.upper().map(lambda x: 1 if "MASCULINO" in x else 0).astype(int)
    metrics = {
        "run_id": run_id,
        "normalizacao": normalization_kind,
        "modelo": model_name,
        "n_individuos": int(len(people)),
        "n_silhuetas": int(len(X)),
        "n_treino_individuos": int(len(train_people)),
        "n_teste_individuos": int(len(test_people)),
        "n_treino_aug": int(len(X_train_aug)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_masculino": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "precision_masculino": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_masculino": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "accuracy_pct": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "f1_pct": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "precision_pct": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "recall_pct": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
        "melhor_val_accuracy_pct": round(float(max(history.history.get("val_accuracy", [0]))) * 100, 2),
        **keras_history_summary(history),
    }
    view_metrics = []
    test_view = by_view[by_view["split"].eq("teste")].copy()
    for side, g in test_view.groupby("lado"):
        yt = g["genero_final"].astype(str).str.upper().map({"FEMININO": 0, "MASCULINO": 1}).astype(int)
        yp = g["prob_masculino_vista"].astype(float).ge(0.5).astype(int)
        view_metrics.append({
            "run_id": run_id,
            "modelo": model_name,
            "lado": side,
            "n_vistas_teste": int(len(g)),
            "accuracy": round(float(accuracy_score(yt, yp)), 6),
            "accuracy_pct": round(float(accuracy_score(yt, yp)) * 100, 2),
            "f1_masculino": round(float(f1_score(yt, yp, zero_division=0)), 6),
            "f1_pct": round(float(f1_score(yt, yp, zero_division=0)) * 100, 2),
        })
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    safe_to_csv(pd.DataFrame([metrics]), sexo_dir(root) / "metricas_modelo_sexo_cranio.csv")
    safe_to_csv(pd.DataFrame(view_metrics), sexo_dir(root) / "metricas_modelo_sexo_cranio_por_vista.csv")
    safe_to_csv(pd.DataFrame(cm, index=["real_FEMININO", "real_MASCULINO"], columns=["pred_FEMININO", "pred_MASCULINO"]).reset_index().rename(columns={"index": "classe_real"}), sexo_dir(root) / "matriz_confusao_sexo_cranio.csv")
    safe_to_csv(pv, sexo_dir(root) / "predicoes_sexo_cranio_por_vista.csv")
    safe_to_csv(pred, sexo_dir(root) / "predicoes_sexo_cranio.csv")
    comp = save_comparison(root, people, pred, normalization_kind)
    append_run_history(
        root,
        run_id,
        "genero_cranio_artigo",
        {"backend": "article_vgg16", "epochs": epochs, "image_size": image_size, "batch_size": batch_size, "weights": weights, "learning_rate": learning_rate, "augmentation": "flip_horizontal_rotacao_pm5", "voto": "CF_CE_CD", "normalizacao": normalization_kind},
        metrics,
        comp,
    )
    print(f"[OK] modo artigo VGG16 treinado: {metrics['accuracy_pct']}% teste por voto", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--backend", choices=["sklearn", "vgg16", "vgg16_finetune", "article_vgg16"], default="vgg16_finetune")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--normalization-kind", default="atual", choices=[NORMALIZATION_ORIGINAL, *NORMALIZATION_KINDS])
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    if args.normalization_kind != NORMALIZATION_ORIGINAL:
        os.environ["CRANIOS_NORMALIZATION_KIND"] = args.normalization_kind
    cache_root = resultados_root(root) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    stale_stop = stop_flag_path(root)
    if stale_stop.exists():
        stale_stop.unlink()
    lock = acquire_lock(root)
    if lock is None:
        raise SystemExit(2)
    try:
        if args.backend == "article_vgg16":
            train_article_vgg16(root, args.min_samples, args.epochs, args.image_size, args.batch_size, args.weights, args.learning_rate, args.normalization_kind)
        elif args.backend == "vgg16_finetune":
            train_vgg16_finetune(root, args.min_samples, args.epochs, args.image_size, args.batch_size, args.weights, args.learning_rate, args.normalization_kind)
        elif args.backend == "vgg16":
            train_vgg16(root, args.min_samples, args.epochs, args.image_size, args.batch_size, args.weights, args.normalization_kind)
        else:
            train_sklearn(root, args.min_samples, args.normalization_kind)
    finally:
        if lock.exists():
            lock.unlink()
        if stop_flag_path(root).exists():
            stop_flag_path(root).unlink()


if __name__ == "__main__":
    main()
