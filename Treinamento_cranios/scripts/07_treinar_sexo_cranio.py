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
        if age < 60 * 60:
            print(f"[AVISO] treino ja esta em execucao ou ficou preso ha {age/60:.1f} min: {lock}", flush=True)
            return None
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


def save_comparison(root: Path, people: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    comp = people[["ano", "id_pessoa", "genero_final"]].drop_duplicates().merge(pred, on=["ano", "id_pessoa"], how="left")
    comp = comp.rename(columns={"genero_final": "genero_definido_face"})
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
        "genero_definido_face",
        "genero_previsto_cranio",
        "resultado_genero",
        "classe_cranio",
        "confianca_cranio",
        "prob_masculino_cranio_pct",
        "prob_feminino_cranio_pct",
        "prob_masculino_cranio",
        "prob_feminino_cranio",
        "modelo_sexo_cranio",
    ]
    comp = comp[[c for c in cols if c in comp.columns]].copy()
    safe_to_csv(comp, sexo_dir(root) / "comparacao_genero_face_cranio.csv")
    return comp


def image_features(path: Path, side: str, size: int = 64) -> np.ndarray | None:
    img = read_image(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    side_vec = np.array([side == "CF", side == "CE", side == "CD"], dtype=np.float32)
    return np.concatenate([small.reshape(-1), side_vec])


def train_sklearn(root: Path, min_samples: int) -> None:
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
    comp = save_comparison(root, people, pred)
    append_run_history(root, run_id, "genero_cranio", {"backend": "sklearn", "min_samples": min_samples, "image_size": 64}, metrics[0], comp)
    print("[OK] modelo sexo cranio rapido treinado", flush=True)


def train_vgg16(root: Path, min_samples: int, epochs: int, image_size: int, batch_size: int, weights: str) -> None:
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
        emb = base.predict(preprocess_input(X.copy()), batch_size=batch_size, verbose=1)
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
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    clf.fit(
        emb[is_train],
        yarr[is_train],
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    prob = clf.predict(emb, batch_size=batch_size, verbose=1).reshape(-1)
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
    comp = save_comparison(root, people, pred)
    append_run_history(
        root,
        run_id,
        "genero_cranio",
        {"backend": "vgg16", "epochs": epochs, "image_size": image_size, "batch_size": batch_size, "weights": weights, "min_samples": min_samples},
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


def train_article_vgg16(root: Path, min_samples: int, epochs: int, image_size: int, batch_size: int, weights: str, learning_rate: float) -> None:
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
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
    )
    prob = model.predict(X, batch_size=batch_size, verbose=1)[:, 1]
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
    comp = save_comparison(root, people, pred)
    append_run_history(
        root,
        run_id,
        "genero_cranio_artigo",
        {"backend": "article_vgg16", "epochs": epochs, "image_size": image_size, "batch_size": batch_size, "weights": weights, "learning_rate": learning_rate, "augmentation": "flip_horizontal_rotacao_pm5", "voto": "CF_CE_CD"},
        metrics,
        comp,
    )
    print(f"[OK] modo artigo VGG16 treinado: {metrics['accuracy_pct']}% teste por voto", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--backend", choices=["sklearn", "vgg16", "article_vgg16"], default="sklearn")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--learning-rate", type=float, default=2e-6)
    args = ap.parse_args()
    root = as_root(args.root)
    ensure_base_dirs(root)
    lock = acquire_lock(root)
    if lock is None:
        return
    try:
        if args.backend == "article_vgg16":
            train_article_vgg16(root, args.min_samples, args.epochs, args.image_size, args.batch_size, args.weights, args.learning_rate)
        elif args.backend == "vgg16":
            train_vgg16(root, args.min_samples, args.epochs, args.image_size, args.batch_size, args.weights)
        else:
            train_sklearn(root, args.min_samples)
    finally:
        if lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
