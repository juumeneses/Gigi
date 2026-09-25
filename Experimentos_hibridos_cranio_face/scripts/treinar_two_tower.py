#!/usr/bin/env python3
"""Treino opcional do baseline two-tower; nao e executado automaticamente."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
sys.path.insert(0, str(LAB))

from src.common import abs_repo, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args(); cfg = load_config(args.config)
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    data = pd.read_csv(LAB / "dataset/manifestos/dataset_pareado_matching.csv")
    pairs = pd.read_csv(LAB / "dataset/manifestos/pares_matching.csv")
    features = [c for c in data.columns if c.startswith("face_") and pd.api.types.is_numeric_dtype(data[c])]
    if not features:
        raise SystemExit("Nenhuma feature facial numerica encontrada.")
    by_id = data.set_index(data["id_pessoa"].astype(str), drop=False)
    size = int(cfg["matching"]["image_size"]); emb = int(cfg["matching"]["embedding_dim"])

    base = tf.keras.applications.VGG16(include_top=False, weights=None if args.weights == "none" else "imagenet", input_shape=(size, size, 3))
    base.trainable = False
    skull_in = layers.Input((size, size, 3), name="cranio_cf"); x = tf.keras.applications.vgg16.preprocess_input(skull_in); x = base(x); x = layers.GlobalAveragePooling2D()(x); x = layers.Dense(emb)(x); skull_emb = layers.UnitNormalization(axis=-1, name="embedding_cranio")(x)
    face_in = layers.Input((len(features),), name="features_face"); y = layers.Dense(512, activation="relu")(face_in); y = layers.Dense(emb)(y); face_emb = layers.UnitNormalization(axis=-1, name="embedding_face")(y)
    sim = layers.Dot(axes=-1, normalize=True)([skull_emb, face_emb]); out = layers.Dense(1, activation="sigmoid", name="prob_match")(layers.Reshape((1,))(sim)); model = Model([skull_in, face_in], out); model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

    def gen(split):
        for _, p in pairs[pairs["split"] == split].iterrows():
            fr, cr = by_id.loc[str(p["face_id"])], by_id.loc[str(p["cranio_id"])]
            image = tf.io.read_file(str(abs_repo(cr["imagem_cf"]))); image = tf.io.decode_image(image, channels=3, expand_animations=False); image = tf.image.resize(image, (size, size))
            fv = np.nan_to_num(fr[features].to_numpy(float), nan=0).astype("float32")
            yield {"cranio_cf": image, "features_face": fv}, np.float32(p["label"])
    sig = ({"cranio_cf": tf.TensorSpec((size,size,3),tf.float32), "features_face": tf.TensorSpec((len(features),),tf.float32)}, tf.TensorSpec((),tf.float32))
    batch = int(cfg["matching"]["batch_size"]); train = tf.data.Dataset.from_generator(lambda: gen("treino"), output_signature=sig).shuffle(1024).batch(batch).prefetch(tf.data.AUTOTUNE); val = tf.data.Dataset.from_generator(lambda: gen("validacao"), output_signature=sig).batch(batch)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=int(cfg["matching"]["early_stopping_patience"]), restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(LAB/"modelos/two_tower_global.keras", monitor="val_auc", mode="max", save_best_only=True)]
    history=model.fit(train, validation_data=val, epochs=args.epochs or int(cfg["matching"]["epochs"]), callbacks=callbacks)
    pd.DataFrame(history.history).to_csv(LAB/"resultados/matching/historico_two_tower.csv", index=False)


if __name__ == "__main__": main()
