#!/usr/bin/env python3
"""Model definition + training (spec 6, tensorflow-cpu).

Architecture (build() MUST match src/ocr/cnn.js LAYERS exactly):
    conv3x3(1->8) same no bias BN ReLU maxpool2   -> 16x80x8
    conv3x3(8->16) same no bias BN ReLU maxpool2  -> 8x40x16
    conv3x3(16->32) same no bias BN ReLU maxpool2 -> 4x20x32
    conv3x3(32->32) same no bias BN ReLU          -> 4x20x32
    average over height (AveragePooling2D((4,1))) -> 1x20x32
    flatten (index = x*32 + channel)              -> 640
    dropout .3, dense 128 ReLU, dropout .2
    head digits: dense 80 -> reshape [8,10] -> softmax
    head isDate: dense 1 -> sigmoid
~107,897 params. Loss = mean over 8 heads sparse CE (isDate=1 only) + BCE(isDate).
Adam, cosine decay from 2e-3, batch 128, input noise augmentation.

Usage:
    python3 train_keras.py <epochs> [--train train.npz --val-seen val_seen.npz
        --val-unseen val_unseen.npz --out .] [--lr 2e-3] [--finetune weights.h5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

EPOCHS_DEFAULT = 12


def build(tr=None):
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=(32, 160, 1))
    x = inp
    for li, (filters, pool) in enumerate([(8, True), (16, True), (32, True), (32, False)], start=1):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{li}")(x)
        x = keras.layers.BatchNormalization(name=f"bn{li}")(x)
        x = keras.layers.ReLU(name=f"relu{li}")(x)
        if pool:
            x = keras.layers.MaxPooling2D(2, name=f"pool{li}")(x)
    # x: 4x20x32 NHWC -> average over height -> 1x20x32
    x = keras.layers.AveragePooling2D((4, 1), name="avg_h")(x)
    # flatten index = x*32 + channel (Keras row-major flatten of [1,20,32] does this)
    x = keras.layers.Flatten(name="flatten")(x)  # 640
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x = keras.layers.Dropout(0.2)(x)
    digits_pre = keras.layers.Dense(80, name="digits_pre")(x)
    digits = keras.layers.Reshape((8, 10), name="digits_reshape")(digits_pre)
    digits = keras.layers.Softmax(name="digits")(digits)
    is_date_logit = keras.layers.Dense(1, name="is_date_logit")(x)
    is_date = keras.layers.Activation("sigmoid", name="is_date")(is_date_logit)
    model = keras.Model(inp, [digits, is_date])
    print(f"params: {model.count_params()}")
    return model


def standardize_batch(X: np.ndarray, NW: np.ndarray) -> np.ndarray:
    """X uint8 (N,32,160) + NW -> float32 standardised like preprocess()."""
    N = X.shape[0]
    out = np.zeros((N, 32, 160, 1), dtype=np.float32)
    for i in range(N):
        nw = int(NW[i])
        region = X[i, :, :nw].astype(np.float32)
        m, s = region.mean(), region.std()
        out[i, :, :nw, 0] = (region - m) / (s + 1e-6)
    return out


def load_npz(path: Path):
    z = np.load(path)
    return standardize_batch(z["X"], z["NW"]), z["D"].astype(np.int64), z["Y"].astype(np.float32)


def confusion_matrix(y_true_8: np.ndarray, y_pred_8: np.ndarray) -> np.ndarray:
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(y_true_8.reshape(-1), y_pred_8.reshape(-1)):
        if 0 <= t < 10:
            cm[t, p] += 1
    return cm


def evaluate(model, X, D, Y, name: str):
    import tensorflow as tf
    digits_p, date_p = model.predict(X, batch_size=256, verbose=0)
    pred_8 = digits_p.argmax(-1)
    date_hat = (date_p[:, 0] >= 0.5).astype(np.int32)
    pos = D[:, 0] >= 0
    exact = ((pred_8[pos] == D[pos]).all(axis=1)).mean() if pos.sum() else 0.0
    per_digit = ((pred_8[pos] == D[pos]).mean() if pos.sum() else 0.0)
    is_acc = (date_hat == Y).mean()
    cm = confusion_matrix(D[pos], pred_8[pos]) if pos.sum() else np.zeros((10, 10), int)
    z05 = cm[0, 5] / max(1, cm[0].sum())
    z50 = cm[5, 0] / max(1, cm[5].sum())
    print(f"[{name}] exact={exact:.4f} per-digit={per_digit:.4f} isDate={is_acc:.4f} "
          f"0->5={z05:.4f} 5->0={z50:.4f}")
    return exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", nargs="?", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--train", default="train.npz")
    ap.add_argument("--val-seen", default="val_seen.npz")
    ap.add_argument("--val-unseen", default="val_unseen.npz")
    ap.add_argument("--out", default=".")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--finetune", default=None)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, Dtr, Ytr = load_npz(Path(args.train))
    Xvs, Dvs, Yvs = load_npz(Path(args.val_seen)) if Path(args.val_seen).exists() else (None, None, None)
    Xvu, Dvu, Yvu = load_npz(Path(args.val_unseen)) if Path(args.val_unseen).exists() else (None, None, None)
    print(f"train {Xtr.shape} pos={(Ytr == 1).mean():.2f}")

    model = build()
    if args.finetune:
        model.load_weights(args.finetune)
        print(f"fine-tuning from {args.finetune}")

    steps_per_epoch = max(1, len(Xtr) // args.batch)
    lr_sched = keras.optimizers.schedules.CosineDecay(args.lr, steps_per_epoch * args.epochs)
    opt = keras.optimizers.Adam(lr_sched)

    @tf.function
    def train_step(xb, db, yb):
        noise = tf.random.normal(tf.shape(xb), stddev=0.02)
        db_safe = tf.where(db < 0, tf.zeros_like(db), db)
        with tf.GradientTape() as tape:
            digit_logits, date_p = model(xb + noise, training=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(db_safe, digit_logits)  # [B,8]
            ce = tf.reduce_mean(ce, axis=-1)  # [B]
            mask = tf.cast(yb[:, 0] == 1, tf.float32)
            digit_loss = tf.reduce_sum(ce * mask) / (tf.reduce_sum(mask) + 1e-6)
            bce = tf.keras.losses.binary_crossentropy(yb, date_p)
            loss = digit_loss + tf.reduce_mean(bce)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    best = -1.0
    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(len(Xtr))
        tot, n = 0.0, 0
        for s in range(0, len(Xtr), args.batch):
            b = idx[s:s + args.batch]
            loss = train_step(Xtr[b], Dtr[b], Ytr[b][:, None])
            tot += float(loss) * len(b)
            n += len(b)
        print(f"epoch {ep}/{args.epochs} loss={tot / max(n, 1):.4f}")
        if Xvs is not None:
            e_seen = evaluate(model, Xvs, Dvs, Yvs, "val-seen")
            e_unseen = evaluate(model, Xvu, Dvu, Yvu, "val-unseen") if Xvu is not None else 0.0
            score = e_seen + e_unseen
            if score >= best:
                best = score
                model.save_weights(str(out / "best.weights.h5"))
                print(f"  saved best ({score:.4f})")
    print("done. best.weights.h5 written.")


if __name__ == "__main__":
    args = sys.argv
    main()
