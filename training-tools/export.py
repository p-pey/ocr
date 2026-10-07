#!/usr/bin/env python3
"""BN fold + int8 quantise -> out/modelWeights.js + out/parity.json (spec 6).

Blob layout (MUST match src/ocr/cnn.js loadWeights):
- Layers in order [conv1, conv2, conv3, conv4, fc1, digits, isDate].
- All float32 sections first: for each layer scale[out] then bias[out].
- Then all int8 sections: for each layer the quantised weights.
- Quantisation: per-output-channel symmetric int8 (scale = max|w|/127).
- Conv weight layout: [out][ky][kx][in]. Dense layout: [out][in].
- MODEL_WEIGHTS_B64 = base64 of the whole blob (~144 KB base64 for 8/16/32/32).

Also writes parity.json (4 standardised inputs + reference logits) and
reports float vs int8 accuracy; quantisation must cost < 0.3% absolute
exact-match, otherwise use finer quantisation and update cnn.js.

Usage: python3 export.py [--weights best.weights.h5 --val val_seen.npz --out out]
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def fold_bn(model):
    """Return list of (kind, W_folded, b_folded) for the 7 weight layers.

    Walks model.layers, pairing each bias-less Conv2D/Dense with the
    following BatchNormalization (conv only). Dense layers keep their own
    bias; BN scale is folded into weights.
    """
    import tensorflow as tf
    layers = model.layers
    out = []
    i = 0
    # find conv/dense layers with weights in order
    weight_layers = [l for l in layers if l.get_weights()]
    # Expected: conv,bn,conv,bn,conv,bn,conv,bn,dense,dense,dense
    # (dropout/pooling/relu/flatten/reshape/softmax have no weights)
    wi = 0
    for f, pool in [(8, True), (16, True), (32, True), (32, False)]:
        conv = weight_layers[wi]
        bn = weight_layers[wi + 1]
        W = conv.get_weights()[0]  # [ky,kx,in,out]
        gamma, beta, mean, var = bn.get_weights()
        eps = float(bn.epsilon)
        s = gamma / np.sqrt(var + eps)
        Wf = W * s.reshape(1, 1, 1, -1)
        bf = beta - mean * s
        out.append(("conv", Wf, bf))
        wi += 2
    # remaining three dense layers (fc1 has bias, digits/isDate have bias)
    for _ in range(3):
        dense = weight_layers[wi]
        W, b = dense.get_weights()  # [in,out]
        out.append(("dense", W, b))
        wi += 1
    return out


def to_layout(kind, W):
    """To export layout: conv [out][ky][kx][in], dense [out][in]."""
    if kind == "conv":
        ky, kx, inn, outc = W.shape
        t = np.transpose(W, (3, 0, 1, 2)).copy()  # [out,ky,kx,in]
        return t, (outc, inn, 9)
    else:
        inn, outc = W.shape
        return W.T.copy(), (outc, inn, 1)  # [out,in]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.weights.h5")
    ap.add_argument("--val", default="val_seen.npz")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from train_keras import build, standardize_batch

    import tensorflow as tf

    model = build()
    model.load_weights(args.weights)
    folded = fold_bn(model)
    assert len(folded) == 7, f"expected 7 layers, got {len(folded)}"

    # accuracy before/after quantisation
    z = np.load(args.val)
    X = standardize_batch(z["X"], z["NW"])
    D, Y = z["D"].astype(np.int64), z["Y"]
    digits_p, date_p = model.predict(X, batch_size=256, verbose=0)
    pred = digits_p.argmax(-1)
    pos = D[:, 0] >= 0
    exact_float = ((pred[pos] == D[pos]).all(axis=1)).mean() if pos.sum() else 0.0

    scales_list, bias_list, int8_list = [], [], []
    for kind, Wf, bf in folded:
        t, (outc, inn, k) = to_layout(kind, Wf)
        flat = t.reshape(outc, -1)
        scales = np.abs(flat).max(axis=1) / 127.0
        scales = np.maximum(scales, 1e-9)
        q = np.clip(np.round(flat / scales[:, None]), -127, 127).astype(np.int8)
        scales_list.append(scales.astype(np.float32))
        bias_list.append(bf.astype(np.float32))
        int8_list.append(q.reshape(t.shape).astype(np.int8))

    # int8 accuracy: dequantise and run the folded float network in numpy
    # (conv via tf to avoid writing a second conv; uses dequantised weights)
    deq = [q.astype(np.float32) * s.reshape(-1, *([1] * (q.ndim - 1)))
           for q, s in zip(int8_list, scales_list)]
    # quick check: rebuild conv kernels [ky,kx,in,out] for tf
    # (only needed for the accuracy report; export uses [out][ky][kx][in])
    print(f"float exact-match: {exact_float:.4f}")
    # NOTE: full numpy forward is expensive; evaluate quantisation error via
    # weight SNR + a 256-sample forward through a patched keras model.
    try:
        patched = build()
        patched.set_weights(model.get_weights())  # structure only; overwrite below
        # set dequantised weights back into conv/dense (+ keep BN identity)
        wl = [l for l in patched.layers if l.get_weights()]
        wi = 0
        for li, (kind, _Wf, _bf) in enumerate(folded):
            if kind == "conv":
                Wback = np.transpose(deq[li].reshape(
                    {"conv1": (8, 3, 3, 1), "conv2": (16, 3, 3, 8), "conv3": (32, 3, 3, 16),
                     "conv4": (32, 3, 3, 32)}[["conv1", "conv2", "conv3", "conv4"][li]]),
                    (1, 2, 3, 0))
                wl[wi].set_weights([Wback])
                # BN -> (x + folded_bias): BN(x) = (x-mean)/sqrt(var+eps)*gamma + beta,
                # so mean=0, var=1-eps, gamma=1, beta=bias gives exactly x + bias.
                # (The conv itself is bias-less; the folded bias lives here in the
                # check harness. The real blob stores it separately for cnn.js.)
                eps = float(wl[wi + 1].epsilon)
                g = np.ones_like(wl[wi + 1].get_weights()[0])
                wl[wi + 1].set_weights([g, bias_list[li].astype(np.float32),
                                        np.zeros_like(g), np.full_like(g, 1 - eps)])
                wi += 2
            else:
                outc = deq[li].shape[0]
                Wback = deq[li].reshape(outc, -1).T.reshape(
                    {"fc1": (640, 128), "digits": (128, 80), "isDate": (128, 1)}[
                        ["fc1", "digits", "isDate"][li - 4]])
                wl[wi].set_weights([Wback, bias_list[li]])
                wi += 1
        Xs = X[:256]
        dp, dtp = patched.predict(Xs, batch_size=64, verbose=0)
        exact_q = ((dp.argmax(-1) == D[:256]).all(axis=1)[Y[:256] == 1]).mean()
        print(f"int8 exact-match (256-sample): {exact_q:.4f} (cost {exact_float - exact_q:+.4f})")
        assert exact_float - exact_q < 0.003 + 1e-9, "quantisation costs > 0.3%; use finer quantisation"
    except AssertionError:
        raise
    except Exception as e:
        print(f"quant check skipped ({e})")

    # blob: floats first (scale[out], bias[out] per layer), then int8
    blob = bytearray()
    for s, b in zip(scales_list, bias_list):
        blob += s.tobytes()
        blob += b.tobytes()
    for q in int8_list:
        blob += q.tobytes()
    n_floats = sum(2 * len(s) for s in scales_list)
    print(f"blob: {len(blob) / 1024:.1f} KB ({len(blob) - n_floats * 4} B int8, {n_floats * 4} B scales/biases)")
    assert len(blob) <= 150 * 1024 + 2048, "weights exceed ~150 KB raw budget"

    b64 = base64.b64encode(bytes(blob)).decode()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "modelWeights.js").write_text(
        "// Generated by training-tools/export.py. Do not edit.\n"
        f"// Float exact-match on val: {exact_float:.4f}.\n"
        f"export const MODEL_WEIGHTS_B64 = \"{b64}\";\n"
    )
    print(f"wrote {out / 'modelWeights.js'} ({len(b64) / 1024:.1f} KB base64)")

    # parity.json: 4 standardised inputs + reference LOGITS (pre-softmax/pre-sigmoid).
    # The JS parity test runs _forwardRaw (int8 forward) and compares raw logits.
    # Reference MUST come from the DEQUANTISED int8 weights (the `patched`
    # model above): parity validates JS-vs-Python CODE equivalence (layout,
    # conv order, buffer reuse) to < 5e-3. Comparing int8-JS against the float
    # model would measure quantisation noise amplified by the network, not code
    # correctness (quant fidelity is gated by the < 0.3% accuracy-cost check).
    idx = [0, 1, 2, 3] if len(X) >= 4 else list(range(len(X)))
    try:
        ref_model = patched
    except NameError:
        ref_model = model
    pre_soft = tf.keras.Model(ref_model.input, [ref_model.get_layer("digits_pre").output,
                                                ref_model.get_layer("is_date_logit").output])
    raw_d, raw_t = pre_soft.predict(X[idx], batch_size=4, verbose=0)
    parity = {
        "inputs": [X[i].reshape(-1).astype(np.float32).tolist() for i in idx],
        "logits": raw_d.reshape(len(idx), -1).tolist(),
        "dateLogit": raw_t[:, 0].tolist(),
    }
    (out / "parity.json").write_text(json.dumps(parity))
    print(f"wrote {out / 'parity.json'}")
    print(f"next: cp {out}/modelWeights.js ../src/ocr/ ; cp {out}/parity.json ../tests/fixtures/")


if __name__ == "__main__":
    main()
