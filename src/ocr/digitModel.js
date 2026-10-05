/**
 * Date-line digit recogniser, pure JavaScript (TensorFlow.js).
 *
 * The birth date is always printed as YYYY/MM/DD with zero padding, i.e. exactly
 * 8 digits. So instead of CTC / segmentation, a small CNN looks at the whole line
 * and predicts all 8 digits at once (8 softmax heads of 10 classes), plus an
 * `isDate` score that says "this crop really is a date line". That second output
 * is what stops a 10-digit national-ID line from being mistaken for a date.
 *
 * The same module is used by the browser engine AND by train/train.mjs, so the
 * architecture and the preprocessing can never drift apart.
 */

export const MODEL_H = 32;
export const MODEL_W = 160;
export const NUM_DIGITS = 8;

/**
 * gray: cv.Mat (CV_8UC1), one text line, dark ink on light background.
 * Returns Float32Array(MODEL_H * MODEL_W), row-major, standardised, zero padded right.
 */
export function grayToModelInput(cv, gray) {
  const newW = Math.max(
    8,
    Math.min(MODEL_W, Math.round((gray.cols * MODEL_H) / Math.max(gray.rows, 1))),
  );
  const resized = new cv.Mat();
  try {
    cv.resize(
      gray,
      resized,
      new cv.Size(newW, MODEL_H),
      0,
      0,
      gray.rows > MODEL_H ? cv.INTER_AREA : cv.INTER_LINEAR,
    );
    const px = resized.data;
    const n = px.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += px[i];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) variance += (px[i] - mean) ** 2;
    const std = Math.sqrt(variance / n);

    const out = new Float32Array(MODEL_H * MODEL_W);
    for (let y = 0; y < MODEL_H; y++) {
      for (let x = 0; x < newW; x++) {
        out[y * MODEL_W + x] = (px[y * newW + x] - mean) / (std + 1e-6);
      }
    }
    return out;
  } finally {
    resized.delete();
  }
}

/** Architecture. `tf` is passed in so Node (training) and browser share it. */
export function buildModel(tf) {
  const input = tf.input({ shape: [MODEL_H, MODEL_W, 1] });
  let x = input;
  const convBlock = (filters, pool) => {
    x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", useBias: false }).apply(x);
    x = tf.layers.batchNormalization().apply(x);
    x = tf.layers.activation({ activation: "relu" }).apply(x);
    if (pool) x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);
  };
  convBlock(16, true); // 16 x 80
  convBlock(32, true); // 8 x 40
  convBlock(64, true); // 4 x 20
  convBlock(64, false);
  x = tf.layers.flatten().apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);
  x = tf.layers.dense({ units: 256, activation: "relu" }).apply(x);

  let digits = tf.layers.dense({ units: NUM_DIGITS * 10 }).apply(x);
  digits = tf.layers.reshape({ targetShape: [NUM_DIGITS, 10] }).apply(digits);
  digits = tf.layers.activation({ activation: "softmax", name: "digits" }).apply(digits);
  const isDate = tf.layers.dense({ units: 1, activation: "sigmoid", name: "is_date" }).apply(x);

  return tf.model({ inputs: input, outputs: [digits, isDate] });
}

/**
 * probs: Float32Array [N, 8, 10]; dateProbs: Float32Array [N].
 * confidence (0-100) = mean over digit heads of the winning probability.
 */
export function decodeHeads(probs, dateProbs, N) {
  const results = [];
  for (let n = 0; n < N; n++) {
    let digits = "";
    let sum = 0;
    let minProb = 1;
    for (let h = 0; h < NUM_DIGITS; h++) {
      const base = (n * NUM_DIGITS + h) * 10;
      let best = 0;
      for (let c = 1; c < 10; c++) if (probs[base + c] > probs[base + best]) best = c;
      const p = probs[base + best];
      digits += best;
      sum += p;
      if (p < minProb) minProb = p;
    }
    const dateProb = dateProbs[n];
    results.push({
      digits,
      text: `${digits.slice(0, 4)}/${digits.slice(4, 6)}/${digits.slice(6)}`,
      confidence: (sum / NUM_DIGITS) * 100,
      minProb,
      dateProb,
      isDate: dateProb >= 0.5,
    });
  }
  return results;
}

export class DigitLineRecognizer {
  /** modelUrl points to the model.json written by train/train.mjs */
  constructor({ modelUrl = "/models/date_cnn/model.json" } = {}) {
    this.modelUrl = modelUrl;
    this.tf = null;
    this.model = null;
  }

  get ready() {
    return Boolean(this.model);
  }

  /** Node tests / custom loaders can hand in an already loaded tf + model. */
  attach(tf, model) {
    this.tf = tf;
    this.model = model;
  }

  async load() {
    if (this.model) return;
    const tf = await import("@tensorflow/tfjs");
    await tf.ready();
    this.model = await tf.loadLayersModel(this.modelUrl);
    this.tf = tf;
  }

  /** inputs: Float32Array[] of length MODEL_H*MODEL_W each. */
  async recognize(inputs) {
    if (!inputs.length) return [];
    if (!this.model) await this.load();
    const tf = this.tf;
    const N = inputs.length;
    const data = new Float32Array(N * MODEL_H * MODEL_W);
    inputs.forEach((arr, i) => data.set(arr, i * MODEL_H * MODEL_W));

    const x = tf.tensor4d(data, [N, MODEL_H, MODEL_W, 1]);
    const [digitsT, dateT] = this.model.predict(x);
    const [probs, dateProbs] = await Promise.all([digitsT.data(), dateT.data()]);
    tf.dispose([x, digitsT, dateT]);
    return decodeHeads(probs, dateProbs, N);
  }

  async dispose() {
    this.model?.dispose();
    this.model = null;
  }
}
