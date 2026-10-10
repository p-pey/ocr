/**
 * Tiny dependency-free CNN for date-line recognition (pure JavaScript).
 *
 * Input : 32x160 grayscale line crop, standardised (see `grayToModelInput`).
 * Output: 8 digits (YYYYMMDD) + an "is this a date line" score.
 *
 * Architecture (BatchNorm folded into the convolutions at export time):
 *   conv3x3(1->8)+ReLU+pool2 -> conv3x3(8->16)+ReLU+pool2 -> conv3x3(16->32)+ReLU+pool2
 *   -> conv3x3(32->32)+ReLU -> average over height -> 20x32 -> dense 128 + ReLU
 *   -> digits head (8x10 softmax) + isDate head (sigmoid)
 * ~107k int8 weights (~110 KB, embedded in modelWeights.js). ~4M MACs per line.
 */
import { MODEL_WEIGHTS_B64 } from "./modelWeights.js";
import { CNN_INPUT as CNN_CFG } from "./engineConfig.js";

// CNN input contract (§J). Values come from engineConfig.js — changing them
// without retraining silently destroys accuracy (training/inference contract).
export const MODEL_H = CNN_CFG.heightPx;
export const MODEL_W = CNN_CFG.widthPx;
export const NUM_DIGITS = CNN_CFG.digitSlots;

/**
 * PREPROCESSING CONTRACT (spec section 5, byte-for-byte with training).
 * 1. Crop of one text line, grayscale uint8, dark ink / light paper.
 * 2. newW = clamp(round(cropW * 32 / cropH), 8, 160).
 * 3. resize to (newW, 32), INTER_AREA if cropH > 32 else INTER_LINEAR.
 * 4. Standardise over the resized region ONLY: x = (x - mean) / (std + 1e-6).
 * 5. Place into zero 32x160 tensor at columns [0, newW); rest stays 0.
 * No binarisation / CLAHE / sharpening. Changing this requires retraining.
 */
export function grayToModelInput(cv, gray) {
  const newW = Math.max(
    CNN_CFG.minResizedWidthPx,
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
        // Standardise over the resized region ONLY (§J).
        out[y * MODEL_W + x] =
          (px[y * newW + x] - mean) / (std + CNN_CFG.standardiseEpsilon);
      }
    }
    return out;
  } finally {
    resized.delete();
  }
}

// [outChannels, inChannels, kind]
const LAYERS = [
  { name: "conv1", out: 8, inn: 1, k: 9 },
  { name: "conv2", out: 16, inn: 8, k: 9 },
  { name: "conv3", out: 32, inn: 16, k: 9 },
  { name: "conv4", out: 32, inn: 32, k: 9 },
  { name: "fc1", out: 128, inn: 640, k: 1 },
  { name: "digits", out: 80, inn: 128, k: 1 },
  { name: "isDate", out: 1, inn: 128, k: 1 },
];

let cached = null;

function b64ToBytes(b64) {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return Uint8Array.from(Buffer.from(b64, "base64")); // Node (tests)
}

/** Decodes the embedded weights once: int8 * per-channel scale -> Float32. */
function loadWeights() {
  if (cached) return cached;
  const bytes = b64ToBytes(MODEL_WEIGHTS_B64);
  const nFloats = LAYERS.reduce((s, l) => s + 2 * l.out, 0);
  const floats = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + nFloats * 4));
  const ints = new Int8Array(bytes.buffer, bytes.byteOffset + nFloats * 4);
  let fo = 0;
  let io = 0;
  cached = LAYERS.map((l) => {
    const scales = floats.subarray(fo, fo + l.out); fo += l.out;
    const bias = floats.subarray(fo, fo + l.out); fo += l.out;
    const n = l.out * l.inn * l.k;
    const w = new Float32Array(n);
    const per = l.inn * l.k;
    for (let o = 0; o < l.out; o++) {
      const s = scales[o];
      for (let i = 0; i < per; i++) w[o * per + i] = ints[io + o * per + i] * s;
    }
    io += n;
    return { ...l, w, bias: Float32Array.from(bias) };
  });
  return cached;
}

// Scratch buffers (single-threaded JS, reused between calls).
const buf = {
  a: new Float32Array(MODEL_H * MODEL_W * 8),
  b: new Float32Array(MODEL_H * MODEL_W * 8),
  pad: new Float32Array((MODEL_H + 2) * (MODEL_W + 2) * 32),
};

/** 3x3 same-padding conv + bias + ReLU, NHWC. src: [H,W,inC] -> dst [H,W,outC] */
function conv3x3Relu(src, H, W, inC, layer, dst) {
  const { w, bias, out: outC } = layer;
  const PW = W + 2;
  const pad = buf.pad;
  pad.fill(0, 0, (H + 2) * PW * inC);
  for (let y = 0; y < H; y++) {
    pad.set(src.subarray(y * W * inC, (y + 1) * W * inC), ((y + 1) * PW + 1) * inC);
  }
  const rowLen = 3 * inC; // one kernel row (kx=0..2, ic=0..inC-1) is contiguous in both arrays
  const kLen = 9 * inC;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o0 = (y * W + x) * outC;
      const base0 = (y * PW + x) * inC;
      const base1 = ((y + 1) * PW + x) * inC;
      const base2 = ((y + 2) * PW + x) * inC;
      for (let oc = 0; oc < outC; oc++) {
        const wo = oc * kLen;
        let sum = bias[oc];
        for (let i = 0; i < rowLen; i++) sum += pad[base0 + i] * w[wo + i];
        for (let i = 0; i < rowLen; i++) sum += pad[base1 + i] * w[wo + rowLen + i];
        for (let i = 0; i < rowLen; i++) sum += pad[base2 + i] * w[wo + 2 * rowLen + i];
        dst[o0 + oc] = sum > 0 ? sum : 0;
      }
    }
  }
}

function maxPool2(src, H, W, C, dst) {
  const oh = H >> 1;
  const ow = W >> 1;
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      for (let c = 0; c < C; c++) {
        const i00 = ((2 * y) * W + 2 * x) * C + c;
        const i01 = i00 + C;
        const i10 = i00 + W * C;
        const i11 = i10 + C;
        const a = src[i00] > src[i01] ? src[i00] : src[i01];
        const b = src[i10] > src[i11] ? src[i10] : src[i11];
        dst[(y * ow + x) * C + c] = a > b ? a : b;
      }
    }
  }
}

function dense(src, layer, relu, dst) {
  const { w, bias, out, inn } = layer;
  for (let o = 0; o < out; o++) {
    let sum = bias[o];
    const wo = o * inn;
    for (let i = 0; i < inn; i++) sum += src[i] * w[wo + i];
    dst[o] = relu && sum < 0 ? 0 : sum;
  }
}

const fc1Out = new Float32Array(128);
const headOut = new Float32Array(80);
const dateOut = new Float32Array(1);
const avg = new Float32Array(640);

/** One standardised 32x160 line -> raw logits. */
function forward(input) {
  const L = loadWeights();
  let H = MODEL_H;
  let W = MODEL_W;
  let cur = input;
  // conv1..conv3 with pooling
  for (let li = 0; li < 3; li++) {
    const l = L[li];
    conv3x3Relu(cur, H, W, l.inn, l, buf.a);
    maxPool2(buf.a, H, W, l.out, buf.b);
    H >>= 1; W >>= 1;
    cur = buf.b.subarray(0, H * W * l.out);
    if (li < 2) { buf.a.set(cur); cur = buf.a.subarray(0, H * W * l.out); }
  }
  // conv4 (no pool): [4,20,32]
  const l4 = L[3];
  const conv4Out = new Float32Array(H * W * l4.out);
  conv3x3Relu(cur, H, W, l4.inn, l4, conv4Out);
  // average over height -> [20,32] flattened as x*32+c
  const C = l4.out;
  for (let x = 0; x < W; x++) {
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let y = 0; y < H; y++) s += conv4Out[(y * W + x) * C + c];
      avg[x * C + c] = s / H;
    }
  }
  dense(avg, L[4], true, fc1Out);
  dense(fc1Out, L[5], false, headOut);
  dense(fc1Out, L[6], false, dateOut);
  return { logits: headOut, dateLogit: dateOut[0] };
}

/** probs: Float32Array(8*10) (softmax per digit), dateProb in 0..1 -> result object. */
export function decodeProbs(probs, dateProb) {
  let digits = "";
  let sum = 0;
  let minProb = 1;
  for (let h = 0; h < NUM_DIGITS; h++) {
    let best = 0;
    for (let c = 1; c < 10; c++) if (probs[h * 10 + c] > probs[h * 10 + best]) best = c;
    digits += best;
    const p = probs[h * 10 + best];
    sum += p;
    if (p < minProb) minProb = p;
  }
  return {
    digits,
    text: `${digits.slice(0, 4)}/${digits.slice(4, 6)}/${digits.slice(6)}`,
    confidence: (sum / NUM_DIGITS) * 100,
    minProb,
    dateProb,
    isDate: dateProb >= CNN_CFG.isDateThreshold,
    probs,
  };
}

/** Average several reads of the same line (test-time augmentation). */
export function averageReads(reads) {
  const probs = new Float32Array(NUM_DIGITS * 10);
  let dateProb = 0;
  for (const r of reads) {
    for (let i = 0; i < probs.length; i++) probs[i] += r.probs[i] / reads.length;
    dateProb += r.dateProb / reads.length;
  }
  return decodeProbs(probs, dateProb);
}

/**
 * @param {Float32Array[]} inputs  standardised lines (MODEL_H*MODEL_W each)
 * @returns per line: { digits, text "YYYY/MM/DD", confidence 0-100, minProb, dateProb, isDate, probs }
 */
export function recognizeLines(inputs) {
  const results = [];
  for (const input of inputs) {
    const { logits, dateLogit } = forward(input);
    const probs = new Float32Array(NUM_DIGITS * 10);
    for (let h = 0; h < NUM_DIGITS; h++) {
      let max = -Infinity;
      for (let c = 0; c < 10; c++) if (logits[h * 10 + c] > max) max = logits[h * 10 + c];
      let denom = 0;
      for (let c = 0; c < 10; c++) denom += Math.exp(logits[h * 10 + c] - max);
      for (let c = 0; c < 10; c++) probs[h * 10 + c] = Math.exp(logits[h * 10 + c] - max) / denom;
    }
    results.push(decodeProbs(probs, 1 / (1 + Math.exp(-dateLogit))));
  }
  return results;
}

/** Debug helper for parity tests: raw outputs for one input. */
export function _forwardRaw(input) {
  const { logits, dateLogit } = forward(input);
  return { logits: Float32Array.from(logits), dateLogit };
}
