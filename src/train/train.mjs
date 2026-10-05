#!/usr/bin/env node
/**
 * Train the date-line model in pure JavaScript (Node + TensorFlow.js).
 *
 *   npm i -D @tensorflow/tfjs-node @napi-rs/canvas @techstark/opencv-js
 *   node src/train/train.mjs --fonts ./fonts --out public/models/date_cnn \
 *        [--real ./real_crops] [--backgrounds ./card_textures] [--steps 6000]
 *
 * --fonts        folder of .ttf/.otf fonts that contain the Persian digits ۰-۹
 *                (Vazirmatn, IRANSans, Yekan, ...). Pick ones that look like the
 *                print on the cards. Open <out>/preview.png after the first run
 *                and make sure every sample shows real digits, not empty boxes.
 * --real         optional folder with real cropped lines + labels.csv:
 *                    crop001.png,13750512      (8 digits, slashes optional)
 *                    crop002.png,none          (a line that is NOT a date)
 *                Even a few hundred real crops matter more than millions of
 *                synthetic ones. 20% are held out and reported as REAL-exact.
 * --backgrounds  optional folder of card-texture images (png/jpg).
 *
 * Without @tensorflow/tfjs-node it falls back to plain tfjs (works, but slow).
 */
import { createCanvas, GlobalFonts, loadImage } from "@napi-rs/canvas";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import util from "node:util";
// Fix @tensorflow/tfjs-node@4.22.0 on Node >= 22: it does
// `import { isNullOrUndefined } from 'util'` (Node built-in), but that helper
// was removed from Node's stdlib. Shim it before tfjs-node is loaded.
if (typeof util.isNullOrUndefined !== "function") {
  util.isNullOrUndefined = (v) => v == null;
}
if (typeof util.isArray !== "function") {
  util.isArray = Array.isArray;
}
import {
  buildModel,
  grayToModelInput,
  MODEL_H,
  MODEL_W,
  NUM_DIGITS,
} from "../ocr/digitModel.js";

const require = createRequire(import.meta.url);

/* ---------------- args ---------------- */
const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => {
    if (a.startsWith("--"))
      acc.push([
        a.slice(2),
        all[i + 1]?.startsWith("--") || all[i + 1] === undefined
          ? "true"
          : all[i + 1],
      ]);
    return acc;
  }, []),
);
const OUT = args.out ?? "public/models/date_cnn";
const STEPS = Number(args.steps ?? 6000);
const BATCH = Number(args.batch ?? 64);
const LR = Number(args.lr ?? 2e-3);
const N_VAL = Number(args.val ?? 300);
const EVAL_EVERY = Number(args.evalEvery ?? 100);
if (!args.fonts) throw new Error("--fonts <dir> is required");

/* ---------------- runtimes ---------------- */
let tf;
try {
  tf = await import("@tensorflow/tfjs-node");
  console.log("backend: tfjs-node (native)");
} catch {
  tf = await import("@tensorflow/tfjs");
  console.log(
    "backend: plain tfjs (slow). Install @tensorflow/tfjs-node for speed.",
  );
}
await tf.ready();

let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv;
else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

/* ---------------- random ---------------- */
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const U = (r, a, b) => a + (b - a) * r();
const I = (r, a, b) => Math.floor(U(r, a, b + 1));
const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
function gauss(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

/* ---------------- fonts / assets ---------------- */
const walk = (dir, exts) =>
  fs.existsSync(dir)
    ? fs
        .readdirSync(dir, { withFileTypes: true })
        .flatMap((e) =>
          e.isDirectory()
            ? walk(path.join(dir, e.name), exts)
            : exts.some((x) => e.name.toLowerCase().endsWith(x))
              ? [path.join(dir, e.name)]
              : [],
        )
    : [];
const fontFiles = walk(args.fonts, [".ttf", ".otf"]);
if (!fontFiles.length) throw new Error(`no fonts in ${args.fonts}`);
const families = fontFiles.map((f, i) => {
  GlobalFonts.registerFromPath(f, `fnt${i}`);
  return `fnt${i}`;
});
console.log(`fonts: ${families.length}`);
const bgImages = [];
for (const f of walk(args.backgrounds ?? "", [".png", ".jpg", ".jpeg"])) {
  try {
    bgImages.push(await loadImage(f));
  } catch {
    /* skip */
  }
}

/* ---------------- label generation ---------------- */
const FA = "۰۱۲۳۴۵۶۷۸۹";
const toFa = (s) => s.replace(/\d/g, (d) => FA[Number(d)]);
const LETTERS = "ابپتثجچحخدذرزسشصضطظعغفقکگلمنوهی";
// Card field labels seen next to the digits on real cards (rendered next to
// digit strings in negatives so isDate learns "label + not-a-date" -> reject).
const LABELS = ["تاریخ تولد", "شماره ملی", "تولد", "نام"];
const dim = (y, m) =>
  m <= 6
    ? 31
    : m <= 11
      ? 30
      : [1, 5, 9, 13, 17, 22, 26, 30].includes(y % 33)
        ? 30
        : 29;

function genLabel(r) {
  if (r() < 0.6) {
    // positive: zero padded valid date
    const y = I(r, 1300, 1415),
      m = I(r, 1, 12),
      d = I(r, 1, dim(y, m));
    const digits = `${y}${String(m).padStart(2, "0")}${String(d).padStart(2, "0")}`;
    return {
      text: toFa(
        `${digits.slice(0, 4)}/${digits.slice(4, 6)}/${digits.slice(6)}`,
      ),
      digits,
      isDate: 1,
    };
  }
  // negatives: isDate=0, digit heads are masked out (see toTensors)
  const k = r();
  const cardLabel = pick(r, LABELS);
  let text;
  if (k < 0.2)
    text = `${cardLabel} ${toFa(Array.from({ length: 10 }, () => I(r, 0, 9)).join(""))}`; // national id WITH label (most common confuser)
  else if (k < 0.32)
    text = toFa(Array.from({ length: 10 }, () => I(r, 0, 9)).join("")); // bare national id
  else if (k < 0.44) {
    // near-date: valid shape, impossible month/day (teaches isDate to reject lookalikes)
    const y = I(r, 1300, 1415);
    const bad = r() < 0.5 ? `${y}/${I(r, 13, 19)}/${I(r, 1, 31)}` : `${y}/${I(r, 1, 12)}/${I(r, 32, 39)}`;
    text = `${cardLabel} ${toFa(bad)}`;
  } else if (k < 0.56)
    text = toFa(
      r() < 0.5
        ? `${I(r, 1300, 1415)}/${I(r, 1, 12)}` // truncated date (partial window lookalike)
        : Array.from({ length: pick(r, [5, 6, 7, 9, 11]) }, () => I(r, 0, 9)).join(""),
    )
  else if (k < 0.72)
    text = `${cardLabel} ${Array.from({ length: I(r, 2, 8) }, () => pick(r, [...LETTERS])).join("")}`; // label + word
  else if (k < 0.92)
    text = Array.from({ length: I(r, 6, 24) }, () =>
      r() < 0.15 ? " " : pick(r, [...LETTERS]),
    ).join(""); // words
  else text = ""; // empty crop
  return { text, digits: null, isDate: 0 };
}

/* ---------------- rendering ---------------- */
const scratch = createCanvas(10, 10).getContext("2d");

// Tight text box (paper + optional texture + ink). Returns the canvas.
function renderText(text, r, family, size) {
  scratch.font = `${size}px ${family}`;
  scratch.direction = "ltr";
  const tw = Math.max(20, Math.ceil(scratch.measureText(text).width));
  const th = size;
  const padX = Math.round(th * U(r, 0.15, 0.4)),
    padY = Math.round(th * U(r, 0.2, 0.45));
  const w = tw + 2 * padX,
    h = th + 2 * padY;
  const c = createCanvas(w, h),
    ctx = c.getContext("2d");

  const base = I(r, 165, 245);
  ctx.fillStyle = `rgb(${base},${base},${base})`;
  ctx.fillRect(0, 0, w, h);
  if (bgImages.length && r() < 0.6) {
    const img = pick(r, bgImages);
    const sw = Math.min(
      img.width,
      Math.max(16, Math.round(img.width * U(r, 0.1, 0.4))),
    );
    const sh = Math.min(img.height, Math.max(8, Math.round((sw * h) / w)));
    ctx.drawImage(
      img,
      I(r, 0, img.width - sw),
      I(r, 0, img.height - sh),
      sw,
      sh,
      0,
      0,
      w,
      h,
    );
  } else {
    for (let i = 0, n = I(r, 0, 6); i < n; i++) {
      // guilloche-like fine lines
      const v = base - I(r, 5, 25);
      ctx.strokeStyle = `rgba(${v},${v},${v},0.8)`;
      ctx.lineWidth = U(r, 0.5, 1.5);
      ctx.beginPath();
      const y0 = U(r, 0, h);
      ctx.moveTo(0, y0);
      ctx.lineTo(w, y0 + U(r, -h / 3, h / 3));
      ctx.stroke();
    }
  }
  const ink = I(r, 0, 70);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate((U(r, -2.5, 2.5) * Math.PI) / 180);
  ctx.transform(1, U(r, -0.03, 0.03), U(r, -0.12, 0.12), 1, 0, 0);
  ctx.font = `${size}px ${family}`;
  ctx.direction = "ltr";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = `rgb(${ink},${ink},${ink})`;
  ctx.fillText(text, 0, 0);
  ctx.restore();
  return c;
}

function rawRender(label, r) {
  return renderText(
    label.text || " ",
    r,
    pick(r, families),
    I(r, 26, 60),
  );
}

/**
 * Neighbor-text fragment overlapping one edge of the crop (mostly cut off).
 * Mimics a detector/sliding window that clips adjacent words. Applied to
 * positives so the heads learn to read the date despite edge clutter.
 */
function addEdgeClutter(c, r, family, size) {
  if (r() > 0.5) return c;
  const word = Array.from(
    { length: I(r, 2, 6) },
    () => pick(r, [...LETTERS]),
  ).join("");
  const wc = createCanvas(Math.max(8, Math.ceil(size * word.length * 0.7)), size + 8);
  const wctx = wc.getContext("2d");
  wctx.font = `${size}px ${family}`;
  wctx.direction = "ltr";
  wctx.textAlign = "center";
  wctx.textBaseline = "middle";
  const ink = I(r, 0, 70);
  wctx.fillStyle = `rgb(${ink},${ink},${ink})`;
  wctx.fillText(word, wc.width / 2, wc.height / 2);
  const out = createCanvas(c.width, c.height);
  const ctx = out.getContext("2d");
  ctx.drawImage(c, 0, 0);
  const vis = U(r, 0.1, 0.45); // visible fraction of the fragment
  const x =
    r() < 0.5
      ? -Math.round(wc.width * (1 - vis))
      : c.width - Math.round(wc.width * vis);
  const y = Math.round(U(r, -0.15, 0.15) * c.height);
  ctx.drawImage(wc, x, y);
  return out;
}

/**
 * Paste the line onto a larger paper sheet with random asymmetric margins and
 * scale. Detector crops never have the stereotyped tight padding of rawRender;
 * without this the model memorises digit positions and misfires confidently on
 * any real crop (the "worst output" failure mode).
 */
function rebox(src, r) {
  const scale = U(r, 0.7, 1.25);
  const dw = Math.max(8, Math.round(src.width * scale));
  const dh = Math.max(8, Math.round(src.height * scale));
  const mL = Math.round(dw * U(r, 0, 0.6));
  const mR = Math.round(dw * U(r, 0, 0.6));
  const mT = Math.round(dh * U(r, 0, 0.6));
  const mB = Math.round(dh * U(r, 0, 0.6));
  const W = dw + mL + mR;
  const H = dh + mT + mB;
  const c = createCanvas(W, H);
  const ctx = c.getContext("2d");
  // paper fill sampled from a source corner so margins blend in
  const corner = src.getContext("2d").getImageData(0, 0, 1, 1).data;
  ctx.fillStyle = `rgb(${corner[0]},${corner[1]},${corner[2]})`;
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(src, mL, mT, dw, dh);
  return c;
}

/** Full pre-degradation sample canvas: text -> clutter -> detector-like crop. */
function renderSampleCanvas(label, r) {
  const family = pick(r, families);
  const size = I(r, 26, 60);
  let c = renderText(label.text || " ", r, family, size);
  if (label.isDate) c = addEdgeClutter(c, r, family, size);
  return rebox(c, r);
}

async function degrade(src, r, strength = 1) {
  let c = src;
  const w = c.width,
    h = c.height;
  if (r() < 0.6 * strength) {
    // resolution loss
    const s = U(r, 0.35, 0.9);
    const small = createCanvas(
      Math.max(8, Math.round(w * s)),
      Math.max(8, Math.round(h * s)),
    );
    small.getContext("2d").drawImage(c, 0, 0, small.width, small.height);
    const back = createCanvas(w, h);
    back.getContext("2d").drawImage(small, 0, 0, w, h);
    c = back;
  }
  const filters = [];
  if (r() < 0.5 * strength)
    filters.push(`blur(${U(r, 0.4, 1.4).toFixed(2)}px)`);
  filters.push(
    `contrast(${U(r, 0.6, 1.3).toFixed(2)})`,
    `brightness(${U(r, 0.8, 1.15).toFixed(2)})`,
  );
  const out = createCanvas(w, h),
    ctx = out.getContext("2d");
  ctx.filter = filters.join(" ");
  ctx.drawImage(c, 0, 0);
  ctx.filter = "none";
  if (r() < 0.4 * strength) {
    // glare
    const gx = U(r, 0, w),
      gy = U(r, 0, h),
      rad = U(r, 0.2, 0.7) * Math.max(w, h);
    const g = ctx.createRadialGradient(gx, gy, 0, gx, gy, rad);
    g.addColorStop(0, `rgba(255,255,255,${U(r, 0.2, 0.6).toFixed(2)})`);
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }
  const img = ctx.getImageData(0, 0, w, h),
    d = img.data,
    sigma = U(r, 0, 8) * strength;
  for (let i = 0; i < d.length; i += 4) {
    const n = gauss(r) * sigma;
    d[i] += n;
    d[i + 1] += n;
    d[i + 2] += n;
  }
  ctx.putImageData(img, 0, 0);
  if (r() < 0.5 * strength) {
    // JPEG artefacts
    const jpg = out.toBuffer("image/jpeg", I(r, 25, 90));
    const im = await loadImage(jpg);
    const c2 = createCanvas(w, h);
    c2.getContext("2d").drawImage(im, 0, 0);
    return c2;
  }
  return out;
}

function canvasToInput(c) {
  const { data } = c.getContext("2d").getImageData(0, 0, c.width, c.height);
  const rgba = cv.matFromArray(c.height, c.width, cv.CV_8UC4, Array.from(data));
  const gray = new cv.Mat();
  try {
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    return grayToModelInput(cv, gray);
  } finally {
    rgba.delete();
    gray.delete();
  }
}

async function syntheticSample(r) {
  const label = genLabel(r);
  return {
    x: canvasToInput(await degrade(renderSampleCanvas(label, r), r)),
    digits: label.digits,
    isDate: label.isDate,
  };
}

/* ---------------- real data ---------------- */
async function loadReal(dir) {
  const csv = path.join(dir, "labels.csv");
  if (!fs.existsSync(csv)) throw new Error(`${csv} not found`);
  const items = [];
  for (const line of fs.readFileSync(csv, "utf8").split(/\r?\n/)) {
    const [file, label] = line.split(",").map((s) => s?.trim());
    if (!file || !label) continue;
    const digits = label.replace(/\D/g, "");
    if (label.toLowerCase() === "none")
      items.push({ file: path.join(dir, file), digits: null, isDate: 0 });
    else if (digits.length === 8)
      items.push({ file: path.join(dir, file), digits, isDate: 1 });
  }
  return items;
}
async function realSample(item, r, strength) {
  const im = await loadImage(item.file);
  const c = createCanvas(im.width, im.height);
  c.getContext("2d").drawImage(im, 0, 0);
  return {
    x: canvasToInput(strength > 0 ? await degrade(c, r, strength) : c),
    digits: item.digits,
    isDate: item.isDate,
  };
}

/* ---------------- batching ---------------- */
function toTensors(samples) {
  const B = samples.length;
  const x = new Float32Array(B * MODEL_H * MODEL_W);
  const y = new Float32Array(B * NUM_DIGITS * 10);
  const m = new Float32Array(B),
    isDate = new Float32Array(B);
  samples.forEach((s, i) => {
    x.set(s.x, i * MODEL_H * MODEL_W);
    isDate[i] = s.isDate;
    if (s.digits) {
      m[i] = 1;
      for (let h = 0; h < NUM_DIGITS; h++)
        y[(i * NUM_DIGITS + h) * 10 + Number(s.digits[h])] = 1;
    }
  });
  return {
    x: tf.tensor4d(x, [B, MODEL_H, MODEL_W, 1]),
    y: tf.tensor3d(y, [B, NUM_DIGITS, 10]),
    m: tf.tensor1d(m),
    isDate: tf.tensor2d(isDate, [B, 1]),
  };
}

async function evaluate(model, samples) {
  const t = toTensors(samples);
  const [dg, dt] = model.predict(t.x);
  const pred = await dg.argMax(-1).array(),
    truth = await t.y.argMax(-1).array();
  const dprob = await dt.data(),
    masks = await t.m.data();
  let ok = 0,
    tot = 0,
    dOk = 0;
  samples.forEach((s, i) => {
    dOk += dprob[i] >= 0.5 === (s.isDate === 1) ? 1 : 0;
    if (masks[i]) {
      tot++;
      ok += pred[i].every((p, h) => p === truth[i][h]) ? 1 : 0;
    }
  });
  tf.dispose([t.x, t.y, t.m, t.isDate, dg, dt]);
  return { exact: tot ? ok / tot : 0, dateAcc: dOk / samples.length };
}

async function saveModel(model, dir) {
  fs.mkdirSync(dir, { recursive: true });
  await model.save(
    tf.io.withSaveHandler(async (a) => {
      fs.writeFileSync(
        path.join(dir, "weights.bin"),
        Buffer.from(a.weightData),
      );
      fs.writeFileSync(
        path.join(dir, "model.json"),
        JSON.stringify({
          modelTopology: a.modelTopology,
          format: a.format,
          generatedBy: a.generatedBy,
          convertedBy: a.convertedBy,
          weightsManifest: [{ paths: ["weights.bin"], weights: a.weightSpecs }],
        }),
      );
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
        },
      };
    }),
  );
}

/* ---------------- main ---------------- */
const rng = mulberry32(Date.now() & 0xffffffff);
let real = args.real ? await loadReal(args.real) : [];
real = real
  .map((v) => [v, rng()])
  .sort((a, b) => a[1] - b[1])
  .map((v) => v[0]);
const nVal = Math.min(200, Math.floor(real.length / 5));
const realVal = real.slice(0, nVal),
  realTrain = real.slice(nVal);
console.log(`real crops: ${realTrain.length} train / ${realVal.length} val`);

fs.mkdirSync(OUT, { recursive: true });
{
  // preview grid so you can eyeball the synthetic samples
  const pr = mulberry32(7),
    rows = 12,
    grid = createCanvas(480, rows * 60),
    g = grid.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, 480, rows * 60);
  for (let i = 0; i < rows; i++) {
    // NOTE: uses the full sample pipeline (text -> clutter -> rebox -> degrade)
    // so preview.png honestly shows what the model trains on.
    const c = await degrade(renderSampleCanvas(genLabel(pr), pr), pr);
    g.drawImage(c, 4, i * 60 + 4, 470, 52);
  }
  fs.writeFileSync(path.join(OUT, "preview.png"), grid.toBuffer("image/png"));
}
const vr = mulberry32(12345);
const valSyn = [];
for (let i = 0; i < N_VAL; i++) valSyn.push(await syntheticSample(vr));
const valReal = [];
for (const it of realVal) valReal.push(await realSample(it, vr, 0));

const model = buildModel(tf);
const opt = tf.train.adam(LR);
let best = -1;
for (let step = 1; step <= STEPS; step++) {
  opt.learningRate =
    LR * (0.02 + 0.98 * 0.5 * (1 + Math.cos(Math.PI * (step / STEPS)))); // cosine decay
  const batch = [];
  for (let i = 0; i < BATCH; i++) {
    batch.push(
      realTrain.length && rng() < 0.5
        ? await realSample(pick(rng, realTrain), rng, 0.7)
        : await syntheticSample(rng),
    );
  }
  const t = toTensors(batch);
  const loss = opt.minimize(() => {
    const [dg, dt] = model.apply(t.x, { training: true });
    const ce = tf
      .neg(tf.sum(tf.mul(t.y, tf.log(tf.add(dg, 1e-7))), -1))
      .mean(-1); // [B]
    const digitLoss = tf.sum(tf.mul(ce, t.m)).div(tf.add(tf.sum(t.m), 1e-6));
    const eps = 1e-7,
      p = tf.clipByValue(dt, eps, 1 - eps);
    const bce = tf
      .neg(
        tf.add(
          tf.mul(t.isDate, tf.log(p)),
          tf.mul(tf.sub(1, t.isDate), tf.log(tf.sub(1, p))),
        ),
      )
      .mean();
    return digitLoss.add(bce);
  }, true);
  const lossVal = (await loss.data())[0];
  tf.dispose([t.x, t.y, t.m, t.isDate, loss]);

  if (step % EVAL_EVERY === 0 || step === STEPS) {
    const s = await evaluate(model, valSyn);
    const rl = valReal.length ? await evaluate(model, valReal) : null;
    const score = rl ? rl.exact + rl.dateAcc : s.exact + s.dateAcc;
    console.log(
      `step ${step}/${STEPS} loss ${lossVal.toFixed(3)} | synthetic exact ${s.exact.toFixed(3)} isDate ${s.dateAcc.toFixed(3)}` +
        (rl
          ? ` | REAL exact ${rl.exact.toFixed(3)} isDate ${rl.dateAcc.toFixed(3)}`
          : ""),
    );
    if (score >= best) {
      best = score;
      await saveModel(model, OUT);
    }
  }
}
console.log(
  `done. best model saved in ${OUT} (open preview.png to sanity-check the synthetic data)`,
);
