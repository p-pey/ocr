#!/usr/bin/env node
// Save probe renders + model-input visualizations to inspect the domain gap.
import { createCanvas, GlobalFonts } from "@napi-rs/canvas";
import fs from "node:fs";
import { createRequire } from "node:module";
import * as tf from "@tensorflow/tfjs";
import { DigitLineRecognizer, grayToModelInput, MODEL_H, MODEL_W } from "../src/ocr/digitModel.js";
import { parseJalaliDate } from "../src/ocr/dateParse.js";

const require = createRequire(import.meta.url);
let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv;
else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

GlobalFonts.registerFromPath("./fonts/Vazirmatn-VariableFont_wght.ttf", "vazir");

function renderLoose(text, size = 44, w = 420, h = 90) {
  const c = createCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#111";
  ctx.font = `${size}px vazir`;
  ctx.direction = "ltr";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
  return c;
}

// training-style tight render (mirror of train.mjs rawRender, no degrade)
function renderTight(faText, size = 44) {
  const scratch = createCanvas(10, 10).getContext("2d");
  scratch.font = `${size}px vazir`;
  scratch.direction = "ltr";
  const tw = Math.max(20, Math.ceil(scratch.measureText(faText).width));
  const th = size;
  const padX = Math.round(th * 0.25);
  const padY = Math.round(th * 0.3);
  const w = tw + 2 * padX;
  const h = th + 2 * padY;
  const c = createCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#111";
  ctx.font = `${size}px vazir`;
  ctx.direction = "ltr";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(faText, w / 2, h / 2);
  return c;
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

// visualize model input (unnormalize to 0..255) as PNG
function inputToPng(inp, file) {
  const c = createCanvas(MODEL_W, MODEL_H);
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(MODEL_W, MODEL_H);
  let mn = Infinity;
  let mx = -Infinity;
  for (const v of inp) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  for (let i = 0; i < inp.length; i++) {
    const v = Math.round(((inp[i] - mn) / (mx - mn + 1e-9)) * 255);
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  // 3x upscale for visibility
  const big = createCanvas(MODEL_W * 3, MODEL_H * 3);
  const bctx = big.getContext("2d");
  bctx.imageSmoothingEnabled = false;
  bctx.drawImage(c, 0, 0, big.width, big.height);
  fs.writeFileSync(file, big.toBuffer("image/png"));
}

const dir = "public/models/date_cnn";
const json = JSON.parse(fs.readFileSync(dir + "/model.json", "utf8"));
const w = fs.readFileSync(dir + "/weights.bin");
const model = await tf.loadLayersModel(
  tf.io.fromMemory({
    modelTopology: json.modelTopology,
    weightSpecs: json.weightsManifest[0].weights,
    weightData: w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength),
  }),
);
const rec = new DigitLineRecognizer();
rec.attach(tf, model);

const FA = "۰۱۲۳۴۵۶۷۸۹";
const toFa = (s) => s.replace(/\d/g, (d) => FA[Number(d)]);
const truth = "13750512";
const faDate = toFa("1375/05/12");

const loose = renderLoose(faDate);
const tight = renderTight(faDate);
fs.writeFileSync(".diag/loose.png", loose.toBuffer("image/png"));
fs.writeFileSync(".diag/tight.png", tight.toBuffer("image/png"));
const inpLoose = canvasToInput(loose);
const inpTight = canvasToInput(tight);
inputToPng(inpLoose, ".diag/inp_loose.png");
inputToPng(inpTight, ".diag/inp_tight.png");

for (const [name, inp] of [["loose-centered", inpLoose], ["training-tight", inpTight]]) {
  const out = (await rec.recognize([inp]))[0];
  console.log(`${name}: text=${out.text} conf=${out.confidence.toFixed(1)} isDate=${out.isDate} (truth 1375/05/12)`);
}
