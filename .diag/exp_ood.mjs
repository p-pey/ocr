#!/usr/bin/env node
// OOD probe: feed the trained model (a) tight date crop, (b) label+date line,
// (c) label + national-ID line. Shows what the model does with realistic card lines.
import { createCanvas, GlobalFonts } from "@napi-rs/canvas";
import fs from "node:fs";
import { createRequire } from "node:module";
import * as tf from "@tensorflow/tfjs";
import { DigitLineRecognizer, grayToModelInput } from "/home/pooriya/Documents/ocr/src/ocr/digitModel.js";
import { parseJalaliDate } from "/home/pooriya/Documents/ocr/src/ocr/dateParse.js";

const require = createRequire(import.meta.url);
let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv;
else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

GlobalFonts.registerFromPath("/home/pooriya/Documents/ocr/fonts/Vazirmatn-VariableFont_wght.ttf", "vazir");
GlobalFonts.registerFromPath("/home/pooriya/Documents/ocr/fonts/Yekan.ttf", "yekan");

function renderLine(text, { font = "vazir", size = 44, dir = "rtl", w = 700, h = 90 } = {}) {
  const c = createCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#111";
  ctx.font = `${size}px ${font}`;
  ctx.direction = dir;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
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

const dir = "/home/pooriya/Documents/ocr/public/models/date_cnn";
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

const cases = {
  "tight date only": renderLine("۱۳۷۵/۰۵/۱۲", { dir: "ltr", w: 420 }),
  "label + date (rtl line)": renderLine("تاریخ تولد : ۱۳۷۵/۰۵/۱۲"),
  "label + national ID": renderLine("شماره ملی : ۰۱۲۳۴۵۶۷۸۹"),
  "date, other font": renderLine("۱۳۷۵/۰۵/۱۲", { dir: "ltr", w: 420, font: "yekan" }),
};
for (const [name, c] of Object.entries(cases)) {
  const out = (await rec.recognize([canvasToInput(c)]))[0];
  const parsed = parseJalaliDate(out.text);
  console.log(
    `${name}\n  -> text=${out.text} conf=${out.confidence.toFixed(1)} isDate=${out.isDate} (p=${out.dateProb.toFixed(2)}) parsed=${parsed?.formatted ?? null}`,
  );
}
