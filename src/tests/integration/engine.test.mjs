// Wiring test in Node: rectification + line finding + model load/predict + result building.
// Usage (needs @tensorflow/tfjs, @techstark/opencv-js, @napi-rs/canvas installed):
//   node src/tests/integration/engine.test.mjs <card-photo.jpg|png> <model_dir>
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import * as tf from "@tensorflow/tfjs";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import { DigitLineRecognizer } from "../../ocr/digitModel.js";
import { TesseractOCR, fitToCard, findLineCandidates } from "../../ocr/TesseractOCR.js";

const require = createRequire(import.meta.url);
let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv; else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

const [cardFile, modelDir] = process.argv.slice(2);
const img = await loadImage(cardFile);
const canvas = createCanvas(img.width, img.height);
const g = canvas.getContext("2d"); g.drawImage(img, 0, 0);
const { data } = g.getImageData(0, 0, img.width, img.height);
const src = cv.matFromArray(img.height, img.width, cv.CV_8UC4, Array.from(data));
const { card, rectified } = fitToCard(cv, src);
console.log("rectified:", rectified, `${card.cols}x${card.rows}`);
const gray = new cv.Mat(); cv.cvtColor(card, gray, cv.COLOR_RGBA2GRAY);
console.log("line candidates:", findLineCandidates(cv, gray).length);

const json = JSON.parse(fs.readFileSync(path.join(modelDir, "model.json"), "utf8"));
const weightData = fs.readFileSync(path.join(modelDir, "weights.bin"));
const model = await tf.loadLayersModel(tf.io.fromMemory({
  modelTopology: json.modelTopology,
  weightSpecs: json.weightsManifest[0].weights,
  weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
}));
const recognizer = new DigitLineRecognizer(); recognizer.attach(tf, model);
const ocr = new TesseractOCR({ recognizer });
const result = await ocr.recognizeMat(cv, src);
console.log("best:", result.best?.birthDate?.formatted ?? null, "| dates considered:", result.allDates.map((d) => d.formatted));
