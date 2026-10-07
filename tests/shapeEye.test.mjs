// Visual "eye" engine tests (directives 2 + shape training):
// vertical-projection slot splitting (pure), fixed 8+2 date profiler +
// cv.matchTemplate scoring (needs @techstark/opencv-js + @napi-rs/canvas).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { splitSlotsProjection, splitDateSlots, rankTemplates } from "../src/ocr/shapeGate.js";

const here = path.dirname(fileURLToPath(import.meta.url));

// --- projection splitter on a synthetic 4-slot strip ---
{
  const W = 120;
  const H = 24;
  const px = new Uint8Array(W * H).fill(255);
  for (const [x0, x1] of [[8, 24], [38, 54], [68, 84], [98, 114]]) {
    for (let y = 4; y < H - 4; y++) for (let x = x0; x < x1; x++) px[y * W + x] = 0;
  }
  const slots = splitSlotsProjection(px, W, H, 4);
  assert.equal(slots.length, 4);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(slots[i].x0 - [8, 38, 68, 98][i]) <= 2, `slot ${i} x0`);
    assert.ok(Math.abs(slots[i].x1 - [24, 54, 84, 114][i]) <= 2, `slot ${i} x1`);
  }
}
// --- blank strip -> no slots, never throws ---
assert.deepEqual(splitSlotsProjection(new Uint8Array(50 * 20).fill(255), 50, 20, 8), []);

// --- fixed 8+2 splitter on a synthetic YYYY/MM/DD strip ---
{
  const W = 220;
  const H = 24;
  const px = new Uint8Array(W * H).fill(255);
  // 4 digit bars (16px) + slash (6px) + 2 digits + slash + 2 digits, 8px gaps
  const bars = [];
  let x = 6;
  const put = (w) => { bars.push([x, x + w]); x += w + 8; };
  [16, 16, 16, 16].forEach(put);
  const slash1 = [x, x + 6]; x += 6 + 8;
  [16, 16].forEach(put);
  const slash2 = [x, x + 6]; x += 6 + 8;
  [16, 16].forEach(put);
  for (const [x0, x1] of [...bars, slash1, slash2]) {
    for (let y = 4; y < H - 4; y++) for (let xx = x0; xx < x1; xx++) px[y * W + xx] = 0;
  }
  const r = splitDateSlots(px, W, H);
  assert.equal(r.fixed, true);
  assert.equal(r.digits.length, 8);
  assert.equal(r.slashes.length, 2);
  r.digits.forEach((s, i) => {
    assert.ok(Math.abs(s.x0 - bars[i][0]) <= 2, `digit ${i} x0`);
    assert.ok(Math.abs(s.x1 - bars[i][1]) <= 2, `digit ${i} x1`);
  });
  assert.ok(Math.abs(r.slashes[0].x0 - slash1[0]) <= 2);
  assert.ok(Math.abs(r.slashes[1].x0 - slash2[0]) <= 2);
  // merged strip (no clean 11 runs) falls back gracefully
  const blob = new Uint8Array(W * H).fill(255);
  for (let y = 4; y < H - 4; y++) for (let xx = 10; xx < 200; xx++) blob[y * W + xx] = 0;
  const fb = splitDateSlots(blob, W, H);
  assert.equal(fb.fixed, false);
  assert.equal(fb.digits.length, 8);
  assert.equal(fb.slashes.length, 0);
}

// --- fixed splitter on the REAL card strip (1380/10/18 fixture) ---
{
  const { loadImage, createCanvas } = await import("@napi-rs/canvas");
  const img = await loadImage(path.join(here, "fixtures/real/date_13801018.png"));
  const c = createCanvas(img.width, img.height);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  const { data } = g.getImageData(0, 0, img.width, img.height);
  const px = new Float32Array(img.width * img.height);
  for (let i = 0; i < px.length; i++) px[i] = data[i * 4];
  const mean = px.reduce((s, v) => s + v, 0) / px.length;
  if (mean < 110) for (let i = 0; i < px.length; i++) px[i] = 255 - px[i];
  const r = splitDateSlots(px, img.width, img.height);
  assert.equal(r.fixed, true);
  assert.equal(r.digits.length, 8);
  assert.equal(r.slashes.length, 2);
  // slash centers match the labeled slices (x from glyph filenames + w/2)
  const lab = JSON.parse(fs.readFileSync(
    path.join(here, "../training-tools/glyphs/glyphs_card1380.json"), "utf8"));
  const slashX = [];
  for (const [label, items] of Object.entries(lab.digits)) {
    if (label !== "slash") continue;
    for (const it of items) {
      const m = /_(\d{4})\.png$/.exec(it.png);
      slashX.push(Number(m[1]) + it.w / 2);
    }
  }
  slashX.sort((a, b) => a - b);
  r.slashes.forEach((s, i) => {
    const cx = (s.x0 + s.x1) / 2;
    assert.ok(Math.abs(cx - slashX[i]) <= 8, `slash ${i} center ${cx} vs labeled ${slashX[i]}`);
  });
  // all ranges sorted and non-overlapping
  const all = [...r.digits.map((s) => ({ ...s, k: "d" })), ...r.slashes.map((s) => ({ ...s, k: "s" }))]
    .sort((a, b) => a.x0 - b.x0);
  assert.equal(all.length, 10);
  for (let i = 1; i < all.length; i++) assert.ok(all[i].x0 >= all[i - 1].x1 - 1);
}

// --- matchTemplate matrix on synthetic shapes (circle template set) ---
const require = createRequire(import.meta.url);
let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv;
else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

const mk = (draw) => {
  const m = new cv.Mat(48, 32, cv.CV_8UC1, new cv.Scalar(255));
  draw(m);
  return m;
};
const circle = () => mk((m) => cv.circle(m, new cv.Point(16, 24), 12, new cv.Scalar(0), 3));
const square = () => mk((m) => cv.rectangle(m, new cv.Point(4, 12), new cv.Point(28, 36), new cv.Scalar(0), 3));
const templates = { 0: circle(), 5: square() };
try {
  // query = circle -> template "0" must outscore "5" by a clear margin
  const q = circle();
  try {
    const ranked = rankTemplates(cv, q, templates);
    assert.equal(ranked[0][0], "0");
    assert.ok(ranked[0][1] - ranked[1][1] > 0.15, `margin ${ranked[0][1] - ranked[1][1]}`);
  } finally {
    q.delete();
  }
  // query = square -> "5" wins
  const q2 = square();
  try {
    assert.equal(rankTemplates(cv, q2, templates)[0][0], "5");
  } finally {
    q2.delete();
  }
  // no templates -> empty ranking, never throws
  const q3 = circle();
  try {
    assert.deepEqual(rankTemplates(cv, q3, {}), []);
  } finally {
    q3.delete();
  }
} finally {
  templates[0].delete();
  templates[5].delete();
}
console.log("shape-eye tests passed");
