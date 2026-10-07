// Real card-shape tests (directive: template slicing + shape-gate enforcer).
// Exemplars sliced from a real card date strip (training-tools/glyphs/
// card_shapes/, identity-free by construction). Asserts geometry facts,
// NCC/Hu self-consistency, the pure-Hu port vs cv.HuMoments, and the
// verifySlotShape enforcer verdicts. Needs @techstark/opencv-js +
// @napi-rs/canvas (devDependencies).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import {
  ncc,
  resizeNearest,
  huFromPixels,
  huDistance,
  registerShapeTemplates,
  loadEmbeddedShapeTemplates,
  clearShapeTemplates,
  shapeRegistryEmpty,
  verifySlotShape,
} from "../src/ocr/shapeGate.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(here, "../training-tools/glyphs/card_shapes");
const meta = JSON.parse(fs.readFileSync(path.join(dir, "card_shapes.json"), "utf8"));

// --- geometry facts: hollow-circle zero, unverified identities ---
assert.ok(meta.shapes.length >= 6);
for (const s of meta.shapes) {
  assert.equal(s.identity, null, `${s.png} must never carry a digit label`);
  if (s.shape === "hollow") {
    assert.equal(s.holes, 1, `${s.png} hollow ring must have exactly 1 hole`);
    assert.ok(s.height_ratio < 0.6, `${s.png} zero height ratio < 0.6 (spec 9.2)`);
  }
}

// --- load pixels ---
const require = createRequire(import.meta.url);
let cv = require("@techstark/opencv-js");
if (cv.then) cv = await cv;
else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));
const { loadImage, createCanvas } = await import("@napi-rs/canvas");
const load = async (fn) => {
  const img = await loadImage(path.join(dir, fn));
  const c = createCanvas(img.width, img.height);
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  const { data } = g.getImageData(0, 0, img.width, img.height);
  const px = new Float32Array(img.width * img.height);
  for (let i = 0; i < px.length; i++) px[i] = data[i * 4];
  return { px, w: img.width, h: img.height };
};
const byShape = {};
for (const s of meta.shapes) {
  (byShape[s.shape] ??= []).push({ ...s, ...(await load(s.png)) });
}
const [H1, H2] = byShape.hollow;
const [S1] = byShape.slash;
const [N1] = byShape.narrow;

// --- NCC self-consistency: hollow matches hollow, not slash/narrow ---
{
  const n = (a, b) => ncc(resizeNearest(a.px, a.w, a.h, 24, 32), resizeNearest(b.px, b.w, b.h, 24, 32));
  const hh = n(H1, H2);
  const hs = n(H1, S1);
  const hn = n(H1, N1);
  console.log(`  NCC hollow-hollow=${hh.toFixed(3)} hollow-slash=${hs.toFixed(3)} hollow-narrow=${hn.toFixed(3)}`);
  assert.ok(hh > hs + 0.15 && hh > hn + 0.15, "hollow must match hollow best with margin");
}

// --- Hu self-consistency + pure port vs cv.HuMoments ---
{
  const d = (a, b) => huDistance(huFromPixels(a.px, a.w, a.h), huFromPixels(b.px, b.w, b.h));
  const hh = d(H1, H2);
  const hs = d(H1, S1);
  const hn = d(H1, N1);
  console.log(`  Hu hollow-hollow=${hh.toFixed(3)} hollow-slash=${hs.toFixed(3)} hollow-narrow=${hn.toFixed(3)}`);
  assert.ok(hh < hs && hh < hn, "hollow must be Hu-nearest to hollow");

  // pure port validation against OpenCV on the same dark-ink pixels
  const m = new cv.Mat(48, 32, cv.CV_8UC1, new cv.Scalar(255));
  cv.circle(m, new cv.Point(16, 24), 12, new cv.Scalar(0), 2);
  const md = m.data;
  const mpx = new Float32Array(md.length);
  for (let i = 0; i < md.length; i++) mpx[i] = md[i];
  const mine = huFromPixels(mpx, 32, 48);
  const bin = new cv.Mat();
  cv.threshold(m, bin, 0, 255, cv.THRESH_BINARY_INV);
  const mo = cv.moments(bin, true);
  const cvhu = cv.Mat.zeros(7, 1, cv.CV_64F);
  cv.HuMoments(mo, cvhu);
  let maxrel = 0;
  for (let i = 0; i < 7; i++) {
    const cvv = cvhu.doubleAt(i, 0);
    const unlogged = (mine[i] >= 0 ? 1 : -1) * Math.pow(10, -Math.abs(mine[i]));
    // floor the denominator: near-zero invariants (~1e-23) only carry
    // floating-point noise; anything below 1e-12 is numerically zero.
    maxrel = Math.max(maxrel, Math.abs(unlogged - cvv) / Math.max(1e-12, Math.abs(cvv)));
  }
  console.log(`  pure-Hu vs cv.HuMoments max rel diff: ${maxrel.toExponential(2)}`);
  assert.ok(maxrel < 1e-6, "pure Hu port must match OpenCV");
  m.delete();
  bin.delete();
  cvhu.delete();
}

// --- enforcer verdicts on real shapes ---
registerShapeTemplates({
  hollow: byShape.hollow.map((t) => ({ w: t.w, h: t.h, data: t.px })),
  slash: byShape.slash.map((t) => ({ w: t.w, h: t.h, data: t.px })),
  narrow: byShape.narrow.map((t) => ({ w: t.w, h: t.h, data: t.px })),
});
try {
  assert.equal(shapeRegistryEmpty(), false);
  let v = verifySlotShape(H1.px, H1.w, H1.h, "0");
  console.log("  hollow-as-0:", JSON.stringify(v));
  assert.equal(v.conflict, false);
  v = verifySlotShape(S1.px, S1.w, S1.h, "0");
  console.log("  slash-as-0:", JSON.stringify(v));
  assert.equal(v.conflict, true, "slash slot called 0 must be vetoed");
  v = verifySlotShape(N1.px, N1.w, N1.h, "1");
  console.log("  narrow-as-1:", JSON.stringify(v));
  assert.equal(v.conflict, false);
  v = verifySlotShape(H1.px, H1.w, H1.h, "5");
  assert.equal(v.conflict, false, "unverifiable digits abstain, never veto");
} finally {
  clearShapeTemplates();
}
assert.equal(shapeRegistryEmpty(), true);

// --- embedded bundle loads ---
await loadEmbeddedShapeTemplates();
assert.equal(shapeRegistryEmpty(), false);
clearShapeTemplates();

console.log("card-shape tests passed");
