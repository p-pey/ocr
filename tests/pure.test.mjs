// node tests/pure.test.mjs   (from the project root; no dependencies)
import assert from "node:assert/strict";
import fs from "node:fs";
import { parseJalaliDate, repairStructure, isJalaliLeapYear } from "../src/ocr/dateParse.js";
import { recognizeLines, _forwardRaw, decodeProbs, averageReads } from "../src/ocr/cnn.js";

// --- Jalali parsing / validation ---
assert.equal(parseJalaliDate("۱۳۷۵/۰۵/۱۲").formatted, "1375/05/12");
assert.equal(parseJalaliDate("13750512").formatted, "1375/05/12");      // dropped slashes
assert.equal(parseJalaliDate("12/05/1375").formatted, "1375/05/12");    // visual order flipped
assert.equal(parseJalaliDate("1375/13/01"), null);                       // month 13
assert.equal(parseJalaliDate("1375/07/31"), null);                       // Mehr has 30 days
assert.equal(parseJalaliDate("1375/06/31").formatted, "1375/06/31");     // Shahrivar has 31
assert.equal(parseJalaliDate("1375/12/30").formatted, "1375/12/30");     // 1375 is leap
assert.equal(parseJalaliDate("1376/12/30"), null);                       // 1376 is not
assert.equal(parseJalaliDate("1500/01/01"), null);
assert.ok(isJalaliLeapYear(1375) && isJalaliLeapYear(1403) && !isJalaliLeapYear(1402));
assert.equal(repairStructure("۱۳۷۵ / ۰۵ / ۱۲ ").length, 10);

// --- TTA averaging ---
const p = (d) => { const a = new Float32Array(80).fill(0.01); [...d].forEach((c, h) => { a[h * 10 + Number(c)] = 0.91; }); return a; };
const merged = averageReads([decodeProbs(p("13750512"), 0.9), decodeProbs(p("13750512"), 0.8)]);
assert.equal(merged.text, "1375/05/12");

// --- CNN parity with the Python/TensorFlow reference (quantised weights) ---
const fx = JSON.parse(fs.readFileSync(new URL("./fixtures/parity.json", import.meta.url)));
let maxDiff = 0;
fx.inputs.forEach((inp, i) => {
  const raw = _forwardRaw(Float32Array.from(inp));
  fx.logits[i].forEach((v, k) => { maxDiff = Math.max(maxDiff, Math.abs(v - raw.logits[k])); });
  maxDiff = Math.max(maxDiff, Math.abs(fx.dateLogit[i] - raw.dateLogit));
});
console.log("max |JS - reference| logit diff:", maxDiff.toFixed(5));
assert.ok(maxDiff < 5e-3, "JS inference does not match the reference network");
const t = performance.now();
for (let i = 0; i < 20; i++) recognizeLines([Float32Array.from(fx.inputs[0])]);
console.log("CNN speed (Node):", ((performance.now() - t) / 20).toFixed(1), "ms per line");
console.log("pure tests passed");
