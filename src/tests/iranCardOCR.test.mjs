// node src/tests/iranCardOCR.test.mjs        (or: npm run test:card)
//
// Phase 1 — pure logic (no network, no OpenCV): Jalali validation,
//   positional repair, segment alignment, 4-2-2 pattern search.
// Phase 2 — full pipeline on synthetic card images rendered with the repo
//   fonts: rectification, pattern detection, per-glyph Tesseract OCR,
//   upside-down retry, graceful failure. Needs @napi-rs/canvas and
//   @tesseract.js-data/fas (both devDependencies — local files, no CDN);
//   without them phase 2 is skipped with a notice.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");

const {
  isValidJalaliDate,
  isJalaliLeapYear,
  invalidDateFields,
  chooseDigit,
  alignSegmentDigits,
  findDatePattern,
  findDateInRow,
  clusterByGaps,
  groupIntoRows,
} = await import("../ocr/iranCardOCR.js");

/* ====================================================================== */
/* Phase 1 — pure                                                          */
/* ====================================================================== */

// --- Jalali validation (leap years verified against jalaali-js) ---------
assert.equal(isValidJalaliDate(1366, 6, 22), true);
assert.equal(isValidJalaliDate(1375, 12, 30), true);   // leap Esfand
assert.equal(isValidJalaliDate(1376, 12, 30), false);  // not leap
assert.equal(isValidJalaliDate(1403, 12, 30), true);   // leap
assert.equal(isValidJalaliDate(1402, 12, 30), false);  // not leap
assert.equal(isValidJalaliDate(1370, 6, 31), true);
assert.equal(isValidJalaliDate(1375, 7, 31), false);   // Mehr has 30 days
assert.equal(isValidJalaliDate(1366, 13, 22), false);  // month out of range
assert.equal(isValidJalaliDate(1366, 6, 32), false);
assert.equal(isValidJalaliDate(1299, 6, 22), false);   // below year floor
assert.equal(isValidJalaliDate(1421, 6, 22), false);   // above year cap
assert.equal(isValidJalaliDate(1366.5, 6, 22), false); // non-integer
assert.ok(isJalaliLeapYear(1375) && isJalaliLeapYear(1403) && isJalaliLeapYear(1399));
assert.ok(!isJalaliLeapYear(1376) && !isJalaliLeapYear(1402));

// --- suspect-field aim for the validation-driven repair round -----------
assert.deepEqual(invalidDateFields(1403, 12, 30), []);       // leap Esfand OK
assert.deepEqual(invalidDateFields(1466, 6, 22), ["year"]);
assert.deepEqual(invalidDateFields(1403, 18, 30), ["month"]); // day check skipped
assert.deepEqual(invalidDateFields(1370, 7, 32), ["day"]);
assert.deepEqual(invalidDateFields(1402, 12, 30), ["day", "year"]); // non-leap Esfand-30

// --- positional repair through the confusion table ---------------------
assert.deepEqual(
  chooseDigit([{ digit: "7", conf: 60 }], ["1"]),
  { digit: "1", conf: 48, repaired: true },           // 7 -> 1 (year first digit)
);
assert.deepEqual(
  chooseDigit([{ digit: "9", conf: 90 }], null),
  { digit: "9", conf: 90, repaired: false },          // free position keeps the read
);
assert.deepEqual(
  chooseDigit([{ digit: "9", conf: 90 }, { digit: "4", conf: 50 }], ["3", "4"]),
  { digit: "4", conf: 50, repaired: false },          // direct allowed source beats a repair
);
assert.equal(chooseDigit([{ digit: "6", conf: 70 }], ["1"]), null); // no repair path -> refuse
assert.deepEqual(
  chooseDigit([null, { digit: "0", conf: 92 }], ["0", "1"]),
  { digit: "0", conf: 92, repaired: false },          // zero-dot source used when glyph read failed
);
assert.equal(chooseDigit([null, null], ["0", "1"]), null); // no signal -> never guess

// --- segment alignment --------------------------------------------------
assert.deepEqual(alignSegmentDigits("06", 2, true, false), ["0", "6"]);
assert.deepEqual(alignSegmentDigits("6", 2, true, false), [null, "6"]);   // dropped leading zero
assert.deepEqual(alignSegmentDigits("1366", 4, false, false), ["1", "3", "6", "6"]);
assert.deepEqual(alignSegmentDigits("012", 2, false, false), [null, null]); // unusable length

// --- glyph row helpers --------------------------------------------------
const g = (x, w, h, extra = {}) => ({
  x, right: x + w, y: 0, bottom: h, width: w, height: h,
  centerX: x + w / 2, centerY: h / 2,
  isSeparator: false, isZeroDot: false, ...extra,
});

// Digit metrics taken from the font measurements (heights ≈ 60px, gaps in
// fractions of the digit height) for "۱۳۶۶/۰۶/۲۲".
const refH = 60;
const mkRow = (spec) => {
  let x = 0;
  return spec.map(([w, extra]) => {
    const box = g(x, w, extra?.height ?? refH, extra);
    x += w + (extra?.gap ?? 8);
    return box;
  });
};

// Tight print: separators attached; only the digits-only clustering splits it.
{
  const row = mkRow([
    [12, { gap: 26 }],            // ۱  (narrow, gap ≈ 0.43h)
    [54, { gap: 14 }],            // ۳
    [35, { gap: 24 }],            // ۶
    [35, { gap: 17 }],            // ۶
    [39, { isSeparator: true, gap: 18 }],  // /
    [31, { height: 30, isZeroDot: true, gap: 18 }], // ۰ (short)
    [35, { gap: 26 }],            // ۶
    [38, { isSeparator: true, gap: 17 }],  // /
    [34, { gap: 17 }],            // ۲
    [34, { gap: 17 }],            // ۲
  ]);
  const digitsOnly = row.filter((b) => !b.isSeparator);
  let found = null;
  for (const factor of [0.5, 0.65, 0.8, 0.95, 1.1]) {
    const clusters = clusterByGaps(digitsOnly, refH, factor);
    found = findDatePattern(clusters);
    if (found) break;
  }
  assert.ok(found, "tight-print row must match via digits-only clustering");
  assert.equal(found.reversed, false);
  assert.equal(found.groups.year.length, 4);
  assert.equal(found.groups.month.length, 2);
  assert.equal(found.groups.day.length, 2);
  assert.equal(found.groups.month[0].isZeroDot, true);
}

// Loose print: "/" as its own cluster → [4][/][2][/][2] on the raw row.
{
  const row = mkRow([
    [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 40 }],
    [30, { isSeparator: true, height: 60, gap: 40 }],
    [40, { gap: 6 }], [40, { gap: 40 }],
    [30, { isSeparator: true, height: 60, gap: 40 }],
    [40, { gap: 6 }], [40, { gap: 6 }],
  ]);
  const clusters = clusterByGaps(row, refH, 0.65);
  const found = findDatePattern(clusters);
  assert.ok(found, "loose print must match with standalone separators");
  assert.equal(found.groups.year.length, 4);
  assert.equal(found.groups.day.length, 2);
}

// Upside-down capture: clusters present right-to-left → reversed flag.
{
  const row = mkRow([
    [40, { gap: 6 }], [40, { gap: 45 }],   // day (was rightmost)
    [30, { isSeparator: true, height: 60, gap: 45 }],
    [40, { gap: 6 }], [40, { gap: 45 }],   // month
    [30, { isSeparator: true, height: 60, gap: 45 }],
    [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 6 }], // year
  ]);
  const clusters = clusterByGaps(row, refH, 0.65);
  const found = findDatePattern(clusters);
  assert.ok(found, "reversed row must match");
  assert.equal(found.reversed, true);
  assert.equal(found.groups.year.length, 4);
}

// Label junk before the date must not break the windowed search
// (full attempt matrix, exactly as detection runs it).
{
  const junk = [g(0, 18, 12), g(30, 22, 14), g(70, 15, 10), g(110, 25, 16)];
  const date = mkRow([
    [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 6 }], [40, { gap: 35 }],
    [30, { isSeparator: true, height: 60, gap: 35 }],
    [40, { gap: 6 }], [40, { gap: 35 }],
    [30, { isSeparator: true, height: 60, gap: 35 }],
    [40, { gap: 6 }], [40, { gap: 6 }],
  ]);
  date.forEach((b) => { b.x += 300; b.right += 300; b.centerX += 300; });
  const row = [...junk, ...date].sort((a, b) => a.x - b.x);
  const matches = findDateInRow(row, refH);
  assert.ok(matches.length > 0, "leading label glyphs must not hide the date window");
  assert.equal(matches[0].groups.year[0].x, 300);
}

// Row grouping: digits share a row, stray dots above do not.
{
  const digits = [0, 46, 92, 138].map((x) => g(x, 40, 60));
  const dots = [10, 60, 110].map((x) => ({ ...g(x, 9, 9), centerY: 4, y: 0, bottom: 9 }));
  const rows = groupIntoRows([...digits, ...dots], refH);
  assert.equal(rows.length, 2);
  assert.equal(rows.find((r) => r.glyphs.length === 4)?.glyphs.length, 4);
}

console.log("phase 1 (pure) passed");

/* ====================================================================== */
/* Phase 2 — full pipeline on synthetic cards                              */
/* ====================================================================== */

let fixtures;
try {
  fixtures = await import("./fixtures.mjs");
} catch {
  console.log("phase 2 skipped: @napi-rs/canvas not installed");
  process.exit(0);
}

let fasDir = null;
try {
  const pkg = require.resolve("@tesseract.js-data/fas/package.json");
  fasDir = path.join(path.dirname(pkg), "4.0.0_best_int");
} catch {
  console.log("phase 2 skipped: @tesseract.js-data/fas not installed");
  process.exit(0);
}

const { loadImage } = await import("@napi-rs/canvas");
const { renderCard, toMat } = fixtures;
const { IranCardOCR } = await import("../ocr/iranCardOCR.js");

const ocr = new IranCardOCR({ langPath: fasDir });
const results = [];
const timed = async (name, fn) => {
  const t0 = Date.now();
  const out = await fn();
  const ms = Date.now() - t0;
  results.push({ name, ms, ...out });
  console.log(
    `  ${out.ok ? "OK  " : "FAIL"} ${name.padEnd(34)} ${String(ms).padStart(5)}ms` +
    (out.detail ? `  ${out.detail}` : ""),
  );
  if (!out.ok && out.attempts) {
    for (const a of out.attempts) {
      console.log("    attempt:", JSON.stringify({
        raw: a.rawText, note: a.note, conf: a.confidence,
        glyphReads: a.glyphReads, segReads: a.segReads,
      }));
    }
  }
  assert.equal(out.ok, true, `${name}: ${out.detail ?? ""}`);
};

console.log("phase 2 (synthetic cards):");

// Case A — full photo, card at a slight angle, Vazirmatn.
await timed("photo card (angled, Vazir)", async () => {
  const canvas = renderCard({ font: "Vazir", date: [1366, 6, 22], mode: "photo" });
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success && r.birthDate === "1366/06/22" && r.year === 1366 && r.month === 6 && r.day === 22 &&
        r.confidence >= 40 && typeof r.durationMs === "number" && (r.ocrCalls ?? 99) <= 24,
      detail: r.success
        ? `${r.birthDate} conf=${r.confidence} calls=${r.ocrCalls}`
        : r.error,
      attempts: r.attempts,
    };
  } finally { mat.delete(); }
});

// Case B — tight crop around the date line only (no card quadrilateral).
await timed("date crop (no quad, Yekan)", async () => {
  const canvas = renderCard({ font: "Yekan", date: [1375, 5, 12], mode: "crop" });
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success && r.birthDate === "1375/05/12",
      detail: r.success ? `${r.birthDate} conf=${r.confidence} calls=${r.ocrCalls}` : r.error,
      attempts: r.attempts,
    };
  } finally { mat.delete(); }
});

// Case C — whole photo upside down (180°): reversed-pattern + rotate retry.
await timed("photo upside-down (180°)", async () => {
  const canvas = renderCard({ font: "Vazir", date: [1391, 11, 3], mode: "photo", rotate: 180 });
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success && r.birthDate === "1391/11/03",
      detail: r.success ? `${r.birthDate} conf=${r.confidence} calls=${r.ocrCalls}` : r.error,
      attempts: r.attempts,
    };
  } finally { mat.delete(); }
});

// Case D — third font + leap Esfand date (year '۴', month '۲', day '۳۰').
await timed("photo card (Iranian Sans, 1403)", async () => {
  const canvas = renderCard({ font: "IranianSans", date: [1403, 12, 30], mode: "photo" });
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success && r.birthDate === "1403/12/30",
      detail: r.success ? `${r.birthDate} conf=${r.confidence} calls=${r.ocrCalls}` : r.error,
      attempts: r.attempts,
    };
  } finally { mat.delete(); }
});

// Case E — card without a birth date must fail gracefully, never throw.
await timed("no date present (graceful fail)", async () => {
  const canvas = renderCard({ font: "Vazir", mode: "nodate" });
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success === false && typeof r.error === "string" && r.error.length > 0,
      detail: `error="${r.error}" calls=${r.ocrCalls}`,
    };
  } finally { mat.delete(); }
});

// Case F — featureless image (no card, no ink at all) must fail, not hang.
await timed("blank image (graceful fail)", async () => {
  const { createCanvas } = await import("@napi-rs/canvas");
  const canvas = createCanvas(640, 400);
  const g = canvas.getContext("2d");
  g.fillStyle = "#808080";
  g.fillRect(0, 0, 640, 400);
  const mat = await toMat(canvas);
  try {
    const r = await ocr.recognizeBirthDate(mat);
    return {
      ok: r.success === false && typeof r.error === "string" && r.error.length > 0,
      detail: `error="${r.error}"`,
    };
  } finally { mat.delete(); }
});

// Case G — unsupported input type in Node must not throw.
await timed("bad input (no throw)", async () => {
  const r = await ocr.recognizeBirthDate("data:image/png;base64,AAAA");
  return {
    ok: r.success === false && typeof r.error === "string",
    detail: `error="${r.error}"`,
  };
});

await ocr.terminate();

const total = results.reduce((s, r) => s + r.ms, 0);
console.log(`phase 2 passed — ${results.length} cases, ${total}ms total`);
