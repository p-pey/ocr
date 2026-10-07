// Rule 8.2 / 4.4 tests (field-disambiguation directive):
// birth date (upper row, EARLIEST year) must always beat the expiry date
// (lower row, later year) — even with overlapping boxes or inverted Y.
import assert from "node:assert/strict";
import { selectBest } from "../src/ocr/TesseractOCR.js";
import { parseJalaliDate, isValidJalaliDate } from "../src/ocr/dateParse.js";
import { rtlAnchorScore, suppressExpiryRows } from "../src/ocr/fieldAssign.js";
import { countHoles, verifyDigitTopo } from "../src/ocr/shapeGate.js";

const D = (year, month, day, confidence, dateProb, relY) => ({
  year, month, day,
  formatted: `${year}/${String(month).padStart(2, "0")}/${String(day).padStart(2, "0")}`,
  confidence, dateProb, minProb: 0.9, score: confidence, relY,
});

// 1. Birth above + earlier year beats confident expiry below — even at
// 99% expiry vs 85% birth confidence (directive §4: never confidence-picked).
{
  const birth = D(1376, 7, 5, 85, 0.9, 0.55);
  const expiry = D(1401, 11, 25, 99, 0.99, 0.78);
  assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
}

// 2. Overlapping boxes of the same row: upper box wins the tie.
{
  const lower = D(1376, 7, 5, 95, 0.95, 0.58);
  const upper = D(1376, 7, 5, 90, 0.95, 0.52);
  assert.equal(selectBest([lower, upper]).relY, 0.52);
}

// 3. Inverted Y (upside-down capture: birth BELOW expiry): year dominates.
{
  const birth = D(1376, 7, 5, 90, 0.9, 0.75);
  const expiry = D(1401, 11, 25, 96, 0.98, 0.5);
  assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
}

// 4. Nothing confident: smallest year WITHIN the birth-side pool
// (MASTER SPEC §2.2: never confidence-picked — years > 1400 dropped when
// <= 1400 exists, then smallest year wins outright).
{
  const expiry = D(1401, 11, 25, 70, 0.7, 0.78);
  const birth = D(1376, 7, 5, 65, 0.65, 0.55);
  assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
}
// 4b. No birth-side year at all: smallest year, never highest confidence.
{
  const a = D(1409, 1, 1, 80, 0.7, 0.78);
  const b = D(1402, 5, 5, 70, 0.65, 0.55);
  assert.equal(selectBest([a, b]).formatted, b.formatted);
}
// 4c. Boundary: 1400 is kept, 1401 dropped when a younger year exists.
{
  const edge = D(1400, 1, 1, 90, 0.9, 0.6);
  const over = D(1401, 1, 1, 95, 0.95, 0.5);
  assert.equal(selectBest([over, edge]).formatted, edge.formatted);
}
// 4d. Dev trace flag never affects the verdict.
{
  process.env.OCR_SELECTION_TRACE = "1";
  try {
    const birth = D(1376, 7, 5, 85, 0.9, 0.55);
    const expiry = D(1401, 11, 25, 99, 0.99, 0.78);
    assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
  } finally {
    delete process.env.OCR_SELECTION_TRACE;
  }
}

// 5. Empty -> null; missing relY treated as bottom but year still wins.
assert.equal(selectBest([]), null);
{
  const birth = { ...D(1376, 7, 5, 90, 0.9, undefined), relY: undefined };
  const expiry = D(1401, 11, 25, 90, 0.9, 0.4);
  assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
}

// 6. Directive 1 formats: YYYY-MM-DD with Persian digits parses too.
assert.equal(parseJalaliDate("۱۳۷۶-۰۷-۰۵").formatted, "1376/07/05");
assert.equal(parseJalaliDate("1376/07/05").formatted, "1376/07/05");

// 7. MASTER SPEC §2.2 example: 1360/03/10 beats 1402/04/01 no matter what.
{
  const old = D(1360, 3, 10, 62, 0.61, 0.6);
  const young = D(1402, 4, 1, 99, 0.99, 0.4);
  assert.equal(selectBest([young, old]).formatted, old.formatted);
}

// 8. MASTER SPEC §7 strict enforcer: impossible dates rejected upstream
// (parseJalaliDate returns null; isValidJalaliDate is the same gate).
{
  assert.equal(parseJalaliDate("1311/11/36"), null);
  assert.equal(isValidJalaliDate(1311, 11, 36), false);
  assert.equal(isValidJalaliDate(1363, 1, 1), true);
  assert.equal(isValidJalaliDate(1289, 1, 1), false);
  assert.equal(isValidJalaliDate(1411, 1, 1), false);
}

// 9. MASTER SPEC §2.1 RTL helpers: right-anchored beats left-anchored,
// expiry rows below the birth row are suppressed.
{
  const right = rtlAnchorScore({ x: 700, width: 400 }, 1200, 756);
  const left = rtlAnchorScore({ x: 10, width: 200 }, 1200, 756);
  assert.ok(right > left, `rtl right=${right} left=${left}`);
  const rows = [
    { formatted: "1360/03/10", year: 1360, yMin: 0.50 },
    { formatted: "1402/04/01", year: 1402, yMin: 0.72 },
  ];
  assert.equal(suppressExpiryRows(rows, 0.50).length, 1);
  assert.equal(suppressExpiryRows(rows, 0.50)[0].year, 1360);
}

// 10. MASTER SPEC §6 Shape Gate topology (pure pixels, no OpenCV).
{
  const ring = (s, hole) => {
    const px = new Float32Array(s * s).fill(255);
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
      const edge = x === 0 || y === 0 || x === s - 1 || y === s - 1;
      const h = x >= hole && x < s - hole && y >= hole && y < s - hole;
      px[y * s + x] = edge || !h ? 0 : (h ? 255 : 0);
      if (!edge && h) px[y * s + x] = 255;
      if (edge) px[y * s + x] = 0;
    }
    return px;
  };
  const s = 24;
  const hollow = ring(s, 7);
  assert.equal(countHoles(hollow, s, s).holes, 1);
  assert.equal(verifyDigitTopo(hollow, s, s, "0").conflict, false);
  const solid = new Float32Array(s * s).fill(255);
  for (let y = 2; y < s - 2; y++) for (let x = 9; x < 15; x++) solid[y * s + x] = 0;
  const v1 = verifyDigitTopo(solid, s, s, "9");
  assert.equal(v1.conflict, true, "solid bar as 9 must conflict (no loop)");
}

console.log("rule 8.2 tests passed");
