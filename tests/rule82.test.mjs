// Rule 8.2 / 4.4 tests (field-disambiguation directive):
// birth date (upper row, EARLIEST year) must always beat the expiry date
// (lower row, later year) — even with overlapping boxes or inverted Y.
import assert from "node:assert/strict";
import { selectBest } from "../src/ocr/TesseractOCR.js";
import { parseJalaliDate } from "../src/ocr/dateParse.js";

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

// 4. Nothing confident: highest confidence WITHIN the birth-side pool
// (years > 1400 dropped when <= 1400 exists) — expiry can no longer win
// a fallback on confidence alone.
{
  const expiry = D(1401, 11, 25, 70, 0.7, 0.78);
  const birth = D(1376, 7, 5, 65, 0.65, 0.55);
  assert.equal(selectBest([expiry, birth]).formatted, birth.formatted);
}
// 4b. No birth-side year at all: plain highest confidence, no filtering.
{
  const a = D(1409, 1, 1, 80, 0.7, 0.78);
  const b = D(1402, 5, 5, 70, 0.65, 0.55);
  assert.equal(selectBest([a, b]).formatted, a.formatted);
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

console.log("rule 8.2 tests passed");
