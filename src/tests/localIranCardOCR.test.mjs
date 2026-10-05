// Local browser engine regression tests.
// Run: node src/tests/localIranCardOCR.test.mjs
// Synthetic fixtures only; these tests are not a real-card accuracy benchmark.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";
import { parseBirthDateLine, findDateTextRows, LocalIranCardOCR } from "../ocr/localIranCardOCR.js";

const require = createRequire(import.meta.url);

// Strict parser: accept valid Jalali formats but never trim digits from an
// unrelated national-ID/serial line or repair a digit by guessing.
assert.equal(parseBirthDateLine("۱۳۷۵/۰۵/۱۲").formatted, "1375/05/12");
assert.equal(parseBirthDateLine("١٣٧٥/٠٥/١٢").formatted, "1375/05/12");
assert.equal(parseBirthDateLine("13750512").formatted, "1375/05/12");
assert.equal(parseBirthDateLine("1375 / 5 / 2").formatted, "1375/05/02");
assert.equal(parseBirthDateLine("1375 05 12").formatted, "1375/05/12");
assert.equal(parseBirthDateLine("1375/12/30").formatted, "1375/12/30");
assert.equal(parseBirthDateLine("1376/12/30"), null); // not a leap year
assert.equal(parseBirthDateLine("1375/07/31"), null); // Mehr has 30 days
assert.equal(parseBirthDateLine("0012345678"), null); // ten-digit national ID
assert.equal(parseBirthDateLine("12/05/1375"), null); // do not guess a different order
assert.equal(parseBirthDateLine("1375/05/12 and 1404/05/15"), null); // ambiguous row

let fixtureModule;
try {
  fixtureModule = await import("./fixtures.mjs");
} catch {
  console.log("synthetic pipeline tests skipped: @napi-rs/canvas is not installed");
  process.exit(0);
}

let langPath;
try {
  langPath = path.join(
    path.dirname(require.resolve("@tesseract.js-data/fas/package.json")),
    "4.0.0_best_int",
  );
} catch {
  console.log("synthetic pipeline tests skipped: @tesseract.js-data/fas is not installed");
  process.exit(0);
}

const { renderCard } = fixtureModule;
const { createCanvas } = await import("@napi-rs/canvas");
const readCanvas = (canvas) => canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);

// The lightweight local detector should find a numeric text row without an
// OpenCV runtime. A full-crop row includes label glyphs; whitelist OCR strips them.
const detectionFixture = renderCard({ font: "Yekan", date: [1375, 5, 12], mode: "crop" });
const detectedRows = findDateTextRows(readCanvas(detectionFixture));
assert.ok(detectedRows.length > 0, "birth-date crop should contain a candidate text row");
assert.ok(detectedRows[0].digitLikeCount >= 6);

const ocr = new LocalIranCardOCR({ langPath, minConfidence: 35 });
const cases = [
  {
    name: "tight birth-date crop (Yekan)",
    image: renderCard({ font: "Yekan", date: [1375, 5, 12], mode: "crop" }),
    expected: "1375/05/12",
  },
  {
    name: "card photo (Vazirmatn)",
    image: renderCard({ font: "Vazir", date: [1366, 6, 22], mode: "photo" }),
    expected: "1366/06/22",
  },
  {
    name: "upside-down date crop",
    image: renderCard({ font: "Vazir", date: [1391, 11, 3], mode: "crop", rotate: 180 }),
    expected: "1391/11/03",
  },
  {
    name: "downscaled crop",
    image: (() => {
      const source = renderCard({ font: "Yekan", date: [1400, 11, 3], mode: "crop" });
      const small = createCanvas(460, 150);
      small.getContext("2d").drawImage(source, 0, 0, small.width, small.height);
      return small;
    })(),
    expected: "1400/11/03",
  },
];

try {
  for (const testCase of cases) {
    const result = await ocr.recognizeBirthDate(testCase.image);
    assert.equal(result.success, true, `${testCase.name}: ${result.error ?? ""}`);
    assert.equal(result.birthDate, testCase.expected, testCase.name);
    assert.equal(result.engine, "canvas+tesseract-fas");
    assert.equal(result.requiresUserConfirmation, true);
    assert.ok(result.ocrCalls <= 24, "OCR calls must stay within the per-image budget");
    assert.ok(result.confidence > 0 && result.confidence <= 100);
    assert.ok([
      "two-preprocessing-reads-agree",
      "single-high-confidence-read",
    ].includes(result.confidenceKind));
    if (result.confidenceKind === "two-preprocessing-reads-agree") assert.equal(result.agreement, true);
  }

  const noDate = await ocr.recognizeBirthDate(renderCard({ font: "Vazir", mode: "nodate" }));
  assert.equal(noDate.success, false, "an ID-only card must not be returned as a birth date");
  assert.equal(noDate.requiresUserConfirmation, false);
  assert.ok(noDate.ocrCalls <= 24, "no-date scans must stay within the per-image budget");
  assert.match(noDate.error, /date/i);

  const badInput = await ocr.recognizeBirthDate("data:image/png;base64,AAAA");
  assert.equal(badInput.success, false);
  assert.equal(typeof badInput.error, "string");
} finally {
  await ocr.terminate();
}

console.log("local Canvas + Tesseract tests passed (synthetic fixtures only)");
