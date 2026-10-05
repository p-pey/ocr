# Iranian national card – birth-date OCR (pure JavaScript)

Extracts the Jalali birth date (`YYYY/MM/DD`) from a photo or scan of an Iranian
national card. **No training, no build step, no custom model files** — install two
packages and import one file:

```
npm install tesseract.js @techstark/opencv-js
```

```js
import BirthDateOCR, { IranCardOCR } from "./src/ocr/iranCardOCR.js";

const ocr = new BirthDateOCR();          // or: new IranCardOCR()
const result = await ocr.recognizeBirthDate(imageSrc /* data URL, <img>, canvas, cv.Mat */);

if (result.success) {
  console.log(result.birthDate);  // "1366/06/22"
  console.log(result.confidence); // 0-100
} else {
  console.log(result.error);      // never throws past recognize()
}
await ocr.terminate();            // call when shutting down (one reused worker)
```

In the browser the stock `fas.traineddata` is fetched from the Tesseract CDN
(`@tesseract.js-data/fas` on jsDelivr) and cached by tesseract.js. To self-host,
pass `langPath`:

```js
new BirthDateOCR({ langPath: "/tessdata" });  // expects fas.traineddata[.gz]
```

## How it works

| Stage | What it does |
|---|---|
| Card detection | OpenCV quadrilateral detection → perspective rectification to a 1200×756 work image (a tight crop without a card quad is used as-is; upside-down captures rotate once and re-detect) |
| Date geometry | Otsu ink, glyph contours, center band 0.22–0.78, row grouping, 4-2-2 pattern search with `/` separator gates, zero-dot ۰ detection, Jalali-validated positional rules |
| Glyph OCR | Each of the 8 digits is rendered alone (isolation removes any neighbour fragments) and read by tesseract.js `fas` in SINGLE_WORD mode; any read that is unusable, low-confidence, or violates its position's allowed digits gets a SINGLE_LINE second opinion |
| Reconcile | Positional rules → confusion-table repair (≤1 repair accepted) → `isValidJalaliDate`; candidates are verified in score order (separator evidence outranks look-alike ID rows). If the date fails validation, only the suspect field's glyphs are re-read from an Otsu-binarized rendering (one repair round) and reconciled again |
| Failure | Always returns `{ success:false, error, durationMs, attempts }` — never throws |

The fast path is 8–10 OCR calls (one per glyph + a couple of second opinions);
the hard budget is 24 calls per image across candidates and the flip retry.

## Result contract

```js
// success
{ success: true, birthDate: "1366/06/22", year: 1366, month: 6, day: 22,
  confidence: 86, repairs: 0, durationMs, ocrCalls, attempts, lineImage }

// failure — same shape as the spec: never throws
{ success: false, error: "…", durationMs, attempts }
```

Each `attempts[i]` carries `strategy`, `engine`, `candidateIndex`, `rotation`,
`bounds`, `rawText`, `confidence`, `repairs`, `glyphReads`, `segReads`,
`segmentImages.line` and `preprocessedImage` for debugging/progress UIs.

## Files

```
src/ocr/iranCardOCR.js            ★ the deliverable — single-file engine (no build needed)
src/tests/iranCardOCR.test.mjs    engine test suite: node src/tests/iranCardOCR.test.mjs
src/tests/fixtures.mjs            synthetic-card renderer shared with the tests
src/hooks/useOCR.js               React adapter (feeds ResultDisplay's {best, allDates, allAttempts})
src/components/App.jsx            upload → crop → result flow
src/ocr/TesseractOCR.js           legacy CNN engine (optional; kept for npm test / integration)
src/train/train.mjs               legacy CNN training script (not needed by the new engine)
```

## Tests

```
npm test                          pure helpers (legacy suite — stays green)
node src/tests/iranCardOCR.test.mjs
```

The engine suite runs in two phases:

1. **Pure** — Jalali validation (leap years verified against `jalaali-js`),
   positional repair, segment alignment, 4-2-2 pattern search.
2. **Full pipeline** — synthetic cards rendered with the repo fonts through the
   real OpenCV + Tesseract path: angled photo card, tight crop with no card quad
   (Yekan), upside-down photo (180°), a card without a date (graceful failure)
   and a non-image input (no throw).

Phase 2 needs the devDependencies `@napi-rs/canvas` and `@tesseract.js-data/fas`
(both local files — no network); without them it skips with a notice.

## Setup

```
npm install --ignore-scripts        # tfjs-node postinstall needs network; not required
node src/tests/iranCardOCR.test.mjs
npm run build                       # vite build passes (engine is plain JS)
```

`OCR_TRACE=1` prints detection diagnostics (rows, candidates, per-position OCR
decisions) on stderr — Node only, never enabled in browsers.

## Notes for production

* **One worker, reused.** `initialize()` warms the Tesseract worker; call
  `terminate()` only on shutdown.
* **All OpenCV Mats are freed** in `finally` blocks; `recognize()` never throws.
* **Self-host the language data** if the CDN is unreachable: download
  `fas.traineddata` from `@tesseract.js-data/fas` and pass `langPath` — that is
  the only optional file (≈1–2 MB), still no build step.
* For live camera use, run several frames and accept a date only when 2+ agree.

## What has and has not been verified

Verified in this repo:

* Phase-1 pure tests and the seven phase-2 pipeline cases, deterministic across
  repeated runs (angled photo → `1366/06/22` conf 86 in 9 OCR calls; crop →
  `1375/05/12`; upside-down → `1391/11/03`; Iranian Sans → `1403/12/30` via the
  validation repair round; no-date, blank image and bad input fail safely).
* `npm test` (legacy pure suite) and `npm run build` pass.
* Browser input path (data URL / `<img>` / canvas / Blob), CDN + local `langPath`.

Not verified: accuracy on **real card photos** (the suite uses synthetic cards
rendered with Vazirmatn/Yekan/Iranian Sans). The engine's failure modes are explicit
(`attempts` shows exactly which glyph read failed), so real-world tuning is
observation-driven: run `OCR_TRACE=1`, look at the dumped reads, adjust
`GLYPH_MIN_CONFIDENCE` / rule tables if a font misbehaves.

## Legacy CNN engine (optional)

`src/ocr/TesseractOCR.js` + `src/train/train.mjs` are the previous TensorFlow.js
line-CNN engine with its own training pipeline (`npm run train`). The React app
now uses the new engine; the legacy engine and `public/models/date_cnn/` remain
for the old `npm test` / `npm run test:integration` workflows.
