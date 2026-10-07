# Iranian national card – birth-date OCR (OpenCV + CNN, zero Tesseract)

Extracts the Jalali birth date (`YYYY/MM/DD`) from a photo or scan of an Iranian
national card. No Tesseract, no tfjs, no model hosting, no downloads at runtime —
the only runtime dependency is `@techstark/opencv-js`, with the CNN weights
embedded in the bundle:

```
npm install @techstark/opencv-js
```

```js
import { TesseractOCR } from "./src/ocr/TesseractOCR.js";

const ocr = new TesseractOCR(); // class name kept for UI compatibility
await ocr.initialize();
const result = await ocr.recognize(imageSrc, onProgress, onAttempt);

if (result.best) {
  console.log(result.best.birthDate.formatted); // "1376/07/05"
  console.log(result.best.birthDate.confidence); // 0-100
} else {
  // retake photo — a wrong date is worse than no date
}
await ocr.terminate(); // only on shutdown
```

React: `useOCR()` (`src/hooks/useOCR.js`) wraps the engine; `recognizeConsensus(frames)`
accepts a date only when ≥2 frames agree (live camera).

## How it works

| Stage | What it does |
|---|---|
| Card detection | OpenCV quadrilateral → perspective warp to 1200×756 (tight crops pass through, portrait rotated) |
| Line candidates | Search band → Gaussian → adaptive threshold → CLOSE [13,21,33] → geometry filter → NMS → top 18 by closeness to middle |
| Digit CNN | Dependency-free `cnn.js` reads each line: 8 digit heads (YYYYMMDD) + `isDate` head; int8 weights embedded in `modelWeights.js` (~107 KB raw) |
| Shape gate | Deterministic verifier: digit-slot geometry (dot-zero vs loop), projection splitting, `matchTemplate` scoring vs Persian digit templates |
| TTA + selection | Best 4 reads re-read on shifted/grown crops, probabilities averaged; **earliest confident year wins** (expiry is always later), upper row breaks ties |
| Failure | `best: null` — never a guessed date |

Birth = earliest year among confident valid dates (Rule 8.2/4.4); the birth row
sits above the expiry row (`relY` tiebreak).

## Result contract

```js
// success
{ best: { birthDate: { year, month, day, formatted, confidence, votes, ... },
          confidence, engine: "opencv+cnn", ... },
  allDates: [...], allAttempts: [...], timingMs }
// failure
{ best: null, allDates: [...], allAttempts: [...], timingMs }
```

## Files

```
src/ocr/TesseractOCR.js   engine (card + lines + selection, no Tesseract inside)
src/ocr/cnn.js            dependency-free CNN inference + decode/TTA helpers
src/ocr/modelWeights.js   embedded int8 weights (generated, do not edit)
src/ocr/dateParse.js      digit normalisation + Jalali validation (pure)
src/ocr/shapeGate.js      eye engine: slots, geometry, projection, matchTemplate, Hu
src/ocr/shapeTemplates.js embedded real-card shape exemplars (generated)
src/ocr/fieldAssign.js    birth/expiry disambiguation: anchors, Y-sort, chronology
src/ocr/consensus.js      multi-frame agreement helper
src/hooks/useOCR.js       React adapter (recognize + recognizeConsensus)
training-tools/           offline Python pipeline (fonts → data → train → export)
tests/                    pure + gate + rule82 + shapeEye + noTesseract + e2e
IMPLEMENTATION_SPEC.md    normative spec (sections 0-14)
REPORT.md                 measured acceptance numbers (honest gaps listed)
INTEGRATION_NOTE.md       one-page integration guide
```

## Tests

```
npm test                  pure + gate + rule82 + shapeEye + noTesseract
node tests/integration/engine.test.mjs <cards_dir> [...]
```

`npm run build` passes (engine is plain JS + embedded weights).

## Production notes

- `best === null` → ask for a retake. `minProb` / `gateConflict` in attempts
  can drive stricter retake thresholds.
- Every `cv.Mat` is freed; images never leave the device; no logging of dates.
- Real-card accuracy is unmeasured until user glyph sheets + real crops land
  (see REPORT.md §10.3); synthetic e2e sits at ~34% exact / ~6% wrong —
  read REPORT.md before quoting numbers.
