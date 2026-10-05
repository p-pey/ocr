# Iranian national card – birth-date OCR (pure JavaScript)

Drop-in replacement for the old `TesseractOCR` engine: same class name, same
`initialize / recognize(imageSrc, onProgress, onAttempt) / terminate` API, same
result shape. **No Python anywhere**: inference runs in the browser with
TensorFlow.js, training is a Node script.

## What changed and why

| Old engine | New engine |
|---|---|
| Tesseract (`fas`) on every glyph, ~33 OCR calls per image | A small CNN (TensorFlow.js) reads the whole date line in one batched call |
| Hand-tuned digit-shape rules (zero-dot, hole counts, confusion tables) | The model learns the Persian digit shapes (incl. the dot-zero ۰) from data |
| Fragile 4-2-2 glyph clustering to find the date | Generic text-line finder in a search band; the date is whichever line reads as a **valid Jalali date** |
| Silent digit guessing (`padStart("0")`, confusion "repair") | Never guesses digits. Only fixes structure. Not confident -> `best: null` (ask for a retake) |

How the model works: the birth date is always `YYYY/MM/DD`, zero padded, i.e. exactly
8 digits. The CNN looks at one line crop and predicts all 8 digits at once (8 softmax
heads) plus an `isDate` score. `isDate` keeps a 10-digit national-ID line, a name or a
blank crop from being read as a date. No segmentation, no CTC.
(Limitation: dates printed *without* zero padding, like `1375/5/2`, are not supported by
this model. If your cards do that, tell me and I'll switch to a variable-length decoder.)

## Why output used to be confidently wrong (fixed)

The 8-head dense design has no translation invariance: trained only on tight,
centered crops, it memorised digit *positions* and misfired with ~99% confidence on
any real detector crop (different padding/position/scale) — even ones that look
identical to a human. Two defences now work together:

* **Training** (`src/train/train.mjs`): every synthetic sample is pasted onto a larger
  paper sheet with random asymmetric margins (0–60% per side), scale (0.7–1.25) and
  random placement, plus cut-off neighbour-word fragments at the edges — i.e. the
  model trains on detector-like crops, not tight boxes. 40% of date positives carry
  a full card-field label next to the date (`label + date` and `date + label`, both
  orders) so the heads learn to find the date amid label text. Hard negatives were
  added: `label + national-ID`, `label + impossible date` (month 13–19 / day 32–39),
  truncated dates, and `label + word`, so `isDate` rejects the real confusers.
  `preview.png` honestly shows these inputs — dates must appear at random positions
  with big margins, not centered.
* **Engine** (`src/ocr/TesseractOCR.js`): each detected line is also scanned with
  overlapping date-sized sliding windows (`line-cnn` + `window-cnn` strategies, one
  batched model call). A date is accepted only with **≥2 agreeing boxes or one box
  ≥90% confidence** (plus the 60% floor); lone weak reads mean `best: null` (retake),
  because misfires scatter across windows instead of agreeing. The `isDate` head is
  recorded for debugging but is NOT a gate — measured on label-adjacent windows it
  once vetoed perfectly-read dates, so Jalali validation + agreement decide.
* **Fine-tuning**: `node src/train/train.mjs --init <prev_model_dir> --out ... --steps
  2000 --lr 5e-4` continues from previous weights (same architecture) instead of
  training from scratch — used to adapt the model to the label-adjacent positives.

## Files

```
src/ocr/TesseractOCR.js           engine (OpenCV card + line detection, orchestration, results)
src/ocr/digitModel.js             model architecture, preprocessing, decoding, TF.js recogniser
src/ocr/dateParse.js              digit normalisation + Jalali validation (pure)
src/train/train.mjs               synthetic data generator + training + export (Node)
src/tests/pure.test.mjs           unit tests (npm test)
src/tests/integration/            Node wiring test on a card photo
```

## Setup

1. `npm i @tensorflow/tfjs` (you already have `@techstark/opencv-js` and `tesseract.js`;
   tesseract.js is only a fallback now).
2. Train the model (next section). Serve the output folder as static files, e.g.
   `public/models/date_cnn/{model.json,weights.bin}` (or `new TesseractOCR({ modelUrl })`).
3. Fix imports if your layout differs (`../utils/imageUtils` -> `loadImage`).
   `persianUtils.isValidJalali` is no longer used.

Without the model files the engine logs a warning and falls back to Tesseract line OCR.
That works, but is only a little better than the old engine.

## Training the model (Node)

```
npm i -D @tensorflow/tfjs-node @napi-rs/canvas
node src/train/train.mjs --fonts ./fonts --out public/models/date_cnn --steps 6000 \
     [--real ./real_crops] [--backgrounds ./card_textures]
# shortcuts: npm run train / npm test
```

* `--fonts`: folder of .ttf/.otf fonts that contain Persian digits (Vazirmatn, IRANSans,
  Yekan, ...), ideally resembling the card print. After the first run **open
  `<out>/preview.png`**: every row must show real digits, not empty boxes.
* `--real`: folder with real cropped date lines and `labels.csv` (`crop001.png,13750512`,
  or `crop002.png,none` for a line that is not a date). **This is the most valuable input.**
  A few hundred real crops, mixed 50/50 with synthetic data, decide real-world accuracy;
  20% are held out and reported as `REAL exact`. Trust that number, not the synthetic one.
* Use `@tensorflow/tfjs-node` (native). Plain tfjs works but is extremely slow for
  training (about a minute per step on my 1-core test box).
* Node 22+: `@tensorflow/tfjs-node@4.22.0` imports `isNullOrUndefined` from Node's
  built-in `util`, which no longer provides it. `src/train/train.mjs` shims it
  (`util.isNullOrUndefined ??= (v) => v == null`) before loading tfjs-node.
* Collect real crops with user consent and store them securely (national ID data).

## Calibrate once

`SEARCH_BAND` / `IDEAL_Y` at the top of `TesseractOCR.js` are deliberately wide because I
don't have your card layouts. Rectify ~20 real photos, see where the birth date sits and
tighten the band. Check both card generations.

## Result notes

* `best` is **null** when nothing reached `MIN_CONFIDENCE` (60) **with agreement**
  (≥2 boxes reading the same date, or a single box ≥90%). Treat it as "retake photo".
  A lone mid-confidence read is usually a misfire, not a date.
* `confidence` is 0-100 (mean winning probability over the 8 digits).
* `best.segmentImages` is now `{ line: dataUrl }`; `preprocessedImage` is the same crop.
* `attempt.strategy` is `line-cnn` (full line), `window-cnn` (sliding window) or
  `line-tesseract` (fallback when the model files are missing). The result header shows
  which engine produced it (neural model vs Tesseract fallback).
* For live camera use, run several frames and accept a date only when 2+ agree.

## What has and has not been verified

Verified in a sandbox:
* Unit tests (`npm test`): date parsing/validation and head decoding.
* Card rectification and line-candidate detection with real opencv.js on a synthetic photographed card.
* The training script runs end to end (data generation, loss, evaluation, saving), and the
  saved `model.json`/`weights.bin` load back into TF.js and run through the engine
  (`node src/tests/integration/engine.test.mjs <card.png> <model_dir>`).
* The synthetic samples look right (`preview.png` — dates at random positions with
  detector-like margins, plus label+ID / near-date / truncated hard negatives).
* `npm run build` passes; UI (`ResultDisplay`, `ProgressBar`) uses `best` (null = retake)
  and the new `line-cnn` / `window-cnn` attempt shape, and shows which engine ran.
* Synthetic photographed-card e2e: rectification + line finding + window search +
  voting returns the correct date with a decisive margin (truth 2 votes / 96%
  vs best near-miss 2 votes / 90%).

NOT verified:
* **Real-card accuracy.** The model learns on synthetic data (loss falls, `isDate`
  accuracy rises), but real accuracy is unmeasured until you train with `--real`
  crops and check the `REAL exact` number. Expect to iterate on the training data.
  If the card font differs a lot from Vazirmatn/Yekan/IranianSans, add a closer font.
* In-browser inference timing. Expect ~45-60 min for 6000 steps on a desktop CPU.
* `grayToModelInput` is shared by training and the engine, so preprocessing cannot drift.
