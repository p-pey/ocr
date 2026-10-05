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

* `best` is **null** when nothing reached `MIN_CONFIDENCE` (55). Treat it as "retake photo".
* `confidence` is 0-100 (mean winning probability over the 8 digits).
* `best.segmentImages` is now `{ line: dataUrl }`; `preprocessedImage` is the same crop.
* For live camera use, run several frames and accept a date only when 2+ agree.

## What has and has not been verified

Verified in a sandbox:
* Unit tests (`npm test`): date parsing/validation and head decoding.
* Card rectification and line-candidate detection with real opencv.js on a synthetic photographed card.
* The training script runs end to end (data generation, loss, evaluation, saving), and the
  saved `model.json`/`weights.bin` load back into TF.js and run through the engine
  (`node src/tests/integration/engine.test.mjs <card.png> <model_dir>`).
* The synthetic samples look right (`preview.png` — Persian digits render correctly).
* `npm run build` passes; UI (`ResultDisplay`, `ProgressBar`) uses `best` (null = retake)
  and the new `line-cnn` attempt shape.

NOT verified:
* **Real-card accuracy.** The model learns on synthetic data (loss falls, `isDate`
  accuracy rises), but real accuracy is unmeasured until you train with `--real`
  crops and check the `REAL exact` number. Expect to iterate on the training data.
* In-browser inference timing. Expect ~45 min for 6000 steps on a desktop CPU.
* `grayToModelInput` is shared by training and the engine, so preprocessing cannot drift.
