# Engine Flow — Iranian National-Card Birth-Date OCR (detailed)

> Code map: `src/ocr/TesseractOCR.js` (orchestrator) · `src/ocr/cnn.js`
> (recogniser) · `src/ocr/shapeGate.js` (deterministic verifier) ·
> `src/ocr/dateParse.js` (Jalali parser) · `src/ocr/fieldAssign.js`
> (birth/expiry) · `src/ocr/consensus.js` (multi-frame voting) ·
> `src/ocr/engineConfig.js` (ALL tunable numbers) ·
> `src/hooks/useOCR.js` (React adapter).
> Class name `TesseractOCR` is kept for UI compatibility — there is **no
> Tesseract, no tfjs, no network** inside. Runtime dependency is only
> `@techstark/opencv-js` + embedded int8 weights (`modelWeights.js`).

---

## 0. One-paragraph summary

A photo goes in; `YYYY/MM/DD` (Jalali, Latin digits) plus confidence comes
out — or `best: null` ("retake photo", never a guess). The engine
rectifies the card to a standard 1200×756 image, finds horizontal text-line
boxes, reads each box with a tiny dependency-free CNN (8 digit heads +
`isDate` head), verifies every read with a deterministic shape gate (holes,
stems, aspect, templates), re-reads the best boxes with shifted/grown crops
(TTA), and finally picks the winner by score with birth/expiry
disambiguation (year cutoff + row geometry). Two passes exist: pass 1 is
plain grayscale (clean images untouched); pass 2 (CLAHE/stretch/unsharp)
runs **only** if pass 1 yields zero valid dates; a 180° rotation retry is
the last emergency step.

---

## 1. Entry points and result contract

```
UI / camera
  └─ useOCR().recognize(imageSrc)              src/hooks/useOCR.js
  └─ useOCR().recognizeConsensus(frames[])     ≥2 frames must agree
        └─ TesseractOCR.recognize(src, onProgress, onAttempt)
              └─ cv.imread(loadImage(src))     src/utils/imageUtils.js
              └─ recognizeMat(cv, source, …)   DOM-free core (Node-testable)
                    └─ { best, allDates, allAttempts, fields, timingMs }
```

Result shape (backward compatible, do not change):

```js
// success
{ best: { birthDate: { year, month, day, formatted:"1376/07/05",
                       raw, confidence:0-100, score, corrected:false,
                       correctionCost:0, votes, minProb, dateProb,
                       relY, yMin },
          strategy, candidateIndex, rotation, rectified, bounds:{x,y,w,h},
          engine:"opencv+cnn", rawText, normalizedText, confidence,
          minProb, dateProb, dates, repairs:0, gateConflict, gateReasons,
          segmentImages:{line}, preprocessedImage },
  allDates:  [...per-date objects + field:"birth"|"expiry"|"unknown"],
  allAttempts: [...per-box attempts],
  fields: { birth, expiry, swapped, method, sequenceError },
  timingMs }
// failure
{ best: null, allDates, allAttempts, fields, timingMs }
```

`best === null` means "retake photo". `minProb` (weakest winning digit)
and `gateConflict` can drive stricter retake thresholds in the UI.

Progress callbacks: 2 (init) → 5/15 (load) → 35 (card warp) → 50/75
(rotation 0) → 80/95 (rotation 180, if run) → 100 (done). `onAttempt(n,
attempt)` fires per accepted date read.

---

## 2. Full pipeline (step by step)

### Step 0 — `initialize()` (one-time, excluded from timing targets)

1. Dynamically imports `@techstark/opencv-js`, waits for
   `onRuntimeInitialized` (timeout `RUNTIME.openCvLoadTimeoutMs` = 30 s).
2. Touches `cnn.js` (`recognizeLines` must exist → corrupt weight blob
   fails fast here, not mid-recognition).
3. Best-effort `loadEmbeddedShapeTemplates()` for the NCC/Hu template
   cross-check. Without templates the geometry/topology gate still runs.

### Step 1 — Load + gentle readability lift

```
loadImage(src) → HTMLImageElement → cv.imread → RGBA/RGB Mat `source`
liftCardReadability(source):  out = src * CARD_CONTRAST_GAIN + CARD_BRIGHTNESS_LIFT
```

- `CARD_BRIGHTNESS_LIFT` (+8) / `CARD_CONTRAST_GAIN` (×1.12): deliberately
  gentle — clean cards are visually untouched, degraded ones gain
  ink/paper separation.
- Runs on the **card image only**, before gray conversion. The CNN input
  contract (standardised grayscale, §5) is unchanged.
- Tune in `engineConfig.js` §H.

### Step 2 — `fitToCard`: find the card, warp to 1200×756

1. Downscale so longest side ≤ `CARD_ANALYSIS_MAX_SIDE_PX` (1600).
   Interpolation: `INTER_AREA` when shrinking, `INTER_LINEAR` otherwise.
2. `detectCardQuadrilateral` (§B):
   - gray → Gaussian blur (`blurKernelPx` 5) → Canny (`cannyLowThreshold`
     45 / `cannyHighThreshold` 140) → CLOSE 5×5 → `findContours(RETR_LIST)`.
   - Keep the largest convex quadrilateral with area ≥
     `minQuadAreaFraction` (28%) of the photo and aspect (long/short)
     within 1.25–2.05; polygon epsilon = 2% of perimeter.
   - Corners ordered TL/TR/BR/BL. Records `portrait = width < height`.
3. If a quad is found: perspective warp to `CARD_STANDARD_WIDTH_PX ×
   CARD_STANDARD_HEIGHT_PX` (1200×756, `INTER_CUBIC`,
   `BORDER_REPLICATE`); portrait quads are rotated 90° clockwise back to
   landscape. Returns `{ card, rectified: true }`.
4. Else (tight crop / no border visible): rotate portrait crops to
   landscape, scale width to 1200, return `{ card, rectified: false }`.
   Downstream `tightCrop` heuristics + the lone-expiry guard behave
   differently for this case (see §8).
5. Every intermediate `cv.Mat` is deleted (`deleteMats`); leaks crash
   mobile WebViews.

### Step 3 — Rotation/pass loop (strict order, cheapest first)

```
runRotation(0, pass=1)          # upright, native grayscale
if no date with conf ≥ 80:  runRotation(0, pass=2)   # upright, enhanced
if still zero dates:        runRotation(180, pass=1) # upside-down, native
if still zero dates:        runRotation(180, pass=2) # upside-down, enhanced
```

- 180° is the **final emergency step only**; it is skipped entirely when
  rotation-0 already produced a confident date (conf ≥
  `skipPass2IfConfidenceAtLeast` and dateProb ≥ `confidentDateProbability`).
- Why this order: most photos are upright and clean — pass 1 alone
  answers them in minimum time without enhancement artefacts.

### Step 4 — `adaptiveGrayPass`: grayscale with polarity fix (+ conditional rescue)

- **Pass 1 (always):** `toGray` → if mean < `INK_POLARITY.invertIfMeanBelow`
  (110) invert. That is ALL — no CLAHE, no sharpening, no normalisation.
  Rationale: the model is trained on dark-ink-on-light-paper; enhancements
  blow out ink on clean images, and binarisation drops the tiny `۰` dot
  (the original 0/5 confusion).
- **Pass 2 (only if pass 1 yields zero valid dates):** clone pass-1 gray,
  measure (μ, σ), apply exactly one rescue:
  - μ < `claheIfMeanBelow` (100) → selective CLAHE (clip
    `claheClipLimit` 2.0, tiles `claheTilePx` 8×8; colour path equalises L
    in Lab, gray path equalises directly).
  - else σ < `stretchIfSigmaBelow` (38) → min-max contrast stretch via
    256-entry LUT.
  - else 38 ≤ σ < `unsharpIfSigmaBelow` (65) → unsharp mask
    (`1.5×orig − 0.5×blur(σ1.2)`).
- Returns `{ mat, pass, mu, sigma, applied }`; caller frees `mat`.

### Step 5 — `findLineCandidates`: text-line boxes (no digit assumptions)

Input: the pass gray image. Output: ≤ `maxCandidates` (18) rects in card
pixels.

1. Crop to `LINE_SEARCH_BAND_FRACTIONS` (x 0.08–0.99, y 0.18–0.93).
2. Gaussian blur 3×3 → `adaptiveThreshold(GAUSSIAN_C, BINARY_INV,
   adaptiveBlockSizePx 31, adaptiveConstantC 12)`.
3. For each kernel width in `LINE_MERGE_KERNELS.widthsPx`
   ([23, 21, 33, 40], height 5): MORPH_CLOSE with a `kw×5` rectangle →
   `findContours(RETR_EXTERNAL)` → `boundingRect` (upright only — **zero
   rotation** by design).
4. Geometry filter (§D): keep boxes with height 18–84 px, width ≥ 90 px,
   aspect ≥ 2.2, width ≤ 90% of card width.
5. Pad each kept box by 25% of its height horizontally / 30% vertically
   (glyphs fully inside), then expand by `BIRTH_STRIP_PAD_X/Y_PX`
   (10/13 px, Issue #1 — edge day digits stay inside), clamped to image.
6. De-duplicate with NMS (IoU > `duplicateOverlapIouThreshold` 0.6),
   sort by distance to `LINE_IDEAL_CENTER_Y_FRACTION` (0.52), keep top 18.
7. Back in `recognizeMat`, candidates are re-sorted by `rtlAnchorScore`
   (right-anchored corridor, §O) so birth-label-like strips are read first.
   This affects read **order**, not the winner.

### Step 6 — `grayToModelInput` + `recognizeLines` (the CNN read)

Preprocessing contract (§J — byte-for-byte with training; changing it
requires retraining):

```
crop (grayscale uint8, dark ink / light paper)
newW = clamp(round(cropW * 32 / cropH), 8, 160)
resize to (newW × 32): INTER_AREA if cropH > 32 else INTER_LINEAR
x = (x − mean) / (std + 1e-6)   over the resized region ONLY
place into zero 32×160 tensor at columns [0, newW); rest stays 0
```

No binarisation / CLAHE / sharpening in this path.

Network (`cnn.js`, dependency-free, ~107k int8 params, ~4M MACs/line):

```
conv3×3(1→8, BN-folded, ReLU) → maxpool2      16×80×8
conv3×3(8→16, ReLU) → maxpool2                 8×40×16
conv3×3(16→32, ReLU) → maxpool2                4×20×32
conv3×3(32→32, ReLU)                           4×20×32
average over height → 20×32 → flatten (x*32+c) 640
dense 128 + ReLU (dropout only in training)
heads: digits dense 80 → reshape [8,10] → softmax per slot
       isDate dense 1 → sigmoid
```

Weights: per-output-channel symmetric int8 (`scale = max|w|/127`, BN
folded), base64 blob in `modelWeights.js`. `decodeProbs` turns 8×10
probabilities + dateProb into `{ digits, text "YYYY/MM/DD", confidence
(mean digit prob ×100), minProb, dateProb, isDate (≥0.5), probs }`.

All candidates of one rotation/pass are batched through `recognizeLines`
in a single call.

### Step 7 — Acceptance gates + shape gate (`pushAttempt`)

For each `(read, rect)`:

1. `parseJalaliDate(read.text)` → must parse (structure repair only:
   separators, dropped slashes `13750512 → 1375/05/12`, flipped visual
   order; **never** digit guessing) AND `isValidJalaliDate` (year
   1290–1410, month 1–12, day per Jalali month lengths incl. leap Esfand).
2. `read.isDate && read.dateProb ≥ minDateProbability (0.6)` AND
   `read.confidence ≥ minConfidence (60)` — else dropped silently.
3. **Shape gate** (best-effort, never throws; §7 below for internals):
   crop the gray strip → `extractDigitSlots` → `checkLineGate` (0/5 height
   ratios) + `verifyDigitTopo` per slot (holes via flood-fill, ۹-stem,
   ۱-aspect) + OpenCV `countHolesCv` twin (RETR_CCOMP) on the
   **cleaned** crop for 0/5/9 + `verifySlotShape` (NCC + Hu vs exemplars)
   when 8 slots segment cleanly. Any conflict halves confidence
   (`conflictConfidenceScale` 0.5), sets `gateConflict: true +
   gateReasons[]`. A halved read below 60 is dropped.
4. Survivors get spatial tags: `relY` (centre), `yMin` (top edge —
   drives field assignment), `xMax` (right edge — RTL anchoring), `roiW/H`
   (half-cut guard), `boxKey`, plus a UI preview (`segmentImages.line`)
   rendered from the **padded** strip.
5. `onAttempt?.(count, attempt)` fires; dates accumulate in `allDates`.

### Step 8 — TTA (test-time augmentation)

For the top `topReadsToAugment` (4) direct reads by confidence: build
`variantsPerRead` (4) shifted/grown crops from `TTA.variants`, read each
with the CNN, re-read the base box, `averageReads` (mean of the 8×10
tables + mean dateProb) → merged read. If the merged text is no longer a
valid date, confidence × `invalidMergePenalty` (0.8). Merged reads go
through the same `pushAttempt` gates with strategy `"line-cnn-tta"`.

### Step 9 — Suppression, voting, selection, field assignment

1. **Negative-region suppression:** `birthY = min(yMin)`; drop rows
   clearly below it (`suppressExpiryRows`, margin 0.02) — expiry rows
   never contend for birth when a higher row validated.
2. **Vote aggregation** (`buildFinalResult`): group `allDates` by
   `formatted`; count `votes`, sum scores; keep each group's best
   representative; sort groups by confidence → `ranked`.
3. **`selectBest(ranked)`:**
   - Drop every year > `expiryCutoffYear` (1400) when any ≤ 1400 exists
     (expiry lives in the 1400s — the ONLY year rule).
   - Prefer full strips: rows with `roiW ≥ fullStripMinWidthPx` (140)
     race; fragments only win if no full strip exists (half-cut fix).
   - Winner = highest confidence → highest dateProb → upper row.
4. **Expiry-takeover guard:** if the winner is a lower row but an upper
   birth-side row is within `upperRowTakeoverMaxGap` (15) confidence
   points and ≥ `upperRowMinGapFraction` (0.015) higher, prefer the upper
   row (cleaner expiry print often scores marginally higher).
5. **`assignFields(ranked)`** (Y-sort + chronology): single date →
   `single` (or lone-expiry hint when year > 1400 and deep in card);
   multiple → upper = birth-side, latest lower year = expiry;
   `birth.year > expiry.year` → `sequenceError` ("birth-after-expiry").
   `swapped` flags upside-down captures (earlier year below later year).
6. **Lone-expiry guard:** single validated date, expiry-side year, deep
   (`loneExpiryMinYFraction` 0.65), NOT a tight crop → `birth: null`,
   `expiry: <date>`, method `single-expiry` (never emit expiry AS birth).
   Tight crops (`!rectified` + small/wide card dims) skip the guard — the
   user isolated the line on purpose.
7. **Final `best`:** `effectiveWinner` (+ sequence-error and single-expiry
   vetoes) joined to its **own box's** highest-confidence attempt (not
   just any attempt with the same text — avoids showing the expiry strip
   under birth text). Every ranked date is tagged
   `field: birth|expiry|unknown`.

### Step 10 — React adapter + consensus

- `useOCR().recognize(src)`: resets state, streams `progress` /
  `currentAttempt`, maps the engine result (`finalScore`, `lineImage`,
  `error`, `fields`) into React state.
- `useOCR().recognizeConsensus(sources, { requiredVotes: 2 })`: runs
  `recognize` per frame, feeds `best.birthDate.formatted` into
  `ConsensusReader`; stops early on agreement; returns `{ mapped,
  agreed }`. A date is accepted only when ≥ 2 frames agree (spec 8.5).

---

## 3. Module responsibilities (who owns what)

| Module | Owns | Must NOT do |
|---|---|---|
| `TesseractOCR.js` | card warp, bands, boxes, passes, TTA loop, gates wiring, selection, result shape, Mat lifetime | digit shapes, Jalali math, label reading |
| `cnn.js` | model input contract, int8 forward, softmax/sigmoid, decode, averageReads | thresholds, geometry, dates |
| `shapeGate.js` | slots, holes (flood-fill + CCOMP twin), stems, aspect, NCC/Hu templates | probabilities, selection |
| `dateParse.js` | digit normalisation, structure repair, strict Jalali validation | guessing digits, padding |
| `fieldAssign.js` | RTL anchor score, row suppression, Y-sort + chronology | CNN, OpenCV |
| `consensus.js` | cross-frame vote counting | single-frame logic |
| `engineConfig.js` | every tunable number + safe ranges | logic |
| `useOCR.js` | React state, progress, consensus loop | engine decisions |

---

## 4. Memory + privacy rules

- Every `cv.Mat` / `MatVector` is deleted in `finally` blocks
  (`deleteMats`); clones passed to the result (previews) are data URLs,
  not Mats. Run 200 recognitions in a loop to verify a flat WASM heap.
- Images never leave the device; no image data / full dates in production
  logs. `OCR_SELECTION_TRACE=1` (dev only) logs years, never full dates.

---

## 5. Failure modes (what `best: null` means)

- No quad + no readable strip (glare, blur, occlusion, hologram over the
  row) → no candidates or all reads gated out.
- Birth row missed but expiry row read → reported as `fields.expiry`
  with `best: null` (lone-expiry guard) — retake, not a wrong answer.
- Two rows with inverted chronology → `sequenceError` → `best: null`.
- Frames disagree (live camera) → consensus returns null — hold still
  and retry.

See `TUNING_GUIDE.md` for which knob addresses which failure.
