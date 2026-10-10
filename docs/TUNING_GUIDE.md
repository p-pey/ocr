# Tuning Guide — every knob in `src/ocr/engineConfig.js`

> **Rule 1:** edit ONLY `src/ocr/engineConfig.js`. Every engine module
> imports from there; nothing else holds a magic number.
> **Rule 2:** never change `CNN_INPUT` (§J) or the pass-1 polarity rule
> without retraining — training and inference must stay byte-for-byte
> identical or accuracy silently collapses.
> **Rule 3:** after any change, re-verify on a synthetic card set + a
> real held-out set (exact-date accuracy, wrong-answer rate, ۰↔۵
> confusion), plus speed and a 200-loop memory test.

How to read each entry: **What** (what it controls) · **▲ raise**
(what happens) · **▼ lower** · **Safe range** · **When to touch**.

---

## A. Card geometry

- `CARD_STANDARD_WIDTH_PX / HEIGHT_PX` (1200×756) — rectified card size;
  fixes the meaning of every pixel threshold below.
  ▲ more detail, slower CNN + memory · ▼ faster, thin strokes / `۰` dot
  vanish · 900–1500 wide (keep ≈1.587 aspect) · Touch when: systematically
  changing speed/accuracy budget.
- `CARD_ANALYSIS_MAX_SIDE_PX` (1600) — downscale cap before detection.
  ▲ slower detection, better tiny-card recall · ▼ faster, distant cards
  missed · 1200–2000 · Touch when: detection is slow or small cards miss.

## B. Card detection (`CARD_DETECTION`)

- `blurKernelPx` (5) — pre-Canny smoothing. ▲ fewer noisy edges, weak
  borders lost · ▼ more edges, more false quads · odd 3–7 · Touch when:
  borders fragment or background rectangles win.
- `cannyLowThreshold / cannyHighThreshold` (45/140) — edge hysteresis.
  ▲ fewer/stronger edges · ▼ noisier edge map · 30–60 / 100–180 ·
  Touch when: card border missed in low contrast.
- `closingKernelPx` (5) — seals border gaps. ▲ seals broken borders, merges
  background · ▼ broken borders · 3–9.
- `minQuadAreaFraction` (0.28) — quad must cover 28% of photo. ▲ fewer
  false quads, cropped cards rejected · ▼ background wins · 0.15–0.4.
- `min/maxQuadAspect` (1.25–2.05) — accepted card proportions (ID-1 ≈1.59).
  Widen when: portrait/folded captures rejected.
- `approxEpsilonFraction` (0.02) — polygon strictness. ▲ accepts warped
  cards, more non-quads · ▼ rejects rounded cards · 0.01–0.04.

## C. Line search band (`LINE_SEARCH_BAND_FRACTIONS`, `LINE_IDEAL_CENTER_Y_FRACTION`)

- `left/top/right/bottom` (0.08/0.18/0.99/0.93) — WHERE text rows are
  searched. ▲ wider = finds low rows, more false alarms + slower ·
  ▼ narrower = faster/cleaner, risks cutting birth row · Touch when:
  localisation recall < 98% (log misses first, then widen the missing side).
- `LINE_IDEAL_CENTER_Y_FRACTION` (0.52) — sort order only (closest to
  middle read first). Does NOT pick the winner. Touch when: read order /
  speed only.

## D. Line geometry filter (`LINE_GEOMETRY`)

- `min/maxHeightPx` (18/84) — blob height at 1200px cards (date strips
  ≈25–60). ▲ max admits merged rows · ▼ min admits specks. Touch when:
  rows split or merge systematically.
- `minWidthPx` (90) — minimum blob width. ▼ admits fragments (slower,
  noisier) · ▲ drops year-only fragments entirely.
- `fullStripMinWidthPx` (140) — fragments below this NEVER beat a full
  strip (half-cut fix). ▼ weakens the fix (clipped days win) · ▲ wide
  cards' fragments never contend. Touch when: day-clipped dates win.
- `minAspectRatio` (2.2) — rows are wide/short. ▼ admits square noise.
- `maxWidthFractionOfCard` (0.9) — rejects merged-row blobs.
- `padX/YFractionOfHeight` (0.25/0.30) — blob padding before CNN.
  ▲ full glyphs, neighbouring-row bleed · ▼ clipped glyphs.

## E. Morphology kernels (`LINE_MERGE_KERNELS`)

- `widthsPx` ([23,21,33,40]) — CLOSE passes merging digits into one strip.
  Keep the widest (40) in at least one pass (half-cut fix). ▲ wider merges
  distant digits but fuses adjacent rows · ▼ narrower splits dates into
  fragments · Touch when: dates split (widen) or rows fuse (narrow).
- `heightPx` (5) — tall enough to join digit rows, short enough not to fuse
  birth/expiry rows. Range 3–7.

## F. Candidate ranking (`LINE_RANKING`)

- `duplicateOverlapIouThreshold` (0.6) — NMS dedup. ▲ fewer boxes (risk:
  distinct close rows merged) · ▼ duplicate CNN work.
- `maxCandidates` (18) — boxes sent to CNN. ▲ recall, linear slowdown ·
  ▼ speed, missed rows. Range 10–30.
- `blurKernelPx / adaptiveBlockSizePx / adaptiveConstantC` (3/31/12) —
  line-finder binarisation. Larger block = more tolerant of uneven light,
  less precise on small print. Touch when: finder misses rows under glare.

## G. ROI padding + cleanup (`BIRTH_STRIP_PAD_*`, `ROI_CLEANUP_*`)

- `BIRTH_STRIP_PAD_X_PX / PAD_Y_PX` (10/13) — expansion AFTER detection
  (Issue #1: cut day digits). ▲ edge digits safe, more neighbour bleed ·
  ▼ tighter, clipped days · 0–24 · First knob for clipped day digits.
- `ROI_CLEANUP_KERNEL_PX` (2) — elliptical open dropping specks inside ۰
  (Issue #2: 0→8 phantom bar). 3+ erodes thin print.
- `ROI_CLEANUP_UPSCALE_SMALL/MEDIUM_ROW_PX` (50/80) + `BLUR_PX` (3) —
  upscale tiny ROIs before cleanup so bars separate from strokes.

## H. Readability lift (`CARD_BRIGHTNESS_LIFT`, `CARD_CONTRAST_GAIN`)

- `CARD_BRIGHTNESS_LIFT` (+8, 0–255) — additive lift on card image.
  ▲ washed print appears, whites blow out · 0–20 · Touch when: pale print
  missed (raise) or whites saturate (lower to 0).
- `CARD_CONTRAST_GAIN` (×1.12) — ink/paper separation. Range 1.0–1.3.
  CNN contract unchanged (standardised grayscale downstream).

## I. Polarity + adaptive pass 2 (`INK_POLARITY`, `ADAPTIVE_PASS2`)

- `invertIfMeanBelow` (110) — polarity gate (model = dark-on-light only).
  ▼ fewer inversions (negatives misread) · ▲ more inversions (dark photos
  flipped wrongly). Do not touch without retraining context.
- `skipPass2IfConfidenceAtLeast` (80) — pass 2 skipped when pass 1 already
  confident. ▼ more images get enhanced (slower, riskier) · ▲ fewer rescues.
- `claheIfMeanBelow` (100) + `claheClipLimit` (2.0) + `claheTilePx` (8) —
  dark-image rescue. Higher clip = stronger local contrast + more noise.
- `stretchIfSigmaBelow` (38) — flat-image rescue (min-max stretch).
- `unsharpIfSigmaBelow` (65) + `unsharpSigma` (1.2) + weights (1.5/−0.5) —
  soft-image rescue. Stronger weights = crisper but ringing artefacts.

## J. CNN input (`CNN_INPUT`) — retrain-required

- `heightPx/widthPx` (32×160), `digitSlots` (8), `standardiseEpsilon`
  (1e-6), `minResizedWidthPx` (8), `isDateThreshold` (0.5).
  **Do not change without regenerating data + retraining + re-exporting +
  re-running parity.** Slashes are fixed positions; unpadded dates
  (`1375/5/2`) are unsupported by design (needs +CTC — see spec risk R3).

## K. Acceptance gates (`ACCEPTANCE`)

- `minDateProbability` (0.6) — `isDate` floor (rejects national-ID/word
  lines). ▼ more false dates · ▲ more retakes. First knob for ID-line
  false accepts.
- `minConfidence` (60) — mean digit confidence floor. Below → `best:null`.
  Never lowered to "guess". UI can demand higher (e.g. 80) for payments.
- `confidentDateProbability` (0.8) — fast-path "confident" level (skips
  pass 2 / 180° retry). Lower = fewer retries (faster, more misses).

## L. TTA (`TTA`)

- `topReadsToAugment` (4) × `variantsPerRead` (4) — re-read cost scales
  linearly (≈16 extra CNN forwards worst case). ▲ accuracy on jittery
  boxes, slower · ▼ faster, less robust to box shift.
- `variants` — shift/grow shapes. Wider grows tolerate loose boxes;
  larger shifts risk walking off the strip.
- `invalidMergePenalty` (0.8) — merged-into-non-date confidence scale.

## M. Birth/expiry selection (`SELECTION`)

- `expiryCutoffYear` (1400) — THE only year rule: above-cutoff years drop
  from the birth race when any birth-side year exists. Move only if card
  generations change year ranges (never per-photo).
- `upperRowTakeoverMaxGap` (15) + `upperRowMinGapFraction` (0.015) —
  upper-row preference when scores are close (cleaner expiry print often
  scores marginally higher). ▲ gap = upper row wins more often.
- `loneExpiryMinYFraction` (0.65) — single deep expiry-side date →
  `birth:null, expiry:<date>`. Lower = more lone reads called expiry.
- `tightCropMaxWidth/HeightPx + MaxAspect` (800/400/2.2) — user-crop
  detection (skips lone-expiry guard).

## N. Shape gate (`SHAPE_GATE`)

- `inkThreshold` (128) — ink vs paper. Lower = only dark cores count.
- `zeroMaxHeightRatio` (0.7) — ۰ taller than 70% of median digit height =
  conflict (tall thing can't be the small dot). Lower = stricter ۰.
- `fiveMinHeightRatio` (0.55) — ۵ shorter than 55% = conflict.
- `nineLoopMaxCenterYFraction` (0.55) — ۹ loop must sit in upper 55%.
- `fiveLoopMin/MaxCenterYFraction` (0.3/0.7) — ۵ loop must be central.
- `oneMaxAspect` (0.4) — ۱ must be narrow.
- `stemRightHeavyRatio` (1.5) + `stemMinRightInkPx` (2) +
  `stemMaxRightInkForFivePx` (12) — ۹ needs a descending right stem,
  ۵ must not have one.
- `conflictConfidenceScale` (0.5) — conflict halves confidence (usually →
  retake). Lower = harsher gate.
- `minHoleAreaPx` (2) — dust vs loop floor.
- `templateCompareWidth/HeightPx` (24×32) — NCC/Hu compare size.
- Gate philosophy: topology (holes/stem/aspect) runs FIRST and the CNN
  never overrides it; NCC+HU templates confirm 0/1 classes. Touch when:
  a specific ۰/۵/۹-style confusion survives — adjust that digit's rule,
  add a gate unit test on the confusable pair, and verify clean-set
  accuracy drops ≤ 0.3%.

## O. RTL anchoring (`RTL_ANCHOR`)

- `rightEdgeStartFraction` (0.35) + `rightEdgeSpan` (0.6) — right-anchored
  corridor (label RIGHT, digits LEFT). Affects read order only.
- `rightnessWeight / widthWeight` (0.7/0.3), `min/maxFullWidthFraction`
  (0.12–0.75), `offWidthScore` (0.25) — full-strip preference.
- `expiryRowMarginFraction` (0.02) — rows deeper than birthYMin + margin
  are expiry-side (dropped from birth pool). ▲ wider birth pool (expiry
  leaks in) · ▼ tighter (close rows dropped).

## P. Jalali range (`JALALI`)

- `min/maxBirthYear` (1290–1410) — accepted birth years. Outside →
  rejected (null). Widen only for genuine demographic need; every widen
  admits more expiry-side confusion.
- `trainMin/MaxYear` (1300–1415) — training label range (informational;
  baked into weights, not runtime).

## Q. Consensus (`CONSENSUS`)

- `requiredVotes` (2) — frames that must agree. ▲ 3 = stricter live
  capture (fewer wrong, more retries) · 1 = no protection.
- `maxFrames` (5) — vote buffer.

## R. Runtime (`RUNTIME`)

- `openCvLoadTimeoutMs` (30000) — OpenCV load timeout. Lower for
  fail-fast UX on broken networks (OpenCV is bundled, so this fires only
  on truly stuck runtimes).

---

## Symptom → knob cheat-sheet

| Symptom | First knob | Direction |
|---|---|---|
| Day digits clipped in preview/read | `BIRTH_STRIP_PAD_X_PX` | ▲ 10→14–18 |
| Neighbour row bleeds into strip | `BIRTH_STRIP_PAD_*`, `LINE_GEOMETRY.pad*` | ▼ |
| Date split into fragments | `LINE_MERGE_KERNELS.widthsPx` | widen / add pass |
| Two rows fused into one box | `LINE_MERGE_KERNELS.heightPx`, `LINE_GEOMETRY.maxHeightPx` | ▼ |
| Birth row not among candidates | `LINE_SEARCH_BAND_FRACTIONS`, `LINE_RANKING.adaptive*` | widen |
| National-ID line read as date | `ACCEPTANCE.minDateProbability` | ▲ 0.6→0.7 |
| ۰ read as ۵/۹ (or reverse) | `SHAPE_GATE.zeroMaxHeightRatio` etc. + retrain mix | tighten + unit test |
| 0 read as 8 (phantom bar) | `ROI_CLEANUP_KERNEL_PX` | keep 2, check upscale |
| Pale/washed print missed | `CARD_BRIGHTNESS_LIFT`, pass-2 `clahe*` | ▲ gently |
| Blurry print missed | pass-2 `unsharp*` | strengthen mildly |
| Expiry reported as birth | `SELECTION.expiryCutoffYear`, `upperRowTakeoverMaxGap` | verify, don't guess |
| Good photo, slow | `LINE_RANKING.maxCandidates`, `TTA.topReadsToAugment` | ▼ |
| Live-camera wrong answers | `CONSENSUS.requiredVotes` | ▲ 2→3 |
