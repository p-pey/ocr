/**
 * engineConfig.js — SINGLE SOURCE OF TRUTH for every tunable number in the
 * birth-date OCR engine.
 *
 * HOW TO USE:
 *   1. Want to change behaviour? Edit ONLY this file.
 *   2. Every constant has a clear name + a comment explaining:
 *      - what it controls,
 *      - what happens if you raise / lower it,
 *      - the safe range.
 *   3. All engine modules (`TesseractOCR.js`, `cnn.js`, `dateParse.js`,
 *      `shapeGate.js`, `fieldAssign.js`, `consensus.js`) import from here.
 *      Old export names in `TesseractOCR.js` (CARD_W, SEARCH_BAND, …) are
 *      kept as aliases so existing UI / tests keep working.
 *
 * SECTIONS:
 *   A. Card geometry            — standard rectified card size + analysis cap
 *   B. Card detection           — quad finding (Canny / area / aspect)
 *   C. Line search band         — WHERE on the card we look for text rows
 *   D. Line geometry filter    — WHICH blobs count as a text line
 *   E. Morphology kernels      — how blobs are merged into full date strips
 *   F. Candidate ranking       — NMS + sort + top-N
 *   G. ROI padding / preview   — keep edge digits inside the crop
 *   H. Readability lift        — gentle brightness/contrast on the card
 *   I. Polarity + adaptive pass — dark-on-light normalisation + pass-2 rescue
 *   J. CNN input contract      — MUST match training (retrain if changed)
 *   K. Acceptance gates        — when a CNN read becomes a valid date
 *   L. Test-time augmentation  — shifted/grown re-reads
 *   M. Birth/expiry selection  — which valid date is reported as birth
 *   N. Shape-gate topology     — deterministic ۰/۵/۹/… verifier
 *   O. RTL anchoring / rows    — geometric birth-vs-expiry helpers
 *   P. Jalali calendar range   — accepted birth years
 *   Q. Multi-frame consensus   — live-camera voting
 *   R. Runtime / memory        — OpenCV load timeout
 *
 * RULE: changing anything in J (CNN input) or the polarity rule in I without
 * retraining silently destroys accuracy (training/inference contract).
 */

/* ================================================================== */
/* A. Card geometry                                                    */
/* ================================================================== */

/**
 * Standard rectified card size in pixels.
 * Every detected card is perspective-warped to exactly this size so all
 * downstream pixel thresholds (line height 18-84px, widths, kernels) mean
 * the same thing on every photo.
 * - Larger = more detail but slower CNN + more memory.
 * - Smaller = faster but thin strokes / the small ۰ dot disappear.
 * Safe range: width 900-1500 (keep 1200x756 aspect ≈ 1.587).
 */
export const CARD_STANDARD_WIDTH_PX = 1200;
export const CARD_STANDARD_HEIGHT_PX = 756;

/**
 * Photos bigger than this (longest side) are downscaled before card
 * detection. Keeps Canny/contours fast and memory flat on phones.
 * Raise = slower detection, slightly better tiny-card recall.
 * Safe range: 1200-2000.
 */
export const CARD_ANALYSIS_MAX_SIDE_PX = 1600;

/* ================================================================== */
/* B. Card detection (find the card quadrilateral in the photo)        */
/* ================================================================== */

export const CARD_DETECTION = {
  /** Gaussian blur kernel applied before edge detection (odd number). */
  blurKernelPx: 5,
  /** Canny low/high hysteresis thresholds. Lower = more edges (noisier). */
  cannyLowThreshold: 45,
  cannyHighThreshold: 140,
  /** Closing kernel that seals gaps in the card border edges. */
  closingKernelPx: 5,
  /**
   * Quad must cover at least this fraction of the photo area.
   * Filters out small background rectangles. Lower = more false quads.
   */
  minQuadAreaFraction: 0.28,
  /** Accepted card aspect (long/short side). ID-1 cards ≈ 1.586. */
  minQuadAspect: 1.25,
  maxQuadAspect: 2.05,
  /**
   * Polygon approximation epsilon as a fraction of contour perimeter.
   * Smaller = stricter quadrilateral (rejects rounded/warped cards).
   */
  approxEpsilonFraction: 0.02,
};

/* ================================================================== */
/* C. Line search band — WHERE on the card we look                     */
/* ================================================================== */

/**
 * Fractions of the rectified card inside which text lines are searched.
 * Birth date lives near the vertical middle; national-ID/expiry live lower.
 * - Widening y1 finds low expiry rows but adds background false alarms.
 * - Narrowing speeds up + reduces noise but risks cutting the birth row.
 */
export const LINE_SEARCH_BAND_FRACTIONS = {
  left: 0.08,
  top: 0.18,
  right: 0.99,
  bottom: 0.93,
};

/**
 * Ideal birth-row centre (fraction of card height) used ONLY to order
 * candidates before the CNN reads them (closest-to-middle first).
 * Does NOT decide the winner — selection (section M) does.
 */
export const LINE_IDEAL_CENTER_Y_FRACTION = 0.52;

/* ================================================================== */
/* D. Line geometry filter — WHICH blobs count as a text line          */
/* ================================================================== */

export const LINE_GEOMETRY = {
  /**
   * Accepted blob height AFTER morphological closing, in card pixels.
   * Below min = noise/specks; above max = merged rows / photo borders.
   * At 1200px wide, a date strip is typically ~25-60px tall.
   */
  minHeightPx: 18,
  maxHeightPx: 84,
  /**
   * Minimum blob width. A full YYYY/MM/DD strip is ~200-450px wide.
   * Anything narrower than ~140px is a year-only fragment (half-cut) and
   * may only win if no full strip exists (see FULL_STRIP_MIN_WIDTH_PX).
   */
  minWidthPx: 90,
  /**
   * Fragments narrower than this NEVER beat a full strip in selection,
   * even with higher confidence. This is the "half-cut" fix.
   */
  fullStripMinWidthPx: 140,
  /** Minimum width/height ratio: date rows are wide and short. */
  minAspectRatio: 2.2,
  /** Reject blobs wider than this fraction of the card (merged rows). */
  maxWidthFractionOfCard: 0.9,
  /** Blob padding before the CNN reads it (fraction of blob height). */
  padXFractionOfHeight: 0.25,
  padYFractionOfHeight: 0.3,
};

/* ================================================================== */
/* E. Morphology kernels — merge digits into ONE date strip            */
/* ================================================================== */

export const LINE_MERGE_KERNELS = {
  /**
   * Closing kernel widths tried (each = one full pass). Wide horizontal
   * kernels bridge inter-digit gaps and slashes so YYYY/MM/DD becomes one
   * blob instead of 8 fragments.
   * The widest entry is MANDATORY in at least one pass (fixes half-cut).
   * Wider = longer strips merged, but nearby rows may fuse together.
   */
  widthsPx: [23, 21, 33, 40],
  /** Kernel height: tall enough to join digit rows, short enough not to
   *  fuse the birth row with the row above/below it. */
  heightPx: 5,
};

/* ================================================================== */
/* F. Candidate ranking (NMS + sort + top-N)                           */
/* ================================================================== */

export const LINE_RANKING = {
  /** Two boxes overlapping more than this (IoU) are duplicates: keep one. */
  duplicateOverlapIouThreshold: 0.6,
  /** Maximum line boxes sent to the CNN per rotation/pass. More = slower. */
  maxCandidates: 18,
  /** Line-finder blur + adaptive-threshold block size / constant C. */
  blurKernelPx: 3,
  adaptiveBlockSizePx: 31,
  adaptiveConstantC: 12,
};

/* ================================================================== */
/* G. ROI padding + preview (Issue #1: cut-off day digits)             */
/* ================================================================== */

/**
 * Extra pixels added AROUND every candidate box AFTER detection, clamped
 * to the image. Keeps edge day digits inside the crop + the UI preview.
 * Raise if day digits are clipped; lower if neighbouring rows bleed in.
 * At 1200px cards, ±10px ≈ ±1 digit width. Safe range: 0-24.
 */
export const BIRTH_STRIP_PAD_X_PX = 10;
export const BIRTH_STRIP_PAD_Y_PX = 13;

/**
 * ROI cleanup kernel (elliptical open) used by the shape-gate cross-check
 * and the readability boost. Drops specks inside ۰ without eating strokes.
 * 2 = gentle. 3+ starts eroding thin print.
 */
export const ROI_CLEANUP_KERNEL_PX = 2;

/** Upscale factor for tiny ROIs before cleanup (rows<50 → x3, <80 → x2). */
export const ROI_CLEANUP_UPSCALE_SMALL_ROW_PX = 50;
export const ROI_CLEANUP_UPSCALE_MEDIUM_ROW_PX = 80;
export const ROI_CLEANUP_BLUR_PX = 3;

/* ================================================================== */
/* H. Readability lift — gentle brightness/contrast on the CARD image  */
/* ================================================================== */

/**
 * Additive brightness lift (0-255) applied to the card before detection
 * and recognition. +8 barely moves clean cards, lifts washed-out ones.
 * Raise if pale print is missed; lower (0) if whites blow out.
 * Safe range: 0-20.
 */
export const CARD_BRIGHTNESS_LIFT = 8;

/**
 * Multiplicative contrast gain around the mean. 1.12 = +12% ink/paper
 * separation. CNN input contract (standardised grayscale) is unchanged.
 * Safe range: 1.0-1.3.
 */
export const CARD_CONTRAST_GAIN = 1.12;

/* ================================================================== */
/* I. Polarity + adaptive two-pass preprocessing                       */
/* ================================================================== */

export const INK_POLARITY = {
  /**
   * If mean gray < this, the image is treated as light-ink-on-dark and
   * inverted. The model is trained on dark-ink-on-light-paper ONLY.
   * Lower = fewer inversions (risk: negative images misread).
   */
  invertIfMeanBelow: 110,
};

export const ADAPTIVE_PASS2 = {
  /**
   * Pass 1 = native grayscale only. Pass 2 (CLAHE / stretch / unsharp)
   * runs ONLY when pass 1 yields zero valid dates — clean images are
   * never "enhanced" into failure.
   */
  /** Skip pass 2 entirely if any pass-1 date reaches this confidence. */
  skipPass2IfConfidenceAtLeast: 80,
  /** μ below this → selective CLAHE (dark/washed images). */
  claheIfMeanBelow: 100,
  claheClipLimit: 2.0,
  claheTilePx: 8,
  /** σ below this → min-max contrast stretch (flat/foggy images). */
  stretchIfSigmaBelow: 38,
  /** 38 ≤ σ < this → unsharp masking (soft/blurry images). */
  unsharpIfSigmaBelow: 65,
  unsharpSigma: 1.2,
  unsharpStrongWeight: 1.5,
  unsharpBlurWeight: -0.5,
};

/* ================================================================== */
/* J. CNN input contract — MUST match training byte-for-byte           */
/* ================================================================== */

export const CNN_INPUT = {
  /** Model input height/width (H×W), single channel, row-major float32. */
  heightPx: 32,
  widthPx: 160,
  /** Birth date is always zero-padded YYYY/MM/DD = exactly 8 digits. */
  digitSlots: 8,
  /** Standardisation epsilon: x = (x-mean)/(std+eps). */
  standardiseEpsilon: 1e-6,
  /** Minimum resize width after aspect-preserving scale (px). */
  minResizedWidthPx: 8,
  /** `isDate` head threshold: line counts as date-like above this. */
  isDateThreshold: 0.5,
};

/* ================================================================== */
/* K. Acceptance gates — when a CNN read becomes a valid date          */
/* ================================================================== */

export const ACCEPTANCE = {
  /**
   * Minimum `isDate` probability (national-ID / word lines rejected below).
   * Lower = more candidates survive (more false dates); higher = stricter.
   */
  minDateProbability: 0.6,
  /**
   * Minimum mean per-digit confidence (0-100). Below → `best = null`
   * (ask for retake). Never lowered to "guess" a date.
   */
  minConfidence: 60,
  /** Same as minDateProbability but for the "confident" fast-path checks. */
  confidentDateProbability: 0.8,
};

/* ================================================================== */
/* L. Test-time augmentation (TTA) — shifted/grown re-reads            */
/* ================================================================== */

export const TTA = {
  /** Re-read this many top direct reads with variants. More = slower. */
  topReadsToAugment: 4,
  /** Variants per top read (shifted/grown crops), probabilities averaged. */
  variantsPerRead: 4,
  /**
   * Variant definitions: dx/dy = centre shift (fraction of box),
   * grow = box growth fraction. Deterministic, clamped to the image.
   */
  variants: [
    { dx: -0.1, dy: 0, grow: 0.1 },
    { dx: 0.1, dy: 0, grow: 0.1 },
    { dx: 0, dy: -0.05, grow: 0.12 },
    { dx: 0, dy: 0.05, grow: 0.16 },
  ],
  /** If the TTA-merged read is no longer a valid date, scale × this. */
  invalidMergePenalty: 0.8,
};

/* ================================================================== */
/* M. Birth/expiry selection — WHICH valid date is reported            */
/* ================================================================== */

export const SELECTION = {
  /**
   * Years ABOVE this are expiry-side (smart-card expiry lives in the
   * 1400s; birth dates don't). When any ≤ cutoff year exists, every year
   * above the cutoff is dropped from the birth race. This is the ONLY
   * year-based rule — it never picks between two birth-side years.
   */
  expiryCutoffYear: 1400,
  /**
   * Expiry-takeover guard: when the top-score winner is the LOWER
   * (expiry-side) row but an UPPER birth-side row is within this many
   * confidence points, prefer the upper row (expiry print is often
   * cleaner and scores marginally higher).
   */
  upperRowTakeoverMaxGap: 15,
  /** "Upper" means top edge at least this much higher (fraction of card). */
  upperRowMinGapFraction: 0.015,
  /**
   * Lone-expiry guard: a SINGLE validated date with an expiry-side year
   * sitting this deep in the card is the expiry row with the birth row
   * missed → report it as expiry and return birth=null (never emit expiry
   * AS birth). Tight crops (user isolated the line) skip this guard.
   */
  loneExpiryMinYFraction: 0.65,
  /** Tight-crop heuristic: unrectified card smaller than this = user crop. */
  tightCropMaxWidthPx: 800,
  tightCropMaxHeightPx: 400,
  tightCropMaxAspect: 2.2,
};

/* ================================================================== */
/* N. Shape-gate topology — deterministic ۰/۵/۹/… verifier             */
/* ================================================================== */

export const SHAPE_GATE = {
  /** Pixel < this = ink (0-255 grayscale). */
  inkThreshold: 128,
  /** Minimum ink rows per column / width to count as an ink column run. */
  minInkRowsPerColumn: null, // null → max(2, round(H*0.12)) computed per line
  minRunWidthPx: 2,
  /** Merge neighbouring ink runs separated by ≤ this gap (px). */
  runMergeGapPx: 2,
  /** A ۰ read TALLER than this × median digit height = conflict. */
  zeroMaxHeightRatio: 0.7,
  /** A ۵ read SHORTER than this × median digit height = conflict. */
  fiveMinHeightRatio: 0.55,
  /** ۹ loop must sit above this fraction of slot height (upper 55%). */
  nineLoopMaxCenterYFraction: 0.55,
  /** ۵ loop must sit inside this vertical band (fractions of height). */
  fiveLoopMinCenterYFraction: 0.3,
  fiveLoopMaxCenterYFraction: 0.7,
  /** ۱ must be narrower than this (width/height). */
  oneMaxAspect: 0.4,
  /** Right-heavy ink ratio that counts as a descending stem (۹ vs ۵). */
  stemRightHeavyRatio: 1.5,
  stemMinRightInkPx: 2,
  stemMaxRightInkForFivePx: 12,
  /** Gate conflict halves CNN confidence (suspected misread → retake). */
  conflictConfidenceScale: 0.5,
  /** Minimum believable hole area (px) — smaller = dust, not a loop. */
  minHoleAreaPx: 2,
  /** Template/NCC compare size for slot-vs-exemplar checks. */
  templateCompareWidthPx: 24,
  templateCompareHeightPx: 32,
};

/* ================================================================== */
/* O. RTL anchoring + row suppression (geometric, no label OCR)        */
/* ================================================================== */

export const RTL_ANCHOR = {
  /**
   * Persian cards read right-to-left: the birth label sits RIGHT, digits
   * extend LEFT. Scores how birth-label-like a box is (higher = read first).
   * rightness maps right-edge xMax from [edgeStart, edgeStart+edgeSpan].
   */
  rightEdgeStartFraction: 0.35,
  rightEdgeSpan: 0.6,
  rightnessWeight: 0.7,
  widthWeight: 0.3,
  /** Full-date strips are 12-75% of card width; else scored 0.25. */
  minFullWidthFraction: 0.12,
  maxFullWidthFraction: 0.75,
  offWidthScore: 0.25,
  /** Rows deeper than birthYMin + this are expiry-side (dropped). */
  expiryRowMarginFraction: 0.02,
};

/* ================================================================== */
/* P. Jalali calendar range (strict enforcer, no guessing)             */
/* ================================================================== */

export const JALALI = {
  /** Accepted birth years (adult birth dates). Outside → rejected (null). */
  minBirthYear: 1290,
  maxBirthYear: 1410,
  /** DIGIT-LEVEL training label range (wider than accepted birth range). */
  trainMinYear: 1300,
  trainMaxYear: 1415,
};

/* ================================================================== */
/* Q. Multi-frame consensus (live camera)                              */
/* ================================================================== */

export const CONSENSUS = {
  /** Frames that must agree on the same formatted date before accept. */
  requiredVotes: 2,
  /** Maximum frames buffered for voting. */
  maxFrames: 5,
};

/* ================================================================== */
/* R. Runtime / memory                                                 */
/* ================================================================== */

export const RUNTIME = {
  /** Reject OpenCV.js load if it takes longer than this (ms). */
  openCvLoadTimeoutMs: 30000,
};
