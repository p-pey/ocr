/**
 * Iranian National Card — Birth Date OCR
 * ========================================================================
 *
 * Production-ready, single-file engine. No build step, no custom traineddata,
 * no training, no model files.
 *
 *   npm install tesseract.js @techstark/opencv-js
 *
 * Usage (browser):
 *
 *   import { IranCardOCR } from "./iranCardOCR";
 *
 *   const ocr = new IranCardOCR();
 *   await ocr.initialize();
 *
 *   const result = await ocr.recognizeBirthDate(fileOrDataUrlOrImage);
 *
 *   if (result.success) {
 *     console.log(result.birthDate);        // "1366/06/22"
 *     console.log(result.year, result.month, result.day);
 *     console.log(result.confidence);       // 0-100
 *   } else {
 *     console.warn(result.error);
 *   }
 *
 *   // Keep the instance alive between recognitions. On shutdown:
 *   await ocr.terminate();
 *
 * How it reads the date
 * -----------------------------------------------------------------------
 * 1. OpenCV: card rectification (best effort), ink binarisation, glyph
 *    extraction, row clustering and a 4-2-2 (YYYY MM DD) pattern search
 *    across several gap thresholds — tolerant of label text on the same
 *    row, attached/standalone "/" separators and upside-down captures.
 * 2. Tesseract (stock fas.traineddata from the tesseract.js CDN): ONE
 *    SINGLE_CHAR read per digit glyph (8 calls) runs first — empirically
 *    the most reliable signal for Persian digits. When a glyph read is
 *    unusable the segment is read once with SINGLE_LINE as a fallback.
 * 3. Position rules (YYYY/MM/DD), zero-dot geometry and Jalali validation
 *    reconcile the signals; anything invented is counted as a "repair".
 *
 * Never throws past recognizeBirthDate() — always returns
 * { success, ... } or { success: false, error }.
 */

import { createWorker, PSM, OEM } from "tesseract.js";

/* ========================================================================
   Constants
   ======================================================================== */

const MAX_ANALYSIS_DIMENSION = 1400;
const OPENCV_INIT_TIMEOUT_MS = 30000;

// Birth date sits in the middle of the card; expiry sits near the bottom.
const CENTER_BAND_MIN = 0.22;
const CENTER_BAND_MAX = 0.78;
const CENTER_BAND_IDEAL = 0.5;

// Glyph geometry — Persian script is wide/cursive; digits are tall/narrow.
const GLYPH_MAX_ASPECT = 1.6;
const GLYPH_MIN_HEIGHT_RATIO = 0.18;
const GLYPH_MAX_HEIGHT_RATIO = 1.25;
const GLYPH_MIN_PIXELS = 10;

// Persian "۰" is a short ring/dot — the key to detecting zero geometrically.
// Measured on Yekan / Vazirmatn / IRANSans: ۰ height is 0.37–0.51 of the
// digit height, every other digit is ≥ 0.84.
const ZERO_HEIGHT_RATIO = 0.58;
const ZERO_MAX_WIDTH_RATIO = 0.85;

// Slash separators: measured aspects 0.40–0.68; ink in the top/bottom 30%
// bands covers ≤ 55% of the box width and the two bands are horizontally
// offset (diagonal). Narrow strokes (۱), top bars (۷) and rings (۰) fail
// these tests, in either orientation.
const SLASH_MIN_ASPECT = 0.3;
const SLASH_MAX_ASPECT = 0.95;
const SLASH_MAX_BAND_WIDTH = 0.55;
const SLASH_MIN_BAND_SHIFT = 0.3;

// Gap thresholds (× the digit reference height) for row clustering.
// Multiplied across several values so both tight and spaced prints work:
//  - "all":     keep digits together, optionally split "/" into its own
//               cluster (loose print with spaces around the separators).
//  - "digits":  separators removed first, so the inter-field gap becomes
//               gap + slashWidth + gap — larger than any digit-to-digit
//               gap, which is what tight prints need.
const SEGMENT_GAP_FACTORS = [0.5, 0.65];
const DIGIT_GAP_FACTORS = [0.5, 0.65, 0.8, 0.95, 1.1];

const TARGET_SEGMENT_HEIGHT = 160;
const TARGET_GLYPH_HEIGHT = 100;
const MIN_UPSCALE = 2;
const MAX_UPSCALE = 7;

const DIGITS_WHITELIST = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩";

// Hard positional rules for YYYY/MM/DD.
const POSITION_RULES = {
  year: [["1"], ["3", "4"], null, null],
  month: [["0", "1"], null],
  day: [["0", "1", "2", "3"], null],
};

// Visual confusion map — repairs OCR output that violates the rules above.
const DIGIT_CONFUSIONS = {
  0: [
    { d: "5", c: 1 }, { d: "9", c: 1 }, { d: "6", c: 2 }, { d: "8", c: 2 },
  ],
  1: [{ d: "7", c: 1 }, { d: "4", c: 2 }],
  2: [{ d: "3", c: 2 }, { d: "7", c: 2 }],
  3: [{ d: "8", c: 1 }, { d: "2", c: 2 }],
  4: [{ d: "1", c: 2 }, { d: "9", c: 2 }],
  5: [{ d: "0", c: 1 }, { d: "6", c: 1 }, { d: "9", c: 2 }],
  6: [{ d: "5", c: 1 }, { d: "0", c: 2 }, { d: "8", c: 2 }],
  7: [{ d: "1", c: 1 }, { d: "2", c: 2 }],
  8: [{ d: "0", c: 2 }, { d: "6", c: 2 }, { d: "3", c: 2 }],
  9: [{ d: "0", c: 1 }, { d: "5", c: 2 }, { d: "4", c: 2 }],
};

// 33-year arithmetic break years (same list as jalaali-js; verified to
// agree with it for 1280–1450).
const JALALI_BREAK_YEARS = [
  -61, 9, 38, 199, 426, 686, 756, 818, 1111, 1181, 1210, 1635, 2060, 2097,
  2192, 2262, 2324, 2394, 2456, 3178,
];

const DATE_FIELDS = ["year", "month", "day"];

// OCR budget / quality gates.
const GLYPH_MIN_CONFIDENCE = 45; // per-glyph SINGLE_CHAR reads below this are discarded
const MAX_REPAIRS = 1; // more confusion-repairs → not a real date line
const SEPARATOR_SCORE_BONUS = 35; // "/" evidence outranks center-biased ID rows

// OCR_TRACE=1 → stderr diagnostics (Node only; never enabled in browsers).
const TRACE =
  typeof process !== "undefined" && Boolean(process.env && process.env.OCR_TRACE);
const SEGMENT_TRUST_FACTOR = 0.7; // segment votes are weaker than per-glyph votes
const MIN_MEAN_CONFIDENCE = 35; // below this the whole read is rejected
const MAX_CANDIDATES = 3; // date candidates to OCR-verify per orientation
const MAX_OCR_CALLS = 24; // hard cap per recognition (all candidates/orientations)

let openCVPromise = null;
let nodeCanvasPromise = null;

/* ========================================================================
   OpenCV loader (lazy, cached)
   ======================================================================== */

async function getOpenCV() {
  if (openCVPromise) return openCVPromise;

  openCVPromise = (async () => {
    const imported = await import("@techstark/opencv-js");
    let cv = imported.default ?? imported;
    if (cv && typeof cv.then === "function") cv = await cv;
    if (cv?.Mat) return cv;
    if (!cv) throw new Error("OpenCV runtime unavailable.");

    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("OpenCV initialisation timed out.")),
        OPENCV_INIT_TIMEOUT_MS,
      );
      const previous = cv.onRuntimeInitialized;
      cv.onRuntimeInitialized = () => {
        clearTimeout(timer);
        if (typeof previous === "function") previous();
        resolve();
      };
    });

    return cv;
  })().catch((error) => {
    openCVPromise = null;
    throw error;
  });

  return openCVPromise;
}

/* ========================================================================
   Small utilities
   ======================================================================== */

function deleteMats(...values) {
  for (const value of values) {
    try { value?.delete?.(); } catch { /* already freed */ }
  }
}

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** Packed bytes for an 8-bit mat — .data is only trustworthy when continuous. */
function matBytes(mat) {
  if (mat.isContinuous?.()) return mat.data;
  const ch = mat.channels();
  const packed = new Uint8Array(mat.rows * mat.cols * ch);
  for (let y = 0; y < mat.rows; y++) {
    for (let x = 0; x < mat.cols; x++) {
      const p = mat.ucharPtr(y, x);
      const base = (y * mat.cols + x) * ch;
      for (let c = 0; c < ch; c++) packed[base + c] = p[c];
    }
  }
  return packed;
}

/**
 * Mat -> PNG data URL. Uses the DOM canvas in the browser; in Node (tests)
 * it falls back to @napi-rs/canvas if that package happens to be installed.
 * Never pulls the Node path into a browser bundle (dynamic, ignored import).
 */
async function matToDataUrl(cv, mat) {
  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    canvas.width = mat.cols;
    canvas.height = mat.rows;
    cv.imshow(canvas, mat);
    return canvas.toDataURL("image/png");
  }

  if (!nodeCanvasPromise) {
    const spec = ["@napi-rs", "canvas"].join("/");
    nodeCanvasPromise = import(/* @vite-ignore */ spec).catch(() => null);
  }
  const napi = await nodeCanvasPromise;
  if (!napi) {
    throw new Error(
      "Rendering outside the browser requires the optional @napi-rs/canvas package.",
    );
  }
  const { createCanvas } = napi;
  const canvas = createCanvas(mat.cols, mat.rows);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(mat.cols, mat.rows);
  const bytes = matBytes(mat);
  const ch = mat.channels();
  const out = image.data;
  if (ch === 4) {
    out.set(bytes.subarray(0, mat.cols * mat.rows * 4));
  } else if (ch === 1) {
    for (let i = 0, n = mat.cols * mat.rows; i < n; i++) {
      const v = bytes[i];
      out[i * 4] = v;
      out[i * 4 + 1] = v;
      out[i * 4 + 2] = v;
      out[i * 4 + 3] = 255;
    }
  } else if (ch === 3) {
    for (let i = 0, n = mat.cols * mat.rows; i < n; i++) {
      out[i * 4] = bytes[i * 3];
      out[i * 4 + 1] = bytes[i * 3 + 1];
      out[i * 4 + 2] = bytes[i * 3 + 2];
      out[i * 4 + 3] = 255;
    }
  } else {
    throw new Error(`Unsupported channel count: ${ch}`);
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function toGray(cv, src) {
  const gray = new cv.Mat();
  if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  else src.copyTo(gray);
  return gray;
}

function binarizeInk(cv, grayMat) {
  const binary = new cv.Mat();
  cv.threshold(grayMat, binary, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);
  if (cv.countNonZero(binary) / (binary.rows * binary.cols) > 0.55) {
    cv.bitwise_not(binary, binary);
  }
  return binary;
}

function upscaleGray(cv, grayMat, targetHeight) {
  const scale = Math.max(
    MIN_UPSCALE,
    Math.min(MAX_UPSCALE, targetHeight / Math.max(1, grayMat.rows)),
  );
  const result = new cv.Mat();
  cv.resize(
    grayMat,
    result,
    new cv.Size(
      Math.max(1, Math.round(grayMat.cols * scale)),
      Math.max(1, Math.round(grayMat.rows * scale)),
    ),
    0, 0, cv.INTER_CUBIC,
  );
  return result;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** nth percentile (0..1) of a numeric array. */
function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

function normalizeDigits(text) {
  if (!text) return "";
  return String(text)
    .replace(/[۰-۹]/g, (c) => String("۰۱۲۳۴۵۶۷۸۹".indexOf(c)))
    .replace(/[٠-٩]/g, (c) => String("٠١٢٣٤٥٦٧٨٩".indexOf(c)));
}

function digitString(text) {
  return normalizeDigits(text).replace(/\D/g, "");
}

/**
 * Extract the single digit a glyph OCR read. Word mode sometimes emits the
 * same digit twice in both scripts ("۱1") — collapse duplicates; anything
 * genuinely ambiguous (two different digits) is unusable.
 *
 * @returns {string|null}
 */
export function singleDigitFrom(text) {
  const digits = digitString(text);
  if (digits.length === 1) return digits;
  if (digits.length > 1 && digits.every((ch) => ch === digits[0])) return digits[0];
  return null;
}

/* ========================================================================
   Image loading
   ======================================================================== */

async function loadImageElement(input) {
  if (typeof window === "undefined") {
    throw new Error(
      "Image loading requires a browser environment. In Node, pass a cv.Mat.",
    );
  }

  if (typeof HTMLImageElement !== "undefined" && input instanceof HTMLImageElement) {
    if (input.complete && input.naturalWidth) return input;
    await new Promise((resolve, reject) => {
      input.onload = () => resolve();
      input.onerror = () => reject(new Error("Image failed to load."));
    });
    return input;
  }

  if (typeof HTMLCanvasElement !== "undefined" && input instanceof HTMLCanvasElement) {
    return input;
  }

  let src = input;
  let objectUrl = null;
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    objectUrl = URL.createObjectURL(input);
    src = objectUrl;
  }

  try {
    const image = new Image();
    image.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("Image failed to load."));
      image.src = src;
    });
    return image;
  } finally {
    // Revoke only after the load has completed (or failed).
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}

/* ========================================================================
   Jalali calendar helpers
   ======================================================================== */

export function isJalaliLeapYear(year) {
  let previousBreak = JALALI_BREAK_YEARS[0];
  let currentBreak = previousBreak;
  let jump = 0;

  for (let i = 1; i < JALALI_BREAK_YEARS.length; i++) {
    currentBreak = JALALI_BREAK_YEARS[i];
    jump = currentBreak - previousBreak;
    if (year < currentBreak) break;
    previousBreak = currentBreak;
  }

  if (year >= JALALI_BREAK_YEARS[JALALI_BREAK_YEARS.length - 1]) return false;

  let yearsSinceBreak = year - previousBreak;
  if (jump - yearsSinceBreak < 6) {
    yearsSinceBreak = yearsSinceBreak - jump + Math.floor((jump + 4) / 33) * 33;
  }
  let leap = (((yearsSinceBreak + 1) % 33) - 1) % 4;
  if (leap === -1) leap = 4;
  return leap === 0;
}

export function isValidJalaliDate(year, month, day) {
  if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
    return false;
  }
  if (year < 1300 || year > 1420) return false;
  if (month < 1 || month > 12) return false;
  if (day < 1) return false;

  if (month <= 6) return day <= 31;
  if (month <= 11) return day <= 30;
  // Month 12 (Esfand)
  if (day > 30) return false;
  return day !== 30 || isJalaliLeapYear(year);
}

/* ========================================================================
   Card detection + rectification
   ======================================================================== */

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const next = points[(i + 1) % points.length];
    area += points[i].x * next.y - next.x * points[i].y;
  }
  return Math.abs(area / 2);
}

function orderQuadCorners(points) {
  const sum = (p) => p.x + p.y;
  const diff = (p) => p.x - p.y;
  const tl = points.reduce((b, p) => (sum(p) < sum(b) ? p : b));
  const br = points.reduce((b, p) => (sum(p) > sum(b) ? p : b));
  const tr = points.reduce((b, p) => (diff(p) > diff(b) ? p : b));
  const bl = points.reduce((b, p) => (diff(p) < diff(b) ? p : b));
  const ordered = [tl, tr, br, bl];
  return new Set(ordered).size === 4 ? ordered : null;
}

function detectCardQuad(cv, image) {
  let gray, blurred, edges, closed, kernel, contourInput, contours, hierarchy;
  let best = null;

  try {
    gray = toGray(cv, image);
    blurred = new cv.Mat();
    edges = new cv.Mat();
    closed = new cv.Mat();
    kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();

    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 45, 140);
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
    contourInput = closed.clone();
    cv.findContours(contourInput, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const imageArea = image.cols * image.rows;
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const approx = new cv.Mat();
      try {
        if (Math.abs(cv.contourArea(contour)) / imageArea < 0.28) continue;
        const perimeter = cv.arcLength(contour, true);
        if (perimeter < Math.min(image.cols, image.rows) * 0.5) continue;
        cv.approxPolyDP(contour, approx, perimeter * 0.02, true);
        if (approx.rows !== 4) continue;

        const data = approx.data32S;
        const pts = Array.from({ length: 4 }, (_, k) => ({
          x: data[k * 2], y: data[k * 2 + 1],
        }));
        const area = polygonArea(pts);
        const ordered = orderQuadCorners(pts);
        if (!ordered || area / imageArea < 0.28) continue;

        const [tl, tr, br, bl] = ordered;
        const w = (Math.hypot(tr.x - tl.x, tr.y - tl.y) + Math.hypot(br.x - bl.x, br.y - bl.y)) / 2;
        const h = (Math.hypot(bl.x - tl.x, bl.y - tl.y) + Math.hypot(br.x - tr.x, br.y - tr.y)) / 2;
        const aspect = Math.max(w, h) / Math.min(w, h);
        if (aspect < 1.25 || aspect > 2.05) continue;

        if (!best || area > best.area) {
          best = { points: ordered, area, portrait: w < h };
        }
      } finally {
        deleteMats(contour, approx);
      }
    }
  } catch { /* fall through */ } finally {
    deleteMats(gray, blurred, edges, closed, kernel, contourInput, contours, hierarchy);
  }

  return best;
}

function rectifyCard(cv, image, quad) {
  if (!quad) return null;
  let src, dst, transform, warped, rotated;
  try {
    const outW = quad.portrait ? 760 : 1200;
    const outH = quad.portrait ? 1200 : 760;
    const [tl, tr, br, bl] = quad.points;

    src = cv.matFromArray(4, 1, cv.CV_32FC2, [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]);
    dst = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, 0, outW - 1, 0, outW - 1, outH - 1, 0, outH - 1,
    ]);
    transform = cv.getPerspectiveTransform(src, dst);
    warped = new cv.Mat();
    cv.warpPerspective(
      image, warped, transform,
      new cv.Size(outW, outH),
      cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar(0, 0, 0, 0),
    );

    if (quad.portrait) {
      rotated = new cv.Mat();
      cv.rotate(warped, rotated, cv.ROTATE_90_CLOCKWISE);
      deleteMats(warped);
      const r = rotated;
      rotated = null;
      return r;
    }
    const r = warped;
    warped = null;
    return r;
  } catch {
    return null;
  } finally {
    deleteMats(src, dst, transform, warped, rotated);
  }
}

export function createAnalysisImage(cv, source) {
  let resized;
  try {
    const maxDim = Math.max(source.cols, source.rows);
    const scale = Math.min(1.4, MAX_ANALYSIS_DIMENSION / maxDim);
    resized = new cv.Mat();
    cv.resize(
      source, resized,
      new cv.Size(
        Math.max(1, Math.round(source.cols * scale)),
        Math.max(1, Math.round(source.rows * scale)),
      ),
      0, 0, scale < 1 ? cv.INTER_AREA : cv.INTER_CUBIC,
    );
    const quad = detectCardQuad(cv, resized);
    const rect = rectifyCard(cv, resized, quad);
    if (rect) {
      deleteMats(resized);
      return rect;
    }
    const r = resized;
    resized = null;
    return r;
  } finally {
    deleteMats(resized);
  }
}

function rotate180(cv, mat) {
  const out = new cv.Mat();
  cv.rotate(mat, out, cv.ROTATE_180);
  return out;
}

/* ========================================================================
   Glyph extraction + annotation
   ======================================================================== */

function collectGlyphs(cv, binaryInk) {
  const boxes = [];
  let input, contours, hierarchy;
  try {
    input = binaryInk.clone();
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(input, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      try {
        const r = cv.boundingRect(contour);
        if (r.width * r.height < GLYPH_MIN_PIXELS) continue;
        boxes.push({
          x: r.x, y: r.y, width: r.width, height: r.height,
          centerX: r.x + r.width / 2,
          centerY: r.y + r.height / 2,
          right: r.x + r.width,
          bottom: r.y + r.height,
        });
      } finally {
        deleteMats(contour);
      }
    }
  } catch { /* ignore */ } finally {
    deleteMats(input, contours, hierarchy);
  }
  return boxes.sort((a, b) => a.x - b.x);
}

function countHoles(cv, binaryInk, box) {
  let roi, padded, contours, hierarchy;
  let holes = 0;
  try {
    roi = binaryInk.roi(new cv.Rect(box.x, box.y, box.width, box.height));
    padded = new cv.Mat();
    cv.copyMakeBorder(roi, padded, 2, 2, 2, 2, cv.BORDER_CONSTANT, new cv.Scalar(0));
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(padded, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    const minHoleArea = Math.max(2, box.width * box.height * 0.015);
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      try {
        const parent = hierarchy.intPtr(0, i)[3];
        if (parent !== -1 && cv.contourArea(contour) >= minHoleArea) holes++;
      } finally {
        deleteMats(contour);
      }
    }
  } catch { /* ignore */ } finally {
    deleteMats(roi, padded, contours, hierarchy);
  }
  return holes;
}

/**
 * Ink extents of the top/bottom 30% bands of a glyph box — used to tell a
 * diagonal "/" from vertical strokes (۱) and top bars (۷). Read through
 * ucharPtr: OpenCV.js .data lies about ROI stride.
 */
function bandInk(binaryInk, box) {
  const band = Math.max(2, Math.round(box.height * 0.3));
  const extent = (y0, y1) => {
    let min = -1;
    let max = -1;
    for (let x = 0; x < box.width; x++) {
      const col = box.x + x;
      for (let y = y0; y < y1; y++) {
        if (binaryInk.ucharPtr(y, col)[0]) {
          if (min < 0) min = x;
          max = x;
          break;
        }
      }
    }
    return { width: max >= 0 ? max - min + 1 : 0, center: max >= 0 ? (min + max) / 2 : -1 };
  };
  return {
    top: extent(box.y, box.y + band),
    bottom: extent(box.y + box.height - band, box.y + box.height),
  };
}

function isSlashGlyph(box, bands) {
  const aspect = box.width / Math.max(1, box.height);
  if (aspect < SLASH_MIN_ASPECT || aspect > SLASH_MAX_ASPECT) return false;
  if (box.relativeHeight < 0.65) return false;
  if (box.holes !== 0) return false;
  if (bands.top.width > SLASH_MAX_BAND_WIDTH * box.width) return false;
  if (bands.bottom.width > SLASH_MAX_BAND_WIDTH * box.width) return false;
  if (bands.top.center < 0 || bands.bottom.center < 0) return false;
  const shift = Math.abs(bands.top.center - bands.bottom.center);
  return shift >= SLASH_MIN_BAND_SHIFT * box.width;
}

function annotateGlyph(cv, binaryInk, box, refHeight) {
  const rel = box.height / Math.max(1, refHeight);
  const holes = countHoles(cv, binaryInk, box);
  const widthRatio = box.width / Math.max(1, refHeight);
  const annotated = {
    ...box,
    holes,
    relativeHeight: rel,
    // ۰: short (≤0.58×), not wider than a digit, at most a ring hole.
    isZeroDot:
      rel <= ZERO_HEIGHT_RATIO &&
      widthRatio <= ZERO_MAX_WIDTH_RATIO &&
      holes <= 1,
    isSeparator: false,
  };
  // Slash test needs the band extents (cheap; only for plausible shapes).
  if (
    rel >= 0.65 &&
    holes === 0 &&
    box.width / Math.max(1, box.height) >= SLASH_MIN_ASPECT &&
    box.width / Math.max(1, box.height) <= SLASH_MAX_ASPECT
  ) {
    const bands = bandInk(binaryInk, box);
    annotated.isSeparator = isSlashGlyph(annotated, bands);
    if (TRACE && !annotated.isSeparator) {
      console.error(
        `TRACE slash-reject box=${box.x},${box.y},${box.width}x${box.height} ` +
        `asp=${(box.width / box.height).toFixed(2)} rel=${rel.toFixed(2)} holes=${holes} ` +
        `topW=${bands.top.width}/${box.width} botW=${bands.bottom.width}/${box.width} ` +
        `shift=${Math.abs(bands.top.center - bands.bottom.center).toFixed(1)}`,
      );
    }
  }
  return annotated;
}

/* ========================================================================
   Row clustering + 4-2-2 pattern search
   ======================================================================== */

export function groupIntoRows(glyphs, refHeight) {
  if (!glyphs.length) return [];
  const tolerance = Math.max(6, (refHeight || median(glyphs.map((g) => g.height))) * 0.4);
  const sorted = [...glyphs].sort((a, b) => a.centerY - b.centerY);
  const rows = [];
  for (const glyph of sorted) {
    const row = rows.find((r) => Math.abs(r.centerY - glyph.centerY) <= tolerance);
    if (row) {
      row.glyphs.push(glyph);
      row.centerY = (row.centerY * (row.glyphs.length - 1) + glyph.centerY) / row.glyphs.length;
    } else {
      rows.push({ centerY: glyph.centerY, glyphs: [glyph] });
    }
  }
  return rows.map((r) => ({ ...r, glyphs: r.glyphs.sort((a, b) => a.x - b.x) }));
}

/** Split a x-sorted glyph list into clusters using a gap > refHeight·factor. */
export function clusterByGaps(glyphs, refHeight, factor) {
  if (!glyphs.length) return [];
  const splitGap = Math.max(refHeight * factor, 5);
  const clusters = [[glyphs[0]]];
  for (let i = 1; i < glyphs.length; i++) {
    const gap = glyphs[i].x - glyphs[i - 1].right;
    if (gap > splitGap) clusters.push([glyphs[i]]);
    else clusters[clusters.length - 1].push(glyphs[i]);
  }
  return clusters;
}

/** Drop "/" glyphs hanging off the edges of a cluster (attached print). */
function clusterDigits(cluster) {
  const glyphs = [...cluster];
  while (glyphs.length && glyphs[0].isSeparator) glyphs.shift();
  while (glyphs.length && glyphs[glyphs.length - 1].isSeparator) glyphs.pop();
  return glyphs;
}

/**
 * Match a window of clusters as [4 year][2 month][2 day] digits, tolerating
 * separators attached to the field clusters and one standalone separator
 * cluster between fields ([4][/][2][/][2]).
 */
function matchTriple(window) {
  // After stripping edge-attached separators, a field must contain no
  // separator at all — a year never has a "/" inside it, and interior
  // slashes mean this is garbage geometry (clipped/merged rows).
  const field = (cluster) => {
    const g = clusterDigits(cluster);
    return g.some((gl) => gl.isSeparator) ? null : g;
  };
  const ok = (g, n) => g !== null && g.length === n;
  const withSepFlag = (groups, separators) => {
    groups.separators = separators;
    return groups;
  };
  // Standalone or edge-attached "/" evidence strongly favors this match
  // over bare digit runs (national IDs, serials) in scoring.
  const hasAttachedSep = (clusters) =>
    clusters.some((cl) => cl.some((gl) => gl.isSeparator));

  if (window.length === 1) {
    // No separators detected (thin crops, low contrast): year+month+day
    // collapse into a single 8-digit cluster. Split it by position.
    const g = field(window[0]);
    if (g && g.length === 8) {
      return withSepFlag([g.slice(0, 4), g.slice(4, 6), g.slice(6, 8)], false);
    }
    return null;
  }
  if (window.length === 3) {
    const [a, b, c] = window.map(field);
    if (ok(a, 4) && ok(b, 2) && ok(c, 2)) {
      return withSepFlag([a, b, c], hasAttachedSep(window));
    }
    return null;
  }
  if (window.length === 5) {
    const [a, s1, b, s2, c] = window;
    // The two middle clusters must be pure separators ("/" alone, possibly
    // with dust). Do not run clusterDigits here — it would strip them empty.
    const isSepCluster = (cluster) =>
      cluster.length >= 1 && cluster.every((gl) => gl.isSeparator);
    if (isSepCluster(s1) && isSepCluster(s2)) {
      const A = field(a);
      const B = field(b);
      const C = field(c);
      if (ok(A, 4) && ok(B, 2) && ok(C, 2)) return withSepFlag([A, B, C], true);
    }
  }
  return null;
}

/**
 * Sliding-window search over the row's clusters: the date may sit anywhere
 * in a label+date row, and upside-down captures present it right-to-left.
 *
 * @returns {{groups: {year: [], month: [], day: []}, reversed: boolean}|null}
 */
export function findDatePattern(clusters) {
  const scan = (list) => {
    for (let i = 0; i < list.length; i++) {
      const w1 = matchTriple(list.slice(i, i + 1));
      if (w1) {
        return {
          groups: { year: w1[0], month: w1[1], day: w1[2] },
          reversed: false,
          hadSeparators: Boolean(w1.separators),
        };
      }
      const w3 = matchTriple(list.slice(i, i + 3));
      if (w3) {
        return {
          groups: { year: w3[0], month: w3[1], day: w3[2] },
          reversed: false,
          hadSeparators: Boolean(w3.separators),
        };
      }
      const w5 = matchTriple(list.slice(i, i + 5));
      if (w5) {
        return {
          groups: { year: w5[0], month: w5[1], day: w5[2] },
          reversed: false,
          hadSeparators: Boolean(w5.separators),
        };
      }
    }
    return null;
  };

  const forward = scan(clusters);
  if (forward) return forward;
  const backward = scan([...clusters].reverse());
  if (backward) {
    return {
      groups: backward.groups,
      reversed: true,
      hadSeparators: backward.hadSeparators,
    };
  }
  return null;
}

function scoreCandidate(triple, image) {
  const all = [...triple.year, ...triple.month, ...triple.day];
  const centerY = all.reduce((s, g) => s + g.centerY, 0) / all.length;
  const relativeY = centerY / image.rows;

  // A legitimate zero-dot glyph (۰ in ۰۱–۰۹) is genuinely half-height —
  // excluding it keeps a real date from being outranked by plain digit rows.
  const heights = all.filter((g) => !g.isZeroDot).map((g) => g.height);
  const hs = heights.length || 1;
  const meanH = heights.reduce((a, b) => a + b, 0) / hs;
  const variance = heights.reduce((a, b) => a + (b - meanH) ** 2, 0) / hs;
  const dispersion = Math.sqrt(variance) / Math.max(1, meanH);

  const left = Math.min(...all.map((g) => g.x));
  const right = Math.max(...all.map((g) => g.right));
  const relativeWidth = (right - left) / image.cols;

  const distanceToIdeal = Math.abs(relativeY - CENTER_BAND_IDEAL);
  const centerBonus = Math.max(0, 1 - distanceToIdeal / (CENTER_BAND_MAX - CENTER_BAND_IDEAL));
  const widthBonus = Math.max(0, 1 - Math.abs(relativeWidth - 0.38) / 0.3);
  const uniformBonus = Math.max(0, 1 - dispersion * 2.2);
  const inBand = relativeY >= CENTER_BAND_MIN && relativeY <= CENTER_BAND_MAX;

  return 80 * centerBonus + 35 * widthBonus + 25 * uniformBonus + (inBand ? 20 : -80);
}

/**
 * Full geometric date search: binarise, extract glyphs, cluster rows under
 * several gap thresholds (with and without separators), collect scored
 * candidates. Returns { candidates, reversed } — when the top candidate
 * matched right-to-left the capture is upside down.
 */
export function detectDateCandidates(cv, image) {
  let gray, normalized, binary;
  try {
    gray = toGray(cv, image);
    normalized = new cv.Mat();
    if (cv.mean(gray)[0] < 110) cv.bitwise_not(gray, normalized);
    else gray.copyTo(normalized);
    binary = binarizeInk(cv, normalized);

    const all = collectGlyphs(cv, binary);
    if (TRACE) console.error(`TRACE detect: all=${all.length}`);
    if (!all.length) return { candidates: [], reversed: false };

    const minY = CENTER_BAND_MIN * image.rows;
    const maxY = CENTER_BAND_MAX * image.rows;
    const centered = all.filter((b) => b.centerY >= minY && b.centerY <= maxY);
    if (TRACE) console.error(`TRACE detect: centered=${centered.length}`);
    if (centered.length < 8) return { candidates: [], reversed: false };

    const refHeight = percentile(centered.map((b) => b.height), 0.75) || 1;

    const digitLike = centered.filter((b) => {
      const aspect = b.width / Math.max(1, b.height);
      const rel = b.height / refHeight;
      // Tall-and-thin slashes are allowed even when taller than the digit
      // band (short rows make digits' relative height look large); letters
      // still get rejected later by the separator gates (shift/topW/botW).
      const slashish =
        aspect >= 0.3 && aspect <= 0.95 && rel >= 0.65 &&
        b.height > refHeight * 1.2;
      return (
        (aspect <= GLYPH_MAX_ASPECT &&
          rel >= GLYPH_MIN_HEIGHT_RATIO &&
          rel <= GLYPH_MAX_HEIGHT_RATIO) ||
        slashish
      );
    });
    if (TRACE) console.error(`TRACE detect: digitLike=${digitLike.length} refH=${refHeight.toFixed(1)}`);
    if (digitLike.length < 8) return { candidates: [], reversed: false };

    const annotated = digitLike.map((b) => annotateGlyph(cv, binary, b, refHeight));
    const rows = groupIntoRows(annotated, refHeight);
    if (TRACE) {
      console.error(`TRACE detect: rows=${rows.length} sizes=${rows.map((r) => r.glyphs.length).join(",")}`);
      for (const row of rows) {
        if (row.glyphs.length >= 8) {
          console.error(`TRACE row: ${row.glyphs.map((g) => `${Math.round(g.x)}:${Math.round(g.width)}${g.isSeparator ? "/" : ""}${g.isZeroDot ? "z" : ""}`).join(" ")}`);
          console.error(`TRACE row matches: ${findDateInRow(row.glyphs, refHeight).length}`);
        }
      }
    }

    const candidates = [];
    const seen = new Set();

    for (const row of rows) {
      if (row.glyphs.length < 8) continue;
      for (const match of findDateInRow(row.glyphs, refHeight)) {
        const key = [
          match.groups.year[0].x,
          match.groups.month[0].x,
          match.groups.day[0].x,
          match.reversed ? "r" : "f",
        ].join("|");
        if (seen.has(key)) continue;
        seen.add(key);
        // Real date lines carry "/" separators; bare digit runs (national
        // IDs, serials) do not. The bonus (tuned against the center-position
        // bias of ID rows) decides the OCR-verification order and which
        // orientation gets tried first.
        candidates.push({
          groups: match.groups,
          reversed: match.reversed,
          hadSeparators: Boolean(match.hadSeparators),
          score:
            scoreCandidate(match.groups, image) +
            (match.hadSeparators ? SEPARATOR_SCORE_BONUS : 0),
        });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    if (TRACE) {
      for (const c of candidates) {
        console.error(
          `TRACE cand score=${c.score.toFixed(0)} yearX=${Math.round(c.groups.year[0].x)} ` +
          `monthX=${Math.round(c.groups.month[0].x)} dayX=${Math.round(c.groups.day[0].x)} ` +
          `rev=${c.reversed} sep=${c.hadSeparators}`,
        );
      }
    }
    return {
      candidates: candidates.slice(0, MAX_CANDIDATES * 2),
      reversed: candidates.length > 0 && candidates[0].reversed,
    };
  } catch {
    return { candidates: [], reversed: false };
  } finally {
    deleteMats(gray, normalized, binary);
  }
}

/**
 * The gap-threshold attempt matrix for one row: separators attached /
 * standalone / removed, several split factors, forward and reversed.
 * Pure — exported so tests exercise exactly what detection runs.
 *
 * @returns {{groups: {year: [], month: [], day: []}, reversed: boolean}[]}
 */
export function findDateInRow(glyphs, refHeight) {
  const matches = [];
  const seen = new Set();
  const rowSeparators = glyphs.filter((g) => g.isSeparator);
  const attempts = [
    { list: glyphs, factors: SEGMENT_GAP_FACTORS },
    { list: glyphs.filter((g) => !g.isSeparator), factors: DIGIT_GAP_FACTORS },
  ];
  for (const attempt of attempts) {
    if (attempt.list.length < 8) continue;
    for (const factor of attempt.factors) {
      const clusters = clusterByGaps(attempt.list, refHeight, factor);
      const match = findDatePattern(clusters);
      if (!match) continue;
      // Separator evidence at ROW level: the separator-stripped clustering
      // path above can match a date whose "/" glyphs never enter the window,
      // so check whether any "/" of this row sits inside the date span.
      const groupsAll = [
        ...match.groups.year, ...match.groups.month, ...match.groups.day,
      ];
      const minX = Math.min(...groupsAll.map((g) => g.x)) - refHeight;
      const maxX = Math.max(...groupsAll.map((g) => g.right)) + refHeight;
      const sepNear = rowSeparators.some(
        (s) => s.right >= minX && s.x <= maxX,
      );
      match.hadSeparators = Boolean(match.hadSeparators || sepNear);
      const key = [
        match.groups.year[0].x,
        match.groups.month[0].x,
        match.groups.day[0].x,
        match.reversed ? "r" : "f",
      ].join("|");
      if (seen.has(key)) continue;
      seen.add(key);
      matches.push(match);
    }
  }
  return matches;
}

/* ========================================================================
   Segment + glyph rendering for Tesseract
   ======================================================================== */

/** Crop the bounding box (plus padding) around a set of glyph boxes. */
function cropBoxesMat(cv, image, glyphs) {
  if (!glyphs.length) return null;
  const left = Math.min(...glyphs.map((g) => g.x));
  const top = Math.min(...glyphs.map((g) => g.y));
  const right = Math.max(...glyphs.map((g) => g.right));
  const bottom = Math.max(...glyphs.map((g) => g.bottom));

  const padX = Math.max(5, Math.round((bottom - top) * 0.3));
  const padY = Math.max(5, Math.round((bottom - top) * 0.35));
  const x = Math.max(0, left - padX);
  const y = Math.max(0, top - padY);
  const w = Math.min(image.cols, right + padX) - x;
  const h = Math.min(image.rows, bottom + padY) - y;
  if (w < 10 || h < 10) return null;

  let view, gray, normalized;
  try {
    view = image.roi(new cv.Rect(x, y, w, h));
    gray = toGray(cv, view);
    normalized = new cv.Mat();
    if (cv.mean(gray)[0] < 110) cv.bitwise_not(gray, normalized);
    else gray.copyTo(normalized);
    const r = normalized;
    normalized = null;
    return r;
  } finally {
    deleteMats(view, gray, normalized);
  }
}

async function renderSegment(cv, segmentMat) {
  let upscaled, padded;
  try {
    upscaled = upscaleGray(cv, segmentMat, TARGET_SEGMENT_HEIGHT);
    padded = new cv.Mat();
    const margin = Math.max(10, Math.round(upscaled.rows * 0.2));
    cv.copyMakeBorder(upscaled, padded, margin, margin, margin, margin, cv.BORDER_CONSTANT, new cv.Scalar(255));
    return await matToDataUrl(cv, padded);
  } finally {
    deleteMats(upscaled, padded);
  }
}

/**
 * Zero out every connected component except the largest. Persian digits are
 * always a single component, so this removes any neighbour fragments that
 * leaked into a glyph crop through the padding — SINGLE_CHAR OCR must see
 * exactly one glyph.
 */
function keepLargestComponent(cv, mat) {
  let binary, input, contours, hierarchy, mask;
  try {
    // Fixed threshold instead of Otsu: predictable separation of thin
    // anti-alias bridges between the digit and any leaked fragment.
    binary = new cv.Mat();
    cv.threshold(mat, binary, 175, 255, cv.THRESH_BINARY_INV);
    input = binary.clone();
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(input, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    if (contours.size() <= 1) return;

    let best = 0;
    let bestArea = -1;
    for (let i = 0; i < contours.size(); i++) {
      const area = Math.abs(cv.contourArea(contours.get(i)));
      if (area > bestArea) {
        bestArea = area;
        best = i;
      }
    }
    mask = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC1, new cv.Scalar(0));
    cv.drawContours(mask, contours, best, new cv.Scalar(255), -1);
    cv.bitwise_and(mat, mat, mat, mask);
  } catch { /* best effort */ } finally {
    deleteMats(binary, input, contours, hierarchy, mask);
  }
}

async function renderGlyph(cv, image, glyph) {
  let roi, gray, normalized, padded, upscaled;
  try {
    // Almost no horizontal padding: gaps between digits are only 4–6 px, so
    // a generous pad would pull neighbours into the SINGLE_CHAR image (and
    // anti-alias bridges can even connect them). The white margin added
    // below gives Tesseract all the breathing room it needs.
    const padX = 1;
    const padY = Math.max(2, Math.round(glyph.height * 0.25));
    const x = Math.max(0, glyph.x - padX);
    const y = Math.max(0, glyph.y - padY);
    const w = Math.min(image.cols, glyph.right + padX) - x;
    const h = Math.min(image.rows, glyph.bottom + padY) - y;
    if (w < 3 || h < 3) return null;

    roi = image.roi(new cv.Rect(x, y, w, h));
    gray = toGray(cv, roi);
    normalized = new cv.Mat();
    if (cv.mean(gray)[0] < 110) cv.bitwise_not(gray, normalized);
    else gray.copyTo(normalized);
    keepLargestComponent(cv, normalized);

    padded = new cv.Mat();
    const margin = Math.max(8, Math.round(h * 0.4));
    cv.copyMakeBorder(normalized, padded, margin, margin, margin, margin, cv.BORDER_CONSTANT, new cv.Scalar(255));
    upscaled = upscaleGray(cv, padded, TARGET_GLYPH_HEIGHT);
    return await matToDataUrl(cv, upscaled);
  } finally {
    deleteMats(roi, gray, normalized, padded, upscaled);
  }
}

/* ========================================================================
   Digit choice logic (pure)
   ======================================================================== */

/**
 * Choose the digit for one position from ranked sources.
 *
 * @param {{digit: string, conf: number}[]} sources ordered strongest first
 * @param {string[]|null} allowed positional rule
 * @returns {{digit: string, conf: number, repaired: boolean}|null}
 */
export function chooseDigit(sources, allowed) {
  const allowedSet = allowed ? new Set(allowed) : null;
  const isAllowed = (d) => !allowedSet || allowedSet.has(d);
  const usable = (sources || []).filter((s) => s && s.digit);

  // 1) A source that directly satisfies the positional rule wins.
  for (const source of usable) {
    if (isAllowed(source.digit)) {
      return {
        digit: source.digit,
        conf: source.conf,
        repaired: Boolean(source.repaired),
      };
    }
  }

  // 2) Otherwise repair through the visual confusion table.
  for (const source of usable) {
    const repair = [...(DIGIT_CONFUSIONS[source.digit] || [])]
      .sort((a, b) => a.c - b.c)
      .find((e) => allowedSet.has(e.d));
    if (repair) {
      return { digit: repair.d, conf: source.conf * 0.8, repaired: true };
    }
  }

  return null;
}

/**
 * Align a segment-level digit string with the field's glyph positions.
 * When one digit is missing the usual cause is a dropped leading zero —
 * right-align unless the position-1 glyph is itself a zero dot.
 *
 * @returns {(string|null)[]} per-position segment digit (null = unavailable)
 */
export function alignSegmentDigits(digits, length, pos0ZeroDot, pos1ZeroDot) {
  const out = new Array(length).fill(null);
  if (!digits) return out;
  if (digits.length === length) {
    for (let i = 0; i < length; i++) out[i] = digits[i];
    return out;
  }
  if (digits.length === length - 1) {
    const rightAlign = pos1ZeroDot && !pos0ZeroDot ? false : true;
    const offset = rightAlign ? 1 : 0;
    for (let i = 0; i < digits.length; i++) out[i + offset] = digits[i];
    return out;
  }
  return out; // unusable length — ignore the segment read
}

/* ========================================================================
   Main class
   ======================================================================== */

export class IranCardOCR {
  /**
   * @param {{langPath?: string}} [options] langPath overrides the default
   *   tesseract.js CDN (self-hosted folder containing fas.traineddata.gz).
   */
  constructor(options = {}) {
    this.worker = null;
    this.ready = false;
    this.initPromise = null;
    this.langPath = options.langPath || undefined;
    this._progress = null;
    this._paramKey = null;
    this._queue = Promise.resolve();
    this._logger = (m) => {
      const cb = this._progress;
      if (!cb || !m) return;
      const status = String(m.status ?? "");
      const p = typeof m.progress === "number" ? m.progress : 0;
      if (status.includes("loading tesseract core")) cb(4 + Math.round(p * 8));
      else if (status.includes("loading language")) cb(12 + Math.round(p * 18));
      else if (status.includes("initializing")) cb(33);
      else if (status.includes("recognizing text")) cb(55 + Math.round(p * 40));
    };
  }

  /**
   * Warm up OpenCV + the Tesseract worker (downloads fas.traineddata once,
   * ~400 KB, then cached by tesseract.js). May throw — call it up front if
   * you want fail-fast; recognizeBirthDate() calls it automatically and
   * never throws.
   */
  async initialize(onProgress) {
    if (this.ready) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      const previousProgress = this._progress;
      if (onProgress) this._progress = onProgress;
      try {
        const [, worker] = await Promise.all([
          getOpenCV(),
          createWorker("fas", OEM.LSTM_ONLY, {
            ...(this.langPath ? { langPath: this.langPath } : {}),
            // Keep tesseract.js's traineddata cache out of the CWD
            // (its default is "." → fas.traineddata dropped in the project
            // root). In Node a missing dir just skips caching silently; in
            // the browser the cache lives in IndexedDB under this prefix.
            cachePath: ".tesseract",
            logger: this._logger,
          }),
        ]);
        this.worker = worker;
        this._paramKey = null;
        this.ready = true;
      } finally {
        this._progress = previousProgress;
      }
    })();

    try {
      await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  async terminate() {
    if (this.initPromise) {
      try { await this.initPromise; } catch { /* ignore */ }
    }
    if (this.worker) {
      try { await this.worker.terminate(); } catch { /* ignore */ }
      this.worker = null;
    }
    this.ready = false;
    this.initPromise = null;
    this._paramKey = null;
    this._progress = null;
    this._queue = Promise.resolve();
  }

  /**
   * Main entry point — never throws.
   *
   *   const result = await ocr.recognizeBirthDate(file, onProgress?);
   *
   * success:
   *   { success: true, birthDate: "1366/06/22", year, month, day,
   *     confidence, repairs, durationMs, ocrCalls, attempts, lineImage }
   * failure:
   *   { success: false, error, durationMs, attempts }
   *
   * Accepts: File/Blob, data URL, http(s) URL, HTMLImageElement,
   * HTMLCanvasElement, or a cv.Mat (Node/tests). Recognition calls are
   * serialised internally — concurrent calls queue up safely.
   */
  recognizeBirthDate(input, onProgress) {
    const run = () => this._recognize(input, onProgress);
    const queued = this._queue.then(run, run);
    this._queue = queued.then(() => {}, () => {});
    return queued;
  }

  async _recognize(input, onProgress) {
    const started = now();
    const progress = typeof onProgress === "function" ? onProgress : null;
    this._progress = progress;
    let ocrCalls = 0;
    const attempts = [];

    const finishWith = (payload) => ({
      ...payload,
      durationMs: Math.round(now() - started),
      ...(ocrCalls ? { ocrCalls } : {}),
      ...(attempts.length ? { attempts } : {}),
    });

    try {
      try {
        await this.initialize(progress);
      } catch (error) {
        return finishWith({
          success: false,
          error: `OCR initialisation failed: ${error?.message || error}`,
        });
      }
      progress?.(38);

      const cv = await getOpenCV();

      let source = null;
      let ownedSource = false;
      if (input && typeof input === "object" && input.rows && input.cols && typeof input.delete === "function") {
        source = input; // caller-owned cv.Mat
      } else {
        const imageElement = await loadImageElement(input);
        source = cv.imread(imageElement);
        ownedSource = true;
      }

      let working = null;
      try {
        working = createAnalysisImage(cv, source);
        progress?.(46);

        const ctx = {
          cv,
          attempts,
          get ocrCalls() { return ocrCalls; },
          addCalls(n) { ocrCalls += n; },
          budgetLeft: () => ocrCalls < MAX_OCR_CALLS,
        };

        let outcome = await this.detectAndRun(ctx, working, 0);
        progress?.(60);

        // Second chance: flip 180° when the first orientation failed.
        if (!outcome.success && !outcome.rotated && (outcome.candidateCount > 0 || !attempts.length)) {
          const flipped = rotate180(cv, working);
          deleteMats(working);
          working = flipped;
          outcome = await this.detectAndRun(ctx, working, 180, outcome.error);
        }

        if (outcome.success) {
          progress?.(100);
          return finishWith({
            success: true,
            birthDate: outcome.payload.birthDate,
            year: outcome.payload.year,
            month: outcome.payload.month,
            day: outcome.payload.day,
            confidence: outcome.payload.confidence,
            repairs: outcome.payload.repairs,
            lineImage: outcome.payload.lineImage,
          });
        }

        return finishWith({
          success: false,
          error:
            outcome.error ||
            "Birth date not found. Ensure the card is well-lit and fully visible.",
        });
      } finally {
        deleteMats(ownedSource ? source : null, working);
      }
    } catch (error) {
      return finishWith({
        success: false,
        error: error?.message || String(error),
      });
    } finally {
      if (this._progress === progress) this._progress = null;
    }
  }

  /**
   * Detect date candidates on `image` (already oriented) and OCR-verify
   * them best-effort. Handles an upside-down top candidate by rotating once
   * and re-detecting before any OCR is spent.
   */
  async detectAndRun(ctx, image, rotation, priorError = null) {
    const { cv } = ctx;
    let detection = detectDateCandidates(cv, image);
    let rotated = rotation !== 0;

    if (detection.reversed) {
      // Capture is upside down (the year cluster appears right-to-left).
      // Rotate the working mat in place — 180° keeps rows/cols identical,
      // so copyTo is safe and all coordinates are re-detected afterwards.
      const flipped = rotate180(cv, image);
      flipped.copyTo(image);
      deleteMats(flipped);
      rotated = true;
      detection = detectDateCandidates(cv, image);
    }

    const candidates = detection.candidates;
    let error = priorError;

    for (const candidate of candidates.slice(0, MAX_CANDIDATES)) {
      if (!ctx.budgetLeft()) break;
      const result = await this.runCandidate(ctx, image, candidate, rotated);
      if (result.ok) return { success: true, payload: result.payload, rotated, candidateCount: candidates.length };
      error = result.error || error;
      if (result.reason === "budget") break;
    }

    return { success: false, error, rotated, candidateCount: candidates.length };
  }

  /**
   * OCR-verify one geometric candidate. Per-glyph SINGLE_CHAR reads first,
   * segment SINGLE_LINE fallback for fields whose glyph reads failed, then
   * positional rules + Jalali validation.
   */
  async runCandidate(ctx, image, candidate, rotated) {
    const { cv } = ctx;
    const groups = candidate.groups;
    const positions = [];

    for (const field of DATE_FIELDS) {
      const glyphs = groups[field] || [];
      if (glyphs.length !== POSITION_RULES[field].length) {
        return { ok: false, error: "Malformed date candidate." };
      }
      for (let i = 0; i < glyphs.length; i++) {
        positions.push({ field, index: i, glyph: glyphs[i] });
      }
    }

    // ---- Pass 1: render + per-glyph OCR (all eight glyphs) -------------
    // Two-stage reading, empirically tuned on these upscaled crops:
    //  1) SINGLE_WORD ('8') first — best single-mode score (7/8 here).
    //  2) A SINGLE_LINE second opinion for any glyph whose first read is
    //     unusable, low-confidence, OR violates the position's allowed
    //     digits (e.g. '۳' read as '1' in year#2 — the positional rule
    //     flags it, and the second mode reads '۳' with conf 95).
    const rendered = await Promise.all(
      positions.map((p) => renderGlyph(cv, image, p.glyph)),
    );

    ctx.addCalls(rendered.filter((u) => u !== null).length);

    const glyphReads = await Promise.all(
      rendered.map(async (url) => {
        if (!url) return { text: "", confidence: 0, digit: null };
        try {
          const read = await this.ocrDigits(url, PSM.SINGLE_WORD);
          return { ...read, digit: singleDigitFrom(read.text) };
        } catch {
          return { text: "", confidence: 0, digit: null };
        }
      }),
    );

    await Promise.all(
      positions.map(async (p, i) => {
        const read = glyphReads[i];
        const rule = POSITION_RULES[p.field][p.index];
        const violatesRule =
          read.digit !== null && rule !== null && !rule.includes(read.digit);
        const needsSecondOpinion =
          read.digit === null ||
          read.confidence < GLYPH_MIN_CONFIDENCE ||
          violatesRule;
        if (!rendered[i] || !needsSecondOpinion) return;

        ctx.addCalls(1);
        try {
          const retry = await this.ocrDigits(rendered[i], PSM.SINGLE_LINE);
          const digit = singleDigitFrom(retry.text);
          const usable =
            digit !== null &&
            retry.confidence >= GLYPH_MIN_CONFIDENCE &&
            (rule === null || rule.includes(digit));
          if (usable) glyphReads[i] = { ...retry, digit };
        } catch { /* keep first read */ }
      }),
    );

    positions.forEach((p, i) => {
      const read = glyphReads[i];
      p.glyphDigit =
        read.digit !== null && read.confidence >= GLYPH_MIN_CONFIDENCE
          ? { digit: read.digit, conf: read.confidence }
          : null;
      p.glyphRaw = digitString(read.text);
    });
    const glyphLog = positions.map((p, i) => ({
      f: p.field, i: p.index, t: glyphReads[i].text.trim(), c: glyphReads[i].confidence, d: p.glyphDigit?.digit ?? null,
    }));

    // ---- Pass 2: segment fallback for fields with unusable glyph reads --
    const fieldsNeedingSegment = DATE_FIELDS.filter((field) =>
      positions.some((p) => p.field === field && !p.glyphDigit),
    );

    const segmentDigitsByField = {};
    for (const field of fieldsNeedingSegment) {
      if (!ctx.budgetLeft()) break;
      const mat = cropBoxesMat(cv, image, groups[field]);
      if (!mat) continue;
      try {
        const url = await renderSegment(cv, mat);
        if (!url) continue;
        ctx.addCalls(1);
        try {
          const read = await this.ocrDigits(url, PSM.SINGLE_LINE);
          segmentDigitsByField[field] = {
            digits: digitString(read.text),
            conf: read.confidence || 0,
          };
        } catch {
          /* leave unavailable */
        }
      } finally {
        deleteMats(mat);
      }
    }

    // ---- Reconcile per position ---------------------------------------
    const parts = { year: "", month: "", day: "" };
    let repairs = 0;
    let confTotal = 0;
    let missing = null;

    for (const field of DATE_FIELDS) {
      const rules = POSITION_RULES[field];
      const fieldPositions = positions.filter((p) => p.field === field);
      const seg = segmentDigitsByField[field];
      const segAligned = alignSegmentDigits(
        seg?.digits ?? null,
        rules.length,
        Boolean(fieldPositions[0]?.glyph.isZeroDot),
        Boolean(fieldPositions[1]?.glyph.isZeroDot),
      );

      for (let i = 0; i < rules.length; i++) {
        const p = fieldPositions[i];
        const sources = [
          p.glyphDigit,
          p.glyph.isZeroDot
            ? { digit: "0", conf: 92 }
            : null,
          segAligned[i]
            ? { digit: segAligned[i], conf: Math.max(1, (seg?.conf || 0) * SEGMENT_TRUST_FACTOR), repaired: true }
            : null,
        ];
        const chosen = chooseDigit(sources, rules[i]);
        if (TRACE) {
          console.error(`TRACE ${field}[${i}] rule=${JSON.stringify(rules[i])} sources=${JSON.stringify(sources.map(s => s && `${s.digit}@${s.conf}${s.repaired ? "*" : ""}`))} chosen=${chosen ? chosen.digit + "@" + chosen.conf + (chosen.repaired ? "*" : "") : "null"}`);
        }
        if (!chosen) {
          missing = `${field} #${i + 1}`;
          break;
        }
        if (chosen.repaired) repairs++;
        parts[field] += chosen.digit;
        confTotal += Math.min(100, chosen.conf);
      }
      if (missing) break;
    }

    const allGlyphs = [...groups.year, ...groups.month, ...groups.day];
    const lineImage = await renderLinePreview(cv, image, allGlyphs);

    const attempt = {
      strategy: "glyph-pattern",
      engine: "opencv+tesseract-fas",
      candidateIndex: ctx.attempts.length,
      rotation: rotated ? 180 : 0,
      rotated,
      bounds: boundsOf(allGlyphs, image),
      rawText: `${parts.year}/${parts.month}/${parts.day}`,
      normalizedText: null,
      confidence: 0,
      repairs,
      dates: [],
      glyphReads: glyphLog,
      segReads: Object.fromEntries(
        Object.entries(segmentDigitsByField).map(([k, v]) => [k, `${v.digits}@${Math.round(v.conf)}`]),
      ),
      segmentImages: lineImage ? { line: lineImage } : {},
      preprocessedImage: lineImage || null,
    };

    if (missing) {
      attempt.note = `no reliable read for ${missing}`;
      ctx.attempts.push(attempt);
      return { ok: false, error: `Birth date not found (unreadable ${missing}).` };
    }

    // A bare 8-digit run (national ID, serial) can slip through geometry as
    // a "date" but only by brute-forcing the positional rules — three or
    // more confusion repairs means this is not a real date line.
    if (repairs > MAX_REPAIRS) {
      attempt.note = `rejected: ${repairs} repairs`;
      ctx.attempts.push(attempt);
      return { ok: false, error: `Birth date candidate needed ${repairs} repairs (rejected).` };
    }

    const year = Number(parts.year);
    const month = Number(parts.month);
    const day = Number(parts.day);
    const meanConf = Math.round(confTotal / 8);
    attempt.confidence = meanConf;

    if (!isValidJalaliDate(year, month, day)) {
      attempt.note = "failed Jalali validation";
      ctx.attempts.push(attempt);
      return {
        ok: false,
        error: `Detected an invalid Jalali date: ${parts.year}/${parts.month}/${parts.day}`,
      };
    }

    const confidence = Math.max(
      MIN_MEAN_CONFIDENCE,
      Math.min(100, Math.round(meanConf - repairs * 6)),
    );

    const formatted = `${parts.year}/${parts.month}/${parts.day}`;
    attempt.normalizedText = formatted;
    attempt.dates.push({
      year, month, day, formatted,
      votes: 1,
      finalScore: confidence,
      confidence,
      repairs,
    });
    ctx.attempts.push(attempt);

    if (meanConf < MIN_MEAN_CONFIDENCE) {
      return { ok: false, error: "Confidence too low — please retake the photo." };
    }

    return {
      ok: true,
      payload: { birthDate: formatted, year, month, day, confidence, repairs, lineImage },
    };
  }

  /** Per-glyph (SINGLE_CHAR) or per-segment (SINGLE_LINE) digit OCR. */
  async ocrDigits(dataUrl, psm) {
    if (!this.worker) throw new Error("OCR worker is not initialized.");
    const key = `${psm}|${DIGITS_WHITELIST}`;
    if (this._paramKey !== key) {
      await this.worker.setParameters({
        tessedit_pageseg_mode: String(psm),
        tessedit_char_whitelist: DIGITS_WHITELIST,
      });
      this._paramKey = key;
    }
    const { data } = await this.worker.recognize(dataUrl);
    return { text: data.text || "", confidence: data.confidence || 0 };
  }
}

/* ========================================================================
   Attempt debug helpers
   ======================================================================== */

async function renderLinePreview(cv, image, glyphs) {
  const mat = cropBoxesMat(cv, image, glyphs);
  if (!mat) return null;
  try {
    return await renderSegment(cv, mat);
  } catch {
    return null;
  } finally {
    deleteMats(mat);
  }
}

function boundsOf(glyphs, image) {
  const left = Math.min(...glyphs.map((g) => g.x));
  const top = Math.min(...glyphs.map((g) => g.y));
  const right = Math.max(...glyphs.map((g) => g.right));
  const bottom = Math.max(...glyphs.map((g) => g.bottom));
  return {
    x: left / image.cols,
    y: top / image.rows,
    width: (right - left) / image.cols,
    height: (bottom - top) / image.rows,
  };
}

export { IranCardOCR as BirthDateOCR };
export default IranCardOCR;
