/**
 * Iranian National Card Birth-Date Reader (spec IMPLEMENTATION_SPEC.md sections 4-8).
 *
 * Class name kept for UI compatibility; NO tesseract / tfjs / onnx inside.
 * Runtime dependencies: @techstark/opencv-js only (+ embedded weights blob).
 *
 * Pipeline (spec section 4):
 *  recognize(imageSrc,onProgress,onAttempt) -> loadImage -> cv.imread
 *   -> recognizeMat -> fitToCard -> rotation {0, 180} (180 only if 0
 *   produced nothing confident) -> findLineCandidates
 *   -> grayToModelInput -> recognizeLines (cnn.js) -> gates (8.1)
 *   -> TTA variantRects+averageReads -> selectBest -> {best,allDates,allAttempts,timingMs}
 */
import { loadImage } from "../utils/imageUtils.js";
import { grayToModelInput, recognizeLines, averageReads } from "./cnn.js";
import { parseJalaliDate, isValidJalaliDate } from "./dateParse.js";
import { extractDigitSlots, checkLineGate, verifySlotShape, shapeRegistryEmpty, loadEmbeddedShapeTemplates, splitDateSlots, countHolesTopo, countHolesCv, verifyDigitTopo } from "./shapeGate.js";
import { assignFields, rtlAnchorScore, suppressExpiryRows } from "./fieldAssign.js";

/* ------------------------------------------------------------------ */
/* Config (spec section 4, normative)                                  */
/* ------------------------------------------------------------------ */

export const CARD_W = 1200;
export const CARD_H = 756;

/** Where to look for the birth date, fractions of the rectified card. */
export const SEARCH_BAND = { x0: 0.08, y0: 0.18, x1: 0.99, y1: 0.93 };
const IDEAL_Y = 0.52;

const LINE_MIN_H = 18;
const LINE_MAX_H = 84;
const LINE_MIN_W = 90;
const LINE_MIN_ASPECT = 2.2;
const LINE_MAX_W_FRAC = 0.9;
// MASTER SPEC §3: wide horizontal kernel bridges inter-digit gaps/slashes.
// 40x5 is mandatory for at least one close pass (fixes "half-cut" boxes).
const CLOSE_KERNEL_WIDTHS = [13, 21, 33, 40];
const CLOSE_KERNEL_HEIGHT = 5;
const MAX_CANDIDATES = 18;

const MAX_ANALYSIS_DIMENSION = 1600;
const OPENCV_RUNTIME_TIMEOUT_MS = 30000;

// Safety gates (spec 8.1): only accept a date if ALL hold.
const MIN_DATE_PROB = 0.6;
const MIN_CONFIDENCE = 60;
// MASTER SPEC §2.2: smallest validated year wins outright (confidence only
// orders equal years). Pass-1 skip threshold: pass 2 runs only if no valid
// date reaches this confidence.
const PASS1_SKIP_CONF = 80;
// TTA (spec 4.3e): re-read the best 4 reads with 4 variants each.
const TTA_TOP_N = 4;
const TTA_VARIANTS = 4;

let openCVPromise;

/* ------------------------------------------------------------------ */
/* OpenCV runtime + small helpers                                      */
/* ------------------------------------------------------------------ */

async function getOpenCV() {
  if (!openCVPromise) {
    openCVPromise = (async () => {
      const imported = await import("@techstark/opencv-js");
      let cv = imported.default ?? imported;
      if (cv && typeof cv.then === "function") cv = await cv;
      if (cv?.Mat) return cv;
      if (!cv) throw new Error("OpenCV.js did not provide a runtime module.");
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error("OpenCV.js runtime initialization timed out.")),
          OPENCV_RUNTIME_TIMEOUT_MS,
        );
        const previous = cv.onRuntimeInitialized;
        cv.onRuntimeInitialized = () => {
          clearTimeout(timeout);
          if (typeof previous === "function") previous();
          resolve();
        };
      });
      return cv;
    })().catch((error) => {
      openCVPromise = null;
      throw error;
    });
  }
  return openCVPromise;
}

function deleteMats(...values) {
  for (const v of values) {
    try {
      v?.delete?.();
    } catch {
      /* already freed */
    }
  }
}

function toGray(cv, src) {
  const gray = new cv.Mat();
  if (src.channels() === 4) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  else if (src.channels() === 3) cv.cvtColor(src, gray, cv.COLOR_RGB2GRAY);
  else src.copyTo(gray);
  return gray;
}

/** Dark ink on light paper is what the model is trained on. */
function toDarkInkGray(cv, src) {
  const gray = toGray(cv, src);
  if (cv.mean(gray)[0] < 110) cv.bitwise_not(gray, gray);
  return gray;
}

/**
 * MASTER SPEC §4 — adaptive two-pass preprocessing (memory-safe).
 * Pass 1 (native baseline): grayscale only + polarity inversion if μ<110.
 * No CLAHE/sharpening/normalisation (they blow out ink on clean images).
 * Pass 2 (conditional fallback, ONLY if pass 1 yields 0 valid dates):
 *  μ<100      → selective CLAHE (clip 2.0, 8x8 tiles);
 *  σ<38       → min-max contrast stretch;
 *  38≤σ<65    → unsharp masking.
 * Every allocated Mat is deleted before return (WebView memory safety).
 * @returns {{mat: cv.Mat, pass: 1|2, mu: number, sigma: number, applied: string}}
 */
export function adaptiveGrayPass(cv, src, forcePass = 0) {
  const stat = (m) => {
    const mean = cv.mean(m)[0];
    let std = null;
    let m2 = null, s2 = null;
    try {
      m2 = new cv.Mat();
      s2 = new cv.Mat();
      cv.meanStdDev(m, m2, s2);
      std = s2.doubleAt(0, 0);
    } catch { std = 40; }
    finally { try { m2?.delete?.(); } catch {} try { s2?.delete?.(); } catch {} }
    return { mu: mean, sigma: std };
  };
  const base = toDarkInkGray(cv, src);
  const s0 = stat(base);
  if (forcePass === 1) return { mat: base, pass: 1, mu: s0.mu, sigma: s0.sigma, applied: "native" };
  if (forcePass !== 2) return { mat: base, pass: 1, mu: s0.mu, sigma: s0.sigma, applied: "native" };
  // ---- Pass 2 fallbacks (operate on a clone, delete intermediates) ----
  let out = base.clone();
  let applied = "none";
  try {
    if (s0.mu < 100) {
      let lab = null, ch = null, eq = null, merged = null, rgb = null, g2 = null;
      try {
        const isGray = src.channels ? src.channels() === 1 : true;
        if (!isGray && cv.cvtColor) {
          rgb = new cv.Mat();
          cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB ?? cv.COLOR_RGB2BGR ?? 4);
          lab = new cv.Mat();
          cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab ?? 48);
          ch = new cv.MatVector();
          cv.split(lab, ch);
          const L = ch.get(0);
          eq = new cv.Mat();
          const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
          try { clahe.apply(L, eq); } finally { try { clahe.delete?.(); } catch {} }
          L.delete();
          ch.set(0, eq);
          merged = new cv.Mat();
          cv.merge(ch, merged);
          const back = new cv.Mat();
          cv.cvtColor(merged, back, cv.COLOR_Lab2RGB ?? 57);
          g2 = toDarkInkGray(cv, back);
          back.delete();
          out.delete();
          out = g2;
          applied = "clahe";
        } else {
          eq = new cv.Mat();
          const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));
          try { clahe.apply(out, eq); } finally { try { clahe.delete?.(); } catch {} }
          out.delete();
          out = eq;
          eq = null;
          applied = "clahe-gray";
        }
      } catch { applied = "clahe-failed"; }
      finally {
        try { lab?.delete?.(); } catch {}
        try { ch?.delete?.(); } catch {}
        try { eq?.delete?.(); } catch {}
        try { merged?.delete?.(); } catch {}
        try { rgb?.delete?.(); } catch {}
        try { g2 && g2 !== out && g2.delete?.(); } catch {}
      }
    } else if (s0.sigma < 38) {
      let mask = null;
      try {
        mask = new cv.Mat();
        const mm = cv.minMaxLoc(out, mask);
        const lo = mm.minVal, hi = mm.maxVal;
        if (hi > lo + 1e-6) {
          const lut = new cv.Mat(1, 256, cv.CV_8U);
          const d = lut.data;
          for (let i = 0; i < 256; i++) d[i] = Math.max(0, Math.min(255, Math.round(((i - lo) / (hi - lo)) * 255)));
          const stretched = new cv.Mat();
          cv.LUT(out, lut, stretched);
          lut.delete();
          out.delete();
          out = stretched;
          applied = "minmax-stretch";
        }
      } catch { applied = "stretch-failed"; }
      finally { try { mask?.delete?.(); } catch {} }
    } else if (s0.sigma < 65) {
      let blur = null, sharp = null;
      try {
        blur = new cv.Mat();
        cv.GaussianBlur(out, blur, new cv.Size(0, 0), 1.2);
        sharp = new cv.Mat();
        cv.addWeighted(out, 1.5, blur, -0.5, 0, sharp);
        out.delete();
        out = sharp;
        sharp = null;
        applied = "unsharp";
      } catch { applied = "unsharp-failed"; }
      finally {
        try { blur?.delete?.(); } catch {}
        try { sharp && sharp !== out && sharp.delete?.(); } catch {}
      }
    }
  } catch { /* fallback keeps pass-1 clone */ }
  base.delete();
  const s1 = stat(out);
  return { mat: out, pass: 2, mu: s1.mu, sigma: s1.sigma, applied };
}

function matToDataUrl(cv, mat) {
  if (typeof document === "undefined") return null; // Node tests: no debug previews
  try {
    const canvas = document.createElement("canvas");
    canvas.width = mat.cols;
    canvas.height = mat.rows;
    cv.imshow(canvas, mat);
    return canvas.toDataURL("image/png");
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Card detection + rectification (spec 4.2)                           */
/* ------------------------------------------------------------------ */

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

function detectCardQuadrilateral(cv, image) {
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
        cv.approxPolyDP(contour, approx, perimeter * 0.02, true);
        if (approx.rows !== 4) continue;

        const d = approx.data32S;
        const pts = Array.from({ length: 4 }, (_, k) => ({ x: d[k * 2], y: d[k * 2 + 1] }));
        const area = polygonArea(pts);
        const ordered = orderQuadCorners(pts);
        if (!ordered || area / imageArea < 0.28) continue;

        const [tl, tr, br, bl] = ordered;
        const width = (Math.hypot(tr.x - tl.x, tr.y - tl.y) + Math.hypot(br.x - bl.x, br.y - bl.y)) / 2;
        const height = (Math.hypot(bl.x - tl.x, bl.y - tl.y) + Math.hypot(br.x - tr.x, br.y - tr.y)) / 2;
        const aspect = Math.max(width, height) / Math.min(width, height);
        if (aspect < 1.25 || aspect > 2.05) continue;

        if (!best || area > best.area) best = { points: ordered, area, portrait: width < height };
      } finally {
        deleteMats(contour, approx);
      }
    }
  } catch {
    return null;
  } finally {
    deleteMats(gray, blurred, edges, closed, kernel, contourInput, contours, hierarchy);
  }
  return best;
}

function warpCard(cv, image, quad) {
  let src, dst, transform, warped;
  try {
    const w = quad.portrait ? CARD_H : CARD_W;
    const h = quad.portrait ? CARD_W : CARD_H;
    const [tl, tr, br, bl] = quad.points;
    src = cv.matFromArray(4, 1, cv.CV_32FC2, [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]);
    dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1]);
    transform = cv.getPerspectiveTransform(src, dst);
    warped = new cv.Mat();
    cv.warpPerspective(image, warped, transform, new cv.Size(w, h), cv.INTER_CUBIC, cv.BORDER_REPLICATE, new cv.Scalar(0, 0, 0, 0));
    if (quad.portrait) {
      const rotated = new cv.Mat();
      cv.rotate(warped, rotated, cv.ROTATE_90_CLOCKWISE);
      deleteMats(warped);
      return rotated;
    }
    return warped;
  } catch {
    deleteMats(warped);
    return null;
  } finally {
    deleteMats(src, dst, transform);
  }
}

/**
 * Returns { card, rectified }. Downscales to <= 1600 px, warps quad to
 * 1200x756 (portrait quad => rotate 90 deg). If no quad: tight crop scaled
 * to width 1200 (rotated if portrait).
 */
export function fitToCard(cv, source) {
  const scale = Math.min(1, MAX_ANALYSIS_DIMENSION / Math.max(source.cols, source.rows));
  const analysis = new cv.Mat();
  try {
    cv.resize(
      source,
      analysis,
      new cv.Size(Math.round(source.cols * scale), Math.round(source.rows * scale)),
      0, 0,
      scale < 1 ? cv.INTER_AREA : cv.INTER_LINEAR,
    );
    const quad = detectCardQuadrilateral(cv, analysis);
    const warped = quad ? warpCard(cv, analysis, quad) : null;
    if (warped) return { card: warped, rectified: true };

    // Tight-crop fallback: rotate portrait crops, scale to width 1200.
    let oriented = analysis.clone();
    if (oriented.rows > oriented.cols) {
      const rotated = new cv.Mat();
      cv.rotate(oriented, rotated, cv.ROTATE_90_CLOCKWISE);
      oriented.delete();
      oriented = rotated;
    }
    const fallback = new cv.Mat();
    const k = CARD_W / oriented.cols;
    cv.resize(oriented, fallback, new cv.Size(CARD_W, Math.max(1, Math.round(oriented.rows * k))), 0, 0, cv.INTER_CUBIC);
    oriented.delete();
    return { card: fallback, rectified: false };
  } finally {
    deleteMats(analysis);
  }
}

/* ------------------------------------------------------------------ */
/* Text-line candidates (spec 4.3b)                                    */
/* ------------------------------------------------------------------ */

function iou(a, b) {
  const x0 = Math.max(a.x, b.x);
  const y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.width, b.x + b.width);
  const y1 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Finds horizontal text lines inside SEARCH_BAND. No digit-shape
 * assumptions: Gaussian 3x3 -> adaptiveThreshold(GAUSSIAN_C, INV, 31, 12) ->
 * CLOSE (kw x 3) for kw in [13,21,33] -> external contours -> geometry
 * filter -> pad -> NMS (IoU > 0.6) -> sort by closeness to IDEAL_Y -> top 18.
 * @returns rects in card pixel coordinates.
 */
export function findLineCandidates(cv, gray) {
  const bx0 = Math.round(SEARCH_BAND.x0 * gray.cols);
  const by0 = Math.round(SEARCH_BAND.y0 * gray.rows);
  const bx1 = Math.round(SEARCH_BAND.x1 * gray.cols);
  const by1 = Math.round(SEARCH_BAND.y1 * gray.rows);

  let roi, blurred, binary;
  const rects = [];
  try {
    roi = gray.roi(new cv.Rect(bx0, by0, bx1 - bx0, by1 - by0));
    blurred = new cv.Mat();
    binary = new cv.Mat();
    cv.GaussianBlur(roi, blurred, new cv.Size(3, 3), 0);
    cv.adaptiveThreshold(blurred, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 12);

    for (const kw of CLOSE_KERNEL_WIDTHS) {
      let kernel, closed, input, contours, hierarchy;
      try {
        // MASTER SPEC §3: heavily rectangular element grouping the full
        // 10-character date string (kw x 5).
        kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(kw, CLOSE_KERNEL_HEIGHT));
        closed = new cv.Mat();
        cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel);
        input = closed.clone();
        contours = new cv.MatVector();
        hierarchy = new cv.Mat();
        cv.findContours(input, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        for (let i = 0; i < contours.size(); i++) {
          const c = contours.get(i);
          try {
            const r = cv.boundingRect(c);
            if (r.height < LINE_MIN_H || r.height > LINE_MAX_H) continue;
            if (r.width < LINE_MIN_W || r.width / r.height < LINE_MIN_ASPECT) continue;
            if (r.width > LINE_MAX_W_FRAC * gray.cols) continue;
            const padX = Math.round(r.height * 0.25);
            const padY = Math.round(r.height * 0.3);
            const x = Math.max(0, bx0 + r.x - padX);
            const y = Math.max(0, by0 + r.y - padY);
            const w = Math.min(gray.cols, bx0 + r.x + r.width + padX) - x;
            const h = Math.min(gray.rows, by0 + r.y + r.height + padY) - y;
            if (w < 8 || h < 8) continue;
            rects.push({ x, y, width: w, height: h });
          } finally {
            deleteMats(c);
          }
        }
      } finally {
        deleteMats(kernel, closed, input, contours, hierarchy);
      }
    }
  } finally {
    deleteMats(roi, blurred, binary);
  }

  const unique = [];
  for (const r of rects) if (!unique.some((u) => iou(u, r) > 0.6)) unique.push(r);

  const idealPx = IDEAL_Y * gray.rows;
  unique.sort(
    (a, b) => Math.abs(a.y + a.height / 2 - idealPx) - Math.abs(b.y + b.height / 2 - idealPx),
  );
  return unique.slice(0, MAX_CANDIDATES);
}

function cropGray(cv, gray, rect) {
  const x = Math.max(0, Math.min(Math.round(rect.x), gray.cols - 1));
  const y = Math.max(0, Math.min(Math.round(rect.y), gray.rows - 1));
  const w = Math.max(1, Math.min(Math.round(rect.width), gray.cols - x));
  const h = Math.max(1, Math.min(Math.round(rect.height), gray.rows - y));
  const view = gray.roi(new cv.Rect(x, y, w, h));
  const copy = view.clone();
  view.delete();
  return copy;
}

/**
 * 4 slightly shifted/grown crops for TTA (spec 4.3e). Deterministic,
 * clamped to the image. Returns 4 rects.
 */
export function variantRects(rect, cols, rows) {
  const out = [];
  const variants = [
    { dx: -0.06, dy: 0, grow: 0 },
    { dx: 0.06, dy: 0, grow: 0 },
    { dx: 0, dy: -0.05, grow: 0.08 },
    { dx: 0, dy: 0.05, grow: 0.12 },
  ];
  for (const v of variants.slice(0, TTA_VARIANTS)) {
    const gw = rect.width * (1 + v.grow);
    const gh = rect.height * (1 + v.grow);
    const cx = rect.x + rect.width / 2 + v.dx * rect.width;
    const cy = rect.y + rect.height / 2 + v.dy * rect.height;
    let x = Math.round(cx - gw / 2);
    let y = Math.round(cy - gh / 2);
    let w = Math.round(gw);
    let h = Math.round(gh);
    x = Math.max(0, Math.min(x, cols - 1));
    y = Math.max(0, Math.min(y, rows - 1));
    w = Math.max(8, Math.min(w, cols - x));
    h = Math.max(8, Math.min(h, rows - y));
    out.push({ x, y, width: w, height: h });
  }
  return out;
}

/**
 * Selection (spec 4.4 + 8.1/8.2 + MASTER SPEC §2.2 selectBest override):
 * candidates must pass dateProb >= 0.6, valid Jalali, confidence >= 60
 * (enforced upstream).
 *  1. Year filter: drop every year > 1400 when a year <= 1400 exists —
 *     expiry dates live in the 1400s, birth dates don't. A 99%-confident
 *     expiry must never beat an 85% birth date.
 *  2. MASTER OVERRIDE — never confidence-picked: when multiple valid dates
 *     pass calendar validation, ALWAYS choose the smallest year integer
 *     (e.g. 1360/03/10 beats 1402/04/01 regardless of confidence). Ties
 *     (same year, overlapping boxes) break towards the UPPER row (smaller
 *     top-left yMin = birth sits higher), then higher confidence. Year
 *     always dominates vertical position (upside-down captures still win).
 * Null when empty. Dev-only: OCR_SELECTION_TRACE=1 logs every accept/drop
 * decision with years (never full dates) to stderr — never set in production.
 */
const CUTOFF_YEAR = 1400;
export function selectBest(dates) {
  if (!dates.length) return null;
  const trace = (msg) => {
    if (typeof process !== "undefined" && process.env?.OCR_SELECTION_TRACE) {
      console.error(`TRACE select: ${msg}`);
    }
  };
  // Directive §4.2 pre-filter (silently drops expiry-side years).
  const young = dates.filter((d) => d.year <= CUTOFF_YEAR);
  const pool = young.length ? young : dates;
  if (pool.length !== dates.length) {
    trace(`${dates.length - pool.length} date(s) over ${CUTOFF_YEAR} dropped (expiry side)`);
  }
  // Directive Step 3.2 orders by top-left Y (yMin); centre relY kept as
  // fallback for dates assembled without bounds (e.g. unit mocks).
  const yOf = (d) => (Number.isFinite(d.yMin) ? d.yMin : Number.isFinite(d.relY) ? d.relY : 1);
  // MASTER SPEC §2.2 override: smallest validated year wins outright.
  // Confidence only orders equal years (upper row first, then confidence).
  const byYear = [...pool].sort((a, b) => a.year - b.year || yOf(a) - yOf(b) || b.confidence - a.confidence);
  const w = byYear[0] ?? null;
  if (w) trace(`pick smallest year=${w.year} conf=${Math.round(w.confidence)} relY~${yOf(w).toFixed(2)} from ${pool.length} valid`);
  return w;
}

/**
 * Pure result finalization (also unit-tested directly): aggregate votes per
 * formatted date, pick the winner (selectBest), assign birth/expiry fields
 * (assignFields, directive steps 3.1-3.3) and attach the winning attempt.
 * An unresolvable date sequence forces best=null (retake) no matter what
 * selectBest preferred. A lone validated date low on a rectified card with
 * an expiry-side year is the expiry row with the birth row missed — report
 * it as expiry and null the birth (never emit expiry AS birth).
 */
export function buildFinalResult(allDates, allAttempts, opts = {}) {
  const byKey = {};
  for (const d of allDates) {
    const e = (byKey[d.formatted] ??= { ...d, votes: 0, totalScore: 0, totalConfidence: 0 });
    e.votes++;
    e.totalScore += d.score ?? d.confidence;
    e.totalConfidence += d.confidence;
      if ((d.score ?? d.confidence) > (e.score ?? e.confidence)) {
        Object.assign(e, { score: d.score, raw: d.raw, confidence: d.confidence, minProb: d.minProb, dateProb: d.dateProb, year: d.year, month: d.month, day: d.day, relY: d.relY, yMin: d.yMin, xMax: d.xMax });
      }
  }
  const ranked = Object.values(byKey)
    .map((d) => ({ ...d, averageConfidence: d.totalConfidence / d.votes }))
    .sort((a, b) => b.confidence - a.confidence);

  const winner = selectBest(ranked);
  const fieldResult = assignFields(ranked, null);
  const fields = {
    birth: fieldResult.birth?.formatted ?? null,
    expiry: fieldResult.expiry?.formatted ?? null,
    swapped: fieldResult.swapped,
    method: fieldResult.method,
    sequenceError: fieldResult.sequenceError?.message ?? null,
  };
  // Lone-expiry guard: a single validated date low on a photo-scale image
  // with an expiry-side year is the expiry row with the birth row missed —
  // report it as expiry, never emit it as birth. Tight crops (the user
  // isolated the line) keep the lone read.
  if (
    winner && fieldResult.method === "single" && !opts.tightCrop &&
    winner.year > 1400 && (winner.yMin ?? 0) > 0.65
  ) {
    fields.birth = null;
    fields.expiry = winner.formatted;
    fields.method = "single-expiry";
  }
  if (opts.tightCrop && fieldResult.method === "single" && winner) {
    fields.method = "single-crop";
  }
  // Hard rule: expiry-inline dates never contend for birth. Tag every row;
  // the fallback in selectBest already excludes >1400 years, and the tags
  // let callers/UI audit which row won and why.
  const tagged = ranked.map((d) => ({
    ...d,
    field: fields.birth && d.formatted === fields.birth
      ? "birth"
      : fields.expiry && d.formatted === fields.expiry
        ? "expiry"
        : "unknown",
  }));
  let best = null;
  if (winner && !fieldResult.sequenceError && fields.method !== "single-expiry") {
    const bestAttempt =
      allAttempts
        .filter((a) => a.dates.some((d) => d.formatted === winner.formatted))
        .sort((a, b) => b.confidence - a.confidence)[0] ?? null;
    if (bestAttempt) {
      best = {
        ...bestAttempt,
        birthDate: {
          year: winner.year,
          month: winner.month,
          day: winner.day,
          formatted: winner.formatted,
          raw: winner.raw,
          confidence: winner.confidence,
          score: winner.score ?? winner.confidence,
          corrected: false,
          correctionCost: 0,
          votes: winner.votes,
          minProb: winner.minProb,
          dateProb: winner.dateProb,
          relY: winner.relY,
          yMin: winner.yMin,
        },
      };
    }
  }
  return { best, allDates: tagged, allAttempts, fields };
}

/* ------------------------------------------------------------------ */
/* Engine                                                              */
/* ------------------------------------------------------------------ */

/**
 * Drop-in replacement for the previous engine: same class name, same
 * initialize / recognize(imageSrc, onProgress, onAttempt) / terminate API.
 * No tesseract / tfjs inside: OpenCV + dependency-free CNN (cnn.js).
 */
export class TesseractOCR {
  constructor() {
    this.ready = false;
  }

  async initialize(onProgress) {
    if (this.ready) return;
    // One-time OpenCV load is excluded from recognize() timing targets.
    await getOpenCV();
    // Touch weights so a corrupt blob fails fast here, not mid-recognition.
    const { recognizeLines: rl } = await import("./cnn.js");
    if (typeof rl !== "function") throw new Error("CNN module failed to load.");
    // Shape-exemplar templates for the gate enforcer (best-effort: without
    // them the geometry gate still runs and behaviour is unchanged).
    try {
      await loadEmbeddedShapeTemplates();
    } catch {
      /* gate skips template checks when unloaded */
    }
    onProgress?.(2);
    this.ready = true;
  }

  async recognize(imageSrc, onProgress, onAttempt) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (!this.ready) await this.initialize(onProgress);
    onProgress?.(5);

    const cv = await getOpenCV();
    onProgress?.(15);

    const imageElement = await loadImage(imageSrc);
    const source = cv.imread(imageElement);
    try {
      const res = await this.recognizeMat(cv, source, onProgress, onAttempt);
      const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
      res.timingMs = Math.round(t1 - t0);
      return res;
    } finally {
      deleteMats(source);
    }
  }

  /** DOM-free core: takes an RGBA/RGB cv.Mat (not deleted here). */
  async recognizeMat(cv, source, onProgress, onAttempt) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    if (!this.ready) await this.initialize(onProgress);
    let card;
    let rectified = false;
    ({ card, rectified } = fitToCard(cv, source));
    onProgress?.(35);
    // Capture dims now: card is deleted before finalization below.
    const cardDims = { w: card.cols, h: card.rows };

    const allAttempts = [];
    const allDates = [];
    // readKey -> { read, rect, rotation, rectified, previewIdx }
    const readsByKey = new Map();
    let attemptCount = 0;

    // Keep gray refs for previews/gate (freed at the end of each rotation).
    let grayRef0 = null;
    let grayRef1 = null;

    const pushAttempt = (read, rect, rotation, strategy) => {
      const parsed = parseJalaliDate(read.text);
      if (!parsed) return null;
      // MASTER SPEC §7 strict enforcer wired to the final probability array:
      // reject impossible triples even if the string parsed (defence in depth).
      if (!isValidJalaliDate(parsed.year, parsed.month, parsed.day)) return null;
      // Safety gates (spec 8.1): ALL must hold.
      if (!(read.isDate && read.dateProb >= MIN_DATE_PROB)) return null;
      if (read.confidence < MIN_CONFIDENCE) return null;

      let gated = read;
      let gateConflict = false;
      let gateReasons = [];
      try {
        const probe = cropGray(cv, rotation === 0 ? grayRef0?.gray : grayRef1?.gray, rect);
        // Note: slots need the gray crop; if unavailable, skip the gate.
        if (probe) {
          try {
            const slots = extractDigitSlots(cv, probe);
            if (slots) {
              const reasons = [];
              const geo = checkLineGate(read.digits ?? read.text?.replace(/\D/g, ""), slots);
              if (geo.conflict) reasons.push(...geo.reasons);
              // Template enforcer: a predicted 0 MUST match the hollow-ring
              // exemplars (and 1/8/3 their classes) — confirmed by BOTH NCC
              // template matching and Hu moments.
              // MASTER SPEC §6 Shape Gate: deterministic topological verifier
              // runs FIRST (holes/stem/aspect) — neural probabilities never
              // override geometric/topological rules. A topo conflict zeroes
              // that digit class implied confidence via gate halving below.
              let topoConflict = false;
              const topoReasons = [];
              try {
                const W = probe.cols, H = probe.rows, P = probe.data;
                const prof = splitDateSlots(P, W, H);
                if (prof.fixed && prof.digits.length === 8 && read.digits?.length === 8) {
                  for (let i = 0; i < 8; i++) {
                    const s = prof.digits[i];
                    const sw = Math.max(1, s.x1 - s.x0);
                    const sh = H;
                    const spx = new Float32Array(sw * sh);
                    for (let yy = 0; yy < sh; yy++) {
                      for (let xx = 0; xx < sw; xx++) spx[yy*sw+xx] = P[yy*W + s.x0 + xx];
                    }
                    const v = verifyDigitTopo(spx, sw, sh, read.digits[i]);
                    if (v.conflict) {
                      topoReasons.push(`pos${i}:${read.digits[i]}-topo:${v.reasons.join("+")}`);
                      topoConflict = true;
                    }
                    // OpenCV RETR_CCOMP twin cross-check for 0/5/9 (holes).
                    if ("059".includes(read.digits[i])) {
                      let dm = null;
                      try {
                        dm = cropGray(cv, probe, { x: s.x0, y: 0, width: sw, height: sh });
                        const hc = countHolesCv(cv, dm);
                        const want = 1;
                        if (hc.holes !== want) {
                          topoReasons.push(`pos${i}:${read.digits[i]}-cvholes:${hc.holes}`);
                          topoConflict = true;
                        }
                      } catch { /* cross-check best-effort */ }
                      finally { try { dm?.delete?.(); } catch {} }
                    }
                  }
                }
              } catch { /* topology gate best-effort */ }
              if (topoConflict) reasons.push(...topoReasons);
              let tplConflict = topoConflict;
              if (slots.length === 8 && !shapeRegistryEmpty() && read.digits?.length === 8 &&
                  probe.isContinuous?.() !== false) {
                const px = probe.data;
                const PW = probe.cols;
                const PH = probe.rows;
                for (let i = 0; i < 8; i++) {
                  const s = slots[i];
                  const sx = Math.max(0, Math.min(Math.round(s.x), PW - 1));
                  const sy = Math.max(0, Math.min(Math.round(s.y ?? 0), PH - 1));
                  const sw = Math.max(4, Math.min(Math.round(s.width), PW - sx));
                  const sh = Math.max(4, Math.min(Math.round(s.height), PH - sy));
                  if (sx + sw > PW || sy + sh > PH) continue;
                  const spx = new Float32Array(sw * sh);
                  for (let yy = 0; yy < sh; yy++) {
                    for (let xx = 0; xx < sw; xx++) spx[yy * sw + xx] = px[(sy + yy) * PW + sx + xx];
                  }
                  const v = verifySlotShape(spx, sw, sh, read.digits[i]);
                  if (v.conflict) {
                    reasons.push(`pos${i}:${read.digits[i]}~${v.top}/${v.huTop}`);
                    tplConflict = true;
                  }
                }
              }
              gateReasons = reasons;
              const halve = tplConflict || geo.conflict;
              if (halve && reasons.length) {
                gated = { ...read, confidence: read.confidence * 0.5, gateConflict: true, gateReasons: [...reasons] };
                gateConflict = true;
              }
            }
          } finally {
            probe.delete();
          }
        }
      } catch {
        /* gate is best-effort; never fail recognition because of it */
      }

      // A gate-halved read below the acceptance floor is not a successfully
      // parsed date (spec 8.1): drop it so fallback selection can never crown
      // a suspected misread as birth.
      if (gated.confidence < MIN_CONFIDENCE) return null;

      const ref = rotation === 0 ? grayRef0 : grayRef1;
      const date = {
        ...parsed,
        raw: read.text,
        confidence: gated.confidence,
        minProb: read.minProb,
        dateProb: read.dateProb,
        score: gated.confidence,
        corrected: false,
        correctionCost: 0,
        votes: 1,
        // Spatial position: birth date sits ABOVE the expiry date on smart
        // cards, so the row's vertical position disambiguates the fields.
        // yMin (top-left Y) drives field assignment; relY (centre) is kept
        // for compatibility/debug.
        relY: (rect.y + rect.height / 2) / ref.rows,
        yMin: rect.y / ref.rows,
        // Right edge for RTL inline anchoring (value sits LEFT of its label).
        xMax: (rect.x + rect.width) / ref.cols,
      };

      let preview = null;
      try {
        const grayForPreview = rotation === 0 ? grayRef0?.gray : grayRef1?.gray;
        if (grayForPreview) {
          const crop = cropGray(cv, grayForPreview, rect);
          try {
            preview = matToDataUrl(cv, crop);
          } finally {
            crop.delete();
          }
        }
      } catch {
        preview = null;
      }

      attemptCount++;
      const attempt = {
        strategy,
        candidateIndex: allAttempts.length,
        rotation,
        rectified,
        bounds: {
          x: rect.x / (rotation === 0 ? grayRef0.cols : grayRef1.cols),
          y: rect.y / (rotation === 0 ? grayRef0.rows : grayRef1.rows),
          width: rect.width / (rotation === 0 ? grayRef0.cols : grayRef1.cols),
          height: rect.height / (rotation === 0 ? grayRef0.rows : grayRef1.rows),
        },
        engine: "opencv+cnn",
        rawText: read.text,
        normalizedText: date.formatted,
        confidence: gated.confidence,
        minProb: read.minProb,
        dateProb: read.dateProb,
        dates: [date],
        repairs: 0,
        gateConflict,
        gateReasons,
        segmentImages: { line: preview },
        preprocessedImage: preview,
      };
      allAttempts.push(attempt);
      allDates.push({ ...date, boxKey: `${rotation}:${rect.x},${rect.y},${rect.width},${rect.height}` });
      readsByKey.set(`${rotation}:${rect.x},${rect.y},${rect.width},${rect.height}`, { read, rect, rotation });
      onAttempt?.(attemptCount, attempt);
      return date;
    };

    try {
      // MASTER SPEC §4 + §5: strict upright, zero rotation on the primary
      // pass. Pass 1 = native grayscale only; Pass 2 (CLAHE/stretch/unsharp)
      // runs ONLY if pass 1 yields 0 valid dates. A 180° retry is the final
      // emergency step only (both passes failed).
      const runRotation = (rotation, pass) => {
        if (rotation === 180 && allDates.some((d) => d.confidence >= PASS1_SKIP_CONF && d.dateProb >= 0.8)) return;
        let oriented = card;
        let owned = false;
        if (rotation === 180) {
          oriented = new cv.Mat();
          cv.rotate(card, oriented, cv.ROTATE_180);
          owned = true;
        }
        const gp = adaptiveGrayPass(cv, oriented, pass);
        const gray = gp.mat;
        try {
          if (rotation === 0) grayRef0 = { gray, cols: gray.cols, rows: gray.rows };
          else grayRef1 = { gray, cols: gray.cols, rows: gray.rows };

          const rects = findLineCandidates(cv, gray);
          // MASTER SPEC §2.1: RTL corridor — prefer right-anchored strips
          // (value LEFT of the birth label); order candidates by anchor score.
          try {
            rects.sort((a, b) => rtlAnchorScore(b, gray.cols, gray.rows) - rtlAnchorScore(a, gray.cols, gray.rows));
          } catch { /* anchor sort best-effort */ }
          onProgress?.(rotation === 0 ? 50 : 80);
          if (!rects.length) return;

          // Batch all candidates through the CNN at once.
          const inputs = [];
          const order = [];
          for (const rect of rects) {
            const crop = cropGray(cv, gray, rect);
            try {
              inputs.push(grayToModelInput(cv, crop));
              order.push(rect);
            } finally {
              crop.delete();
            }
          }
          const decoded = recognizeLines(inputs);
          onProgress?.(rotation === 0 ? 75 : 95);

          // Direct reads.
          const directDates = [];
          decoded.forEach((read, i) => {
            const d = pushAttempt(read, order[i], rotation, "line-cnn");
            if (d) directDates.push({ read, rect: order[i], date: d });
          });

          // TTA: best 4 reads -> 4 variants each -> averageReads.
          directDates.sort((a, b) => b.date.confidence - a.date.confidence);
          for (const top of directDates.slice(0, TTA_TOP_N)) {
            try {
              const vars = variantRects(top.rect, gray.cols, gray.rows);
              const varInputs = [];
              for (const v of vars) {
                const crop = cropGray(cv, gray, v);
                try {
                  varInputs.push(grayToModelInput(cv, crop));
                } finally {
                  crop.delete();
                }
              }
              const varReads = recognizeLines(varInputs);
              const baseCrop = cropGray(cv, gray, top.rect);
              let baseArr;
              try {
                baseArr = grayToModelInput(cv, baseCrop);
              } finally {
                baseCrop.delete();
              }
              const baseRead = recognizeLines([baseArr])[0];
              const merged = averageReads([baseRead, ...varReads]);
              const mergedParsed = parseJalaliDate(merged.text);
              let mergedConf = merged.confidence;
              if (!mergedParsed) mergedConf *= 0.8;
              const mergedRead = { ...merged, confidence: mergedConf };
              pushAttempt(mergedRead, top.rect, rotation, "line-cnn-tta");
            } catch {
              /* TTA is best-effort */
            }
          }
        } finally {
          deleteMats(gray);
          if (rotation === 0) grayRef0 = null;
          else grayRef1 = null;
          if (owned) deleteMats(oriented);
        }
      };
      // Pass 1 (native baseline). Skip pass 2 entirely if a valid date with
      // confidence ≥80 was found (MASTER SPEC §4).
      runRotation(0, 1);
      const pass1Valid = allDates.filter((d) => d.confidence >= PASS1_SKIP_CONF);
      if (!pass1Valid.length) {
        // Pass 2 conditional fallback: same upright rotation, enhanced gray.
        runRotation(0, 2);
      }
      // Absolute final emergency step: 180° deskew (MASTER SPEC §5).
      if (!allDates.length) {
        runRotation(180, 1);
        if (!allDates.length) runRotation(180, 2);
      }
    } finally {
      deleteMats(card);
    }

    // Aggregate votes, select winner, assign birth/expiry fields
    // (MASTER SPEC §2: RTL anchoring proxy + Y-min order + chronology;
    // unresolvable sequence forces best=null).
    // Zero external OCR: Persian label words cannot be read by the digit
    // CNN, so anchors are geometric (right-anchored corridor via
    // rtlAnchorScore, expiry rows suppressed below the birth row) plus
    // Y-sort + smallest-year chronology with invert-or-fallback.
    // (card was freed above; dims were captured before the rotation loop.)
    // Negative-region suppression: drop rows clearly below the birth row.
    try {
      const birthY = Math.min(...allDates.map((d) => Number.isFinite(d.yMin) ? d.yMin : 1));
      if (Number.isFinite(birthY) && allDates.length > 1) {
        const kept = suppressExpiryRows(allDates, birthY);
        if (kept.length && kept.length !== allDates.length) {
          allDates.length = 0;
          allDates.push(...kept);
        }
      }
    } catch { /* suppression best-effort */ }
    const finalized = buildFinalResult(allDates, allAttempts, {
      tightCrop: !rectified && (cardDims.w < 800 || cardDims.h < 400 || cardDims.w / cardDims.h > 2.2),
    });

    onProgress?.(100);
    const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
    return { ...finalized, timingMs: Math.round(t1 - t0) };
  }

  async terminate() {
    this.ready = false;
  }
}

export { TesseractOCR as BirthDateOCR };
export { ConsensusReader } from "./consensus.js";
