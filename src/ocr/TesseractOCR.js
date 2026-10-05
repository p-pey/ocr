import Tesseract, { createWorker } from "tesseract.js";
import { loadImage } from "../utils/imageUtils.js";
import { DigitLineRecognizer, grayToModelInput } from "./digitModel.js";
import { parseJalaliDate } from "./dateParse.js";

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */

const LSTM_ONLY = Tesseract.OEM?.LSTM_ONLY ?? 1;
const SINGLE_LINE_PSM = Tesseract.PSM?.SINGLE_LINE ?? 7;
const DIGITS_ONLY_WHITELIST = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩/";

const OPENCV_RUNTIME_TIMEOUT_MS = 30000;
const MAX_ANALYSIS_DIMENSION = 1600;

/** Canonical size every card is rectified to (ID-1 ratio). */
export const CARD_W = 1200;
export const CARD_H = 756;

/**
 * Where to look for the birth date, as fractions of the rectified card.
 * The expiry date sits at the very bottom, so the vertical band excludes it.
 * Calibrate on real cards (draw the band on a few rectified samples) and
 * tighten x0/x1 once you know the exact layout.
 */
export const SEARCH_BAND = { x0: 0.12, y0: 0.24, x1: 0.99, y1: 0.70 };
const IDEAL_Y = 0.48;

/** Text-line candidate filter (pixels on the rectified card). */
const LINE_MIN_H = 18;
const LINE_MAX_H = 84;
const LINE_MIN_W = 90;
const LINE_MIN_ASPECT = 2.2;
const CLOSE_KERNEL_WIDTHS = [21, 41]; // merge glyphs into one blob per line
const MAX_CANDIDATES = 14;

const MIN_CONFIDENCE = 60; // 0-100, mean softmax prob of emitted chars: the floor for a correct answer
const HIGH_CONFIDENCE = 90; // single-box acceptance (no agreement needed)
const MIN_VOTES = 2; // distinct boxes that must agree otherwise
const WINDOW_ISDATE_THRESHOLD = 0.5; // windows are tight crops — isDate is reliable there
const MIN_PROB_FLOOR = 0.28; // per-digit minimum probability below which the read is unreliable

// NOTE: `isDate` is used ONLY to gate window crops. Full-line crops contain
// label text ("تاریخ تولد") and the head correctly learns "label present but
// still a date" — but windows are date-sized sub-crops where isDate *does*
// separate a date fragment from a label/ID fragment. Jalali validation +
// multi-box agreement handle the remaining filtering; dateProb is also used
// as a soft score bonus.

/**
 * Sliding date-sized windows inside a line blob. A detected line is usually
 * "label + date" (or has detector padding), but the CNN was trained on
 * tight date crops — feeding the whole blob makes it misfire confidently.
 * Overlapping windows of date-like aspect let at least one window frame the
 * date tightly; the agreement gate in buildFinalResult then outvotes the rest.
 */
const WIN_ASPECTS = [3.2, 4.3, 5.4]; // window width = line height * aspect
const WIN_STRIDE_FRAC = 0.5; // stride as a fraction of window width
const MAX_WINDOW_LINES = 8; // only the most promising lines get windows
const MAX_WINDOWS_TOTAL = 48;

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

/** Dark ink on light paper, which is what the recogniser was trained on. */
function toDarkInkGray(cv, src) {
  const gray = toGray(cv, src);
  if (cv.mean(gray)[0] < 110) cv.bitwise_not(gray, gray);
  return gray;
}

function matToDataUrl(cv, mat) {
  if (typeof document === "undefined") return null; // Node tests: no debug previews
  const canvas = document.createElement("canvas");
  canvas.width = mat.cols;
  canvas.height = mat.rows;
  cv.imshow(canvas, mat);
  return canvas.toDataURL("image/png");
}

/* ------------------------------------------------------------------ */
/* Card detection + rectification                                      */
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
  } catch (error) {
    console.warn("Card edge detection failed.", error);
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
  } catch (error) {
    console.warn("Perspective correction failed.", error);
    deleteMats(warped);
    return null;
  } finally {
    deleteMats(src, dst, transform);
  }
}

/**
 * Returns a card image of CARD_W x CARD_H (rectified) or, when no card outline
 * is found, the input scaled to CARD_W wide (assumed to be a tight crop).
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

    const fallback = new cv.Mat();
    const k = CARD_W / analysis.cols;
    cv.resize(analysis, fallback, new cv.Size(CARD_W, Math.round(analysis.rows * k)), 0, 0, cv.INTER_CUBIC);
    return { card: fallback, rectified: false };
  } finally {
    deleteMats(analysis);
  }
}

/* ------------------------------------------------------------------ */
/* Text-line candidates                                                */
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
 * Finds horizontal text lines inside the search band. No assumption about
 * digit shapes: ink is binarised, glyphs are merged horizontally into one blob
 * per line, and blobs with line-like geometry become candidates. The recogniser
 * + date validation later decides which one is the birth date.
 *
 * @param gray dark-ink grayscale Mat of the whole rectified card
 * @returns rects in card pixel coordinates, ordered by closeness to IDEAL_Y
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
        kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(kw, 3));
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
            // pad: the recogniser was trained with some margin around the text
            const padX = Math.round(r.height * 0.25);
            const padY = Math.round(r.height * 0.3);
            const x = Math.max(0, bx0 + r.x - padX);
            const y = Math.max(0, by0 + r.y - padY);
            const w = Math.min(gray.cols, bx0 + r.x + r.width + padX) - x;
            const h = Math.min(gray.rows, by0 + r.y + r.height + padY) - y;
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

  // de-duplicate boxes found by both kernel widths
  const unique = [];
  for (const r of rects) if (!unique.some((u) => iou(u, r) > 0.6)) unique.push(r);

  const idealPx = IDEAL_Y * gray.rows;
  unique.sort(
    (a, b) => Math.abs(a.y + a.height / 2 - idealPx) - Math.abs(b.y + b.height / 2 - idealPx),
  );
  return unique.slice(0, MAX_CANDIDATES);
}

function cropGray(cv, gray, rect) {
  const view = gray.roi(new cv.Rect(rect.x, rect.y, rect.width, rect.height));
  const copy = view.clone();
  view.delete();
  return copy;
}

/**
 * Sliding date-sized windows inside a line blob. Returns window rects
 * (card pixel coordinates, clamped to the image). See WIN_ASPECTS above.
 */
export function windowsForRect(rect, cols, rows) {
  const wins = [];
  const seen = new Set();
  const push = (x, y, w, h) => {
    x = Math.max(0, Math.min(Math.round(x), cols - 1));
    y = Math.max(0, Math.min(Math.round(y), rows - 1));
    w = Math.max(8, Math.min(Math.round(w), cols - x));
    h = Math.max(8, Math.min(Math.round(h), rows - y));
    const key = `${x},${y},${w},${h}`;
    if (!seen.has(key)) {
      seen.add(key);
      wins.push({ x, y, width: w, height: h, key });
    }
  };
  for (const a of WIN_ASPECTS) {
    const ww = Math.round(rect.height * a);
    if (ww >= rect.width || ww < rect.height * 1.8) continue;
    const step = Math.max(8, Math.round(ww * WIN_STRIDE_FRAC));
    let last = -Infinity;
    for (let x = rect.x; x + ww <= rect.x + rect.width + 1; x += step) {
      push(x, rect.y, ww, rect.height);
      last = x;
    }
    const endX = rect.x + rect.width - ww;
    if (last < endX - 1) push(endX, rect.y, ww, rect.height);
  }
  return wins;
}

/**
 * Full-line boxes plus sliding windows for the most promising lines.
 * Each box: { rect, key, lineIndex, kind: "line" | "window", windowIndex }.
 * rects must already be ordered by closeness to IDEAL_Y.
 */
export function collectBoxes(rects, cols, rows, windowsOnly = false) {
  const boxes = [];
  const seen = new Set();
  const addBox = (rect, lineIndex, kind, windowIndex) => {
    const key = `${rect.key ?? `${rect.x},${rect.y},${rect.width},${rect.height}`}`;
    if (seen.has(key)) return;
    seen.add(key);
    boxes.push({ rect, key, lineIndex, kind, windowIndex });
  };
  rects.forEach((rect, lineIndex) => addBox(rect, lineIndex, "line", -1));
  if (!windowsOnly) {
    let total = 0;
    for (
      let lineIndex = 0;
      lineIndex < Math.min(rects.length, MAX_WINDOW_LINES) && total < MAX_WINDOWS_TOTAL;
      lineIndex++
    ) {
      for (const w of windowsForRect(rects[lineIndex], cols, rows)) {
        if (total >= MAX_WINDOWS_TOTAL) break;
        addBox(w, lineIndex, "window", total);
        total++;
      }
    }
  }
  return boxes;
}

/* ------------------------------------------------------------------ */
/* Engine                                                              */
/* ------------------------------------------------------------------ */

/**
 * Drop-in replacement for the previous engine: same class name, same
 * initialize / recognize / terminate API, same result shape.
 *
 * Options:
 *   modelUrl   - URL of the TensorFlow.js model.json written by src/train/train.mjs
 *                (default /models/date_cnn/model.json)
 *   recognizer - optional pre-built recognizer (tests / custom loading)
 */
export class TesseractOCR {
  constructor({ modelUrl, recognizer } = {}) {
    this.recognizer = recognizer ?? new DigitLineRecognizer({ modelUrl });
    this.tesseractWorker = null; // lazy, fallback only
    this.useTesseractFallback = false;
    this.ready = false;
  }

  async initialize(onProgress) {
    if (this.ready) return;
    try {
      await this.recognizer.load();
    } catch (error) {
      console.warn(
        "Digit model could not be loaded; falling back to Tesseract line OCR " +
          "(much less accurate).",
        error,
      );
      this.useTesseractFallback = true;
    }
    onProgress?.(2);
    this.ready = true;
  }

  async getTesseractWorker() {
    if (!this.tesseractWorker) {
      this.tesseractWorker = await createWorker("fas", LSTM_ONLY);
    }
    return this.tesseractWorker;
  }

  /** One batch of candidate boxes -> reads with their box attached */
  async readLines(cv, gray, boxes) {
    if (!this.useTesseractFallback) {
      const inputs = boxes.map((box) => {
        const crop = cropGray(cv, gray, box.rect);
        try {
          return grayToModelInput(cv, crop);
        } finally {
          crop.delete();
        }
      });
      const decoded = await this.recognizer.recognize(inputs);
      return decoded.map((read, i) => ({ ...read, box: boxes[i] }));
    }

    // Tesseract fallback: full lines only (one OCR call per window would be far too slow).
    const worker = await this.getTesseractWorker();
    await worker.setParameters({
      tessedit_pageseg_mode: String(SINGLE_LINE_PSM),
      tessedit_char_whitelist: DIGITS_ONLY_WHITELIST,
      user_defined_dpi: "300",
    });
    const out = [];
    for (const box of boxes.filter((b) => b.kind === "line").slice(0, 5)) {
      const crop = cropGray(cv, gray, box.rect);
      try {
        const scaled = new cv.Mat();
        const k = Math.max(2, 120 / crop.rows);
        cv.resize(crop, scaled, new cv.Size(Math.round(crop.cols * k), Math.round(crop.rows * k)), 0, 0, cv.INTER_CUBIC);
        const { data } = await worker.recognize(matToDataUrl(cv, scaled));
        scaled.delete();
        out.push({ text: data.text || "", confidence: data.confidence || 0, box });
      } finally {
        crop.delete();
      }
    }
    return out;
  }

  async recognize(imageSrc, onProgress, onAttempt) {
    if (!this.ready) await this.initialize(onProgress);
    onProgress?.(5);

    const cv = await getOpenCV();
    onProgress?.(15);

    const imageElement = await loadImage(imageSrc);
    const source = cv.imread(imageElement);
    try {
      return await this.recognizeMat(cv, source, onProgress, onAttempt);
    } finally {
      deleteMats(source);
    }
  }

  /** DOM-free core: takes an RGBA/RGB cv.Mat (not deleted here). */
  async recognizeMat(cv, source, onProgress, onAttempt) {
    if (!this.ready) await this.initialize(onProgress);
    let card;
    let rectified = false;
    ({ card, rectified } = fitToCard(cv, source));
    onProgress?.(35);

    const attempts = [];
    const dates = [];
    let attemptCount = 0;

    try {
      // Cards are normally upright; only flip 180° if the first pass finds nothing.
      for (const rotation of [0, 180]) {
        let oriented = card;
        if (rotation === 180) {
          oriented = new cv.Mat();
          cv.rotate(card, oriented, cv.ROTATE_180);
        }
        const gray = toDarkInkGray(cv, oriented);
        try {
          const rects = findLineCandidates(cv, gray);
          onProgress?.(rotation === 0 ? 50 : 80);
          if (!rects.length) continue;

          const boxes = collectBoxes(rects, gray.cols, gray.rows);
          const reads = await this.readLines(cv, gray, boxes);
          onProgress?.(rotation === 0 ? 75 : 95);

          const passVotes = {};
          reads.forEach((read) => {
            const parsed = parseJalaliDate(read.text);
            if (!parsed) return;
            // ——— reliability gates ———
            // Windows are tightly cropped date-sized fragments: isDate is
            // meaningful there. Full lines contain label text; isDate is less
            // reliable but still useful to reject obvious non-dates.
            const isWindow = read.box.kind === "window";
            if (isWindow && read.dateProb < WINDOW_ISDATE_THRESHOLD) return;
            if (!isWindow && read.dateProb < 0.15) {
              // extremely low isDate on a line → likely a national-ID line
              // keep it only if minProb is also very high (model sure)
              if (read.minProb < 0.55) return;
            }
            if (read.minProb < MIN_PROB_FLOOR) return;
            if (read.confidence < 48) return; // below this digit heads disagree

            const box = read.box;
            const rect = box.rect;
            const relY = (rect.y + rect.height / 2) / gray.rows;
            const centerBonus = Math.max(0, 1 - Math.abs(relY - IDEAL_Y) / 0.3);
            // Line reads have a larger receptive field and are more reliable
            // than windows: give them a +18 bonus so a single confident line
            // outranks scattered window misfires.
            const kindBonus = isWindow ? 0 : 18;
            // isDate soft bonus (0..12) — helps true windows stand out
            const dateBonus = Math.max(0, (read.dateProb - 0.5) * 24);
            // minProb bonus rewards reads where every digit head agreed
            const certaintyBonus = Math.max(0, (read.minProb - 0.4) * 15);
            const date = {
              ...parsed,
              raw: read.text,
              confidence: read.confidence,
              minProb: read.minProb,
              dateProb: read.dateProb,
              boxKey: `${rotation}:${box.key}`,
              score: read.confidence + 40 * centerBonus + kindBonus + dateBonus + certaintyBonus,
              corrected: false,
              correctionCost: 0,
            };

            let preview = null;
            const crop = cropGray(cv, gray, rect);
            try {
              preview = matToDataUrl(cv, crop);
            } finally {
              crop.delete();
            }

            attemptCount++;
            const attempt = {
              strategy: this.useTesseractFallback
                ? "line-tesseract"
                : box.kind === "window"
                  ? "window-cnn"
                  : "line-cnn",
              candidateIndex: box.lineIndex,
              window: box.kind,
              windowIndex: box.windowIndex,
              rotation,
              rectified,
              bounds: {
                x: rect.x / gray.cols,
                y: rect.y / gray.rows,
                width: rect.width / gray.cols,
                height: rect.height / gray.rows,
              },
              engine: this.useTesseractFallback ? "opencv+tesseract" : "opencv+tfjs-cnn",
              rawText: read.text,
              normalizedText: date.formatted,
              confidence: read.confidence,
              dates: [date],
              repairs: 0,
              segmentImages: { line: preview },
              preprocessedImage: preview,
            };
            attempts.push(attempt);
            dates.push(date);
            passVotes[date.formatted] = (passVotes[date.formatted] ?? 0) + 1;
            onAttempt?.(attemptCount, attempt);
          });

          // Only skip the 180° pass when this pass already produced an
          // agreed (or very confident) date. A lone weak read is usually a
          // misfire — keep looking instead of locking it in.
          const passBest = Object.entries(passVotes).some(([formatted]) => {
            const ds = dates.filter((d) => d.formatted === formatted);
            return (
              ds.length >= MIN_VOTES ||
              ds.some((d) => d.confidence >= HIGH_CONFIDENCE)
            );
          });
          if (passBest) break;
        } finally {
          deleteMats(gray);
          if (oriented !== card) deleteMats(oriented);
        }
      }
    } finally {
      deleteMats(card);
    }

    onProgress?.(100);
    return this.buildFinalResult(attempts, dates);
  }

  buildFinalResult(attempts, allDates) {
    const byKey = {};
    for (const d of allDates) {
      const e = (byKey[d.formatted] ??= {
        ...d,
        votes: 0,
        totalScore: 0,
        totalConfidence: 0,
        bestMinProb: 0,
        bestDateProb: 0,
        lineVotes: 0,
      });
      e.votes++;
      e.totalScore += d.score;
      e.totalConfidence += d.confidence;
      e.bestMinProb = Math.max(e.bestMinProb, d.minProb);
      e.bestDateProb = Math.max(e.bestDateProb, d.dateProb);
      if (d.boxKey && String(d.boxKey).includes("line")) {} // placeholder
      // track whether this formatted date had at least one line vote
      const isLine = attempts.some(
        (a) => a.dates?.some((x) => x.formatted === d.formatted) && a.window === "line",
      );
      if (isLine) e.lineVotes = Math.max(e.lineVotes, 1);
      if (d.score > e.score)
        Object.assign(e, {
          score: d.score,
          raw: d.raw,
          confidence: d.confidence,
          minProb: d.minProb,
          dateProb: d.dateProb,
        });
    }
    // recompute lineVotes properly
    for (const key of Object.keys(byKey)) {
      byKey[key].lineVotes = attempts.filter(
        (a) => a.dates?.some((x) => x.formatted === key) && a.window === "line",
      ).length;
    }

    const ranked = Object.values(byKey)
      .map((d) => {
        // finalScore: prefer dates that have a line read + multiple agreements
        const lineBonus = d.lineVotes > 0 ? 22 : 0;
        const agreementBonus = 15 * (d.votes - 1);
        const certaintyBonus = d.bestMinProb > 0.6 ? 10 : 0;
        return {
          ...d,
          averageConfidence: d.totalConfidence / d.votes,
          finalScore: d.score + agreementBonus + lineBonus + certaintyBonus,
        };
      })
      .sort((a, b) => b.finalScore - a.finalScore);

    // Acceptance gate: must be confident AND (agreed or very confident single read)
    // Dates with a line vote are much more trustworthy — single line reads
    // at 90+ confidence are accepted. Pure window dates need 2 votes.
    const bestDate =
      ranked.find((d) => {
        if (d.confidence < MIN_CONFIDENCE) return false;
        if (d.minProb !== undefined && d.minProb < 0.25) return false;
        if (d.lineVotes > 0) {
          return d.votes >= 2 || d.confidence >= HIGH_CONFIDENCE;
        }
        // window-only dates: stricter
        return d.votes >= 2 && d.confidence >= 62;
      }) ?? null;
    const bestAttempt = bestDate
      ? attempts
          .filter((a) => a.dates.some((d) => d.formatted === bestDate.formatted))
          .sort((a, b) => b.confidence - a.confidence)[0] ?? null
      : null;

    return {
      best: bestAttempt ? { ...bestAttempt, birthDate: bestDate } : null,
      allDates: ranked,
      allAttempts: attempts,
    };
  }

  async terminate() {
    try {
      await this.tesseractWorker?.terminate();
    } catch {
      /* ignore */
    }
    this.tesseractWorker = null;
    await this.recognizer.dispose();
    this.ready = false;
    this.useTesseractFallback = false;
  }
}

export { TesseractOCR as BirthDateOCR };
