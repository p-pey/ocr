/**
 * Lightweight, local-only birth-date OCR for Iranian national cards.
 *
 * The finder and image preparation use the browser's Canvas API (no OpenCV,
 * TensorFlow model, or custom model files). Only date-like text rows are sent
 * to a Tesseract.js worker, which runs locally in a Web Worker. The worker
 * needs the Persian `fas` language pack; by default Tesseract.js downloads
 * static runtime/language assets, never the supplied image. Pass local asset
 * paths in strict-offline deployments.
 */

import { parseJalaliDate } from "./dateParse.js";

const MAX_IMAGE_DIMENSION = 1400;
const MAX_IMAGE_PIXELS = 2_000_000;
const MAX_ROWS_TO_READ = 4;
const MAX_COMPONENTS = 20_000;
const MAX_OCR_CALLS = 24;
const MIN_CONFIDENCE = 35;
const TARGET_ROW_HEIGHT = 150;
const DATE_CHARACTERS = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩/\\|.,:؛-–−";
const SEARCH_BAND = { min: 0.12, max: 0.8 };
const AVERAGE_ROW_Y = 0.52;

const LATIN_DIGITS = "0123456789";
const FARSI_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩";

const OTSU_MODE = "otsu";
const ADAPTIVE_MODE = "adaptive";

let canvasModulePromise = null;

function now() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function normalizeDigitCharacters(text) {
  return String(text ?? "")
    .replace(/[۰-۹]/g, (character) => String(FARSI_DIGITS.indexOf(character)))
    .replace(/[٠-٩]/g, (character) => String(ARABIC_DIGITS.indexOf(character)))
    .replace(/[／⁄\\|]/g, "/")
    .replace(/[‐‑‒–—−]/g, "-");
}

/**
 * Parse one OCR line without repairing or inventing digits. The line must
 * contain exactly one 4-2-2 Jalali date, or one unseparated 8-digit run.
 *
 * Exported for deterministic tests and downstream callers.
 */
export function parseBirthDateLine(text, options = {}) {
  const normalized = normalizeDigitCharacters(text);
  const groups = normalized.match(/\d+/g) ?? [];
  let yearText;
  let monthText;
  let dayText;

  if (
    groups.length === 3 &&
    groups[0].length === 4 &&
    groups[1].length >= 1 && groups[1].length <= 2 &&
    groups[2].length >= 1 && groups[2].length <= 2
  ) {
    [yearText, monthText, dayText] = groups;
  } else if (groups.length === 1 && groups[0].length === 8) {
    yearText = groups[0].slice(0, 4);
    monthText = groups[0].slice(4, 6);
    dayText = groups[0].slice(6, 8);
  } else if (groups.length === 8 && groups.every((group) => group.length === 1)) {
    // A few recognizer configurations put a space between every character.
    const digits = groups.join("");
    yearText = digits.slice(0, 4);
    monthText = digits.slice(4, 6);
    dayText = digits.slice(6, 8);
  } else {
    return null;
  }

  const parsed = parseJalaliDate(
    `${yearText}/${monthText}/${dayText}`,
    { minYear: 1280, maxYear: 1410, ...options },
  );
  return parsed ? { ...parsed, rawText: String(text ?? "").trim() } : null;
}

async function getCanvasFactory() {
  if (typeof document !== "undefined") {
    return (width = 1, height = 1) => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      return canvas;
    };
  }
  const module = await getNodeCanvas();
  return module.createCanvas;
}

async function getNodeCanvas() {
  if (!canvasModulePromise) {
    const moduleName = ["@napi-rs", "canvas"].join("/");
    canvasModulePromise = import(/* @vite-ignore */ moduleName).catch(() => null);
  }
  const module = await canvasModulePromise;
  if (!module) {
    throw new Error("In Node.js, pass a canvas or install the optional @napi-rs/canvas package.");
  }
  return module;
}

function isCanvasLike(value) {
  return Boolean(value && typeof value.getContext === "function" && value.width && value.height);
}

async function loadImageElement(input) {
  if (typeof window === "undefined") {
    const { loadImage } = await getNodeCanvas();
    if (typeof Blob !== "undefined" && input instanceof Blob) {
      return loadImage(Buffer.from(await input.arrayBuffer()));
    }
    if (input instanceof Uint8Array) return loadImage(Buffer.from(input));
    if (typeof input === "string") return loadImage(input);
    throw new Error("In Node.js, pass a canvas, image buffer, or data URL.");
  }

  if (typeof HTMLImageElement !== "undefined" && input instanceof HTMLImageElement) {
    if (input.complete && input.naturalWidth) return input;
    await new Promise((resolve, reject) => {
      input.onload = resolve;
      input.onerror = () => reject(new Error("Image failed to load."));
    });
    return input;
  }

  if (typeof ImageBitmap !== "undefined" && input instanceof ImageBitmap) return input;

  let source = input;
  let objectUrl = null;
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    objectUrl = URL.createObjectURL(input);
    source = objectUrl;
  }

  try {
    const image = new Image();
    image.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Image failed to load."));
      image.src = source;
    });
    return image;
  } finally {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}

async function toAnalysisCanvas(input, makeCanvas) {
  const source = isCanvasLike(input) ? input : await loadImageElement(input);
  const sourceWidth = source.width || source.naturalWidth;
  const sourceHeight = source.height || source.naturalHeight;
  if (!sourceWidth || !sourceHeight) throw new Error("The selected image is empty.");

  const scale = Math.min(
    1,
    MAX_IMAGE_DIMENSION / Math.max(sourceWidth, sourceHeight),
    Math.sqrt(MAX_IMAGE_PIXELS / (sourceWidth * sourceHeight)),
  );
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = makeCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not create a 2D canvas context.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height);
  return canvas;
}

function grayscalePixels(imageData) {
  const source = imageData.data;
  const gray = new Uint8Array(imageData.width * imageData.height);
  for (let pixel = 0, i = 0; i < source.length; i += 4, pixel++) {
    // Composite transparency onto white; keep dark-red security ink via luma.
    const alpha = source[i + 3] / 255;
    const luma = source[i] * 0.299 + source[i + 1] * 0.587 + source[i + 2] * 0.114;
    gray[pixel] = Math.round(luma * alpha + 255 * (1 - alpha));
  }
  return gray;
}

function otsuThreshold(gray) {
  const histogram = new Uint32Array(256);
  let totalSum = 0;
  for (const value of gray) {
    histogram[value]++;
    totalSum += value;
  }

  let backgroundWeight = 0;
  let backgroundSum = 0;
  let maximumVariance = -1;
  let threshold = 127;
  for (let value = 0; value < 256; value++) {
    backgroundWeight += histogram[value];
    if (!backgroundWeight) continue;
    const foregroundWeight = gray.length - backgroundWeight;
    if (!foregroundWeight) break;
    backgroundSum += value * histogram[value];
    const meanBackground = backgroundSum / backgroundWeight;
    const meanForeground = (totalSum - backgroundSum) / foregroundWeight;
    const variance = backgroundWeight * foregroundWeight * (meanBackground - meanForeground) ** 2;
    if (variance > maximumVariance) {
      maximumVariance = variance;
      threshold = value;
    }
  }
  return threshold;
}

function makeIntegralImage(gray, width, height) {
  const stride = width + 1;
  const integral = new Uint32Array(stride * (height + 1));
  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    const sourceOffset = (y - 1) * width;
    const targetOffset = y * stride;
    const previousOffset = (y - 1) * stride;
    for (let x = 1; x <= width; x++) {
      rowSum += gray[sourceOffset + x - 1];
      integral[targetOffset + x] = integral[previousOffset + x] + rowSum;
    }
  }
  return integral;
}

function localThresholdMask(gray, width, height, windowSize = 31, offset = 9) {
  const integral = makeIntegralImage(gray, width, height);
  const stride = width + 1;
  const radius = Math.floor(windowSize / 2);
  const mask = new Uint8Array(gray.length);

  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height, y + radius + 1);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width, x + radius + 1);
      const sum =
        integral[y1 * stride + x1] -
        integral[y0 * stride + x1] -
        integral[y1 * stride + x0] +
        integral[y0 * stride + x0];
      const area = (x1 - x0) * (y1 - y0);
      mask[y * width + x] = gray[y * width + x] < sum / area - offset ? 1 : 0;
    }
  }
  return mask;
}

function globalThresholdMask(gray, threshold) {
  const mask = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) mask[i] = gray[i] < threshold ? 1 : 0;
  return mask;
}

/**
 * Connected-component labeling with a single reusable work stack. This is
 * deliberately limited to coarse row geometry; it does not classify glyphs.
 */
function connectedComponents(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  const components = [];

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;

    let stackSize = 1;
    stack[0] = start;
    visited[start] = 1;
    let area = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumYY = 0;
    let sumXY = 0;

    while (stackSize) {
      const index = stack[--stackSize];
      const y = Math.floor(index / width);
      const x = index - y * width;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      sumX += x;
      sumY += y;
      sumXX += x * x;
      sumYY += y * y;
      sumXY += x * y;

      const y0 = Math.max(0, y - 1);
      const y1 = Math.min(height - 1, y + 1);
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      for (let neighborY = y0; neighborY <= y1; neighborY++) {
        const rowOffset = neighborY * width;
        for (let neighborX = x0; neighborX <= x1; neighborX++) {
          const neighbor = rowOffset + neighborX;
          if (mask[neighbor] && !visited[neighbor]) {
            visited[neighbor] = 1;
            stack[stackSize++] = neighbor;
          }
        }
      }
    }

    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    // Ignore background blobs, long card decorations, and single-pixel noise.
    if (
      area < 3 ||
      area > mask.length * 0.025 ||
      boxWidth > width * 0.42 ||
      boxHeight > height * 0.38
    ) continue;

    const meanX = sumX / area;
    const meanY = sumY / area;
    const varianceY = Math.max(1, sumYY / area - meanY * meanY);
    const covariance = sumXY / area - meanX * meanY;
    components.push({
      x: minX,
      y: minY,
      right: maxX + 1,
      bottom: maxY + 1,
      width: boxWidth,
      height: boxHeight,
      area,
      centerX: meanX,
      centerY: meanY,
      slashSlope: covariance / varianceY,
      fillRatio: area / (boxWidth * boxHeight),
    });
    // Noisy scans can contain a huge number of tiny components. Abstain rather
    // than allocating an unbounded geometry list or risking a spread overflow.
    if (components.length >= MAX_COMPONENTS) return [];
  }

  return components;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function isSlashLike(component, referenceHeight) {
  const aspect = component.width / Math.max(1, component.height);
  const relativeHeight = component.height / Math.max(1, referenceHeight);
  return (
    aspect >= 0.28 &&
    aspect <= 0.95 &&
    relativeHeight >= 0.62 &&
    Math.abs(component.slashSlope) >= 0.15 &&
    component.fillRatio <= 0.72
  );
}

function groupComponentsIntoRows(components, referenceHeight) {
  const tolerance = Math.max(5, referenceHeight * 0.46);
  const rows = [];
  const sorted = [...components].sort((a, b) => a.centerY - b.centerY);

  for (const component of sorted) {
    let best = null;
    let bestDistance = Infinity;
    for (const row of rows) {
      const distance = Math.abs(row.centerY - component.centerY);
      if (distance <= tolerance && distance < bestDistance) {
        best = row;
        bestDistance = distance;
      }
    }

    if (!best) {
      rows.push({ centerY: component.centerY, components: [component] });
    } else {
      best.components.push(component);
      const count = best.components.length;
      best.centerY = (best.centerY * (count - 1) + component.centerY) / count;
    }
  }

  return rows.map((row) => {
    const sortedComponents = row.components.sort((a, b) => a.x - b.x);
    const left = Math.min(...sortedComponents.map((part) => part.x));
    const top = Math.min(...sortedComponents.map((part) => part.y));
    const right = Math.max(...sortedComponents.map((part) => part.right));
    const bottom = Math.max(...sortedComponents.map((part) => part.bottom));
    const slashCount = sortedComponents.filter((part) => isSlashLike(part, referenceHeight)).length;
    const digitLikeCount = sortedComponents.filter((part) => {
      const aspect = part.width / Math.max(1, part.height);
      const relativeHeight = part.height / Math.max(1, referenceHeight);
      return aspect <= 1.75 && relativeHeight >= 0.18 && relativeHeight <= 1.3;
    }).length;
    return {
      components: sortedComponents,
      centerY: row.centerY,
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      right,
      bottom,
      slashCount,
      digitLikeCount,
      referenceHeight,
    };
  });
}

function dateRegionForRow(row) {
  const gapLimit = Math.max(8, row.referenceHeight * 0.95);
  const clusters = [];
  for (const component of row.components) {
    const current = clusters.at(-1);
    if (!current || component.x - current.at(-1).right > gapLimit) clusters.push([component]);
    else current.push(component);
  }

  const ranked = clusters.map((components) => {
    const digitLikeCount = components.filter((part) => {
      const aspect = part.width / Math.max(1, part.height);
      const heightRatio = part.height / row.referenceHeight;
      return aspect <= 1.75 && heightRatio >= 0.16 && heightRatio <= 1.4;
    }).length;
    const slashCount = components.filter((part) => isSlashLike(part, row.referenceHeight)).length;
    const score = digitLikeCount * 2 + slashCount * 8 - Math.max(0, components.length - 14);
    return { components, digitLikeCount, slashCount, score };
  }).filter((cluster) => cluster.digitLikeCount >= 6 && (cluster.slashCount > 0 || cluster.digitLikeCount >= 8));

  ranked.sort((a, b) => b.score - a.score);
  const selected = ranked[0];
  if (!selected) {
    return {
      ...row,
      regionScore: 0,
      contextX: row.x,
      contextY: row.y,
      contextRight: row.right,
      contextBottom: row.bottom,
    };
  }
  const left = Math.min(...selected.components.map((part) => part.x));
  const top = Math.min(...selected.components.map((part) => part.y));
  const right = Math.max(...selected.components.map((part) => part.right));
  const bottom = Math.max(...selected.components.map((part) => part.bottom));
  return {
    components: selected.components,
    x: left,
    y: top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
    digitLikeCount: selected.digitLikeCount,
    slashCount: selected.slashCount,
    regionScore: selected.score,
    contextX: row.x,
    contextY: row.y,
    contextRight: row.right,
    contextBottom: row.bottom,
  };
}

function scoreTextRow(row, imageWidth, imageHeight) {
  const relativeY = row.centerY / imageHeight;
  if (relativeY < SEARCH_BAND.min || relativeY > SEARCH_BAND.max) return null;
  if (row.height < 7 || row.height > imageHeight * 0.3) return null;
  if (row.components.length < 6 || row.digitLikeCount < 6) return null;

  const centerScore = Math.max(0, 1 - Math.abs(relativeY - AVERAGE_ROW_Y) / 0.42);
  const digitScore = Math.min(row.digitLikeCount, 12) * 1.8;
  const punctuationScore = Math.min(row.slashCount, 3) * 13;
  const widthScore = Math.min(12, row.width / Math.max(1, imageWidth) * 18);
  const componentPenalty = Math.max(0, row.components.length - 26) * 0.35;

  return 26 * centerScore + digitScore + punctuationScore + widthScore - componentPenalty;
}

/**
 * Find probable text rows without a document-sized CV runtime. `imageData`
 * is a standard ImageData object; this helper is exported for tests.
 */
export function findDateTextRows(imageData, { maxRows = MAX_ROWS_TO_READ } = {}) {
  const { width, height } = imageData;
  if (!width || !height || width * height > MAX_IMAGE_PIXELS) return [];

  const gray = grayscalePixels(imageData);
  const threshold = otsuThreshold(gray);
  const primary = connectedComponents(globalThresholdMask(gray, threshold), width, height);
  const plausibleHeights = primary
    .filter((part) => part.height >= 7 && part.height <= Math.max(22, height * 0.18))
    .filter((part) => part.width / Math.max(1, part.height) <= 2.2)
    .map((part) => part.height);
  const referenceHeight = percentile(plausibleHeights, 0.75);
  if (!referenceHeight) return [];

  const minY = SEARCH_BAND.min * height;
  const maxY = SEARCH_BAND.max * height;
  const textComponents = primary.filter((part) => {
    if (part.centerY < minY || part.centerY > maxY) return false;
    const aspect = part.width / Math.max(1, part.height);
    const relativeHeight = part.height / referenceHeight;
    const slash = isSlashLike(part, referenceHeight);
    return (
      aspect <= 2.0 &&
      (slash || (relativeHeight >= 0.16 && relativeHeight <= 1.45 && part.area >= 4))
    );
  });

  const rows = groupComponentsIntoRows(textComponents, referenceHeight)
    .map((row) => {
      const score = scoreTextRow(row, width, height);
      const region = dateRegionForRow(row);
      return score == null || !region ? null : { ...row, ...region, score: score + region.regionScore };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score);

  const selected = [];
  for (const row of rows) {
    if (selected.some((other) => Math.abs(other.centerY - row.centerY) < referenceHeight * 0.65)) continue;
    selected.push(row);
    if (selected.length >= maxRows) break;
  }
  return selected;
}

function fitLineAngle(row) {
  const points = row.components.filter((part) => part.height >= row.height * 0.3);
  if (points.length < 4) return 0;
  const meanX = points.reduce((sum, point) => sum + point.centerX, 0) / points.length;
  // Component centers move when glyph heights vary (notably Persian ۰).
  // Baselines are a much steadier signal for deskewing a date row.
  const meanY = points.reduce((sum, point) => sum + point.bottom, 0) / points.length;
  let covariance = 0;
  let varianceX = 0;
  for (const point of points) {
    covariance += (point.centerX - meanX) * (point.bottom - meanY);
    varianceX += (point.centerX - meanX) ** 2;
  }
  if (!varianceX) return 0;
  const angle = Math.atan(covariance / varianceX);
  return Math.abs(angle) <= (12 * Math.PI) / 180 ? angle : 0;
}

function cropTextRow(source, row, variant = "gray", makeCanvas) {
  const padX = Math.max(5, Math.round(row.height * 0.22));
  const padY = Math.max(5, Math.round(row.height * 0.3));
  const left = Math.max(0, Math.min(row.x, row.contextX ?? row.x) - padX);
  const top = Math.max(0, Math.min(row.y, row.contextY ?? row.y) - padY);
  const right = Math.min(source.width, Math.max(row.right, row.contextRight ?? row.right) + padX);
  const bottom = Math.min(source.height, Math.max(row.bottom, row.contextBottom ?? row.bottom) + padY);
  const cropWidth = right - left;
  const cropHeight = bottom - top;
  if (cropWidth < 12 || cropHeight < 8) return null;

  const cropCanvas = makeCanvas(cropWidth, cropHeight);
  const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });
  cropContext.drawImage(source, left, top, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

  const scale = Math.min(7, Math.max(2, TARGET_ROW_HEIGHT / Math.max(1, cropHeight)));
  const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
  const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
  const pad = Math.max(12, Math.round(scaledHeight * 0.18));
  const lineCanvas = makeCanvas(Math.min(8000, scaledWidth + pad * 2), scaledHeight + pad * 2);
  const context = lineCanvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "#fff";
  context.fillRect(0, 0, lineCanvas.width, lineCanvas.height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  const angle = fitLineAngle(row);
  context.save();
  context.translate(lineCanvas.width / 2, lineCanvas.height / 2);
  context.rotate(-angle);
  context.drawImage(cropCanvas, -scaledWidth / 2, -scaledHeight / 2, scaledWidth, scaledHeight);
  context.restore();

  if (variant !== "gray") {
    const image = context.getImageData(0, 0, lineCanvas.width, lineCanvas.height);
    const gray = grayscalePixels(image);
    const threshold = variant === OTSU_MODE ? otsuThreshold(gray) : null;
    const adaptiveMask = variant === ADAPTIVE_MODE
      ? localThresholdMask(gray, lineCanvas.width, lineCanvas.height)
      : null;
    for (let i = 0, pixel = 0; i < image.data.length; i += 4, pixel++) {
      const ink = variant === OTSU_MODE
        ? gray[pixel] < threshold
        : Boolean(adaptiveMask[pixel]);
      const value = ink ? 0 : 255;
      image.data[i] = value;
      image.data[i + 1] = value;
      image.data[i + 2] = value;
      image.data[i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  } else {
    // Mild contrast stretch; unlike hard thresholding this preserves thin
    // strokes and anti-aliased Persian numerals for the LSTM recognizer.
    const image = context.getImageData(0, 0, lineCanvas.width, lineCanvas.height);
    const gray = grayscalePixels(image);
    let low = 255;
    let high = 0;
    for (const value of gray) {
      if (value < low) low = value;
      if (value > high) high = value;
    }
    const span = Math.max(24, high - low);
    for (let i = 0, pixel = 0; i < image.data.length; i += 4, pixel++) {
      const value = Math.max(0, Math.min(255, ((gray[pixel] - low) * 255) / span));
      image.data[i] = value;
      image.data[i + 1] = value;
      image.data[i + 2] = value;
      image.data[i + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  return lineCanvas.toDataURL("image/png");
}

function rotateCanvas180(source, makeCanvas) {
  const canvas = makeCanvas(source.width, source.height);
  const context = canvas.getContext("2d");
  context.translate(source.width, source.height);
  context.rotate(Math.PI);
  context.drawImage(source, 0, 0);
  return canvas;
}

function rowsForCanvas(canvas, options) {
  const imageData = canvas.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, canvas.width, canvas.height);
  return findDateTextRows(imageData, options);
}

function meanSymbolConfidence(data) {
  const symbols = (data.blocks ?? [])
    .flatMap((block) => block.paragraphs ?? [])
    .flatMap((paragraph) => paragraph.lines ?? [])
    .flatMap((line) => line.words ?? [])
    .flatMap((word) => word.symbols ?? [])
    .filter((symbol) => /[0-9۰-۹٠-٩]/.test(symbol.text ?? ""));
  const scores = symbols
    .map((symbol) => Number(symbol.confidence))
    .filter((score) => Number.isFinite(score));
  if (!scores.length) return Number(data.confidence) || 0;
  return scores.reduce((sum, score) => sum + score, 0) / scores.length;
}

export class LocalIranCardOCR {
  constructor(options = {}) {
    this.worker = null;
    this.ready = false;
    this.initPromise = null;
    this.langPath = options.langPath;
    this.workerPath = options.workerPath;
    this.corePath = options.corePath;
    this.minConfidence = options.minConfidence ?? MIN_CONFIDENCE;
    this.maxYear = options.maxYear ?? 1410;
    this.minYear = options.minYear ?? 1280;
    this._progress = null;
    this._queue = Promise.resolve();
    this._ocrCalls = 0;
    this._logger = (message) => {
      const status = String(message?.status ?? "");
      const value = typeof message?.progress === "number" ? message.progress : 0;
      if (status.includes("loading tesseract core")) this._progress?.(32 + Math.round(value * 10));
      else if (status.includes("loading language")) this._progress?.(42 + Math.round(value * 15));
      else if (status.includes("initializing")) this._progress?.(58);
      else if (status.includes("recognizing text")) this._progress?.(65 + Math.round(value * 30));
    };
  }

  async initialize(onProgress) {
    if (this.ready) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      const previousProgress = this._progress;
      if (onProgress) this._progress = onProgress;
      try {
        // Lazy import keeps the app's initial JavaScript small. Tesseract's
        // WASM core and `fas` data are fetched only when a date row is found.
        const tesseract = await import("tesseract.js");
        this.worker = await tesseract.createWorker("fas", tesseract.OEM.LSTM_ONLY, {
          ...(this.langPath ? { langPath: this.langPath } : {}),
          ...(this.workerPath ? { workerPath: this.workerPath } : {}),
          ...(this.corePath ? { corePath: this.corePath } : {}),
          cachePath: ".tesseract",
          logger: this._logger,
        });
        await this.worker.setParameters({
          tessedit_char_whitelist: DATE_CHARACTERS,
          tessedit_pageseg_mode: String(tesseract.PSM.SINGLE_LINE),
          preserve_interword_spaces: "1",
          user_defined_dpi: "300",
        });
        this.ready = true;
      } finally {
        this._progress = previousProgress;
      }
    })();

    try {
      await this.initPromise;
    } catch (error) {
      this.worker = null;
      this.ready = false;
      throw error;
    } finally {
      this.initPromise = null;
    }
  }

  async terminate() {
    if (this.initPromise) {
      try { await this.initPromise; } catch { /* ignore shutdown errors */ }
    }
    if (this.worker) {
      try { await this.worker.terminate(); } catch { /* ignore shutdown errors */ }
    }
    this.worker = null;
    this.ready = false;
    this._progress = null;
    this._queue = Promise.resolve();
  }

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
    this._ocrCalls = 0;
    const attempts = [];
    let analysisCanvas = null;

    const result = (payload) => ({
      ...payload,
      engine: "canvas+tesseract-fas",
      requiresUserConfirmation: Boolean(payload.success),
      confidenceKind: payload.confidenceKind ?? "recognizer-score-not-calibrated-probability",
      durationMs: Math.round(now() - started),
      ocrCalls: this._ocrCalls,
      attempts,
    });

    try {
      progress?.(4);
      const makeCanvas = await getCanvasFactory();
      analysisCanvas = await toAnalysisCanvas(input, makeCanvas);
      progress?.(18);

      let best = null;
      const firstRows = rowsForCanvas(analysisCanvas);
      for (let orientationIndex = 0; orientationIndex < 2; orientationIndex++) {
        if (orientationIndex > 0 && best) break;
        const rotation = orientationIndex === 0 ? 0 : 180;
        const orientationCanvas = rotation === 0
          ? analysisCanvas
          : rotateCanvas180(analysisCanvas, makeCanvas);
        const rows = rotation === 0 ? firstRows : rowsForCanvas(orientationCanvas);
        if (!rows.length) continue;
        progress?.(25);

        if (!this.ready) {
          try {
            await this.initialize(progress);
          } catch (error) {
            return result({
              success: false,
              error: `OCR initialisation failed: ${error?.message || error}`,
            });
          }
        }

        const orientationResults = [];
        for (let index = 0; index < rows.length && this._ocrCalls < MAX_OCR_CALLS; index++) {
          const row = rows[index];
          const readVariants = ["gray", OTSU_MODE, ADAPTIVE_MODE];
          let selected = null;
          const validReads = [];

          for (const variant of readVariants) {
            if (this._ocrCalls >= MAX_OCR_CALLS) break;
            const lineImage = cropTextRow(orientationCanvas, row, variant, makeCanvas);
            if (!lineImage) continue;

            this._ocrCalls++;
            let read;
            try {
              const { data } = await this.worker.recognize(lineImage, {}, { text: true, blocks: true });
              read = {
                text: data.text || "",
                confidence: meanSymbolConfidence(data),
              };
            } catch (error) {
              attempts.push({
                strategy: `row-${variant}`,
                engine: "canvas+tesseract-fas",
                candidateIndex: index,
                rotation: rotation,
                bounds: { x: row.x, y: row.y, width: row.width, height: row.height },
                rawText: "",
                confidence: 0,
                note: error?.message || String(error),
              });
              continue;
            }

            const parsed = parseBirthDateLine(read.text, {
              minYear: this.minYear,
              maxYear: this.maxYear,
            });
            const attempt = {
              strategy: `row-${variant}`,
              engine: "canvas+tesseract-fas",
              candidateIndex: index,
              rotation: rotation,
              bounds: { x: row.x, y: row.y, width: row.width, height: row.height },
              rawText: read.text.trim(),
              confidence: Math.round(read.confidence),
              dates: parsed ? [{ formatted: parsed.formatted }] : [],
              lineImage,
              preprocessedImage: lineImage,
            };
            attempts.push(attempt);

            if (!parsed || read.confidence < this.minConfidence) {
              if (
                variant === OTSU_MODE &&
                validReads.length === 1 &&
                validReads[0].confidence >= 82
              ) break;
              continue;
            }
            const current = { parsed, confidence: read.confidence, row, rotation, lineImage, variant };
            validReads.push(current);
            const matchingReads = validReads.filter(
              (candidateRead) => candidateRead.parsed.formatted === parsed.formatted,
            );
            if (matchingReads.length >= 2) {
              selected = {
                ...current,
                confidence: Math.round(
                  matchingReads.reduce((sum, candidateRead) => sum + candidateRead.confidence, 0) /
                    matchingReads.length,
                ),
                agreement: true,
              };
              break;
            }
          }

          const candidate = selected ?? (
            validReads.length === 1 && validReads[0].confidence >= 82
              ? { ...validReads[0], agreement: false }
              : null
          );
          if (!candidate) continue;

          const geomScore = row.score ?? 0;
          const candidateScore = geomScore + candidate.confidence * 0.35 + (candidate.agreement ? 20 : 0);
          const successful = {
            success: true,
            birthDate: candidate.parsed.formatted,
            year: candidate.parsed.year,
            month: candidate.parsed.month,
            day: candidate.parsed.day,
            confidence: Math.round(candidate.confidence),
            confidenceKind: candidate.agreement ? "two-preprocessing-reads-agree" : "single-high-confidence-read",
            repairs: 0,
            durationMs: 0,
            ocrCalls: 0,
            attempts,
            lineImage: candidate.lineImage,
            requiresUserConfirmation: true,
            rotation: candidate.rotation,
            agreement: candidate.agreement,
            _rank: candidateScore,
          };
          orientationResults.push(successful);

          // The DOB field is normally the only date-like center-band row.
          // If the best row has a separator pattern and two agreeing views,
          // there is little value in processing more rows on a slow device.
          if (candidate.agreement && row.slashCount > 0) break;
        }

        orientationResults.sort((a, b) => b._rank - a._rank);
        const orientationBest = orientationResults[0];
        if (orientationBest) {
          best = orientationBest;
          break;
        }
      }

      progress?.(100);
      if (best) {
        delete best._rank;
        return result(best);
      }

      const hasAttempts = attempts.length > 0;
      return result({
        success: false,
        error: hasAttempts
          ? "No unambiguous Jalali birth date was read. Crop the birth-date line tightly and try again."
          : "No date-like text row was found. Crop the front of the card so the birth-date line is visible.",
      });
    } catch (error) {
      return result({ success: false, error: error?.message || String(error) });
    } finally {
      this._progress = null;
    }
  }
}

export { LocalIranCardOCR as IranCardOCR };
export default LocalIranCardOCR;
