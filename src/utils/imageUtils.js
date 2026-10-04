// src/utils/imageUtils.js

export async function getCroppedImg(imageSrc, pixelCrop, rotation = 0) {
  const image = await loadImage(imageSrc);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const rotRad = (rotation * Math.PI) / 180;
  const { width: bBoxW, height: bBoxH } = getRotatedBoundingBox(
    image.width,
    image.height,
    rotation,
  );

  canvas.width = bBoxW;
  canvas.height = bBoxH;

  ctx.translate(bBoxW / 2, bBoxH / 2);
  ctx.rotate(rotRad);
  ctx.translate(-image.width / 2, -image.height / 2);
  ctx.drawImage(image, 0, 0);

  const croppedData = ctx.getImageData(
    pixelCrop.x,
    pixelCrop.y,
    pixelCrop.width,
    pixelCrop.height,
  );

  canvas.width = pixelCrop.width;
  canvas.height = pixelCrop.height;
  ctx.putImageData(croppedData, 0, 0);

  return {
    canvas,
    dataUrl: canvas.toDataURL("image/png"),
    blob: await canvasToBlob(canvas),
  };
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    if (src instanceof Blob) img.src = URL.createObjectURL(src);
    else img.src = src;
  });
}

export function getRotatedBoundingBox(w, h, rotation) {
  const rad = (rotation * Math.PI) / 180;
  const cos = Math.abs(Math.cos(rad));
  const sin = Math.abs(Math.sin(rad));
  return {
    width: Math.round(w * cos + h * sin),
    height: Math.round(w * sin + h * cos),
  };
}

function canvasToBlob(canvas) {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));

// ═══════════════════════════════════════════════════════════
// PREPROCESSING STRATEGIES — Each returns a processed canvas
// ═══════════════════════════════════════════════════════════

export const PREPROCESS_STRATEGIES = {
  /** Strategy 1: Simple upscale + grayscale (good for high quality images) */
  async simpleEnhance(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2000 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyContrast(data, 40);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 2: Aggressive binarization (good for printed text) */
  async binarize(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2000 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyContrast(data, 70);
    applyOtsuThreshold(data);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 3: Adaptive threshold (good for uneven lighting) */
  async adaptiveThreshold(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2000 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyAdaptiveThreshold(data, canvas.width, canvas.height, 21, 10);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 4: Inverted (for dark backgrounds with light text) */
  async inverted(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2000 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyInvert(data);
    applyContrast(data, 50);
    applyAdaptiveThreshold(data, canvas.width, canvas.height, 15, 5);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 5: Red channel isolation (Iranian ID cards have red text) */
  async redChannel(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2000 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = data.data;

    // Highlight red/dark text — convert everything else to white
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i],
        g = d[i + 1],
        b = d[i + 2];
      // Red text: high R, low G, low B
      const isRed = r > 100 && r > g + 20 && r > b + 20;
      // Dark text
      const isDark = r < 100 && g < 100 && b < 100;

      if (isRed || isDark) {
        d[i] = d[i + 1] = d[i + 2] = 0;
      } else {
        d[i] = d[i + 1] = d[i + 2] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 6: Sharpening + edge enhancement */
  async sharpen(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2500 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    let data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyUnsharpMask(data, canvas.width, canvas.height);
    applyContrast(data, 50);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 7: High contrast + dilate (makes digits bolder) */
  async boldText(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 2500 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyGrayscale(data);
    applyContrast(data, 80);
    applyOtsuThreshold(data);
    applyErode(data, canvas.width, canvas.height); // Thicken dark pixels
    ctx.putImageData(data, 0, 0);

    return canvas;
  },

  /** Strategy 8: Original with mild enhance (sometimes OCR likes original color) */
  async mild(imageSrc) {
    const img = await loadImage(imageSrc);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const scale = Math.max(1, 1800 / Math.max(img.width, img.height));

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyContrast(data, 25);
    ctx.putImageData(data, 0, 0);

    return canvas;
  },
};

// ═══════════════════════════════════════════════════════════
// IMAGE PROCESSING ALGORITHMS
// ═══════════════════════════════════════════════════════════

function applyGrayscale(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    d[i] = d[i + 1] = d[i + 2] = g;
  }
}

function applyContrast(imageData, amount) {
  const d = imageData.data;
  const f = (259 * (amount + 255)) / (255 * (259 - amount));
  for (let i = 0; i < d.length; i += 4) {
    d[i] = clamp(f * (d[i] - 128) + 128);
    d[i + 1] = clamp(f * (d[i + 1] - 128) + 128);
    d[i + 2] = clamp(f * (d[i + 2] - 128) + 128);
  }
}

function applyInvert(imageData) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = 255 - d[i];
    d[i + 1] = 255 - d[i + 1];
    d[i + 2] = 255 - d[i + 2];
  }
}

function applyOtsuThreshold(imageData) {
  const d = imageData.data;
  const histogram = new Array(256).fill(0);
  const totalPixels = d.length / 4;

  for (let i = 0; i < d.length; i += 4) histogram[d[i]]++;

  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i];

  let sumB = 0,
    wB = 0,
    maxVar = 0,
    threshold = 0;

  for (let t = 0; t < 256; t++) {
    wB += histogram[t];
    if (wB === 0) continue;
    const wF = totalPixels - wB;
    if (wF === 0) break;

    sumB += t * histogram[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const variance = wB * wF * (mB - mF) ** 2;

    if (variance > maxVar) {
      maxVar = variance;
      threshold = t;
    }
  }

  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] > threshold ? 255 : 0;
    d[i] = d[i + 1] = d[i + 2] = v;
  }
}

function applyAdaptiveThreshold(imageData, w, h, blockSize = 15, C = 8) {
  const d = imageData.data;
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < gray.length; i++) gray[i] = d[i * 4];

  const integral = new Float64Array((w + 1) * (h + 1));
  for (let y = 1; y <= h; y++)
    for (let x = 1; x <= w; x++)
      integral[y * (w + 1) + x] =
        gray[(y - 1) * w + (x - 1)] +
        integral[(y - 1) * (w + 1) + x] +
        integral[y * (w + 1) + (x - 1)] -
        integral[(y - 1) * (w + 1) + (x - 1)];

  const half = Math.floor(blockSize / 2);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const x1 = Math.max(0, x - half);
      const y1 = Math.max(0, y - half);
      const x2 = Math.min(w, x + half + 1);
      const y2 = Math.min(h, y + half + 1);
      const area = (x2 - x1) * (y2 - y1);
      const sum =
        integral[y2 * (w + 1) + x2] -
        integral[y1 * (w + 1) + x2] -
        integral[y2 * (w + 1) + x1] +
        integral[y1 * (w + 1) + x1];
      const val = gray[y * w + x] > sum / area - C ? 255 : 0;
      const idx = (y * w + x) * 4;
      d[idx] = d[idx + 1] = d[idx + 2] = val;
    }
  }
}

function applyUnsharpMask(imageData, w, h) {
  const d = imageData.data;
  const out = new Uint8ClampedArray(d);
  const k = [-1, -1, -1, -1, 9, -1, -1, -1, -1];

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sum = 0;
      for (let ky = -1; ky <= 1; ky++)
        for (let kx = -1; kx <= 1; kx++)
          sum += d[((y + ky) * w + (x + kx)) * 4] * k[(ky + 1) * 3 + (kx + 1)];
      const idx = (y * w + x) * 4;
      out[idx] = out[idx + 1] = out[idx + 2] = clamp(sum);
    }
  }
  imageData.data.set(out);
}

function applyErode(imageData, w, h) {
  const d = imageData.data;
  const out = new Uint8ClampedArray(d);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let minVal = 255;
      for (let ky = -1; ky <= 1; ky++)
        for (let kx = -1; kx <= 1; kx++)
          minVal = Math.min(minVal, d[((y + ky) * w + (x + kx)) * 4]);
      const idx = (y * w + x) * 4;
      out[idx] = out[idx + 1] = out[idx + 2] = minVal;
    }
  }
  imageData.data.set(out);
}

// ═══════════════════════════════════════════════════════════
// MULTI-REGION CROPPING FOR BIRTH DATE
// ═══════════════════════════════════════════════════════════

/**
 * Iranian national ID card has birth date in specific regions.
 * We'll try multiple regions since the format varies between card versions.
 */
export async function cropBirthDateRegions(imageSrc) {
  const img = await loadImage(imageSrc);
  const regions = [
    // Full card
    { name: "full", x: 0, y: 0, w: 1, h: 1 },

    // Right side (where Persian text usually is)
    { name: "right-half", x: 0.5, y: 0, w: 0.5, h: 1 },

    // Middle-right vertical band (common birth date location)
    { name: "mid-right", x: 0.35, y: 0.3, w: 0.5, h: 0.5 },

    // Bottom-right (new card format)
    { name: "bottom-right", x: 0.3, y: 0.5, w: 0.6, h: 0.4 },

    // Center area
    { name: "center", x: 0.2, y: 0.3, w: 0.6, h: 0.5 },

    // Top-right (old cards)
    { name: "top-right", x: 0.4, y: 0.1, w: 0.55, h: 0.5 },

    // Horizontal strip in middle (for date rows)
    { name: "mid-strip", x: 0.1, y: 0.4, w: 0.8, h: 0.25 },
    { name: "mid-strip-2", x: 0.1, y: 0.55, w: 0.8, h: 0.25 },

    // Lower horizontal strip
    { name: "low-strip", x: 0.1, y: 0.65, w: 0.8, h: 0.25 },
  ];

  const results = [];

  for (const region of regions) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const sx = Math.round(img.width * region.x);
    const sy = Math.round(img.height * region.y);
    const sw = Math.round(img.width * region.w);
    const sh = Math.round(img.height * region.h);

    canvas.width = sw;
    canvas.height = sh;
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

    results.push({
      name: region.name,
      canvas,
      dataUrl: canvas.toDataURL("image/png"),
      bounds: { x: sx, y: sy, w: sw, h: sh },
    });
  }

  return results;
}
