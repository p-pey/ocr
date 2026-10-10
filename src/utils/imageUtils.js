// src/utils/imageUtils.js
//
// Active helpers used by the app + engine:
//   loadImage             — Blob/URL/data-URL -> HTMLImageElement
//                          (also used by src/ocr/TesseractOCR.js)
//   getCroppedImg         — crop/rotate helper for the ImageEditor flow
//   getRotatedBoundingBox — bounding-box math for getCroppedImg
// Legacy Tesseract-era code was removed (nothing imported it):
// PREPROCESS_STRATEGIES, cropBirthDateRegions + pixel helpers
// (applyGrayscale/Contrast/Brightness/Invert/Otsu/Adaptive/Unsharp/Erode).
// Engine preprocessing lives in src/ocr/TesseractOCR.js (tuned via
// src/ocr/engineConfig.js).

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
