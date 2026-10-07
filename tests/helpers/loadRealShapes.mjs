// Node/test-only loader for real card-shape PNG exemplars.
// Lives outside src/ so vite never bundles @napi-rs/canvas.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { registerShapeTemplates, clearShapeTemplates, shapeRegistryEmpty } from "../../src/ocr/shapeGate.js";
import { loadImage, createCanvas } from "@napi-rs/canvas";

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(here, "../../training-tools/glyphs/card_shapes");

export async function loadRealShapeTemplates() {
  if (!shapeRegistryEmpty()) return;
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "card_shapes.json"), "utf8"));
  const groups = {};
  for (const s of meta.shapes || []) {
    const img = await loadImage(path.join(dir, s.png));
    const c = createCanvas(img.width, img.height);
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const dd = g.getImageData(0, 0, img.width, img.height).data;
    const px = new Float32Array(img.width * img.height);
    for (let i = 0; i < px.length; i++) px[i] = dd[i * 4];
    (groups[s.shape] ??= []).push({ w: img.width, h: img.height, data: px });
  }
  clearShapeTemplates();
  registerShapeTemplates(groups);
}

export { clearShapeTemplates, shapeRegistryEmpty };
