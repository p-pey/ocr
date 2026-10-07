// Zero-Tesseract / zero-generic-OCR verification (field directive 3):
// no tesseract modules, wrappers, traineddata, subprocess calls, nor
// tfjs/onnx runtimes anywhere in shipped source, training tools, or
// declared dependencies. Run: node tests/noTesseract.test.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Import/call patterns (bare words like the legacy class name in comments
// do NOT match — only real module references, data files, and commands).
const BANNED = [
  /tesseract\.js/i,
  /from\s+['"]tesseract/i,
  /require\(\s*['"]tesseract/i,
  /pytesseract/i,
  /leptonica/i,
  /traineddata/i,
  /tessedit_/i,
  /tessdata/i,
  /createWorker\s*\(\s*['"]fas/i,
  /@tensorflow/i,
  /onnxruntime/i,
];

const walk = (dir, out = []) => {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (["node_modules", "dist", ".git"].includes(e.name)) continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (/\.(js|jsx|ts|tsx|mjs|py|json)$/.test(e.name)) out.push(p);
  }
  return out;
};

const violations = [];
for (const f of [...walk(path.join(root, "src")), ...walk(path.join(root, "training-tools"))]) {
  if (f.endsWith("noTesseract.test.mjs")) continue;
  const text = fs.readFileSync(f, "utf8");
  for (const re of BANNED) {
    if (re.test(text)) violations.push(`${path.relative(root, f)} matches ${re}`);
  }
}
assert.deepEqual(violations, [], `banned OCR dependencies found:\n${violations.join("\n")}`);

// Declared dependencies must not include them either.
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const deps = { ...pkg.dependencies, ...pkg.devDependencies };
const bannedDeps = Object.keys(deps).filter((d) => /tesseract|tensorflow|onnx|leptonica/i.test(d));
assert.deepEqual(bannedDeps, [], `banned packages in package.json: ${bannedDeps}`);
console.log("no-tesseract verification passed");
