// Local/private OCR benchmark. Images and labels are read from disk only and
// are never copied into the repository or sent to a service.
// Usage: npm run benchmark -- --manifest ./private-ocr-eval/manifest.json
// Optional: --min-precision 0.90 (exit non-zero if exact precision is lower).
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { LocalIranCardOCR, parseBirthDateLine } from "../ocr/localIranCardOCR.js";

function parseArgs(argv) {
  const options = { manifest: "./private-ocr-eval/manifest.json", minPrecision: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--manifest") {
      options.manifest = argv[++i];
      if (!options.manifest || options.manifest.startsWith("--")) {
        throw new Error("--manifest requires a JSON file path.");
      }
    } else if (argv[i] === "--min-precision") {
      const precision = Number(argv[++i]);
      if (!Number.isFinite(precision) || precision < 0 || precision > 1) {
        throw new Error("--min-precision must be a number between 0 and 1.");
      }
      options.minPrecision = precision;
    } else {
      throw new Error(`Unknown option: ${argv[i]}`);
    }
  }
  return options;
}

const require = createRequire(import.meta.url);
const options = parseArgs(process.argv.slice(2));
const manifestPath = path.resolve(options.manifest);
if (!fs.existsSync(manifestPath)) {
  throw new Error(`Manifest not found: ${manifestPath}`);
}

let langPath;
try {
  langPath = path.join(
    path.dirname(require.resolve("@tesseract.js-data/fas/package.json")),
    "4.0.0_best_int",
  );
} catch {
  langPath = undefined; // falls back to Tesseract.js static assets
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const samples = Array.isArray(manifest) ? manifest : manifest.samples;
if (!Array.isArray(samples) || samples.length === 0) {
  throw new Error("Manifest must be a non-empty array, or contain a non-empty `samples` array.");
}

const ocr = new LocalIranCardOCR({ langPath });
let accepted = 0;
let correct = 0;
let falseAccepts = 0;
let abstained = 0;
let durationTotal = 0;

try {
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index];
    if (!sample?.image || !sample?.birthDate) {
      throw new Error(`Sample ${index + 1} must contain image and birthDate.`);
    }
    const expected = parseBirthDateLine(sample.birthDate)?.formatted;
    if (!expected) throw new Error(`Sample ${index + 1} has an invalid expected Jalali date.`);

    const imagePath = path.resolve(path.dirname(manifestPath), sample.image);
    if (!fs.existsSync(imagePath)) throw new Error(`Sample ${index + 1} image is missing.`);
    const imageBytes = fs.readFileSync(imagePath);
    const result = await ocr.recognizeBirthDate(imageBytes);
    durationTotal += result.durationMs ?? 0;

    if (result.success) {
      accepted++;
      if (result.birthDate === expected) {
        correct++;
        console.log(`#${index + 1}: exact match (${result.durationMs} ms)`);
      } else {
        falseAccepts++;
        console.log(`#${index + 1}: WRONG accepted date (${result.durationMs} ms)`);
      }
    } else {
      abstained++;
      console.log(`#${index + 1}: abstained (${result.durationMs} ms)`);
    }
  }
} finally {
  await ocr.terminate();
}

const precision = accepted ? correct / accepted : 0;
const coverage = accepted / samples.length;
const exactRecall = correct / samples.length;
console.log("\nLocal evaluation summary");
console.log(`  samples:      ${samples.length}`);
console.log(`  accepted:     ${accepted}`);
console.log(`  abstained:    ${abstained}`);
console.log(`  false accepts:${falseAccepts}`);
console.log(`  exact precision: ${(precision * 100).toFixed(1)}%`);
console.log(`  exact coverage:  ${(coverage * 100).toFixed(1)}%`);
console.log(`  exact recall:    ${(exactRecall * 100).toFixed(1)}%`);
console.log(`  mean latency:    ${(durationTotal / samples.length).toFixed(0)} ms`);

if (options.minPrecision != null && precision < options.minPrecision) {
  process.exitCode = 1;
}
