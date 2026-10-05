# On-device Iranian national-card birth-date OCR

A browser OCR flow for extracting a Jalali birth date from an Iranian national
card. The active app asks the user to crop/straighten the card, finds likely
numeric rows with Canvas-based image processing, and recognizes only those rows
with Tesseract.js Persian (`fas`) running in a browser worker.

**The image is not sent to an OCR API.** No custom neural model, training corpus,
TensorFlow runtime, or OpenCV runtime is needed by the active app. The previous
5.6 MB public CNN model was unused by the app and has been removed from the
production output.

## Privacy and network behavior

- Image decoding, row detection, preprocessing, and OCR inference run locally
  on the user's device. The OCR result is returned to the app; the engine does
  not upload image pixels or call an inference endpoint.
- By default, Tesseract.js downloads its worker/WASM runtime and the static
  Persian language data from its configured CDN the first time OCR is used.
  Those are OCR assets, not an OCR API; the user's image is still processed
  locally. Browser caching reduces repeat downloads.
- For strict network isolation, self-host the Tesseract worker, all compatible
  core builds, and `fas.traineddata.gz`, then pass their paths to the engine.
  Tesseract.js documents the asset paths and local setup
  [here](https://github.com/naptha/tesseract.js/blob/master/docs/local-installation.md).

```js
import { LocalIranCardOCR } from "./src/ocr/localIranCardOCR.js";

const ocr = new LocalIranCardOCR({
  langPath: "/assets/tessdata",          // contains fas.traineddata.gz
  workerPath: "/assets/tesseract/worker.min.js",
  corePath: "/assets/tesseract-core",   // directory with the supported core builds
});

const result = await ocr.recognizeBirthDate(fileOrCanvas, (percent) => {
  console.log(`OCR progress: ${percent}%`);
});

if (result.success) {
  console.log(result.birthDate); // e.g. "1375/05/12"
  // Always show this as a candidate and require a human to verify it.
}

await ocr.terminate();
```

The constructor can also take `minYear`, `maxYear`, and `minConfidence` options.
The default accepted year range is 1280–1410 Jalali. Browser input supports
`File`/`Blob`, data URLs, `<img>`, `<canvas>`, and `ImageBitmap`. In Node tests,
pass a canvas, `Buffer`/`Uint8Array`, or data URL.

## Why this approach

The official Iranian civil-registration form shows Jalali input in
`YYYY/MM/DD` form (for example, `1345/04/01`), and document reporting describes
both date of birth and expiry date on the card front. That makes the bottom
expiry row an important false-positive risk:

- [Iranian civil-registration form](https://auth.ncr.ir/hooda-api/ctzRegistration?refId=)
- [Landinfo report on Iranian passports, ID and civil-status documents (PDF)](https://landinfo.no/wp-content/uploads/2021/01/Iran-Passports-ID-and-civil-status-documnents-05012021.pdf)

The active flow uses a bounded center-card search band, recognizes a whole
numeric text line rather than making eight context-free guesses, compares
normal grayscale, Otsu, and local adaptive-threshold reads, and accepts only a
strict, calendar-valid Jalali date. The fast path is usually two Tesseract line
reads; hard cases can try up to 24 candidate reads across row variants and a
180° retry. Tesseract.js runs through a local browser worker; its worker and
model-asset behavior is described in the
[Tesseract.js documentation](https://github.com/naptha/tesseract.js/blob/master/docs/local-installation.md).

I kept the older OpenCV/CNN code available for comparison and training, but
moved OpenCV and TensorFlow.js to development dependencies. They are not
imported by the production app. The legacy trainer uses pure-JavaScript
TensorFlow by default (slower, but without a native TensorFlow install). The
`public/models/date_cnn/` 5.6 MB weights and preview were removed, and
`npm run train` now writes model files to the git-ignored `models/date_cnn/`
directory rather than `public/`.

## Result contract

Success:

```js
{
  success: true,
  birthDate: "1375/05/12",
  year: 1375,
  month: 5,
  day: 12,
  confidence: 85,
  confidenceKind: "two-preprocessing-reads-agree",
  requiresUserConfirmation: true,
  agreement: true,
  engine: "canvas+tesseract-fas",
  durationMs: 840,
  ocrCalls: 2,
  attempts: [/* row-level debug information */],
  lineImage: "data:image/png;base64,..."
}
```

Failure is always returned as `{ success: false, error, durationMs, ocrCalls,
attempts }`; `recognizeBirthDate()` does not throw to its caller.

**`confidence` is a Tesseract character-score, not a calibrated probability
that the date is correct.** A valid date may still be misread as another valid
Jalali date. All successful results set `requiresUserConfirmation: true`; the
React UI also presents the value as an OCR candidate and warns the user to
compare it with the physical card before using it in a financial decision.

## Accuracy and safety limits

The repository has synthetic regression fixtures rendered with Yekan and
Vazirmatn. They check crop-only input, a card-like photo containing a separate
10-digit ID and an expiry date, upside-down and mildly rotated (4°) crops, a
downscaled crop, no-date rejection, and bad input handling. **They do not
establish 90% accuracy on real card photos.**
There is no honest way to promise 90% without a representative, consented,
locally evaluated set of actual capture conditions.

The active app includes manual crop and rotation controls. Use the front of the
card, keep the birth-date row in view, avoid glare/blur, and straighten the row.
Severe perspective distortion, low resolution, occlusion, unfamiliar card
layouts, and damaged prints can cause abstentions or errors. The engine intentionally rejects
ambiguous/unreadable rows instead of guessing; even a successful result needs
human confirmation.

### Measure your own accuracy without uploading card data

Place a local JSON manifest and local sample images in `private-ocr-eval/` (that
folder is git-ignored):

```json
[
  { "image": "case-001.jpg", "birthDate": "1375/05/12" },
  { "image": "case-002.jpg", "birthDate": "1366/06/22" }
]
```

Then run:

```sh
npm run benchmark -- --manifest ./private-ocr-eval/manifest.json
npm run benchmark -- --manifest ./private-ocr-eval/manifest.json --min-precision 0.90
```

The benchmark reads only those local files and reports exact precision among
accepted results, coverage, recall, false accepts, and mean latency. It prints
sample numbers rather than birth dates or image paths. For a fintech launch,
include real consenting users' card variations and difficult captures, measure
false accepts separately from abstentions, and do not enable unattended
acceptance solely because synthetic tests pass.

## Development and tests

```sh
npm ci --ignore-scripts
npm test                  # pure parsing + synthetic Canvas/Tesseract regressions
npm run test:legacy       # previous OpenCV/Tesseract engine (optional comparison)
npm run build
npm run lint
```

The active engine is `src/ocr/localIranCardOCR.js`; the app adapter is
`src/hooks/useOCR.js`. `src/tests/localIranCardOCR.test.mjs` contains the active
synthetic regressions. Legacy code remains in `src/ocr/iranCardOCR.js`,
`src/ocr/TesseractOCR.js`, and `src/train/train.mjs` for experiments only.

The Vite production output in this checkout is about 314 KB uncompressed
(roughly 100 KB gzipped) of app JS and CSS and contains no OpenCV bundle or
custom model files. Tesseract's runtime and Persian language pack are separate
runtime assets (or can be self-hosted as above); the exact network transfer
depends on Tesseract.js and browser caching.
