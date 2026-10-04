// src/ocr/tesseractOCR.js

import Tesseract, { createWorker } from "tesseract.js";
import {
  PREPROCESS_STRATEGIES,
  cropBirthDateRegions,
} from "../utils/imageUtils";
import { isValidJalali, persianToEnglish } from "../utils/persianUtils";

const LSTM_ONLY = Tesseract.OEM?.LSTM_ONLY ?? 1;
const SINGLE_LINE_PSM = Tesseract.PSM?.SINGLE_LINE ?? 7;
const RAW_LINE_PSM = Tesseract.PSM?.RAW_LINE ?? 13;

// Iranian cards may use Persian digits (۰۱۲۳...), Arabic-Indic digits (٠١٢٣...),
// or Western digits. Whitelist only those digits, date separators, and spacing.
const DATE_CHAR_WHITELIST = "0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩/-. ";

const DATE_PREPROCESSING_STRATEGIES = ["simpleEnhance", "binarize"];

function normalizeBirthDateText(text) {
  return persianToEnglish(text)
    .replace(/[IiLl|]/g, "1")
    .replace(/[Zz]/g, "2")
    .replace(/[Bb]/g, "8");
}

function extractBirthDates(text) {
  const normalized = normalizeBirthDateText(text);
  if (!normalized) return [];

  const patterns = [
    {
      regex:
        /(^|[^\d])((?:13|14)\d{2})\s*(?:[\/.-]|\s+)\s*(0?[1-9]|1[0-2])\s*(?:[\/.-]|\s+)\s*(0?[1-9]|[12]\d|3[01])(?=$|[^\d])/g,
      score: 100,
    },
    {
      regex:
        /(^|[^\d])((?:13|14)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?=$|[^\d])/g,
      score: 90,
    },
  ];
  const datesByValue = new Map();

  for (const { regex, score } of patterns) {
    let match;
    while ((match = regex.exec(normalized)) !== null) {
      const year = Number(match[2]);
      const month = Number(match[3]);
      const day = Number(match[4]);
      if (!isValidJalali(year, month, day)) continue;

      const formatted = [
        String(year),
        String(month).padStart(2, "0"),
        String(day).padStart(2, "0"),
      ].join("/");
      const date = {
        year,
        month,
        day,
        score,
        raw: match[0].slice(match[1].length).trim(),
        formatted,
      };
      const existing = datesByValue.get(formatted);
      if (!existing || existing.score < score) datesByValue.set(formatted, date);
    }
  }

  return [...datesByValue.values()].sort((a, b) => b.score - a.score);
}

const PAGE_SEGMENT_MODES = [
  { name: "single-line", value: SINGLE_LINE_PSM },
  { name: "raw-line", value: RAW_LINE_PSM },
];

const CROP_PRIORITY = [
  "mid-strip",
  "mid-strip-2",
  "low-strip",
  "mid-right",
  "bottom-right",
  "top-right",
  "center",
  "right-half",
];
const getCropPriority = (name) => {
  const index = CROP_PRIORITY.indexOf(name);
  return index === -1 ? CROP_PRIORITY.length : index;
};

export class TesseractOCR {
  constructor() {
    this.workers = {};
    this.ready = false;
    this.progressCallback = null;
    this.progressContext = null;
  }

  async initialize(onProgress) {
    this.progressCallback = onProgress;
    if (this.ready) return;

    // Persian trained data handles the Persian/Arabic-Indic numerals used on
    // Iranian cards. A single worker avoids loading the same language three times.
    this.workers.persian = await createWorker("fas", LSTM_ONLY, {
      logger: (message) => {
        if (message.status === "recognizing text") {
          const context = this.progressContext;
          const progress = context
            ? (context.completed + message.progress) / context.total
            : message.progress;
          this.progressCallback?.(Math.round(progress * 100));
        }
      },
    });

    this.ready = true;
  }

  async recognize(imageSrc, onProgress, onAttempt) {
    if (!this.ready) await this.initialize(onProgress);
    this.progressCallback = onProgress;
    onProgress?.(0);

    const allAttempts = [];
    const allDates = [];
    let attemptCount = 0;

    // Start with cropped candidate regions; do not OCR the whole card as a block.
    // The crop helper covers common Iranian card layouts, and the narrower strips
    // are tried first because they are more likely to contain a single text line.
    const candidateRegions = (await cropBirthDateRegions(imageSrc))
      .filter((region) => region.name !== "full")
      .sort((a, b) => getCropPriority(a.name) - getCropPriority(b.name));
    const primaryAttempts =
      candidateRegions.length *
      DATE_PREPROCESSING_STRATEGIES.length *
      PAGE_SEGMENT_MODES.length;
    const fallbackAttempts = 8 * PAGE_SEGMENT_MODES.length;
    this.progressContext = {
      completed: 0,
      total: Math.max(1, primaryAttempts + fallbackAttempts),
    };

    for (const region of candidateRegions) {
      for (const strategyName of DATE_PREPROCESSING_STRATEGIES) {
        let dataUrl;
        try {
          // simpleEnhance supplies a grayscale, enlarged, contrast-adjusted view;
          // binarize provides a second view without assuming thresholding is better.
          const processed = await PREPROCESS_STRATEGIES[strategyName](
            region.dataUrl,
          );
          dataUrl = processed.toDataURL("image/png");
        } catch (error) {
          console.warn(
            "Birth-date crop preprocessing failed:",
            region.name,
            strategyName,
            error,
          );
          continue;
        }

        for (const mode of PAGE_SEGMENT_MODES) {
          attemptCount++;
          try {
            const attempt = await this.runOCR(dataUrl, mode.value, {
              strategy: strategyName,
              region: region.name,
              psm: String(mode.value),
              pageSegmentation: mode.name,
              engine: "fas-digit-line",
            });

            allAttempts.push(attempt);
            allDates.push(...attempt.dates);
            onAttempt?.(attemptCount, attempt);
          } catch (error) {
            console.warn(
              "Birth-date OCR attempt failed:",
              region.name,
              strategyName,
              mode.name,
              error,
            );
          }
        }
      }
    }

    // Last-resort horizontal line crops help when the card layout is unfamiliar.
    // They are only needed when the candidate birth-date crops did not yield a
    // structurally valid Jalali date.
    if (allDates.length === 0) {
      try {
        const grayscale = await PREPROCESS_STRATEGIES.simpleEnhance(imageSrc);
        const lineResults = await this.scanByLines(grayscale);
        for (const lineResult of lineResults) {
          attemptCount++;
          allAttempts.push(lineResult);
          allDates.push(...lineResult.dates);
          onAttempt?.(attemptCount, lineResult);
        }
      } catch (error) {
        console.warn("Birth-date line scanning failed:", error);
      }
    }

    this.progressContext = null;
    onProgress?.(100);
    return this.buildFinalResult(allAttempts, allDates);
  }

  async runOCR(dataUrl, psm, meta) {
    const worker = this.workers.persian;
    if (!worker) {
      throw new Error("The Persian Tesseract worker has not been initialized.");
    }

    const progressContext = this.progressContext;
    try {
      await worker.setParameters({
        tessedit_pageseg_mode: String(psm),
        tessedit_char_whitelist: DATE_CHAR_WHITELIST,
        user_defined_dpi: "300",
        load_system_dawg: "0",
        load_freq_dawg: "0",
        preserve_interword_spaces: "1",
      });

      const { data } = await worker.recognize(dataUrl);
      const normalized = normalizeBirthDateText(data.text);
      const dates = extractBirthDates(data.text);

      return {
        ...meta,
        worker: "fas",
        language: "fas",
        rawText: data.text,
        normalizedText: normalized,
        confidence: data.confidence,
        dates,
        preprocessedImage: dataUrl,
      };
    } finally {
      if (progressContext && this.progressContext === progressContext) {
        progressContext.completed = Math.min(
          progressContext.completed + 1,
          progressContext.total,
        );
        this.progressCallback?.(
          Math.round(
            (progressContext.completed / progressContext.total) * 100,
          ),
        );
      }
    }
  }

  /**
   * Scan overlapping horizontal crops as a final fallback for unfamiliar card
   * layouts. Each crop is treated as one line by Tesseract.
   */
  async scanByLines(canvas) {
    const results = [];
    const image = canvas;
    const bandCount = 8;
    const bandHeight = Math.max(1, Math.min(image.height, Math.round(image.height / 4)));
    const maxY = Math.max(0, image.height - bandHeight);

    for (let i = 0; i < bandCount; i++) {
      const sliceCanvas = document.createElement("canvas");
      sliceCanvas.width = image.width;
      sliceCanvas.height = bandHeight;

      const context = sliceCanvas.getContext("2d");
      if (!context) continue;

      context.fillStyle = "#fff";
      context.fillRect(0, 0, sliceCanvas.width, sliceCanvas.height);

      const sourceY = Math.round((maxY * i) / (bandCount - 1));
      context.drawImage(
        image,
        0,
        sourceY,
        image.width,
        bandHeight,
        0,
        0,
        image.width,
        bandHeight,
      );

      const dataUrl = sliceCanvas.toDataURL("image/png");
      for (const mode of PAGE_SEGMENT_MODES) {
        try {
          const attempt = await this.runOCR(dataUrl, mode.value, {
            strategy: "line-scan-grayscale",
            region: `line-${i}`,
            psm: String(mode.value),
            pageSegmentation: mode.name,
            engine: "fas-line-scan",
          });
          results.push(attempt);
        } catch (error) {
          console.warn("Line scan failed for slice", i, mode.name, error);
        }
      }
    }

    return results;
  }

  /**
   * Combine all OCR attempts and pick the best date using validation and voting.
   */
  buildFinalResult(attempts, allDates) {
    // Repeated valid dates across distinct crops and OCR modes receive more votes.
    const voteMap = {};

    for (const date of allDates) {
      const key = date.formatted;
      if (!voteMap[key]) {
        voteMap[key] = {
          ...date,
          votes: 0,
          totalScore: 0,
          foundIn: [],
        };
      }
      voteMap[key].votes++;
      voteMap[key].totalScore += date.score;
    }

    // Attach source information to each voted date.
    for (const attempt of attempts) {
      for (const date of attempt.dates) {
        const entry = voteMap[date.formatted];
        if (entry && entry.foundIn.length < 10) {
          entry.foundIn.push({
            strategy: attempt.strategy,
            region: attempt.region,
            engine: attempt.engine || attempt.worker,
            confidence: attempt.confidence,
          });
        }
      }
    }

    const rankedDates = Object.values(voteMap)
      .map((date) => ({
        ...date,
        finalScore: date.totalScore + date.votes * 50,
      }))
      .sort((a, b) => b.finalScore - a.finalScore);

    const bestDate = rankedDates[0];
    const matchingAttempts = bestDate
      ? attempts
          .filter((attempt) =>
            attempt.dates.some((date) => date.formatted === bestDate.formatted),
          )
          .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
      : [];

    const bestAttempt = bestDate
      ? matchingAttempts[0] || null
      : attempts.reduce(
          (best, attempt) =>
            (best?.confidence || 0) > (attempt?.confidence || 0)
              ? best
              : attempt,
          null,
        );

    return {
      best: bestAttempt
        ? {
            ...bestAttempt,
            birthDate: bestDate || null,
          }
        : null,
      allDates: rankedDates,
      allAttempts: attempts,
      stats: {
        totalAttempts: attempts.length,
        uniqueDatesFound: rankedDates.length,
        bestDateVotes: bestDate?.votes || 0,
      },
    };
  }

  async terminate() {
    for (const worker of new Set(Object.values(this.workers))) {
      try {
        await worker.terminate();
      } catch {
        // Ignore workers that are already terminated.
      }
    }
    this.workers = {};
    this.ready = false;
    this.progressCallback = null;
    this.progressContext = null;
  }
}
