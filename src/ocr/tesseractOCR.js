// src/ocr/tesseractOCR.js

import Tesseract, { createWorker } from "tesseract.js";
import {
  PREPROCESS_STRATEGIES,
  cropBirthDateRegions,
} from "../utils/imageUtils";
import { extractAllDates, persianToEnglish } from "../utils/persianUtils";

export class TesseractOCR {
  constructor() {
    this.workers = {};
    this.ready = false;
  }

  async initialize(onProgress) {
    if (this.ready) return;

    const progressTracker = {
      stages: { persian: 0, digits: 0, latin: 0 },
      update: (stage, p) => {
        this.stages = this.stages || progressTracker.stages;
        progressTracker.stages[stage] = p;
        const total =
          (progressTracker.stages.persian +
            progressTracker.stages.digits +
            progressTracker.stages.latin) /
          3;
        onProgress?.(Math.round(total));
      },
    };

    // Worker 1: Full Persian + Arabic + English for context
    this.workers.full = await createWorker("fas", 1, {
      logger: (m) =>
        m.status === "recognizing text" &&
        progressTracker.update("persian", m.progress * 100),
    });

    // Worker 2: Digit-focused (English + numbers only - more accurate for digits)
    this.workers.digits = await createWorker("fas", 1, {
      logger: (m) =>
        m.status === "recognizing text" &&
        progressTracker.update("digits", m.progress * 100),
    });

    // Worker 3: Persian only (sometimes better for Persian digits)
    this.workers.persian = await createWorker("fas", 1, {
      logger: (m) =>
        m.status === "recognizing text" &&
        progressTracker.update("latin", m.progress * 100),
    });

    this.ready = true;
  }

  async recognize(imageSrc, onProgress, onAttempt) {
    if (!this.ready) await this.initialize(onProgress);

    const allAttempts = [];
    const allDates = [];
    let attemptCount = 0;

    // Preprocess strategies to try
    const strategies = [
      "sharpen",
      "adaptiveThreshold",
      "binarize",
      "simpleEnhance",
      "redChannel",
      "boldText",
      "mild",
      "inverted",
    ];

    // Crop regions to try
    const regions = await cropBirthDateRegions(imageSrc);

    // ─────────────────────────────────────────────────
    // PASS 1: Full image with multiple preprocessing
    // ─────────────────────────────────────────────────
    for (const strategyName of strategies.slice(0, 4)) {
      try {
        const processed = await PREPROCESS_STRATEGIES[strategyName](imageSrc);
        const dataUrl = processed.toDataURL("image/png");

        // Try with multiple PSM modes
        for (const psm of ["6", "11", "4"]) {
          attemptCount++;
          const attempt = await this.runOCR("full", dataUrl, psm, {
            strategy: strategyName,
            region: "full-image",
            psm,
          });

          allAttempts.push(attempt);
          allDates.push(...attempt.dates);
          onAttempt?.(attemptCount, attempt);

          // Early exit if we found high-confidence date
          if (
            attempt.dates.length > 0 &&
            attempt.dates[0].score >= 110 &&
            attempt.confidence > 70
          ) {
            onProgress?.(100);
            return this.buildFinalResult(allAttempts, allDates);
          }
        }
      } catch (err) {
        console.warn("Full image strategy failed:", strategyName, err);
      }
    }

    // ─────────────────────────────────────────────────
    // PASS 2: Region cropping with best strategies
    // ─────────────────────────────────────────────────
    for (const region of regions.filter((r) => r.name !== "full")) {
      for (const strategyName of ["sharpen", "adaptiveThreshold", "binarize"]) {
        try {
          const processed = await PREPROCESS_STRATEGIES[strategyName](
            region.dataUrl,
          );
          const dataUrl = processed.toDataURL("image/png");

          // Try digits-only worker for cropped regions (more reliable for dates)
          attemptCount++;
          const attempt = await this.runOCR("digits", dataUrl, "7", {
            strategy: strategyName,
            region: region.name,
            psm: "7",
            engine: "digits-only",
          });

          allAttempts.push(attempt);
          allDates.push(...attempt.dates);
          onAttempt?.(attemptCount, attempt);

          // Try persian-only worker too
          attemptCount++;
          const attempt2 = await this.runOCR("persian", dataUrl, "6", {
            strategy: strategyName,
            region: region.name,
            psm: "6",
            engine: "persian-only",
          });

          allAttempts.push(attempt2);
          allDates.push(...attempt2.dates);
          onAttempt?.(attemptCount, attempt2);

          if (attempt.dates.length > 0 && attempt.dates[0].score >= 110) {
            onProgress?.(100);
            return this.buildFinalResult(allAttempts, allDates);
          }
        } catch (err) {
          console.warn(
            "Region strategy failed:",
            region.name,
            strategyName,
            err,
          );
        }
      }
    }

    // ─────────────────────────────────────────────────
    // PASS 3: Line-by-line scanning (last resort)
    // ─────────────────────────────────────────────────
    try {
      const processed = await PREPROCESS_STRATEGIES.sharpen(imageSrc);
      const lineResults = await this.scanByLines(processed);
      for (const lineResult of lineResults) {
        allAttempts.push(lineResult);
        allDates.push(...lineResult.dates);
      }
    } catch (err) {
      console.warn("Line scanning failed:", err);
    }

    onProgress?.(100);
    return this.buildFinalResult(allAttempts, allDates);
  }

  async runOCR(workerName, dataUrl, psm, meta) {
    const worker = (this.workers = this.workers.full);

    // Set parameters based on worker type
    const params = {
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE,
      preserve_interword_spaces: "1",
      user_defined_dpi: "300",
    };

    if (workerName === "digits") {
      params.tessedit_char_whitelist = "۰۱۲۳۴۵۶۷۸۹/";
    }

    await worker.setParameters(params);

    const { data } = await worker.recognize(dataUrl);
    const normalized = persianToEnglish(data.text);
    const dates = extractAllDates(data.text);

    return {
      ...meta,
      worker: workerName,
      rawText: data.text,
      normalizedText: normalized,
      confidence: data.confidence,
      dates,
      preprocessedImage: dataUrl,
    };
  }

  /**
   * Scan the image line-by-line to find date patterns.
   * Useful when the whole image OCR fails.
   */
  async scanByLines(canvas) {
    const results = [];
    const img = canvas;
    const h = img.height;
    const sliceHeight = Math.round(h / 8);

    for (let i = 0; i < 8; i++) {
      const sliceCanvas = document.createElement("canvas");
      sliceCanvas.width = img.width;
      sliceCanvas.height = sliceHeight * 2; // Overlap slices

      const ctx = sliceCanvas.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, sliceCanvas.width, sliceCanvas.height);

      const srcY = Math.max(0, i * sliceHeight - sliceHeight / 2);
      ctx.drawImage(
        img,
        0,
        srcY,
        img.width,
        sliceHeight * 2,
        0,
        0,
        img.width,
        sliceHeight * 2,
      );

      try {
        const dataUrl = sliceCanvas.toDataURL("image/png");
        const attempt = await this.runOCR("digits", dataUrl, "7", {
          strategy: "line-scan",
          region: `line-${i}`,
          psm: "7",
          engine: "line-scan",
        });
        results.push(attempt);
      } catch (err) {
        console.warn("Line scan failed for slice", i);
      }
    }

    return results;
  }

  /**
   * Combine all OCR attempts and pick the best date using voting.
   */
  buildFinalResult(attempts, allDates) {
    // Vote on dates — same date found by multiple strategies = more reliable
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

    // Attach source info to each voted date
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

    // Compute final ranking
    const rankedDates = Object.values(voteMap)
      .map((d) => ({
        ...d,
        finalScore: d.totalScore + d.votes * 50,
      }))
      .sort((a, b) => b.finalScore - a.finalScore);

    const bestDate = rankedDates[0];

    // Build best attempt object
    const bestAttempt = bestDate
      ? attempts.find((a) =>
          a.dates.some((d) => d.formatted === bestDate.formatted),
        )
      : attempts.reduce(
          (a, b) => ((a?.confidence || 0) > (b?.confidence || 0) ? a : b),
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
    for (const w of Object.values(this.workers)) {
      try {
        await w.terminate();
      } catch {}
    }
    this.workers = {};
    this.ready = false;
  }
}
