// src/hooks/useOCR.js
//
// React adapter for TesseractOCR (OpenCV + dependency-free CNN, no Tesseract).
// Engine result contract:
//   { best: attempt+{birthDate} | null, allDates, allAttempts, timingMs }
//   birthDate: { year, month, day, formatted, raw, confidence, score,
//                corrected:false, correctionCost:0, votes, minProb, dateProb }
// best === null means "retake photo" — never shown as success.
// Multi-frame voting (spec 8.5) via recognizeConsensus: a date is accepted
// only when >= requiredVotes frames agree.

import { useCallback, useEffect, useRef, useState } from "react";
import { TesseractOCR } from "../ocr/TesseractOCR";
import { ConsensusReader } from "../ocr/consensus.js";

function toUiResult(engineResult) {
  const allAttempts = engineResult.allAttempts || [];
  const allDates = (engineResult.allDates || []).map((d) => ({
    ...d,
    finalScore: d.score ?? d.confidence,
  }));
  const rawBest = engineResult.best;
  const best = rawBest
    ? {
        ...rawBest,
        birthDate: {
          ...rawBest.birthDate,
          finalScore: rawBest.birthDate.score ?? rawBest.birthDate.confidence,
        },
        durationMs: engineResult.timingMs,
      }
    : null;
  return {
    best,
    allDates,
    allAttempts,
    lineImage:
      rawBest?.segmentImages?.line || rawBest?.preprocessedImage || null,
    error: best ? null : "Birth date not found. Please retake the photo.",
    reason: engineResult.reason ?? null,
    quality: engineResult.quality ?? null,
    hints: engineResult.hints ?? [],
    telemetry: engineResult.telemetry ?? null,
    fields: engineResult.fields ?? null,
    engineResult,
  };
}

export function useOCR() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [currentAttempt, setCurrentAttempt] = useState(null);
  const engineRef = useRef(null);

  useEffect(() => {
    engineRef.current = new TesseractOCR();
    return () => engineRef.current?.terminate();
  }, []);

  const recognize = useCallback(async (imageSrc) => {
    setIsProcessing(true);
    setProgress(0);
    setResult(null);
    setError(null);
    setAttempts([]);
    setCurrentAttempt(null);

    try {
      if (!engineRef.current) engineRef.current = new TesseractOCR();
      const raw = await engineRef.current.recognize(
        imageSrc,
        (p) => setProgress(Math.max(0, Math.min(100, Math.round(p)))),
        (_n, attempt) => setCurrentAttempt(attempt),
      );
      const mapped = toUiResult(raw);
      setAttempts(mapped.allAttempts);
      setResult(mapped);
      if (mapped.error) setError(mapped.error);
      return mapped;
    } catch (err) {
      // The engine never throws past recognize(); this is a safety net.
      console.error("OCR error:", err);
      setError(err?.message || String(err));
    } finally {
      setIsProcessing(false);
      setCurrentAttempt(null);
    }
  }, []);

  // Live-camera helper: recognize several frames, accept a date only when
  // >= requiredVotes frames agree (spec 8.5). Returns { mapped, agreed }.
  const recognizeConsensus = useCallback(
    async (imageSources, { requiredVotes = 2 } = {}) => {
      setIsProcessing(true);
      setProgress(0);
      setResult(null);
      setError(null);
      setAttempts([]);
      try {
        if (!engineRef.current) engineRef.current = new TesseractOCR();
        const reader = new ConsensusReader({ requiredVotes });
        let mapped = null;
        for (let i = 0; i < imageSources.length; i++) {
          const raw = await engineRef.current.recognize(
            imageSources[i],
            (p) =>
              setProgress(
                Math.max(
                  0,
                  Math.min(
                    100,
                    Math.round(((i + p / 100) / imageSources.length) * 100),
                  ),
                ),
              ),
          );
          mapped = toUiResult(raw);
          reader.addFrameResult(mapped);
          if (reader.status().agreed) break;
        }
        const { agreed } = reader.status();
        const finalMapped = agreed
          ? { ...mapped, best: { ...mapped.best, birthDate: agreed.birthDate } }
          : mapped;
        setAttempts(finalMapped?.allAttempts ?? []);
        setResult(finalMapped);
        if (!agreed) setError("Frames disagree. Please hold still and retry.");
        return { mapped: finalMapped, agreed };
      } catch (err) {
        console.error("OCR error:", err);
        setError(err?.message || String(err));
        return { mapped: null, agreed: null };
      } finally {
        setIsProcessing(false);
        setCurrentAttempt(null);
      }
    },
    [],
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
    setAttempts([]);
    setCurrentAttempt(null);
  }, []);

  return {
    recognize,
    recognizeConsensus,
    isProcessing,
    progress,
    result,
    error,
    reset,
    attempts,
    currentAttempt,
  };
}
