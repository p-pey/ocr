// src/hooks/useOCR.js
//
// React adapter for IranCardOCR (OpenCV geometry + Tesseract fas).
// The engine's result contract is:
//   success → { success, birthDate, year, month, day, confidence, repairs,
//               durationMs, ocrCalls, attempts, lineImage }
//   failure → { success: false, error, durationMs, attempts }
// ResultDisplay consumes the legacy { best, allDates, allAttempts } shape,
// so results are mapped here once.

import { useCallback, useEffect, useRef, useState } from "react";
import { IranCardOCR } from "../ocr/iranCardOCR";

function toUiResult(engineResult) {
  const attempts = engineResult.attempts || [];
  const ok = Boolean(engineResult.success);
  const date = ok
    ? {
        year: engineResult.year,
        month: engineResult.month,
        day: engineResult.day,
        formatted: engineResult.birthDate,
        votes: Math.max(
          1,
          attempts.filter((a) => a.rawText === engineResult.birthDate).length,
        ),
        finalScore: engineResult.confidence,
        score: engineResult.confidence,
      }
    : null;

  return {
    best: ok
      ? {
          birthDate: date,
          confidence: engineResult.confidence,
          engine: "opencv+tesseract-fas",
          repairs: engineResult.repairs,
          durationMs: engineResult.durationMs,
          ocrCalls: engineResult.ocrCalls,
        }
      : null,
    allDates: date ? [date] : [],
    allAttempts: attempts,
    lineImage: engineResult.lineImage || null,
    error: ok ? null : engineResult.error,
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
    engineRef.current = new IranCardOCR();
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
      if (!engineRef.current) engineRef.current = new IranCardOCR();
      const raw = await engineRef.current.recognizeBirthDate(
        imageSrc,
        (p) => setProgress(Math.max(0, Math.min(100, Math.round(p)))),
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

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(0);
    setAttempts([]);
    setCurrentAttempt(null);
  }, []);

  return {
    recognize,
    isProcessing,
    progress,
    result,
    error,
    reset,
    attempts,
    currentAttempt,
  };
}
