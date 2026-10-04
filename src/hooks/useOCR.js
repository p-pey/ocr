// src/hooks/useOCR.js

import { useCallback, useEffect, useRef, useState } from "react";
import { TesseractOCR } from "../ocr/tesseractOCR";

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
      const res = await engineRef.current.recognize(
        imageSrc,
        setProgress,
        (attemptNum, attempt) => {
          setCurrentAttempt({ number: attemptNum, ...attempt });
          setAttempts((prev) => [...prev, attempt]);
        },
      );
      setResult(res);
      return res;
    } catch (err) {
      console.error("OCR error:", err);
      setError(err.message);
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
