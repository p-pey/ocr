// src/hooks/useImageEditor.js

import { useCallback, useState } from "react";
import { getCroppedImg } from "../utils/imageUtils";

export function useImageEditor() {
  const [originalImage, setOriginalImage] = useState(null);
  const [croppedImage, setCroppedImage] = useState(null);
  const [rotation, setRotation] = useState(0);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [croppedAreaPixels, setCroppedAreaPixels] = useState(null);
  const [isEditing, setIsEditing] = useState(false);

  const loadImage = useCallback((src) => {
    setOriginalImage(src);
    setCroppedImage(null);
    setRotation(0);
    setCrop({ x: 0, y: 0 });
    setZoom(1);
    setIsEditing(true);
  }, []);

  const handleCropComplete = useCallback((_, croppedPixels) => {
    setCroppedAreaPixels(croppedPixels);
  }, []);

  const rotateLeft = useCallback(() => {
    setRotation((prev) => (prev - 90) % 360);
  }, []);

  const rotateRight = useCallback(() => {
    setRotation((prev) => (prev + 90) % 360);
  }, []);

  const rotateCustom = useCallback((deg) => {
    setRotation((prev) => prev + deg);
  }, []);

  const applyCrop = useCallback(async () => {
    if (!originalImage || !croppedAreaPixels) return null;

    try {
      const result = await getCroppedImg(
        originalImage,
        croppedAreaPixels,
        rotation,
      );
      setCroppedImage(result.dataUrl);
      setIsEditing(false);
      return result;
    } catch (err) {
      console.error("Crop failed:", err);
      return null;
    }
  }, [originalImage, croppedAreaPixels, rotation]);

  const resetEditor = useCallback(() => {
    setOriginalImage(null);
    setCroppedImage(null);
    setRotation(0);
    setCrop({ x: 0, y: 0 });
    setZoom(1);
    setCroppedAreaPixels(null);
    setIsEditing(false);
  }, []);

  const reEdit = useCallback(() => {
    setCroppedImage(null);
    setIsEditing(true);
  }, []);

  return {
    originalImage,
    croppedImage,
    rotation,
    crop,
    zoom,
    isEditing,
    setCrop,
    setZoom,
    loadImage,
    handleCropComplete,
    rotateLeft,
    rotateRight,
    rotateCustom,
    applyCrop,
    resetEditor,
    reEdit,
  };
}
