// src/components/ImageEditor.jsx

import { useCallback, useRef, useState } from 'react';
import Cropper from 'react-easy-crop';
import { CardFrameOverlay } from './CardFrameOverlay';

// Iranian National ID Card aspect ratio = 85.6mm × 53.98mm ≈ 1.586
const ID_CARD_ASPECT = 85.6 / 53.98;

export function ImageEditor({
  image,
  crop,
  zoom,
  rotation,
  onCropChange,
  onZoomChange,
  onCropComplete,
  onRotateLeft,
  onRotateRight,
  onRotateCustom,
  onApply,
  onCancel,
}) {
  const [fineRotation, setFineRotation] = useState(0);
  const [showGrid, setShowGrid] = useState(true);
  const [cropSize, setCropSize] = useState(null);
  const containerRef = useRef(null);

  const handleFineRotation = useCallback((value) => {
    const deg = parseFloat(value);
    setFineRotation(deg);
    onRotateCustom(deg - fineRotation);
  }, [fineRotation, onRotateCustom]);

  const handleMediaLoaded = useCallback((mediaSize) => {
    // Auto-zoom to fit card frame nicely
    const idealZoom = Math.max(
      1,
      Math.min(
        mediaSize.naturalWidth / (mediaSize.width * 0.8),
        2
      )
    );
    // onZoomChange(idealZoom);
  }, []);

  return (
    <div style={styles.container} dir="rtl">
      <div style={styles.header}>
        <h2 style={styles.title}>✂️ برش و تنظیم تصویر</h2>
        <p style={styles.subtitle}>
          کارت ملی را داخل کادر سبز قرار دهید. می‌توانید تصویر را بکشید، زوم و چرخش دهید.
        </p>
      </div>

      {/* Crop Area */}
      <div style={styles.cropContainer} ref={containerRef}>
        <Cropper
          image={image}
          crop={crop}
          zoom={zoom}
          rotation={rotation}
          aspect={ID_CARD_ASPECT}
          onCropChange={onCropChange}
          onZoomChange={onZoomChange}
          onCropComplete={onCropComplete}
          onMediaLoaded={handleMediaLoaded}
          cropShape="rect"
          showGrid={showGrid}
          style={{
            containerStyle: styles.cropperContainer,
            cropAreaStyle: styles.cropArea,
            mediaStyle: {},
          }}
          classes={{
            containerClassName: 'cropper-container',
            cropAreaClassName: 'crop-area',
          }}
          objectFit="contain"
          restrictPosition={false}
          minZoom={0.5}
          maxZoom={5}
          zoomSpeed={0.3}
        />

        {/* Card Frame Overlay on top of crop area */}
        <CardFrameOverlay
          containerWidth={cropSize?.width}
          containerHeight={cropSize?.height}
        />
      </div>

      {/* Controls */}
      <div style={styles.controls}>

        {/* Zoom Control */}
        <div style={styles.controlGroup}>
          <label style={styles.controlLabel}>🔍 زوم</label>
          <div style={styles.sliderRow}>
            <span style={styles.sliderValue}>{zoom.toFixed(1)}x</span>
            <input
              type="range"
              min="0.5"
              max="5"
              step="0.1"
              value={zoom}
              onChange={(e) => onZoomChange(parseFloat(e.target.value))}
              style={styles.slider}
            />
            <div style={styles.sliderButtons}>
              <button
                onClick={() => onZoomChange(Math.max(0.5, zoom - 0.1))}
                style={styles.smallBtn}
              >
                −
              </button>
              <button
                onClick={() => onZoomChange(Math.min(5, zoom + 0.1))}
                style={styles.smallBtn}
              >
                +
              </button>
            </div>
          </div>
        </div>

        {/* Rotation Controls */}
        <div style={styles.controlGroup}>
          <label style={styles.controlLabel}>🔄 چرخش</label>

          {/* Quick rotation buttons */}
          <div style={styles.rotateButtons}>
            <button onClick={onRotateLeft} style={styles.rotateBtn}>
              ↺ ۹۰° چپ
            </button>
            <button
              onClick={() => {
                setFineRotation(0);
                onRotateCustom(-rotation);
              }}
              style={styles.rotateBtnReset}
            >
              ↻ ریست
            </button>
            <button onClick={onRotateRight} style={styles.rotateBtn}>
              ↻ ۹۰° راست
            </button>
          </div>

          {/* Fine rotation slider */}
          <div style={styles.sliderRow}>
            <span style={styles.sliderValue}>{rotation.toFixed(1)}°</span>
            <input
              type="range"
              min="-180"
              max="180"
              step="0.5"
              value={rotation}
              onChange={(e) => {
                const newRotation = parseFloat(e.target.value);
                onRotateCustom(newRotation - rotation);
              }}
              style={styles.slider}
            />
          </div>

          {/* Fine adjustment buttons */}
          <div style={styles.fineRotateRow}>
            <button onClick={() => onRotateCustom(-1)} style={styles.fineBtn}>-1°</button>
            <button onClick={() => onRotateCustom(-0.5)} style={styles.fineBtn}>-0.5°</button>
            <button onClick={() => onRotateCustom(-0.1)} style={styles.fineBtn}>-0.1°</button>
            <button onClick={() => onRotateCustom(0.1)} style={styles.fineBtn}>+0.1°</button>
            <button onClick={() => onRotateCustom(0.5)} style={styles.fineBtn}>+0.5°</button>
            <button onClick={() => onRotateCustom(1)} style={styles.fineBtn}>+1°</button>
          </div>
        </div>

        {/* Grid Toggle */}
        <div style={styles.controlGroup}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
            />
            نمایش خطوط راهنما
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div style={styles.actions}>
        <button onClick={onCancel} style={styles.cancelBtn}>
          ❌ انصراف
        </button>
        <button onClick={onApply} style={styles.applyBtn}>
          ✅ تایید و ادامه
        </button>
      </div>

      {/* Tips */}
      <div style={styles.tips}>
        <h4 style={styles.tipsTitle}>💡 راهنمایی:</h4>
        <ul style={styles.tipsList}>
          <li>تصویر را با کشیدن (drag) جابجا کنید</li>
          <li>با اسکرول موس یا لغزنده زوم کنید</li>
          <li>اگر تصویر کج است، از چرخش دقیق استفاده کنید</li>
          <li>کارت ملی باید کامل داخل کادر سبز باشد</li>
        </ul>
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: '#1a1a2e',
    borderRadius: 16,
    overflow: 'hidden',
    border: '1px solid #333',
  },
  header: {
    padding: '20px 24px 12px',
    textAlign: 'center',
  },
  title: {
    color: '#fff',
    fontSize: 20,
    margin: 0,
  },
  subtitle: {
    color: '#aaa',
    fontSize: 13,
    margin: '8px 0 0',
  },
  cropContainer: {
    position: 'relative',
    width: '100%',
    height: 400,
    background: '#0f0f23',
  },
  cropperContainer: {
    borderRadius: 0,
  },
  cropArea: {
    border: '2px solid #00e676',
    boxShadow: '0 0 0 9999px rgba(0,0,0,0.6)',
    borderRadius: 8,
  },
  controls: {
    padding: '16px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  controlGroup: {
    background: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 14,
  },
  controlLabel: {
    color: '#ccc',
    fontSize: 13,
    fontWeight: 'bold',
    display: 'block',
    marginBottom: 10,
    fontFamily: 'Tahoma, sans-serif',
  },
  sliderRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  slider: {
    flex: 1,
    accentColor: '#00e676',
    height: 6,
    cursor: 'pointer',
  },
  sliderValue: {
    color: '#00e676',
    fontSize: 13,
    fontWeight: 'bold',
    minWidth: 50,
    textAlign: 'center',
    fontFamily: 'monospace',
    direction: 'ltr',
  },
  sliderButtons: {
    display: 'flex',
    gap: 4,
  },
  smallBtn: {
    width: 30,
    height: 30,
    border: '1px solid #555',
    borderRadius: 6,
    background: 'transparent',
    color: '#fff',
    fontSize: 18,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  rotateButtons: {
    display: 'flex',
    gap: 8,
    marginBottom: 12,
  },
  rotateBtn: {
    flex: 1,
    padding: '8px 12px',
    border: '1px solid #555',
    borderRadius: 8,
    background: 'rgba(255,255,255,0.05)',
    color: '#ccc',
    fontSize: 12,
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
    transition: 'all 0.2s',
  },
  rotateBtnReset: {
    flex: 1,
    padding: '8px 12px',
    border: '1px solid #e74c3c',
    borderRadius: 8,
    background: 'rgba(231,76,60,0.1)',
    color: '#e74c3c',
    fontSize: 12,
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
  },
  fineRotateRow: {
    display: 'flex',
    gap: 6,
    marginTop: 10,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  fineBtn: {
    padding: '5px 10px',
    border: '1px solid #444',
    borderRadius: 6,
    background: 'rgba(255,255,255,0.05)',
    color: '#aaa',
    fontSize: 11,
    cursor: 'pointer',
    fontFamily: 'monospace',
    direction: 'ltr',
  },
  checkboxLabel: {
    color: '#aaa',
    fontSize: 13,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
  },
  actions: {
    display: 'flex',
    gap: 12,
    padding: '0 24px 20px',
  },
  cancelBtn: {
    flex: 1,
    padding: 14,
    border: '1px solid #555',
    borderRadius: 10,
    background: 'transparent',
    color: '#ccc',
    fontSize: 15,
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
  },
  applyBtn: {
    flex: 2,
    padding: 14,
    border: 'none',
    borderRadius: 10,
    background: 'linear-gradient(135deg, #00e676, #00bcd4)',
    color: '#000',
    fontSize: 15,
    fontWeight: 'bold',
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
  },
  tips: {
    padding: '0 24px 20px',
    borderTop: '1px solid rgba(255,255,255,0.05)',
    marginTop: 4,
    paddingTop: 16,
  },
  tipsTitle: {
    color: '#888',
    fontSize: 13,
    margin: '0 0 8px',
  },
  tipsList: {
    color: '#666',
    fontSize: 12,
    lineHeight: 2,
    paddingRight: 20,
    margin: 0,
  },
};

export default ImageEditor;