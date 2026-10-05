// src/components/App.jsx

import { useCallback, useState } from 'react';
import { useImageEditor } from '../hooks/useImageEditor';
import { useOCR } from '../hooks/useOCR';
import { CameraCapture } from './CameraCapture';
import { ImageEditor } from './ImageEditor';
import { ImageUploader } from './ImageUploader';
import { ProgressBar } from './ProgressBar';
import { ResultDisplay } from './ResultDisplay';

export default function App() {
  const [showCamera, setShowCamera] = useState(false);
  const [step, setStep] = useState('upload'); // upload | edit | result

  const editor = useImageEditor();
  const ocr = useOCR();

  // ── Step 1: User provides image ────────────────────

  const handleImageSelected = useCallback((dataUrl) => {
    editor.loadImage(dataUrl);
    setStep('edit');
  }, [editor]);

  const handleCameraCapture = useCallback((blob) => {
    setShowCamera(false);
    const reader = new FileReader();
    reader.onload = (e) => {
      editor.loadImage(e.target.result);
      setStep('edit');
    };
    reader.readAsDataURL(blob);
  }, [editor]);

  // ── Step 2: User crops & rotates ───────────────────

  const handleApplyCrop = useCallback(async () => {
    const result = await editor.applyCrop();
    if (result) {
      setStep('result');
      // Automatically start OCR on the cropped image
      ocr.recognize(result.dataUrl);
    }
  }, [editor, ocr]);

  const handleCancelEdit = useCallback(() => {
    editor.resetEditor();
    ocr.reset();
    setStep('upload');
  }, [editor, ocr]);

  // ── Step 3: Show result ────────────────────────────

  const handleRetry = useCallback(() => {
    if (editor.originalImage) {
      editor.reEdit();
      ocr.reset();
      setStep('edit');
    } else {
      editor.resetEditor();
      ocr.reset();
      setStep('upload');
    }
  }, [editor, ocr]);

  const handleStartOver = useCallback(() => {
    editor.resetEditor();
    ocr.reset();
    setStep('upload');
  }, [editor, ocr]);

  return (
    <div style={styles.page} dir="rtl">
      <div style={styles.card}>
        {/* Header */}
        <div style={styles.header}>
          <h1 style={styles.title}>🪪 خوانش کارت ملی</h1>
          <p style={styles.subtitle}>
            استخراج خودکار تاریخ تولد از تصویر کارت ملی
          </p>
        </div>

        {/* Steps indicator */}
        <div style={styles.steps}>
          {['بارگذاری تصویر', 'برش و تنظیم', 'نتیجه'].map((label, i) => {
            const stepNames = ['upload', 'edit', 'result'];
            const currentIdx = stepNames.indexOf(step);
            const isActive = i === currentIdx;
            const isDone = i < currentIdx;
            return (
              <div key={i} style={styles.stepItem}>
                <div style={{
                  ...styles.stepCircle,
                  ...(isActive ? styles.stepActive : {}),
                  ...(isDone ? styles.stepDone : {}),
                }}>
                  {isDone ? '✓' : i + 1}
                </div>
                <span style={{
                  ...styles.stepLabel,
                  color: isActive ? '#667eea' : isDone ? '#27ae60' : '#bbb',
                }}>
                  {label}
                </span>
                {i < 2 && <div style={{
                  ...styles.stepLine,
                  background: isDone ? '#27ae60' : '#eee',
                }} />}
              </div>
            );
          })}
        </div>

        {/* ── STEP: Upload ── */}
        {step === 'upload' && (
          <div style={styles.section}>
            <ImageUploader
              onImageSelected={handleImageSelected}
              disabled={false}
            />

            <div style={styles.divider}>
              <span style={styles.dividerText}>یا</span>
            </div>

            <button
              onClick={() => setShowCamera(true)}
              style={styles.cameraBtn}
            >
              📷 عکس با دوربین
            </button>
          </div>
        )}

        {/* ── STEP: Edit (Crop + Rotate) ── */}
        {step === 'edit' && editor.originalImage && editor.isEditing && (
          <ImageEditor
            image={editor.originalImage}
            crop={editor.crop}
            zoom={editor.zoom}
            rotation={editor.rotation}
            onCropChange={editor.setCrop}
            onZoomChange={editor.setZoom}
            onCropComplete={editor.handleCropComplete}
            onRotateLeft={editor.rotateLeft}
            onRotateRight={editor.rotateRight}
            onRotateCustom={editor.rotateCustom}
            onApply={handleApplyCrop}
            onCancel={handleCancelEdit}
          />
        )}

        {/* ── STEP: Result ── */}
        {step === 'result' && (
  <div style={styles.section}>
    {editor.croppedImage && (
      <div style={styles.croppedPreview}>
        <h3 style={styles.previewTitle}>تصویر برش‌خورده:</h3>
        <img src={editor.croppedImage} alt="Cropped" style={styles.croppedImg} />
      </div>
    )}

    <ProgressBar
      progress={ocr.progress}
      isProcessing={ocr.isProcessing}
      currentAttempt={ocr.currentAttempt}
      attempts={ocr.attempts}
    />

    {ocr.error && (
      <div style={styles.errorBox}>⚠️ {ocr.error}</div>
    )}

    {!ocr.isProcessing && ocr.result && (
      <ResultDisplay
        result={ocr.result}
        onRetry={handleRetry}
        onSelectDate={(date) => console.log('Selected:', date)}
      />
    )}

    {!ocr.isProcessing && (
      <button onClick={handleStartOver} style={styles.startOverBtn}>
        🏠 شروع مجدد
      </button>
    )}
  </div>
)}
      </div>

      {/* Camera Modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}

      {/* Global animation styles */}
      <style>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        .crop-area { transition: box-shadow 0.3s !important; }
        * { box-sizing: border-box; }
        body { margin: 0; }
      `}</style>
    </div>
  );
}

const styles = {
  page: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    padding: 16,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
    fontFamily: 'Tahoma, Arial, sans-serif',
  },
  card: {
    background: '#fff',
    borderRadius: 20,
    width: '100%',
    maxWidth: 600,
    marginTop: 16,
    marginBottom: 40,
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
    overflow: 'hidden',
  },
  header: {
    padding: '28px 24px 8px',
    textAlign: 'center',
  },
  title: {
    fontSize: 22,
    margin: 0,
    color: '#2c3e50',
  },
  subtitle: {
    fontSize: 13,
    color: '#888',
    margin: '6px 0 0',
  },

  // Steps
  steps: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '20px 24px 10px',
    gap: 0,
  },
  stepItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  stepCircle: {
    width: 28,
    height: 28,
    borderRadius: '50%',
    background: '#eee',
    color: '#bbb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 12,
    fontWeight: 'bold',
    flexShrink: 0,
  },
  stepActive: {
    background: 'linear-gradient(135deg, #667eea, #764ba2)',
    color: '#fff',
  },
  stepDone: {
    background: '#27ae60',
    color: '#fff',
  },
  stepLabel: {
    fontSize: 11,
    whiteSpace: 'nowrap',
  },
  stepLine: {
    width: 30,
    height: 2,
    background: '#eee',
    margin: '0 4px',
    flexShrink: 0,
  },

  section: {
    padding: '16px 24px 28px',
  },

  divider: {
    display: 'flex',
    alignItems: 'center',
    margin: '20px 0',
    gap: 16,
  },
  dividerText: {
    color: '#ccc',
    fontSize: 13,
    flex: 1,
    textAlign: 'center',
    position: 'relative',
  },

  cameraBtn: {
    width: '100%',
    padding: 14,
    borderRadius: 12,
    border: '2px solid #667eea',
    background: 'transparent',
    color: '#667eea',
    fontSize: 15,
    cursor: 'pointer',
    fontFamily: 'Tahoma, sans-serif',
    transition: 'all 0.2s',
  },

  croppedPreview: {
    textAlign: 'center',
    marginBottom: 16,
  },
  previewTitle: {
    fontSize: 14,
    color: '#666',
    margin: '0 0 10px',
  },
  croppedImg: {
    width: '100%',
    maxHeight: 220,
    objectFit: 'contain',
    borderRadius: 10,
    border: '2px solid #e8e8e8',
    background: '#fafafa',
  },

  errorBox: {
    background: '#fff3cd',
    border: '1px solid #ffc107',
    borderRadius: 8,
    padding: 12,
    fontSize: 13,
    color: '#856404',
    marginBottom: 16,
  },

  startOverBtn: {
    width: '100%',
    padding: 12,
    borderRadius: 10,
    border: '1px solid #ddd',
    background: '#f8f9fa',
    color: '#555',
    fontSize: 14,
    cursor: 'pointer',
    marginTop: 12,
    fontFamily: 'Tahoma, sans-serif',
  },
};