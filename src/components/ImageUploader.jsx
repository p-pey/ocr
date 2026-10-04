// src/components/ImageUploader.jsx

import { useCallback, useRef } from 'react';

export function ImageUploader({ onImageSelected, disabled }) {
  const fileRef = useRef(null);
  const dropRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => onImageSelected(e.target.result);
    reader.readAsDataURL(file);
  }, [onImageSelected]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    dropRef.current.style.borderColor = '#ccc';
    dropRef.current.style.background = 'rgba(102,126,234,0.02)';
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  return (
    <div dir="rtl">
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(e) => handleFile(e.target.files[0])}
        style={{ display: 'none' }}
      />

      <div
        ref={dropRef}
        style={styles.dropZone}
        onClick={() => !disabled && fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) {
            dropRef.current.style.borderColor = '#667eea';
            dropRef.current.style.background = 'rgba(102,126,234,0.08)';
          }
        }}
        onDragLeave={() => {
          dropRef.current.style.borderColor = '#ccc';
          dropRef.current.style.background = 'rgba(102,126,234,0.02)';
        }}
        onDrop={disabled ? (e) => e.preventDefault() : handleDrop}
      >
        <div style={styles.dropIcon}>📁</div>
        <p style={styles.dropText}>
          تصویر کارت ملی را اینجا بکشید
          <br />
          یا کلیک کنید
        </p>
        <p style={styles.dropHint}>
          JPG, PNG — حداکثر ۱۰ مگابایت
        </p>
      </div>
    </div>
  );
}

const styles = {
  dropZone: {
    border: '2px dashed #ccc',
    borderRadius: 14,
    padding: '40px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 0.3s',
    background: 'rgba(102,126,234,0.02)',
  },
  dropIcon: { fontSize: 40, marginBottom: 8 },
  dropText: { color: '#555', fontSize: 14, margin: '0 0 8px', lineHeight: 1.8 },
  dropHint: { color: '#aaa', fontSize: 12, margin: 0 },
};

export default ImageUploader;