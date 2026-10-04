// src/components/CameraCapture.jsx

import { useCallback, useEffect, useRef, useState } from 'react';

const ID_CARD_ASPECT = 85.6 / 53.98;

export function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [facing, setFacing] = useState('environment');
  const [ready, setReady] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        setStream(s);
        setReady(true);
      }
    } catch {
      alert('دسترسی به دوربین ممکن نیست');
    }
  }, [facing]);

  const stopCamera = useCallback(() => {
    stream?.getTracks().forEach(t => t.stop());
    setStream(null);
    setReady(false);
  }, [stream]);

  useEffect(() => { startCamera(); return stopCamera; }, [facing]);

  const capture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const v = videoRef.current;
    const c = canvasRef.current;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob(blob => {
      if (blob) { onCapture(blob); stopCamera(); }
    }, 'image/jpeg', 0.95);
  }, [onCapture, stopCamera]);

  return (
    <div style={styles.container}>
      <div style={styles.videoWrap}>
        <video ref={videoRef} autoPlay playsInline muted style={styles.video} />
        {/* Card frame overlay */}
        <div style={styles.overlay}>
          <div style={styles.frame}>
            <div style={{ ...styles.c, ...styles.tl }} />
            <div style={{ ...styles.c, ...styles.tr }} />
            <div style={{ ...styles.c, ...styles.bl }} />
            <div style={{ ...styles.c, ...styles.br }} />
            <span style={styles.frameText}>
              کارت ملی را در این کادر قرار دهید
            </span>
          </div>
        </div>
      </div>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <div style={styles.bar}>
        <button onClick={onClose} style={styles.btn}>انصراف</button>
        <button onClick={capture} disabled={!ready} style={styles.captureBtn}>📷</button>
        <button onClick={() => { stopCamera(); setFacing(f => f === 'environment' ? 'user' : 'environment'); }} style={styles.btn}>
          🔄
        </button>
      </div>
    </div>
  );
}

const cs = 35;
const styles = {
  container: { position:'fixed', inset:0, background:'#000', zIndex:1000, display:'flex', flexDirection:'column' },
  videoWrap: { flex:1, position:'relative', overflow:'hidden' },
  video: { width:'100%', height:'100%', objectFit:'cover' },
  overlay: { position:'absolute', inset:0, display:'flex', alignItems:'center', justifyContent:'center' },
  frame: {
    width:'85%', maxWidth:500, aspectRatio: `${ID_CARD_ASPECT}`,
    border:'2px dashed rgba(0,230,118,0.8)', borderRadius:12,
    position:'relative', display:'flex', alignItems:'flex-end', justifyContent:'center', padding:12,
  },
  c: { position:'absolute', width:cs, height:cs, borderColor:'#00e676', borderStyle:'solid' },
  tl: { top:-2, left:-2, borderWidth:'3px 0 0 3px', borderRadius:'8px 0 0 0' },
  tr: { top:-2, right:-2, borderWidth:'3px 3px 0 0', borderRadius:'0 8px 0 0' },
  bl: { bottom:-2, left:-2, borderWidth:'0 0 3px 3px', borderRadius:'0 0 0 8px' },
  br: { bottom:-2, right:-2, borderWidth:'0 3px 3px 0', borderRadius:'0 0 8px 0' },
  frameText: {
    color:'#fff', fontSize:13, background:'rgba(0,0,0,0.5)',
    padding:'4px 14px', borderRadius:6, fontFamily:'Tahoma',
  },
  bar: { display:'flex', justifyContent:'space-around', alignItems:'center', padding:20, background:'#111' },
  btn: { padding:'10px 20px', borderRadius:8, border:'none', background:'#333', color:'#fff', fontSize:14, cursor:'pointer', fontFamily:'Tahoma' },
  captureBtn: {
    width:65, height:65, borderRadius:'50%', border:'4px solid #fff',
    background:'#e74c3c', fontSize:24, cursor:'pointer',
  },
};

export default CameraCapture;