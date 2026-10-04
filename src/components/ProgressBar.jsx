// src/components/ProgressBar.jsx


export function ProgressBar({ progress, isProcessing, currentAttempt, attempts }) {
  if (!isProcessing) return null;

  return (
    <div style={styles.container} dir="rtl">
      <div style={styles.barOuter}>
        <div style={{ ...styles.barInner, width: `${Math.max(progress, 5)}%` }} />
      </div>

      <div style={styles.textRow}>
        <span style={styles.label}>
          {progress < 10 ? '⏳ آماده‌سازی موتورهای OCR...'
            : progress < 30 ? '🔍 تلاش ۱: پردازش کامل تصویر...'
            : progress < 60 ? '🎯 تلاش ۲: پردازش ناحیه‌ای...'
            : progress < 90 ? '📝 تلاش ۳: اسکن خط به خط...'
            : '✨ تحلیل نتایج...'}
        </span>
        <span style={styles.percent}>{progress}%</span>
      </div>

      {currentAttempt && (
        <div style={styles.attemptInfo}>
          <div style={styles.attemptBadge}>تلاش #{currentAttempt.number}</div>
          <div style={styles.attemptDetails}>
            <span>🎨 {currentAttempt.strategy}</span>
            <span>📍 {currentAttempt.region}</span>
            <span>⚙️ PSM {currentAttempt.psm}</span>
            {currentAttempt.dates?.length > 0 && (
              <span style={styles.foundBadge}>
                ✅ یافت شد: {currentAttempt.dates[0].formatted}
              </span>
            )}
          </div>
        </div>
      )}

      {attempts.length > 0 && (
        <div style={styles.stats}>
          مجموع تلاش‌ها: {attempts.length} •
          تاریخ‌های یافت‌شده: {attempts.reduce((sum, a) => sum + (a.dates?.length || 0), 0)}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { padding: '16px 0' },
  barOuter: { height: 10, background: '#e0e0e0', borderRadius: 5, overflow: 'hidden' },
  barInner: {
    height: '100%',
    background: 'linear-gradient(90deg, #667eea, #764ba2, #00e676)',
    borderRadius: 5,
    transition: 'width 0.4s ease',
  },
  textRow: { display: 'flex', justifyContent: 'space-between', marginTop: 8 },
  label: { fontSize: 13, color: '#666' },
  percent: { fontSize: 13, color: '#764ba2', fontWeight: 'bold', direction: 'ltr' },
  attemptInfo: {
    marginTop: 12,
    padding: 10,
    background: 'rgba(102,126,234,0.08)',
    borderRadius: 8,
    fontSize: 11,
  },
  attemptBadge: {
    display: 'inline-block',
    background: '#667eea',
    color: '#fff',
    padding: '2px 8px',
    borderRadius: 12,
    fontSize: 10,
    marginBottom: 6,
    direction: 'ltr',
  },
  attemptDetails: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 10,
    color: '#555',
  },
  foundBadge: {
    background: '#00e676',
    color: '#000',
    padding: '2px 8px',
    borderRadius: 10,
    fontWeight: 'bold',
    direction: 'ltr',
  },
  stats: {
    marginTop: 8,
    fontSize: 11,
    color: '#888',
    textAlign: 'center',
  },
};

export default ProgressBar;