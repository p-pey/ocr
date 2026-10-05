// src/components/ResultDisplay.jsx

import { useState } from 'react';
import { englishToPersian, formatJalali } from '../utils/persianUtils';

export function ResultDisplay({ result, onRetry, onSelectDate }) {
  const [showDebug, setShowDebug] = useState(false);
  const [selectedDate, setSelectedDate] = useState(null);

  if (!result) return null;

  const { best, allDates = [], allAttempts = [] } = result;
  // Per README: `best` is null when nothing reached MIN_CONFIDENCE (60).
  // Treat null as "retake photo" — never show a low-confidence allDates[0] as success.
  const topDate = best?.birthDate ?? null;

  const handleSelectDate = (date) => {
    setSelectedDate(date);
    onSelectDate?.(date);
  };

  return (
    <div style={styles.container} dir="rtl">
      {topDate ? (
        <div style={styles.successBox}>
          <div style={styles.icon}>✅</div>
          <h2 style={styles.heading}>تاریخ تولد شناسایی شد</h2>

          <div style={styles.dateBox}>
            <span style={styles.dateLabel}>تاریخ تولد:</span>
            <span style={styles.datePersian}>
              {englishToPersian(formatJalali(topDate))}
            </span>
            <span style={styles.dateEnglish}>({formatJalali(topDate)})</span>
          </div>

          <div style={styles.badges}>
            <span style={styles.badge}>
              🗳️ {topDate.votes} رای
            </span>
            <span style={styles.badge}>
              💯 امتیاز: {Math.round(topDate.finalScore)}
            </span>
            {best?.confidence && (
              <span style={styles.badge}>
                🎯 اطمینان: {Math.round(best.confidence)}%
              </span>
            )}
            {best?.engine && (
              <span style={styles.badge} title={best.engine}>
                🤖 {best.engine.includes('tesseract') ? 'Tesseract (fallback)' : 'مدل عصبی'}
              </span>
            )}
          </div>

          {/* Alternative dates */}
          {allDates.length > 1 && (
            <div style={styles.alternatives}>
              <h4 style={styles.altTitle}>گزینه‌های دیگر:</h4>
              <div style={styles.altList}>
                {allDates.slice(1, 6).map((date, i) => (
                  <button
                    key={i}
                    onClick={() => handleSelectDate(date)}
                    style={{
                      ...styles.altItem,
                      ...(selectedDate?.formatted === date.formatted ? styles.altSelected : {}),
                    }}
                  >
                    <span style={styles.altDate}>
                      {englishToPersian(formatJalali(date))}
                    </span>
                    <span style={styles.altMeta}>
                      {date.votes} رای • {date.finalScore} امتیاز
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div style={styles.failBox}>
          <div style={styles.icon}>❌</div>
          <h2 style={styles.heading}>تاریخ تولد یافت نشد</h2>
          <p style={styles.failText}>
            پس از {allAttempts.length} تلاش — اطمینان به حد نصاب (60) نرسید. لطفاً دوباره عکس بگیرید.
          </p>
          <ul style={styles.tips}>
            <li>تصویر را دوباره برش دهید — فقط ناحیه تاریخ تولد را انتخاب کنید</li>
            <li>مطمئن شوید تصویر چرخش ندارد</li>
            <li>کنتراست و نور تصویر را بررسی کنید</li>
            <li>اگر کارت از پشت است، سمت جلو را اسکن کنید</li>
          </ul>
        </div>
      )}

      {/* Debug Toggle */}
      <button
        onClick={() => setShowDebug(!showDebug)}
        style={styles.debugToggle}
      >
        {showDebug ? '🔽 پنهان‌سازی' : '🔍 نمایش'} جزئیات ({allAttempts.length} تلاش)
      </button>

      {/* Debug Information */}
      {showDebug && (
        <div style={styles.debugPanel}>
          <h4 style={styles.debugTitle}>📊 جزئیات تلاش‌ها</h4>

          {allAttempts.map((attempt, i) => (
            <div key={i} style={styles.attemptCard}>
              <div style={styles.attemptHeader}>
                <span style={styles.attemptNum}>#{i + 1}</span>
                <span style={styles.chip}>🎨 {attempt.strategy}</span>
                {attempt.candidateIndex !== undefined && (
                  <span style={styles.chip}>📍 line #{attempt.candidateIndex}{attempt.window === 'window' ? ` · win ${attempt.windowIndex}` : ''}{attempt.rotation ? ` ↻${attempt.rotation}°` : ''}</span>
                )}
                <span style={styles.chip}>🤖 {attempt.engine}</span>
                <span style={styles.confChip}>
                  {Math.round(attempt.confidence || 0)}%
                </span>
              </div>

              {attempt.dates?.length > 0 && (
                <div style={styles.datesFound}>
                  ✅ یافت شد:
                  {attempt.dates.map((d, j) => (
                    <span key={j} style={styles.foundDate}>
                      {d.formatted} (score: {d.score})
                    </span>
                  ))}
                </div>
              )}

              <details style={styles.textDetails}>
                <summary style={styles.textSummary}>متن استخراج‌شده</summary>
                <pre style={styles.rawText}>
                  {attempt.normalizedText || attempt.rawText || '(خالی)'}
                </pre>
              </details>

              {attempt.preprocessedImage && (
                <details style={styles.textDetails}>
                  <summary style={styles.textSummary}>تصویر پردازش‌شده</summary>
                  <img
                    src={attempt.preprocessedImage}
                    alt="Preprocessed"
                    style={styles.processedImg}
                  />
                </details>
              )}
            </div>
          ))}
        </div>
      )}

      <button onClick={onRetry} style={styles.retryBtn}>
        🔄 تلاش مجدد
      </button>
    </div>
  );
}

const styles = {
  container: { marginTop: 20 },
  successBox: {
    background: 'linear-gradient(135deg, #e8f5e9, #c8e6c9)',
    borderRadius: 14,
    padding: 24,
    textAlign: 'center',
    border: '1px solid #a5d6a7',
  },
  failBox: {
    background: '#fdecea',
    borderRadius: 14,
    padding: 24,
    textAlign: 'center',
    border: '1px solid #f5c6cb',
  },
  icon: { fontSize: 48, marginBottom: 10 },
  heading: { fontSize: 18, color: '#2c3e50', margin: '0 0 16px' },
  dateBox: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 4,
  },
  dateLabel: { color: '#666', fontSize: 13 },
  datePersian: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#27ae60',
    letterSpacing: 2,
    direction: 'ltr',
  },
  dateEnglish: { color: '#999', fontSize: 13, direction: 'ltr' },
  badges: {
    display: 'flex',
    justifyContent: 'center',
    gap: 8,
    flexWrap: 'wrap',
    marginTop: 14,
  },
  badge: {
    background: 'rgba(0,0,0,0.08)',
    padding: '4px 10px',
    borderRadius: 12,
    fontSize: 11,
    color: '#555',
  },
  alternatives: {
    marginTop: 20,
    paddingTop: 16,
    borderTop: '1px solid rgba(0,0,0,0.08)',
    textAlign: 'right',
  },
  altTitle: { fontSize: 13, color: '#666', margin: '0 0 10px' },
  altList: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  altItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 12px',
    background: 'rgba(255,255,255,0.6)',
    border: '1px solid rgba(0,0,0,0.08)',
    borderRadius: 8,
    cursor: 'pointer',
    fontFamily: 'Tahoma',
    fontSize: 12,
  },
  altSelected: {
    background: '#fff',
    border: '2px solid #27ae60',
  },
  altDate: {
    direction: 'ltr',
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  altMeta: { color: '#888', fontSize: 10 },
  failText: { color: '#666', fontSize: 14 },
  tips: {
    textAlign: 'right',
    fontSize: 13,
    lineHeight: 2.2,
    color: '#555',
    paddingRight: 20,
    marginTop: 16,
  },
  debugToggle: {
    width: '100%',
    padding: 10,
    marginTop: 12,
    border: '1px dashed #ccc',
    borderRadius: 8,
    background: 'transparent',
    color: '#666',
    fontSize: 12,
    cursor: 'pointer',
    fontFamily: 'Tahoma',
  },
  debugPanel: {
    marginTop: 12,
    padding: 14,
    background: '#f8f9fa',
    borderRadius: 10,
    maxHeight: 400,
    overflowY: 'auto',
  },
  debugTitle: {
    fontSize: 13,
    color: '#555',
    margin: '0 0 12px',
  },
  attemptCard: {
    background: '#fff',
    padding: 10,
    borderRadius: 8,
    marginBottom: 8,
    border: '1px solid #eee',
    fontSize: 11,
  },
  attemptHeader: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    alignItems: 'center',
    marginBottom: 8,
  },
  attemptNum: {
    background: '#667eea',
    color: '#fff',
    padding: '2px 8px',
    borderRadius: 10,
    fontSize: 10,
    direction: 'ltr',
  },
  chip: {
    background: '#eee',
    padding: '2px 8px',
    borderRadius: 10,
    fontSize: 10,
    color: '#666',
  },
  confChip: {
    background: '#fff3cd',
    padding: '2px 8px',
    borderRadius: 10,
    fontSize: 10,
    color: '#856404',
    marginRight: 'auto',
    direction: 'ltr',
  },
  datesFound: {
    background: '#d4edda',
    padding: 6,
    borderRadius: 6,
    fontSize: 11,
    color: '#155724',
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    alignItems: 'center',
    marginBottom: 6,
  },
  foundDate: {
    background: '#fff',
    padding: '2px 8px',
    borderRadius: 10,
    fontFamily: 'monospace',
    direction: 'ltr',
    fontSize: 11,
  },
  textDetails: { marginTop: 6 },
  textSummary: {
    cursor: 'pointer',
    color: '#888',
    fontSize: 11,
    padding: 4,
  },
  rawText: {
    background: '#f0f0f0',
    padding: 8,
    borderRadius: 4,
    fontSize: 10,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    direction: 'ltr',
    textAlign: 'left',
    maxHeight: 100,
    overflow: 'auto',
    margin: 0,
  },
  processedImg: {
    width: '100%',
    maxHeight: 120,
    objectFit: 'contain',
    borderRadius: 4,
    marginTop: 4,
    border: '1px solid #ddd',
  },
  retryBtn: {
    width: '100%',
    padding: 14,
    borderRadius: 10,
    border: 'none',
    background: '#3498db',
    color: '#fff',
    fontSize: 15,
    cursor: 'pointer',
    marginTop: 16,
    fontFamily: 'Tahoma',
  },
};

export default ResultDisplay;