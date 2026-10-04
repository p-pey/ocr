// src/components/CardFrameOverlay.jsx


/**
 * SVG overlay that shows the ID card placement guide
 * Renders on top of the crop area to help user align the card
 */
export function CardFrameOverlay({ containerWidth, containerHeight }) {
  if (!containerWidth || !containerHeight) return null;

  return (
    <div style={styles.overlay}>
      {/* Corner markers */}
      <div style={{ ...styles.corner, ...styles.topLeft }} />
      <div style={{ ...styles.corner, ...styles.topRight }} />
      <div style={{ ...styles.corner, ...styles.bottomLeft }} />
      <div style={{ ...styles.corner, ...styles.bottomRight }} />

      {/* Field guides */}
      <div style={styles.fieldGuide}>
        <div style={styles.guideRow}>
          <span style={styles.guideLabel}>نام و نام خانوادگی</span>
        </div>
        <div style={styles.guideRow}>
          <span style={styles.guideLabelHighlight}>📅 تاریخ تولد</span>
        </div>
        <div style={styles.guideRow}>
          <span style={styles.guideLabel}>شماره ملی</span>
        </div>
      </div>

      {/* Center hint */}
      <div style={styles.centerHint}>
        کارت ملی را در کادر قرار دهید
      </div>
    </div>
  );
}

const cornerSize = 30;
const cornerBorder = 3;
const cornerColor = '#00e676';

const styles = {
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none',
    zIndex: 10,
  },
  corner: {
    position: 'absolute',
    width: cornerSize,
    height: cornerSize,
    borderColor: cornerColor,
    borderStyle: 'solid',
  },
  topLeft: {
    top: 8, left: 8,
    borderWidth: `${cornerBorder}px 0 0 ${cornerBorder}px`,
    borderRadius: '6px 0 0 0',
  },
  topRight: {
    top: 8, right: 8,
    borderWidth: `${cornerBorder}px ${cornerBorder}px 0 0`,
    borderRadius: '0 6px 0 0',
  },
  bottomLeft: {
    bottom: 8, left: 8,
    borderWidth: `0 0 ${cornerBorder}px ${cornerBorder}px`,
    borderRadius: '0 0 0 6px',
  },
  bottomRight: {
    bottom: 8, right: 8,
    borderWidth: `0 ${cornerBorder}px ${cornerBorder}px 0`,
    borderRadius: '0 0 6px 0',
  },
  fieldGuide: {
    position: 'absolute',
    right: 20,
    top: '30%',
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  guideRow: {
    textAlign: 'right',
  },
  guideLabel: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 11,
    fontFamily: 'Tahoma, sans-serif',
  },
  guideLabelHighlight: {
    color: '#00e676',
    fontSize: 12,
    fontFamily: 'Tahoma, sans-serif',
    fontWeight: 'bold',
    textShadow: '0 0 8px rgba(0,230,118,0.5)',
  },
  centerHint: {
    position: 'absolute',
    bottom: 16,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    fontFamily: 'Tahoma, sans-serif',
    background: 'rgba(0,0,0,0.3)',
    padding: '4px 0',
  },
};

export default CardFrameOverlay;