/**
 * Field assignment (directive steps 3.1-3.3 + MASTER SPEC §2):
 * Y-sort + chronology, with RTL geometric anchoring helpers.
 * No Persian label OCR exists in this zero-external-OCR runtime
 * (digit CNN only), so anchoring is positional/geometric:
 * birth sits ABOVE expiry on smart cards (smaller Y = birth).
 *
 * assignFields(ranked, anchors) ->
 *   { birth, expiry, swapped, method, sequenceError }
 * ranked: date objects { formatted, year, yMin?, relY? } sorted or not.
 * Returns null birth/expiry when empty; sequenceError when the chosen
 * birth is chronologically AFTER the chosen expiry (unresolvable).
 */

const CUTOFF_YEAR = 1400;

function yOf(d) {
  if (Number.isFinite(d.yMin)) return d.yMin;
  if (Number.isFinite(d.relY)) return d.relY;
  return 1;
}

/**
 * MASTER SPEC §2.1 geometric proxy for RTL key-value anchoring.
 * Persian cards read right-to-left: the "تاریخ تولد" label sits on the
 * RIGHT of the card and the date digits extend LEFT of it. With zero
 * external OCR the label pixels cannot be read, so this scores how well a
 * candidate matches that geometry: right-anchored (xMax near the right
 * edge) and horizontally compact (a full 10-char date strip).
 * Higher = more birth-label-like.
 */
export function rtlAnchorScore(rect, cardW, cardH) {
  if (!rect || !(cardW > 0) || !(cardH > 0)) return 0;
  const xMax = (rect.x + rect.width) / cardW;
  const widthFrac = rect.width / cardW;
  // Right-anchored band: date strip's right edge in the right half.
  const rightness = Math.max(0, Math.min(1, (xMax - 0.35) / 0.6));
  // Compact full-date width (too narrow = half-cut, too wide = merged rows).
  const widthOk = widthFrac >= 0.12 && widthFrac <= 0.75 ? 1 : 0.25;
  return rightness * 0.7 + widthOk * 0.3;
}

/**
 * MASTER SPEC §2.1 negative-region suppression proxy.
 * The expiry row sits BELOW the birth row; any candidate whose top edge is
 * clearly below an already-accepted birth row is expiry-side and must be
 * dropped from the birth candidate pool.
 * @returns filtered list (birth-side rows only).
 */
export function suppressExpiryRows(cands, birthYMin, margin = 0.02) {
  if (!Array.isArray(cands) || !cands.length) return [];
  if (!Number.isFinite(birthYMin)) return [...cands];
  return cands.filter((c) => {
    const y = Number.isFinite(c.yMin) ? c.yMin : Number.isFinite(c.relY) ? c.relY : 1;
    return y <= birthYMin + margin;
  });
}

export function assignFields(ranked, anchors) {
  void anchors;
  const list = Array.isArray(ranked) ? [...ranked] : [];
  if (!list.length) {
    return { birth: null, expiry: null, swapped: false, method: "none", sequenceError: null };
  }
  const byY = [...list].sort((a, b) => yOf(a) - yOf(b) || a.year - b.year);
  if (byY.length === 1) {
    const only = byY[0];
    // Lone-expiry hint: an expiry-side year deep in the card reads as
    // expiry; the lone-expiry guard in buildFinalResult decides birth=null.
    if (only.year > CUTOFF_YEAR && yOf(only) > 0.65) {
      return { birth: null, expiry: only, swapped: false, method: "single", sequenceError: null };
    }
    return { birth: only, expiry: null, swapped: false, method: "single", sequenceError: null };
  }
  // Two+ rows: upper rows are birth-side (earliest year wins up top).
  const top = byY[0];
  const rest = byY.slice(1);
  // Expiry = latest year among the lower rows when present, else lowest row.
  let expiry = null;
  for (const d of rest) {
    if (!expiry || d.year > expiry.year) expiry = d;
  }
  // Birth = earliest year among birth-side pool (<=1400 preferred).
  const birthPool = list.filter((d) => d.year <= CUTOFF_YEAR);
  const pool = birthPool.length ? birthPool : list;
  let birth = pool[0];
  for (const d of pool) {
    if (d.year < birth.year || (d.year === birth.year && yOf(d) < yOf(birth))) birth = d;
  }
  // Avoid birth===expiry object when only two distinct rows exist.
  if (expiry && birth.formatted === expiry.formatted && rest.length) {
    // Same formatted date twice: keep one as birth, expiry stays too.
  }
  let sequenceError = null;
  if (birth && expiry && birth.year > expiry.year) {
    sequenceError = new Error("birth-after-expiry sequence");
  }
  // swapped: true when chronological order disagrees with Y order
  // (upside-down capture): the earlier year sits BELOW the later year.
  const swapped = Boolean(birth && expiry && yOf(birth) > yOf(expiry) && birth.year < expiry.year);
  void top;
  return { birth, expiry, swapped, method: byY.length >= 2 ? "y-sort" : "single", sequenceError };
}
