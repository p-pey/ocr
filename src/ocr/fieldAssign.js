/**
 * Field assignment (directive steps 3.1-3.3): Y-sort + chronology.
 * No Persian label OCR exists in this runtime (digit CNN only), so
 * anchoring is positional: birth sits ABOVE expiry on smart cards.
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
