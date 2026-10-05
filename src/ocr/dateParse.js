/**
 * Pure helpers (no DOM / OpenCV / Tesseract): digit normalisation and Jalali
 * birth-date parsing + validation. Unit-testable in Node.
 */

const FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹";
const AR_DIGITS = "٠١٢٣٤٥٦٧٨٩";

// Separators that OCR / fonts commonly produce instead of "/".
const SEPARATOR_RE = /[\\|\-.:،٫٬⁄／]/g;

export function toLatinDigits(text) {
  if (!text) return "";
  return String(text)
    .replace(/[۰-۹]/g, (c) => String(FA_DIGITS.indexOf(c)))
    .replace(/[٠-٩]/g, (c) => String(AR_DIGITS.indexOf(c)));
}

/** 33-year arithmetic rule; matches the official calendar for ~1244-1472. */
export function isJalaliLeapYear(year) {
  return [1, 5, 9, 13, 17, 22, 26, 30].includes(year % 33);
}

export function daysInJalaliMonth(year, month) {
  if (month <= 6) return 31;
  if (month <= 11) return 30;
  return isJalaliLeapYear(year) ? 30 : 29;
}

/**
 * Turn noisy recogniser output into "YYYY/M(M)/D(D)" when that is possible
 * without guessing digits. Only fixes *structure* (separators, dropped
 * slashes, visual order), never individual digits.
 */
export function repairStructure(raw) {
  let s = toLatinDigits(raw).replace(SEPARATOR_RE, "/");
  s = s.replace(/[^0-9/]/g, "").replace(/\/+/g, "/").replace(/^\/|\/$/g, "");

  // Slashes dropped: 13750512 -> 1375/05/12
  if (!s.includes("/") && s.length === 8) {
    s = `${s.slice(0, 4)}/${s.slice(4, 6)}/${s.slice(6)}`;
  }
  // Visual order flipped (DD/MM/YYYY): reverse the groups.
  const parts = s.split("/");
  if (parts.length === 3 && parts[0].length <= 2 && parts[2].length === 4) {
    s = parts.reverse().join("/");
  }
  return s;
}

/**
 * @returns {{year:number, month:number, day:number, formatted:string}|null}
 */
export function parseJalaliDate(raw, { minYear = 1280, maxYear = 1410 } = {}) {
  const s = repairStructure(raw);
  const m = /^(\d{4})\/(\d{1,2})\/(\d{1,2})$/.exec(s);
  if (!m) return null;
  const [year, month, day] = [Number(m[1]), Number(m[2]), Number(m[3])];
  if (year < minYear || year > maxYear) return null;
  if (month < 1 || month > 12) return null;
  if (day < 1 || day > daysInJalaliMonth(year, month)) return null;
  return {
    year,
    month,
    day,
    formatted: `${year}/${String(month).padStart(2, "0")}/${String(day).padStart(2, "0")}`,
  };
}
