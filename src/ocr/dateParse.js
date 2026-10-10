/**
 * Pure helpers (no DOM / OpenCV / Tesseract): digit normalisation and Jalali
 * birth-date parsing + validation. Unit-testable in Node.
 *
 * MASTER SPEC §7 — Strict Jalali Calendar Validation (deterministic enforcer):
 *  Year JALALI.minBirthYear..JALALI.maxBirthYear (adult birth dates),
 *  Month 1..12, days per month (see daysInJalaliMonth).
 * Any fully-read string that fails this logic is rejected entirely (null).
 * Tune the year range in engineConfig.js §P (JALALI).
 */
import { JALALI as JALALI_CFG } from "./engineConfig.js";

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
 * MASTER SPEC §7 enforcer alias (wired to the final probability array):
 * strict Jalali bounds — Year 1290..1410, Month 1..12, Day per month lengths
 * (1–6 ≤31, 7–11 ≤30, Esfand ≤29, 30 only in leap years).
 * @returns {boolean} true iff the triple is a real Jalali birth date.
 */
export function isValidJalaliDate(year, month, day) {
  if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) return false;
  if (year < JALALI_CFG.minBirthYear || year > JALALI_CFG.maxBirthYear) return false;
  if (month < 1 || month > 12) return false;
  if (day < 1 || day > daysInJalaliMonth(year, month)) return false;
  return true;
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
 *
 * MASTER SPEC §7 / repo spec §6 (strict Jalali enforcer):
 *  Year 1290..1410, Month 1..12, Day per Jalali month lengths
 *  (1-6 ≤31, 7-11 ≤30, Esfand ≤29, 30 only in leap years).
 *  Anything failing this logic is rejected entirely (null) — never guessed.
 */
export function parseJalaliDate(raw, { minYear = JALALI_CFG.minBirthYear, maxYear = JALALI_CFG.maxBirthYear } = {}) {
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
