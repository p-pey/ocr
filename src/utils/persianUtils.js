// src/utils/persianUtils.js
//
// Active helpers:
//   englishToPersian — Latin digits -> Persian digits (display only,
//                      used by ResultDisplay)
//   formatJalali     — { year, month, day } -> "YYYY/MM/DD"
// Legacy Tesseract-era text parsing was removed (nothing imported it):
// persianToEnglish, extractAllDates, extractBirthDate, isValidJalali.
// The engine normalises + validates dates in src/ocr/dateParse.js.

const persianDigits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];

export function englishToPersian(str) {
  if (!str) return "";
  let result = str.toString();
  for (let i = 0; i < 10; i++) {
    result = result.replace(new RegExp(i.toString(), "g"), persianDigits[i]);
  }
  return result;
}

export function formatJalali(obj) {
  if (!obj) return "";
  return `${obj.year}/${String(obj.month).padStart(2, "0")}/${String(obj.day).padStart(2, "0")}`;
}
