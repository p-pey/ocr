// src/utils/persianUtils.js

const persianDigits = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
const arabicDigits = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];

export function persianToEnglish(str) {
  if (!str) return "";
  let result = str.toString();
  for (let i = 0; i < 10; i++) {
    result = result.replace(new RegExp(persianDigits[i], "g"), i.toString());
    result = result.replace(new RegExp(arabicDigits[i], "g"), i.toString());
  }
  // Normalize common OCR confusions
  result = result
    .replace(/[Oo٥]/g, "0") // O→0
    .replace(/[Il|]/g, "1") // I,l,| → 1
    .replace(/[sS]/g, "5") // Could be 5
    .replace(/[gq]/g, "9"); // g,q → 9
  return result;
}

export function englishToPersian(str) {
  if (!str) return "";
  let result = str.toString();
  for (let i = 0; i < 10; i++) {
    result = result.replace(new RegExp(i.toString(), "g"), persianDigits[i]);
  }
  return result;
}

/**
 * Extract ALL possible dates from text with scoring
 */
export function extractAllDates(text) {
  if (!text) return [];
  const dates = [];
  const normalized = persianToEnglish(text);

  const patterns = [
    // Strict: YYYY/MM/DD with separators
    {
      regex:
        /(13\d{2}|14\d{2})\s*[\/\-\.\\_،,:]\s*(0?[1-9]|1[0-2])\s*[\/\-\.\\_،,:]\s*(0?[1-9]|[12]\d|3[01])/g,
      score: 100,
    },
    // DD/MM/YYYY (reversed, less common for IR cards)
    {
      regex:
        /(0?[1-9]|[12]\d|3[01])\s*[\/\-\.\\_،,:]\s*(0?[1-9]|1[0-2])\s*[\/\-\.\\_،,:]\s*(13\d{2}|14\d{2})/g,
      score: 80,
      reversed: true,
    },
    // With label
    {
      regex:
        /(?:تاریخ[\s]*تولد|تولد|Birth|Date)\D{0,10}(13\d{2}|14\d{2})\D{1,4}(\d{1,2})\D{1,4}(\d{1,2})/gi,
      score: 120,
    },
    // 8 consecutive digits: 13700512
    {
      regex: /\b(13\d{2}|14\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b/g,
      score: 90,
    },
    // Loose spaces: 1370 05 12
    {
      regex: /(13\d{2}|14\d{2})\s+(0?[1-9]|1[0-2])\s+(0?[1-9]|[12]\d|3[01])/g,
      score: 70,
    },
    // Any 4 then 2 then 2 digits anywhere
    {
      regex: /(13\d{2}|14\d{2})[^\d]{0,5}(\d{1,2})[^\d]{0,5}(\d{1,2})/g,
      score: 50,
    },
    // Persian month names
    {
      regex:
        /(\d{1,2})\s*(فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|آذر|دی|بهمن|اسفند)\s*(\d{4})/g,
      score: 110,
    },
  ];

  const persianMonths = {
    فروردین: 1,
    اردیبهشت: 2,
    خرداد: 3,
    تیر: 4,
    مرداد: 5,
    شهریور: 6,
    مهر: 7,
    آبان: 8,
    آذر: 9,
    دی: 10,
    بهمن: 11,
    اسفند: 12,
  };

  for (const { regex, score, reversed } of patterns) {
    let match;
    const r = new RegExp(regex.source, regex.flags);
    while ((match = r.exec(normalized)) !== null) {
      let year, month, day;

      if (persianMonths[match[2]]) {
        year = parseInt(match[3]);
        month = persianMonths[match[2]];
        day = parseInt(match[1]);
      } else if (reversed) {
        day = parseInt(match[1]);
        month = parseInt(match[2]);
        year = parseInt(match[3]);
      } else {
        year = parseInt(match[1]);
        month = parseInt(match[2]);
        day = parseInt(match[3]);
      }

      if (isValidJalali(year, month, day)) {
        dates.push({
          year,
          month,
          day,
          score,
          raw: match[0],
          formatted: `${year}/${String(month).padStart(2, "0")}/${String(day).padStart(2, "0")}`,
        });
      }
    }
  }

  // Deduplicate — keep highest score
  const unique = {};
  for (const d of dates) {
    const key = d.formatted;
    if (!unique[key] || unique[key].score < d.score) {
      unique[key] = d;
    } else if (unique[key].score === d.score) {
      unique[key].score += 10; // bonus for duplicate detection
    }
  }

  return Object.values(unique).sort((a, b) => b.score - a.score);
}

export function extractBirthDate(text) {
  const dates = extractAllDates(text);
  return dates[0] || null;
}

export function isValidJalali(y, m, d) {
  if (!y || !m || !d) return false;
  if (y < 1280 || y > 1420) return false; // Reasonable birth years
  if (m < 1 || m > 12) return false;
  if (d < 1 || d > 31) return false;
  if (m > 6 && m < 12 && d > 30) return false;
  if (m === 12 && d > 30) return false;
  return true;
}

export function formatJalali(obj) {
  if (!obj) return "";
  return `${obj.year}/${String(obj.month).padStart(2, "0")}/${String(obj.day).padStart(2, "0")}`;
}
