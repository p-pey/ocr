// Strict Jalali validation (directive: post-processing rule enforcers).
// Doc examples: 1363/01/01 must parse (no hallucination into 1388/11/01 at
// this layer); 1311/11/36 must be rejected (day exceeds month bounds).
// month <= 12 and day <= 31 enforced with per-month Jalali month lengths.
import assert from "node:assert/strict";
import {
  parseJalaliDate,
  repairStructure,
  isJalaliLeapYear,
  daysInJalaliMonth,
} from "../src/ocr/dateParse.js";

// --- directive's own examples ---
assert.equal(parseJalaliDate("1363/01/01").formatted, "1363/01/01");
assert.equal(parseJalaliDate("۱۳۶۳/۰۱/۰۱").formatted, "1363/01/01");
assert.equal(parseJalaliDate("1311/11/36"), null); // Aban has 30 days
assert.equal(parseJalaliDate("1311/11/31"), null);
assert.equal(parseJalaliDate("1311/11/30").formatted, "1311/11/30");

// --- month <= 12 ---
assert.equal(parseJalaliDate("1376/13/01"), null);
assert.equal(parseJalaliDate("1376/00/10"), null);
assert.equal(parseJalaliDate("1376/12/01").formatted, "1376/12/01");

// --- day <= 31 with Jalali month lengths ---
assert.equal(parseJalaliDate("1376/01/31").formatted, "1376/01/31"); // Farvardin 31
assert.equal(parseJalaliDate("1376/06/31").formatted, "1376/06/31"); // Shahrivar 31
assert.equal(parseJalaliDate("1376/07/31"), null); // Mehr 30
assert.equal(parseJalaliDate("1376/11/30").formatted, "1376/11/30"); // Bahman 30
assert.equal(parseJalaliDate("1376/11/31"), null);
assert.equal(parseJalaliDate("1376/12/29").formatted, "1376/12/29");
assert.equal(parseJalaliDate("1375/12/30").formatted, "1375/12/30"); // 1375 leap
assert.equal(parseJalaliDate("1376/12/30"), null); // 1376 not leap
assert.equal(parseJalaliDate("1376/05/32"), null);
assert.equal(parseJalaliDate("1376/05/00"), null);
assert.equal(daysInJalaliMonth(1375, 12), 30);
assert.equal(daysInJalaliMonth(1376, 12), 29);
assert.ok(isJalaliLeapYear(1375) && !isJalaliLeapYear(1376));

// --- year bounds (spec 1280-1410) ---
assert.equal(parseJalaliDate("1279/01/01"), null);
assert.equal(parseJalaliDate("1411/01/01"), null);
assert.equal(parseJalaliDate("1280/01/01").formatted, "1280/01/01");
assert.equal(parseJalaliDate("1410/12/29").formatted, "1410/12/29");

// --- structure-only repair: never invent digits ---
assert.equal(parseJalaliDate("garbage"), null);
assert.equal(parseJalaliDate("13a5/05/12"), null); // letter stripped -> 3-digit year
assert.equal(parseJalaliDate("13750512").formatted, "1375/05/12"); // dropped slashes
assert.equal(parseJalaliDate("12/05/1375").formatted, "1375/05/12"); // flipped order
assert.equal(repairStructure("۱۳۸۰ / ۰۱ / ۰۶"), "1380/01/06");

console.log("strict Jalali validation tests passed");
