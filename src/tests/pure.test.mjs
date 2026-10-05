// node src/tests/pure.test.mjs   (run from the project root: npm test)
import assert from "node:assert/strict";
import { parseJalaliDate, repairStructure, isJalaliLeapYear } from "../ocr/dateParse.js";
import { decodeHeads, NUM_DIGITS } from "../ocr/digitModel.js";

// --- parsing / validation ---
assert.equal(parseJalaliDate("۱۳۷۵/۰۵/۱۲").formatted, "1375/05/12");
assert.equal(parseJalaliDate("1375/5/2").formatted, "1375/05/02");
assert.equal(parseJalaliDate("13750512").formatted, "1375/05/12");     // dropped slashes
assert.equal(parseJalaliDate("12/05/1375").formatted, "1375/05/12");   // visual order flipped
assert.equal(parseJalaliDate("۱۳۷۵-۰۵-۱۲").formatted, "1375/05/12");   // other separators
assert.equal(parseJalaliDate("1375/13/01"), null);                      // month 13
assert.equal(parseJalaliDate("1375/07/31"), null);                      // Mehr has 30 days
assert.equal(parseJalaliDate("1375/06/31").formatted, "1375/06/31");    // Shahrivar has 31
assert.equal(parseJalaliDate("1375/12/30").formatted, "1375/12/30");    // 1375 is leap
assert.equal(parseJalaliDate("1376/12/30"), null);                      // 1376 is not
assert.equal(parseJalaliDate("1500/01/01"), null);                      // out of year range
assert.equal(parseJalaliDate("garbage"), null);
assert.ok(isJalaliLeapYear(1375) && isJalaliLeapYear(1403) && !isJalaliLeapYear(1402));
assert.equal(repairStructure("۱۳۷۵ / ۰۵ / ۱۲ ").length, 10);

// --- head decoding: two samples, [N,8,10] probabilities ---
const N = 2;
const probs = new Float32Array(N * NUM_DIGITS * 10).fill(0.01);
["13750512", "14001103"].forEach((d, n) =>
  [...d].forEach((ch, h) => { probs[(n * NUM_DIGITS + h) * 10 + Number(ch)] = 0.91; }));
const out = decodeHeads(probs, new Float32Array([0.97, 0.2]), N);
assert.equal(out[0].text, "1375/05/12");
assert.equal(out[1].text, "1400/11/03");
assert.equal(out[0].isDate, true);
assert.equal(out[1].isDate, false);                 // low isDate score => rejected by the engine
assert.ok(Math.abs(out[0].confidence - 91) < 1e-3);
console.log("pure tests passed");
