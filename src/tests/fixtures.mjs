// Shared synthetic-card renderer for phase-2 tests and debug harnesses.
// Imports @napi-rs/canvas itself (throws if absent).
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");

const napi = await import("@napi-rs/canvas");
const { createCanvas, GlobalFonts } = napi;

GlobalFonts.registerFromPath(path.join(root, "fonts", "Yekan.ttf"), "Yekan");
GlobalFonts.registerFromPath(path.join(root, "fonts", "Vazirmatn-VariableFont_wght.ttf"), "Vazir");

const FA = "۰۱۲۳۴۵۶۷۸۹";
const toFa = (s) => String(s).replace(/\d/g, (d) => FA[Number(d)]);
const dateFa = (y, m, d) =>
  toFa(`${y}/${String(m).padStart(2, "0")}/${String(d).padStart(2, "0")}`);

function drawText(ctx, text, x, y, size, font, align = "left", color = "#102030") {
  ctx.fillStyle = color;
  ctx.font = `${size}px ${font}`;
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  ctx.direction = "ltr";
  ctx.fillText(text, x, y);
}

/** Field row: RTL label with an LTR digit value to its left. */
function drawField(ctx, label, value, x, y, size, font) {
  drawText(ctx, label, x, y, size, font, "right", "#183048");
  const labelW = ctx.measureText(label).width;
  drawText(ctx, value, x - labelW - size * 0.6, y, Math.round(size * 1.15), font, "right", "#0a1420");
}

/**
 * Iranian national card drawn onto a canvas (no external assets).
 * mode: "photo" (card on a desk) | "crop" (date line only) | "nodate".
 * rotate: degrees applied to the whole composition (upside-down case).
 */
function renderCard({
  font = "Vazir",
  date = [1366, 6, 22],
  expiry = [1404, 5, 15],
  idLine = "0012345678",
  mode = "photo",
  rotate = 0,
}) {
  let w, h;
  if (mode === "photo") { w = 1500; h = 1000; }
  // 920: value (≈450px at font 74) + RTL label (≈310px) + margins must fit.
  else if (mode === "crop") { w = 920; h = 300; }
  else { w = 1500; h = 1000; }

  const canvas = createCanvas(w, h);
  const ctx = canvas.getContext("2d");

  if (mode === "crop") {
    ctx.fillStyle = "#f4f1ea";
    ctx.fillRect(0, 0, w, h);
    const size = 74;
    ctx.save();
    drawField(ctx, "تاریخ تولد:", dateFa(...date), w - 40, Math.round(h * 0.62), size, font);
    ctx.restore();
  } else {
    // Desk background.
    ctx.fillStyle = "#3a4048";
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 40; i++) {
      ctx.strokeStyle = `rgba(${180 - i * 2},${180 - i * 2},${180 - i * 2},0.08)`;
      ctx.beginPath();
      ctx.moveTo(0, (i * 37) % h);
      ctx.lineTo(w, (i * 61) % h);
      ctx.stroke();
    }

    // Card geometry (>28% of the frame so quadrilateral detection runs).
    const cx = w / 2, cy = h / 2;
    const cw = 1200, ch = 756;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate((rotate * Math.PI) / 180);
    ctx.translate(-cw / 2, -ch / 2);

    // Card body with a light guilloche-ish background.
    ctx.fillStyle = "#eef2f5";
    ctx.fillRect(0, 0, cw, ch);
    ctx.strokeStyle = "#9fb4c4";
    ctx.lineWidth = 3;
    ctx.strokeRect(4, 4, cw - 8, ch - 8);
    for (let i = 0; i < 26; i++) {
      ctx.strokeStyle = `rgba(120,150,175,${0.10 + (i % 3) * 0.04})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 20 * i);
      ctx.bezierCurveTo(cw * 0.3, 20 * i + 30, cw * 0.7, 20 * i - 30, cw, 20 * i);
      ctx.stroke();
    }

    if (mode !== "nodate") {
      // Header.
      drawText(ctx, "جمهوری اسلامی ایران", cw / 2, 64, 40, font, "center", "#123a5c");
      drawText(ctx, "کارت ملی", cw / 2, 112, 34, font, "center", "#123a5c");

      // Photo box (right side).
      ctx.fillStyle = "#c8d2da";
      ctx.fillRect(cw - 250, 150, 190, 250);
      ctx.strokeStyle = "#7f93a4";
      ctx.lineWidth = 2;
      ctx.strokeRect(cw - 250, 150, 190, 250);
      drawText(ctx, "عکس", cw - 155, 285, 30, font, "center", "#5f7486");

      // Personal fields (values are LTR digit runs).
      const right = cw - 300;
      const fs = 34;
      drawField(ctx, "نام:", "علی", right, 190, fs, font);
      drawField(ctx, "نام خانوادگی:", "رضایی", right, 250, fs, font);
      drawField(ctx, "نام پدر:", "محمد", right, 310, fs, font);
      drawField(ctx, "تاریخ شناسنامه:", "12345678", right, 370, fs, font);
      // Birth date — the target, in the middle band.
      drawField(ctx, "تاریخ تولد:", dateFa(...date), right, 470, fs, font);
      // National ID line (10 digits) — the classic confuser, still in band.
      drawField(ctx, "شماره ملی:", toFa(idLine), right, 560, fs, font);
      drawField(ctx, "سریال:", "123-456-789", right, 630, Math.round(fs * 0.9), font);

      // Expiry near the bottom edge (outside the search band).
      drawText(ctx, `تاریخ انقضا: ${dateFa(...expiry)}`, 40, ch - 40, 30, font, "left", "#33506a");
    } else {
      // No birth date anywhere: header + names + ID only.
      drawText(ctx, "جمهوری اسلامی ایران", cw / 2, 64, 40, font, "center", "#123a5c");
      drawField(ctx, "نام:", "علی", cw - 300, 220, 34, font);
      drawField(ctx, "نام خانوادگی:", "رضایی", cw - 300, 300, 34, font);
      drawField(ctx, "شماره ملی:", toFa(idLine), cw - 300, 460, 34, font);
    }

    ctx.restore(); // card transform
  }

  return canvas;
}

let cvModule = null;
async function getCV() {
  if (cvModule) return cvModule;
  const imported = await import("@techstark/opencv-js");
  let cv = imported.default ?? imported;
  if (cv && typeof cv.then === "function") cv = await cv;
  if (!cv.Mat) await new Promise((r) => { cv.onRuntimeInitialized = r; });
  cvModule = cv;
  return cv;
}

async function toMat(canvas) {
  const { data } = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  const cv = await getCV();
  const mat = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
  mat.data.set(data);
  return mat;
}

export { renderCard, toMat, getCV, dateFa, toFa, drawText, drawField };
