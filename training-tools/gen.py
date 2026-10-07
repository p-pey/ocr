#!/usr/bin/env python3
"""Synthetic date-line generator + augmentation (spec 7) and the
preprocessing twin of JS grayToModelInput (spec 5).

Also hosts render_with_glyphs (spec 9.3): once the user supplies
glyph_sheet.png variants, glyph-based renders become the MAIN data source
(>= 60% of training). Without glyphs it falls back to font rendering.
"""
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_H = 32
MODEL_W = 160

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_LETTERS = "ابپتثجچحخدرزسشصضطظعغفقکگلمنوهی"
LABELS = ["تاریخ تولد", "شماره ملی", "تولد", "نام"]
EXCLUDE = {"lalezar", "kufam", "reem-kufi"}
# Held out for the font-generalisation ("UNSEEN-fonts") metric (spec 7).
HELD_OUT = {"tanha", "gandom", "mirza"}

# Confusable pairs oversampled >= 3x natural rate (spec 9.3).
CONFUSABLE = ["05", "50", "09", "90", "17", "71", "34", "43", "42", "24", "69", "96", "78", "87"]


def is_jalali_leap(year: int) -> bool:
    return year % 33 in (1, 5, 9, 13, 17, 22, 26, 30)


def days_in_month(y: int, m: int) -> int:
    if m <= 6:
        return 31
    if m <= 11:
        return 30
    return 30 if is_jalali_leap(y) else 29


def to_fa(s: str) -> str:
    return "".join(FA_DIGITS[int(c)] if c.isdigit() else c for c in s)


# ---------------------------------------------------------------- preprocessing twin (spec 5)

def resize_for_model(crop: np.ndarray) -> tuple[np.ndarray, int]:
    """crop: grayscale uint8, dark ink / light paper. Returns (resized, newW)."""
    h, w = crop.shape[:2]
    new_w = max(8, min(MODEL_W, round(w * MODEL_H / max(h, 1))))
    interp = cv2.INTER_AREA if h > MODEL_H else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, MODEL_H), interpolation=interp)
    return resized, new_w


def standardize(resized: np.ndarray) -> np.ndarray:
    x = resized.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def preprocess(crop: np.ndarray) -> tuple[np.ndarray, int]:
    """Full contract: standardised 32x160 float32 + newW. Mirrors JS exactly."""
    resized, new_w = resize_for_model(crop)
    std = standardize(resized)
    out = np.zeros((MODEL_H, MODEL_W), dtype=np.float32)
    out[:, :new_w] = std
    return out, new_w


# ---------------------------------------------------------------- fonts

def load_fonts(font_dir: Path):
    fonts = []
    for p in sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf")):
        low = p.stem.lower()
        if any(x in low for x in EXCLUDE):
            continue
        fonts.append(p)
    return fonts


def split_fonts(fonts, seed=0):
    train, held = [], []
    for p in fonts:
        (held if any(h in p.stem.lower() for h in HELD_OUT) else train).append(p)
    if not train:
        train = fonts
    return train, held


# ---------------------------------------------------------------- labels (spec 7: 74% dates, 26% negatives)

def random_date(rng: random.Random) -> str:
    y = rng.randint(1300, 1415)
    m = rng.randint(1, 12)
    d = rng.randint(1, days_in_month(y, m))
    return f"{y:04d}{m:02d}{d:02d}"


def biased_digits(rng: random.Random) -> str:
    """Sample digits with confusable pairs at >= 3x natural rate (spec 9.3)."""
    if rng.random() < 0.45:
        pair = rng.choice(CONFUSABLE)
        y = rng.randint(1300, 1415)
        m = rng.randint(1, 12)
        d = rng.randint(1, days_in_month(y, m))
        base = f"{y:04d}{m:02d}{d:02d}"
        pos = rng.randint(0, 6)
        base = base[:pos] + pair + base[pos + 2:]
        # re-validate month/day stay legal; else fall back to plain date
        try:
            yy, mm, dd = int(base[:4]), int(base[4:6]), int(base[6:8])
            if 1 <= mm <= 12 and 1 <= dd <= days_in_month(yy, mm):
                return base
        except ValueError:
            pass
        return random_date(rng)
    return random_date(rng)


def gen_label(rng: random.Random):
    if rng.random() < 0.74:
        digits = biased_digits(rng)
        text = to_fa(f"{digits[:4]}/{digits[4:6]}/{digits[6:]}")
        if rng.random() < 0.4:
            lab = rng.choice(LABELS)
            text = f"{text} {lab}" if rng.random() < 0.5 else f"{lab} {text}"
        return text, digits, 1
    k = rng.random()
    lab = rng.choice(LABELS)
    if k < 0.25:
        text = f"{lab} {to_fa(''.join(str(rng.randint(0, 9)) for _ in range(10)))}"
    elif k < 0.4:
        text = to_fa("".join(str(rng.randint(0, 9)) for _ in range(10)))
    elif k < 0.55:
        text = to_fa("".join(str(rng.randint(0, 9)) for _ in range(rng.choice([5, 6, 7, 9, 11]))))
    elif k < 0.75:
        text = f"{lab} {''.join(rng.choice(AR_LETTERS) for _ in range(rng.randint(2, 8)))}"
    elif k < 0.92:
        text = "".join(rng.choice(AR_LETTERS + " ") for _ in range(rng.randint(6, 24)))
    else:
        text = ""
    return text, None, 0


# ---------------------------------------------------------------- rendering

def render_text(text: str, font_path: Path, size: int, rng: random.Random) -> np.ndarray:
    try:
        font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.BASIC)
    except Exception:
        font = ImageFont.load_default()
    tmp = Image.new("L", (8, 8), 255)
    d = ImageDraw.Draw(tmp)
    try:
        bbox = d.textbbox((0, 0), text or " ", font=font)
        tw, th = max(20, bbox[2] - bbox[0]), max(8, bbox[3] - bbox[1])
    except Exception:
        tw, th = max(20, size * len(text or " ")), size
    pad_x = round(th * rng.uniform(0.15, 0.4))
    pad_y = round(th * rng.uniform(0.2, 0.45))
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    img = Image.new("L", (w, h), rng.randint(165, 245))
    dr = ImageDraw.Draw(img)
    ink = rng.randint(0, 70)
    # guilloche-ish fine lines
    for _ in range(rng.randint(0, 6)):
        v = max(0, int(np.array(img).mean()) - rng.randint(5, 25))
        y0 = rng.uniform(0, h)
        dr.line([(0, y0), (w, y0 + rng.uniform(-h / 3, h / 3))], fill=v, width=1)
    dr.text((w / 2, h / 2), text or " ", font=font, fill=ink, anchor="mm")
    arr = np.array(img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BILINEAR, fillcolor=255))
    return arr


def load_glyph_variants(glyph_dir: Path):
    """Load per-variant glyph JSONs (spec 9.2 output). Returns {variant: data}."""
    out = {}
    if not glyph_dir.exists():
        return out
    for js in sorted(glyph_dir.glob("glyphs_*.json")):
        try:
            out[js.stem] = json.loads(js.read_text())
        except Exception:
            continue
    return out


def render_with_glyphs(digits_or_text: str, glyphs: dict, rng: random.Random,
                       height: int = 40, slash: str = "/") -> np.ndarray:
    """Compose a date line from user glyph bitmaps (spec 9.3).

    digits_or_text: e.g. '1375/05/12' (slashes kept) or negative strings.
    glyphs: {'0': {'png': path, 'bitmap': [...], 'w':.., 'h':.., 'baseline':..}, ...}
    Keeps the dot-zero's real size ratio; jitter spacing +-8%, per-glyph
    scale +-3%, rotation +-2deg, stroke dilate/erode +-1px + bleed blur.
    """
    items = []
    for ch in digits_or_text:
        key = ch if ch in glyphs else ("slash" if ch == "/" else None)
        if key is None or key not in glyphs:
            continue
        g = glyphs[key]
        png = Path(g["png"])
        if not png.exists():
            continue
        im = Image.open(png).convert("L")
        # per-glyph scale jitter
        s = rng.uniform(0.97, 1.03)
        nw, nh = max(2, round(im.width * s * height / g.get("h", im.height))), max(2, round(im.height * s * height / g.get("h", im.height)))
        im = im.resize((nw, nh), Image.BILINEAR)
        if rng.random() < 0.3:
            import cv2 as _cv
            arr = np.array(im)
            k = 1
            if rng.random() < 0.5:
                arr = _cv.dilate(arr, np.ones((k, k), np.uint8))
            else:
                arr = _cv.erode(arr, np.ones((k, k), np.uint8))
            im = Image.fromarray(arr)
        im = im.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, fillcolor=255)
        items.append(im)
    if not items:
        return np.full((height + 16, 64), 255, np.uint8)
    gap = max(2, round(height * 0.12))
    widths = [im.width for im in items]
    jgaps = [round(gap * rng.uniform(0.92, 1.08)) for _ in items]
    W = sum(widths) + sum(jgaps) + 16
    H = height + 16
    canvas = Image.new("L", (W, H), 255)
    x = 8
    for im, jg in zip(items, jgaps):
        canvas.paste(im, (x, (H - im.height) // 2 + rng.randint(-1, 1)), im if im.mode == "RGBA" else None)
        x += im.width + jg
    return np.array(canvas)


# ---------------------------------------------------------------- augmentation (spec 7)

def augment(crop: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return np.full((32, 64), 255, np.uint8)
    h, w = crop.shape[:2]
    out = crop.copy()
    # low-resolution realism for the dot (spec 9.3): dot may be 2-3 px wide
    if rng.random() < 0.6 * strength:
        s = rng.uniform(0.3, 0.9)
        small = cv2.resize(out, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5 * strength:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    # illumination gradient + glare
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = (rng.uniform(-30, 30) * xx / max(w, 1) + rng.uniform(-20, 20) * yy / max(h, 1))
        out = np.clip(out.astype(np.float32) + grad * strength, 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rad = rng.uniform(0.2, 0.7) * max(w, h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rad / 2) ** 2))
        out = np.clip(out.astype(np.float32) + mask * rng.uniform(40, 120), 0, 255).astype(np.uint8)
    # rotation +-3deg, shear, perspective (spec 7)
    if rng.random() < 0.7:
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 2] += rng.uniform(-1, 1)
        # shear
        shx = rng.uniform(-0.12, 0.12)
        M = np.array([[1, shx, -shx * h / 2], [0, 1, 0]], dtype=np.float32) @ np.vstack([M, [0, 0, 1]])
        M = M[:2]
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.3 * strength:
        # mild perspective (photographed-card double resampling)
        j = lambda v: rng.uniform(-v, v)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[j(w * 0.03), j(h * 0.06)] for _ in range(4)])
        Mp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, Mp, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    # contrast / gamma
    alpha = rng.uniform(0.6, 1.3)
    gamma = rng.uniform(0.7, 1.4)
    out = np.clip((((out.astype(np.float32) / 255) ** gamma) * 255 * alpha), 0, 255).astype(np.uint8)
    # noise
    sigma = rng.uniform(0, 8) * strength
    if sigma > 0.3:
        out = np.clip(out.astype(np.float32) + rng_np(rng).normal(0, sigma, out.shape), 0, 255).astype(np.uint8)
    # JPEG
    if rng.random() < 0.5 * strength:
        q = rng.randint(25, 90)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def rng_np(rng: random.Random):
    return np.random.default_rng(rng.randint(0, 2 ** 31 - 1))


def detector_crop_margins(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mimic engine blob boxes: tight ink bbox + random margins
    (0.05-0.8h horizontally, 0.1-0.6h vertically)."""
    ys, xs = np.where(crop < 200)
    if len(xs) == 0:
        return crop
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h = max(8, y1 - y0)
    mx = lambda: rng.uniform(0.05, 0.8) * h
    my = lambda: rng.uniform(0.1, 0.6) * h
    nx0 = max(0, int(x0 - mx()))
    nx1 = min(crop.shape[1], int(x1 + mx()))
    ny0 = max(0, int(y0 - my()))
    ny1 = min(crop.shape[0], int(y1 + my()))
    if nx1 <= nx0 or ny1 <= ny0:
        return crop
    return crop[ny0:ny1, nx0:nx1]


def render_row(font_path: Path, rng: random.Random, kind: str = "date") -> tuple[np.ndarray, str | None, int]:
    """Full card-row image as the engine's CLOSE-merged blobs produce it
    (spec 4.3b boxes are full label+value rows, 600-1100 px wide at card
    scale, NOT tight crops). Returns (crop, digits_or_None, isDate).

    kind=date  -> label + YYYY/MM/DD date (isDate=1, digits=date)
    kind=id    -> label + 10-digit national id (isDate=0)
    kind=words -> label + Persian words / serial (isDate=0)
    """
    W = rng.randint(800, 1000)
    H = rng.randint(50, 65)
    size = rng.randint(30, 38)
    img = Image.new("L", (W, H), rng.randint(200, 245))
    dr = ImageDraw.Draw(img)
    # guilloche arcs behind the text (like the card background)
    for i in range(rng.randint(8, 20)):
        y0 = rng.uniform(0, H)
        dr.arc([0, y0 - 20, W, y0 + 20], rng.randint(180, 220), rng.randint(300, 360),
               fill=max(0, int(np.array(img).mean()) - rng.randint(5, 25)))
    try:
        font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.BASIC)
    except Exception:
        font = ImageFont.load_default()
    ink = rng.randint(0, 70)
    # Date sits right-of-center with small jitter, like engine blobs
    # (padding is only 0.25h/0.3h, so geometry is stable).
    right = W - rng.randint(30, 50)
    ymid = H / 2 + rng.uniform(-2, 2)
    if kind == "date":
        digits = biased_digits(rng)
        value = to_fa(f"{digits[:4]}/{digits[4:6]}/{digits[6:]}")
        label = rng.choice(LABELS)
        is_date = 1
    elif kind == "id":
        digits, value, label, is_date = None, to_fa("".join(str(rng.randint(0, 9)) for _ in range(10))), rng.choice(LABELS), 0
    else:
        n = rng.randint(2, 6)
        digits, value, label, is_date = None, "".join(rng.choice(AR_LETTERS) for _ in range(n)), rng.choice(LABELS), 0
    try:
        lw = dr.textlength(label, font=font)
    except Exception:
        lw = size * len(label) * 0.6
    dr.text((right, ymid), label, font=font, fill=ink, anchor="rm")
    try:
        vw = dr.textlength(value, font=font)
    except Exception:
        vw = size * len(value) * 0.6
    dr.text((right - lw - size * 0.6, ymid + rng.uniform(-2, 2)), value, font=font, fill=ink, anchor="rm")
    # occasional neighbour-field fragment at the row edge (detector overlap)
    if rng.random() < 0.3:
        frag = to_fa(str(rng.randint(100, 9999)))
        dr.text((rng.choice([8, W - 60]), ymid), frag, font=font, fill=ink, anchor="lm")
    arr = np.array(img)
    if rng.random() < 0.5:
        ang = rng.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        arr = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    return arr, digits, is_date


def contact_sheet(fonts, out: Path, n: int = 40) -> None:
    rng = random.Random(7)
    rows = []
    for _ in range(n):
        text, _, _ = gen_label(rng)
        rows.append(render_text(text or " ", rng.choice(fonts), rng.randint(26, 48), rng))
    H = max(r.shape[0] for r in rows)
    W = max(r.shape[1] for r in rows)
    sheet = np.full((H * n + 8 * (n - 1), W, 3), 255, np.uint8)
    y = 0
    for r in rows:
        sheet[y:y + r.shape[0], :r.shape[1], :] = r[:, :, None]
        y += H + 8
    cv2.imwrite(str(out), sheet)
