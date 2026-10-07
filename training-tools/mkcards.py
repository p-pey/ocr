#!/usr/bin/env python3
"""Synthetic 'photographed card' scenes for end-to-end engine tests (spec 7/11).

Produces full synthetic photographed cards (perspective, rotation, table,
blur, glare, JPEG; birth date near the middle, national id below, expiry at
the bottom, label words next to values) + labels.json with ground truth.

Usage:
    python3 mkcards.py <N> <out_dir> <seen|held|glyphs> [seed]
    python3 mkcards.py 300 /tmp/cards_seen seen 1
    python3 mkcards.py 300 /tmp/cards_unseen held 2
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from gen import (days_in_month, load_fonts, load_glyph_variants, render_with_glyphs,
                 split_fonts, to_fa)

CARD_W, CARD_H = 1200, 756


def pick_font(fonts, rng):
    return rng.choice(fonts)


def draw_field(draw: ImageDraw.ImageDraw, label: str, value: str, x: int, y: int,
               size: int, font, color=(16, 32, 48)):
    draw.text((x, y), label, font=font, fill=color, anchor="ra")
    try:
        lw = draw.textlength(label, font=font)
    except Exception:
        lw = size * len(label) * 0.6
    draw.text((x - lw - size * 0.6, y), value, font=font, fill=(10, 20, 32), anchor="ra")


def render_card(fonts, rng: random.Random, glyph_data=None):
    y = rng.randint(1300, 1415)
    m = rng.randint(1, 12)
    d = rng.randint(1, days_in_month(y, m))
    birth = f"{y:04d}{m:02d}{d:02d}"
    ey = rng.randint(min(y + 1, 1415), 1415)
    expiry = f"{ey:04d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    nid = "".join(str(rng.randint(0, 9)) for _ in range(10))

    try:
        font = ImageFont.truetype(str(pick_font(fonts, rng)), 34, layout_engine=ImageFont.Layout.BASIC)
        font_big = ImageFont.truetype(str(pick_font(fonts, rng)), 40, layout_engine=ImageFont.Layout.BASIC)
    except Exception:
        font = ImageFont.load_default()
        font_big = font

    card = Image.new("RGB", (CARD_W, CARD_H), (238, 242, 245))
    dr = ImageDraw.Draw(card)
    dr.rectangle([4, 4, CARD_W - 8, CARD_H - 8], outline=(159, 180, 196), width=3)
    for i in range(26):  # guilloche-ish background
        y0 = 20 * i
        dr.arc([0, y0 - 20, CARD_W, y0 + 20], 200, 340, fill=(120 + (i % 3) * 10, 150, 175))
    dr.text((CARD_W / 2, 64), "جمهوری اسلامی ایران", font=font_big, fill=(18, 58, 92), anchor="mm")
    dr.text((CARD_W / 2, 112), "کارت ملی", font=font, fill=(18, 58, 92), anchor="mm")
    dr.rectangle([CARD_W - 250, 150, CARD_W - 60, 400], fill=(200, 210, 218), outline=(127, 147, 164), width=2)
    right = CARD_W - 300
    draw_field(dr, "نام:", "علی", right, 190, 34, font)
    draw_field(dr, "نام خانوادگی:", "رضایی", right, 250, 34, font)
    if glyph_data is not None:
        # paste glyph-composed date lines (user shapes) for glyph scenes
        for txt, yy in [(f"{birth[:4]}/{birth[4:6]}/{birth[6:]}", 470),
                        (f"{nid[:5]} {nid[5:]}", 560)]:
            line = render_with_glyphs(to_fa(txt), glyph_data, rng, height=34)
            card.paste(Image.fromarray(np.stack([line] * 3, -1)), (right - line.shape[1] - 200, yy - 20))
        draw_field(dr, "تاریخ تولد:", "", right, 470, 34, font)
    else:
        draw_field(dr, "تاریخ شناسنامه:", "12345678", right, 370, 34, font)
        draw_field(dr, "تاریخ تولد:", to_fa(f"{birth[:4]}/{birth[4:6]}/{birth[6:]}"), right, 470, 34, font)
        draw_field(dr, "شماره ملی:", to_fa(nid), right, 560, 34, font)
    dr.text((40, CARD_H - 40), f"تاریخ انقضا: {to_fa(f'{expiry[:4]}/{expiry[4:6]}/{expiry[6:]}')}",
            font=font, fill=(51, 80, 106))
    return card, birth


def photograph(card: Image.Image, rng: random.Random) -> np.ndarray:
    """Perspective + rotation + table + blur + glare + JPEG (spec 7)."""
    W, H = 1500, 1000
    desk = Image.new("RGB", (W, H), (58, 64, 72))
    dr = ImageDraw.Draw(desk)
    for i in range(40):
        dr.line([(0, (i * 37) % H), (W, (i * 61) % H)], fill=(180 - i * 2,) * 3)
    cw, ch = card.size
    # perspective warp of the card
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    j = lambda v: rng.uniform(-v, v)
    dst = src + np.float32([[j(30), j(30)], [j(30), j(30)], [j(30), j(30)], [j(30), j(30)]])
    M = cv2.getPerspectiveTransform(src, dst + np.float32([[(W - cw) / 2, (H - ch) / 2]] * 4))
    warped = cv2.warpPerspective(np.array(card), M, (W, H), borderValue=(58, 64, 72))
    if rng.random() < 0.15:
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    if rng.random() < 0.5:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)
    if rng.random() < 0.3:  # glare
        h, w = warped.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mask = np.exp(-((xx - w / 2) ** 2 + (yy - h / 3) ** 2) / (2 * (w / 3) ** 2))
        warped = np.clip(warped.astype(np.float32) + mask[:, :, None] * 60, 0, 255).astype(np.uint8)
    q = rng.randint(30, 90)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else warped


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("mode")
    ap.add_argument("seed", nargs="?", type=int, default=1)
    ap.add_argument("--fonts", default="./fonts")
    a = ap.parse_args()
    N, out_dir, mode = a.n, Path(a.out), a.mode
    seed = a.seed
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    from pathlib import Path as P
    fonts = load_fonts(P(a.fonts))
    train_fonts, held_fonts = split_fonts(fonts)
    pool = held_fonts if mode == "held" and held_fonts else train_fonts
    glyph_data = None
    if mode == "glyphs":
        gv = load_glyph_variants(P("./glyphs"))
        if gv:
            glyph_data = next(iter(gv.values())).get("digits", next(iter(gv.values())))
    print(f"scene fonts: {len(pool)} (mode={mode})")

    labels = []
    for i in range(N):
        card, birth = render_card(pool, rng, glyph_data)
        photo = photograph(card, rng)
        fn = f"card_{i:04d}.jpg"
        cv2.imwrite(str(out_dir / fn), photo)
        labels.append({"file": fn, "birth": birth})
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {N} scenes to {out_dir}")


if __name__ == "__main__":
    main()
