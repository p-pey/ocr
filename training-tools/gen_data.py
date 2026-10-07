#!/usr/bin/env python3
"""Writes train/val .npz datasets (spec 7 + 9.3 mix).

Usage:
    python3 gen_data.py <N> <out.npz> <seed> <train|held> [--glyphs ./glyphs] [--real ./real_crops]

Arrays: X(uint8 N,32,160) NW(newW) D(N,8) Y(isDate).
Used: 110k train, 3k val seen-fonts, 3k val unseen-fonts.
Data mix (when glyphs+real available): >=60% user-glyph, 25% other fonts, 15% real (x5 oversampled).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gen import (augment, detector_crop_margins, gen_label, load_fonts, load_glyph_variants,
                 preprocess, render_row, render_text, render_with_glyphs, split_fonts, to_fa)


def load_real(real_dir: Path):
    items = []
    csv_path = real_dir / "labels.csv"
    if not csv_path.exists():
        return items
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            fn, lab = row[0].strip(), row[1].strip()
            p = real_dir / fn
            if not p.exists():
                continue
            digits = "".join(c for c in lab if c.isdigit())
            if lab.lower() == "none":
                items.append((p, None, 0))
            elif len(digits) == 8:
                items.append((p, digits, 1))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("seed", type=int)
    ap.add_argument("mode", choices=["train", "held"])
    ap.add_argument("--glyphs", default="./glyphs")
    ap.add_argument("--real", default="./real_crops")
    ap.add_argument("--fonts", default="./fonts")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = load_fonts(Path(args.fonts))
    if not fonts:
        sys.exit(f"no fonts in {args.fonts}; run collect_fonts.py first")
    train_fonts, held_fonts = split_fonts(fonts)
    pool = held_fonts if args.mode == "held" and held_fonts else train_fonts
    print(f"fonts: {len(train_fonts)} train / {len(held_fonts)} held; using {len(pool)} for mode={args.mode}")

    glyph_variants = load_glyph_variants(Path(args.glyphs))
    glyph_data = None
    if glyph_variants:
        # Use the first variant as the main source (multi-variant held-out in train script).
        first = next(iter(glyph_variants.values()))
        glyph_data = first.get("digits", first)
        print(f"glyph renders enabled ({len(glyph_variants)} variants)")
    else:
        print("glyph renders disabled (no glyph JSONs); font renders only")

    real_items = load_real(Path(args.real)) if args.mode == "train" else []
    print(f"real crops: {len(real_items)}")

    N = args.n
    X = np.zeros((N, 32, 160), dtype=np.uint8)
    NW = np.zeros((N,), dtype=np.int16)
    D = np.full((N, 8), -1, dtype=np.int8)
    Y = np.zeros((N,), dtype=np.int8)

    for i in range(N):
        for _attempt in range(5):
            try:
                use_real = bool(real_items) and rng.random() < 0.15
                use_glyph = (not use_real) and glyph_data is not None and rng.random() < 0.60
                # Full card-row renders matching the engine's CLOSE-merged boxes
                # (spec 4.3b): wide label+value rows the CNN must read directly.
                use_row = (not use_real) and (not use_glyph) and rng.random() < 0.50
                if use_real:
                    p, digits, is_date = rng.choice(real_items)
                    crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if crop is None:
                        text, digits, is_date = gen_label(rng)
                        crop = render_text(text or " ", rng.choice(pool), rng.randint(26, 60), rng)
                    else:
                        crop = augment(crop, rng, strength=0.7)
                elif use_glyph:
                    text, digits, is_date = gen_label(rng)
                    src = text if is_date == 0 and digits is None else (to_fa(f"{digits[:4]}/{digits[4:6]}/{digits[6:]}") if digits else text)
                    line = render_with_glyphs(src or " ", glyph_data, rng)
                    crop = augment(detector_crop_margins(line, rng), rng)
                elif use_row:
                    # 70% date rows (isDate=1), 30% id/word rows (isDate=0)
                    u = rng.random()
                    kind = "date" if u < 0.70 else ("id" if u < 0.85 else "words")
                    crop, digits, is_date = render_row(rng.choice(pool), rng, kind)
                    crop = augment(crop, rng)
                else:
                    text, digits, is_date = gen_label(rng)
                    line = render_text(text or " ", rng.choice(pool), rng.randint(26, 60), rng)
                    crop = augment(detector_crop_margins(line, rng), rng)

                if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                    raise ValueError("degenerate crop")
                proc, new_w = preprocess(crop)
                # Store uint8 resized crop; NW keeps newW.
                resized_u8 = cv2.resize(crop, (new_w, 32),
                                       interpolation=cv2.INTER_AREA if crop.shape[0] > 32 else cv2.INTER_LINEAR)
                X[i, :, :new_w] = resized_u8
                NW[i] = new_w
                Y[i] = is_date
                if digits is not None:
                    D[i] = np.array([int(c) for c in digits], dtype=np.int8)
                break
            except Exception as e:
                if _attempt == 4:
                    print(f"sample {i} failed 5x ({e}); writing blank negative")
                    NW[i] = 8
                    Y[i] = 0
                continue

        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{N}")

    # Recompute standardised float inputs at load time from X+NW (keeps .npz small).
    np.savez_compressed(args.out, X=X, NW=NW, D=D, Y=Y)
    pos = (Y == 1).mean()
    print(f"wrote {args.out}: N={N} pos_rate={pos:.2f}")


if __name__ == "__main__":
    main()
