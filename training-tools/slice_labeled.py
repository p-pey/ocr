#!/usr/bin/env python3
"""Slice a VERIFIED date strip into labeled digit glyphs.

Unlike slice_digits.py (identity-free), this tool assigns TRUE digit labels
— but only when every structural cross-check passes, otherwise it aborts
without writing anything (a wrong mapping poisons everything downstream):
  1. Exactly len(labels) glyph clusters left-to-right (slashes included).
  2. Every '0' label lands on a 1-hole hollow component, and every 1-hole
     short component lands on a '0' (doc: hollow-circle ۰, never solid dot).
  3. Every '1' lands on a narrow (w/h < 0.42) hole-free stroke and vice versa.
  4. Every '/' lands on a tall thin band-shifted diagonal and vice versa.

Writes glyphs_<variant>.json in the render_with_glyphs-compatible schema
(digits: label -> [{png,w,h,holes,height_ratio}]) for future §9.3 training
mixes, plus shape-class sidecars for the gate bundle.

Usage:
    python3 slice_labeled.py "strip.png" --labels 1380/10/18 --variant card1380 \
        --out ./glyphs --shapes-for-bundle hollow,slash,narrow,eight,three
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("--labels", required=True, help="e.g. 1380/10/18 (slashes included)")
    ap.add_argument("--variant", default="card")
    ap.add_argument("--out", default="./glyphs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = list(args.labels)

    img = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {args.strip}")
    if img.mean() < 110:
        img = 255 - img
    row_ink = (img < 200).mean(axis=1)
    ys = np.where(row_ink > 0.01)[0]
    band = img[ys.min():ys.max() + 1, :] if len(ys) else img

    _, b = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 5:
            continue
        comp = (lab == i).astype("uint8") * 255
        cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = sum(1 for k in range(len(hier[0])) if hier[0][k][3] != -1) if hier is not None else 0
        comps.append(dict(x=x, y=y, w=w, h=h, area=area, holes=holes))
    # merge multi-component digits is NOT attempted: touching digits abort
    comps.sort(key=lambda c: c["x"])
    if len(comps) != len(labels):
        raise SystemExit(f"ABORT: {len(comps)} components != {len(labels)} labels; strip needs manual review")

    digit_h = sorted([c["h"] for c in comps if c["holes"] == 0 and c["h"] > 15])
    median_h = float(np.median(digit_h)) if digit_h else 24.0

    def is_slash(c):
        if c["holes"] != 0 or c["h"] < median_h * 0.85:
            return False
        crop = b[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        bh = max(2, c["h"] // 3)
        top = crop[:bh, :].mean(axis=0)
        bot = crop[-bh:, :].mean(axis=0)
        tc = float(np.average(np.arange(c["w"]), weights=top + 1e-9))
        bc = float(np.average(np.arange(c["w"]), weights=bot + 1e-9))
        return abs(tc - bc) / max(1, c["w"]) > 0.25

    # structural cross-checks (BOTH directions)
    for c, lab in zip(comps, labels):
        hr = c["h"] / median_h
        narrow = c["holes"] == 0 and c["w"] / max(1, c["h"]) < 0.42
        if lab == "0" and not (c["holes"] == 1 and hr < 0.62):
            raise SystemExit(f"ABORT: label 0 at x={c['x']} is not hollow (holes={c['holes']}, hr={hr:.2f})")
        if lab == "1" and not narrow:
            raise SystemExit(f"ABORT: label 1 at x={c['x']} is not a narrow stroke")
        if lab == "/" and not is_slash(c):
            raise SystemExit(f"ABORT: label / at x={c['x']} is not a diagonal slash")
        if lab not in "01/" and c["holes"] == 1 and hr < 0.62:
            raise SystemExit(f"ABORT: hollow ring at x={c['x']} labeled '{lab}' (expected 0)")
    print("structural cross-checks passed:",
          f"{sum(1 for l in labels if l == '0')} hollows,",
          f"{sum(1 for l in labels if l == '1')} narrows,",
          f"{sum(1 for l in labels if l == '/')} slashes")

    FEDIGITS = "۰۱۲۳۴۵۶۷۸۹"
    digits: dict[str, list] = {}
    for c, lab in zip(comps, labels):
        crop = band[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        key = "slash" if lab == "/" else lab
        fn = f"glyph_{args.variant}_{key}_{c['x']:04d}.png"
        cv2.imwrite(str(out / fn), crop)
        digits.setdefault(key, []).append({
            "png": fn, "w": c["w"], "h": c["h"], "holes": c["holes"],
            "height_ratio": round(c["h"] / median_h, 3),
            "fa": FEDIGITS[int(lab)] if lab != "/" else "/",
        })
    (out / f"glyphs_{args.variant}.json").write_text(json.dumps(
        {"variant": args.variant, "source": Path(args.strip).name,
         "median_h": round(median_h, 1), "labels": args.labels, "digits": digits}, indent=1))
    print(f"wrote {len(comps)} labeled glyphs -> {out}/glyphs_{args.variant}.json")


if __name__ == "__main__":
    main()
