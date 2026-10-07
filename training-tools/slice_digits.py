#!/usr/bin/env python3
"""Slice verified-geometry shape exemplars from a real card date strip.

Unlike glyphs.py (which needs a full 0-9 sheet with KNOWN order), this
tool extracts shapes whose GEOMETRY is self-verifying, without claiming
digit identity:
  hollow  : ring with exactly 1 hole, short (doc's hollow-circle ۰ structure)
  slash   : tall thin diagonal (taller than digits, no holes, band-shifted)
  narrow  : tall thin stroke, no holes (۱-like; no other Persian digit is one)
  looped  : tall digit-sized blob WITH a hole (looped bowl, e.g. ۶-like)

Each exemplar is stored WITHOUT a digit label (identity unverified) so it
can never poison label training; it serves geometry calibration (spec 9.2
assertions: zero height ratio < 0.6) and shape-gate template matching.
A contact sheet is always written for human review (spec 9.2 human gate).

Usage:
    python3 slice_digits.py <strip.png> --out ./glyphs/card_shapes
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("--out", default="./glyphs/card_shapes")
    ap.add_argument("--min-area", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"cannot read {args.strip}")
    if img.mean() < 110:
        img = 255 - img
    # focus on the ink band (drop quiet top/bottom margins)
    row_ink = (img < 200).mean(axis=1)
    ys = np.where(row_ink > 0.01)[0]
    band = img[ys.min():ys.max() + 1, :] if len(ys) else img

    _, b = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < args.min_area:
            continue
        comp = (lab == i).astype("uint8") * 255
        cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = sum(1 for k in range(len(hier[0])) if hier[0][k][3] != -1) if hier is not None else 0
        comps.append(dict(x=int(x), y=int(y), w=int(w), h=int(h), area=int(area), holes=holes))
    comps.sort(key=lambda c: c["x"])
    digit_h = sorted([c["h"] for c in comps if c["holes"] == 0 and c["h"] > 40], reverse=True)
    median_h = float(np.median(digit_h)) if digit_h else 70.0
    digit_w = sorted([c["w"] for c in comps if c["holes"] == 0 and c["h"] > 40])
    median_w = float(np.median(digit_w)) if digit_w else 44.0

    shapes = []
    for c in comps:
        hr = c["h"] / median_h
        shape = None
        if c["holes"] == 1 and hr < 0.62 and c["w"] < median_h:
            shape = "hollow"  # doc's hollow-circle ۰ structure
        elif (c["holes"] == 0 and c["h"] >= median_h * 0.9 and c["w"] / c["h"] > 0.42
                and c["w"] < median_w * 1.3):
            # tall diagonal separator (slash); merged digit pairs are wider
            # and are excluded here, then verified by band shift below
            shape = "slash?"
        elif c["holes"] == 0 and c["w"] / max(1, c["h"]) < 0.42 and c["h"] >= median_h * 0.85:
            shape = "narrow"  # ۱-like bare stroke
        elif c["holes"] >= 1 and c["h"] >= median_h * 0.85:
            shape = "looped"  # looped bowl digit (۶-like), identity unverified
        c["height_ratio"] = round(hr, 3)
        c["shape"] = shape
        if shape:
            shapes.append(c)

    # slash verification: ink concentrated in top/bottom bands with lateral shift
    for c in [s for s in shapes if s["shape"] == "slash?"]:
        crop = b[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        band_h = max(2, c["h"] // 3)
        top = crop[:band_h, :].mean(axis=0)
        bot = crop[-band_h:, :].mean(axis=0)
        top_c = float(np.average(np.arange(c["w"]), weights=top + 1e-9))
        bot_c = float(np.average(np.arange(c["w"]), weights=bot + 1e-9))
        c["band_shift"] = round(abs(top_c - bot_c) / max(1, c["w"]), 3)
        c["shape"] = "slash" if c["band_shift"] > 0.25 else None

    shapes = [s for s in shapes if s["shape"]]
    data = {"source": str(Path(args.strip).name), "median_h": round(median_h, 1), "shapes": []}
    thumbs = []
    for i, c in enumerate(shapes):
        crop = band[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        fn = f"{c['shape']}_{i:02d}.png"
        cv2.imwrite(str(out / fn), crop)
        data["shapes"].append({
            "id": i, "shape": c["shape"], "png": fn,
            "w": c["w"], "h": c["h"], "holes": c["holes"],
            "height_ratio": c["height_ratio"],
            "identity": None,
        })
        thumbs.append(crop)

    # contact sheet (human gate)
    if thumbs:
        H = max(t.shape[0] for t in thumbs) + 30
        W = sum(t.shape[1] for t in thumbs) + 10 * len(thumbs)
        sheet = np.full((H, W, 3), 255, np.uint8)
        x = 0
        for t, c in zip(thumbs, [s for s in shapes]):
            sheet[:t.shape[0], x:x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(sheet, c["shape"], (x + 2, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            x += t.shape[1] + 10
        cv2.imwrite(str(out / "contact.png"), sheet)
    (out / "card_shapes.json").write_text(json.dumps(data, indent=1))
    counts = {}
    for s in data["shapes"]:
        counts[s["shape"]] = counts.get(s["shape"], 0) + 1
    print(f"sliced {len(data['shapes'])} exemplars {counts} -> {out}")
    print("median_h:", data["median_h"])
    print("HUMAN GATE: inspect contact.png before using these anywhere.")


if __name__ == "__main__":
    main()
