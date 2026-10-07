#!/usr/bin/env python3
"""Glyph extraction from user-supplied glyph sheets (spec 9.2).

Input (spec 9.1A): glyph_sheet_<variant>.png — black ink on white, SINGLE ROW
in exact order left to right: 0 1 2 3 4 5 6 7 8 9 /
glyph height >= 150px, >= 20px gaps, common baseline, NOT tight-cropped
vertically (dot-zero keeps its real position/size).

Steps:
1. Binarise (Otsu), connected components, cluster into glyphs by x-overlap/gap
   (digits like 4 may have several components; 0 is one small component).
   Expect 11 clusters; else stop and show a labelled contact sheet.
2. HUMAN GATE: render contact sheet with assigned label under each glyph;
   the user MUST confirm mapping 0..9,/ before any training.
3. Store per glyph: grayscale bitmap (ink alpha), advance box (x-range),
   baseline offset + height relative to median digit height. Verify:
   zero height ratio < 0.6, five height ratio > 0.8 (unit-test assertions).
4. Per-variant JSON (glyphs_<variant>.json) + PNGs under training-tools/glyphs/.

Fallback (spec T2): if only ~10 labelled date crops are available, segment
digits by connected components and label by known dates (warn: coverage may
be incomplete; every digit 0-9 must appear).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "slash"]
EXPECTED = 11


def cluster_components(boxes, gap_frac=0.5):
    """Cluster component boxes into glyphs by x-overlap/gap."""
    boxes = sorted(boxes, key=lambda b: b[0])
    if not boxes:
        return []
    med_w = sorted(b[2] for b in boxes)[len(boxes) // 2]
    clusters = [[boxes[0]]]
    for b in boxes[1:]:
        prev = clusters[-1]
        prev_right = max(p[0] + p[2] for p in prev)
        gap = b[0] - prev_right
        if gap > max(4, med_w * gap_frac):
            clusters.append([b])
        else:
            prev.append(b)
    # merge clusters into glyph boxes
    glyphs = []
    for cl in clusters:
        x0 = min(b[0] for b in cl)
        y0 = min(b[1] for b in cl)
        x1 = max(b[0] + b[2] for b in cl)
        y1 = max(b[1] + b[3] for b in cl)
        glyphs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return glyphs


def extract(sheet_path: Path, out_dir: Path, variant: str):
    img = cv2.imread(str(sheet_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {sheet_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8:
            continue
        boxes.append((x, y, w, h))
    glyphs = cluster_components(boxes)
    print(f"components={len(boxes)} clusters={len(glyphs)} (expect {EXPECTED})")

    out_dir.mkdir(parents=True, exist_ok=True)
    # contact sheet for the HUMAN GATE (spec 9.2 step 2)
    sheet_h = 220
    cols = []
    for g in glyphs:
        crop = img[g["y"]:g["y"] + g["h"], g["x"]:g["x"] + g["w"]]
        s = sheet_h / max(1, crop.shape[0])
        cols.append(cv2.resize(crop, (max(8, round(crop.shape[1] * s)), sheet_h)))
    contact = np.full((sheet_h + 30, sum(c.shape[1] for c in cols) + 10 * len(cols), 3), 255, np.uint8)
    x = 0
    for i, c in enumerate(cols):
        contact[:sheet_h, x:x + c.shape[1]] = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        label = LABELS[i] if i < len(LABELS) else "?"
        cv2.putText(contact, label, (x + 4, sheet_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        x += c.shape[1] + 10
    contact_path = out_dir / f"contact_{variant}.png"
    cv2.imwrite(str(contact_path), contact)
    print(f"contact sheet: {contact_path}")

    if len(glyphs) != EXPECTED:
        sys.exit(f"STOP: expected {EXPECTED} glyph clusters, got {len(glyphs)}. "
                 f"Inspect {contact_path}, fix the sheet, re-run.")

    # assign labels left-to-right, store bitmaps
    median_h = sorted(g["h"] for g in glyphs[:10])[5]
    data = {"variant": variant, "median_h": median_h, "digits": {}}
    for label, g in zip(LABELS, glyphs):
        crop = img[g["y"]:g["y"] + g["h"], g["x"]:g["x"] + g["w"]]
        png = out_dir / f"glyph_{variant}_{label}.png"
        cv2.imwrite(str(png), crop)
        data["digits"][label] = {
            "png": str(png),
            "w": g["w"], "h": g["h"],
            "height_ratio": g["h"] / max(1, median_h),
            "y": g["y"],
        }
    zr = data["digits"]["0"]["height_ratio"]
    fr = data["digits"]["5"]["height_ratio"]
    print(f"zero height ratio={zr:.2f} (must be < 0.6); five height ratio={fr:.2f} (must be > 0.8)")
    assert zr < 0.6, f"zero glyph too tall ({zr:.2f}); sheet may be wrong"
    assert fr > 0.8, f"five glyph too short ({fr:.2f}); sheet may be wrong"

    js = out_dir / f"glyphs_{variant}.json"
    js.write_text(json.dumps(data, indent=1))
    print(f"wrote {js}")
    print("HUMAN GATE: open the contact sheet and confirm 0..9,/ mapping before training.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--out", default="./glyphs")
    args = ap.parse_args()
    extract(Path(args.sheet), Path(args.out), args.variant)


if __name__ == "__main__":
    main()
