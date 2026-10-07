#!/usr/bin/env python3
"""Collect TTF fonts that contain Persian digits U+06F0..U+06F9 (spec 7).

Scans npm font packages (see IMPLEMENTATION_SPEC.md section 14) plus the
repo ./fonts folder, keeps only fonts whose cmap contains U+06F0..U+06F9
with 10 distinct glyphs. Copies kept fonts to ./fonts/.

Usage:
    python3 collect_fonts.py [--npm /tmp/fonts_npm] [--out ./fonts]
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    from fontTools.ttLib import TTFont
except ImportError:
    sys.exit("fonttools is required: pip install fonttools brotli")

PERSIAN_DIGITS = list(range(0x06F0, 0x06FA))

# Decorative families that render tofu / Arabic-shaped digits (spec 7).
EXCLUDE = {
    "lalezar",
    "kufam",
    "reem-kufi",
    "rubik",
    "cairo",
    "changa",
    "mottos",
}


def font_has_persian_digits(path: Path) -> bool:
    try:
        f = TTFont(str(path), lazy=True)
        cmap = f.getBestCmap()
        glyphs = {cmap.get(cp) for cp in PERSIAN_DIGITS}
        if any(g is None for g in glyphs):
            return False
        return len(glyphs) == 10
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npm", default="/tmp/fonts_npm")
    ap.add_argument("--out", default="./fonts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    for root in [Path(args.npm), Path("../fonts"), Path("./fonts")]:
        if root.exists():
            candidates += list(root.rglob("*.ttf")) + list(root.rglob("*.otf"))
    # dedupe
    seen = set()
    uniq = []
    for c in candidates:
        if c.resolve() not in seen:
            seen.add(c.resolve())
            uniq.append(c)

    kept = 0
    for src in uniq:
        name = src.stem.lower()
        if any(x in name for x in EXCLUDE):
            print(f"skip (excluded): {src}")
            continue
        if not font_has_persian_digits(src):
            print(f"skip (no fa digits): {src}")
            continue
        dst = out / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        kept += 1
        print(f"keep: {dst}")

    print(f"kept {kept} fonts in {out}")
    print("Next: eyeball a contact sheet of rendered digits per font before training.")


if __name__ == "__main__":
    main()
