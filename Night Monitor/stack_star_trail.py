#!/usr/bin/env python3
"""
Star trail stack via lighten blend (per-pixel maximum).

This is the algorithm that produced Star_Trail.tiff / Star_Trail.jpg
from the Night Monitor sequence (170 frames, seq0001–seq0171).

Method:
  - Read frames in sequence order from a folder of TIFFs
  - Drop dawn / bright frames (seq >= dawn_seq, or mean gray >= bright_mean)
  - Accumulate with cv2.max (classic lighten composite)

Example (night folder layout):
  python3 stack_star_trail.py --src 7.19
  python3 stack_star_trail.py --src 7.20 --dawn-seq 180 --bright-mean 48
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def seq_of(path: Path) -> int:
    m = re.search(r"seq(\d+)of", path.name)
    return int(m.group(1)) if m else 10**9


def select_frames(
    files: list[Path],
    dawn_seq: int,
    bright_mean: float,
) -> list[Path]:
    """Drop dawn and abnormally bright frames."""
    selected: list[Path] = []
    for p in files:
        if seq_of(p) >= dawn_seq:
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if float(g.mean()) >= bright_mean:
            continue
        selected.append(p)
    return selected


def stack_lighten(frames: list[Path]) -> np.ndarray:
    """Classic star-trail lighten: each pixel keeps the brightest value seen."""
    acc = None
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"read fail: {p}")
        if acc is None:
            acc = img.copy()
        else:
            acc = cv2.max(acc, img)
        if (i + 1) % 25 == 0 or i + 1 == len(frames):
            print(f"  stacked {i + 1}/{len(frames)}", flush=True)
    if acc is None:
        raise RuntimeError("no frames stacked")
    return acc


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Stack star trails with lighten (cv2.max)")
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Night folder of input TIFFs (e.g. 7.19)",
    )
    ap.add_argument(
        "--dawn-seq",
        type=int,
        default=172,
        help="Drop frames with seq >= this (default: 172)",
    )
    ap.add_argument(
        "--bright-mean",
        type=float,
        default=48.0,
        help="Drop frames whose mean gray >= this (default: 48)",
    )
    ap.add_argument(
        "--out-tiff",
        type=Path,
        default=None,
        help="Default: <src>/Star_Trail.tiff",
    )
    ap.add_argument(
        "--out-jpg",
        type=Path,
        default=None,
        help="Default: <src>/Star_Trail.jpg",
    )
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
    )
    ap.add_argument(
        "--max-seq",
        type=int,
        default=None,
        help="Include frames with seq <= this",
    )
    args = ap.parse_args()
    src = args.src if args.src.is_absolute() else here / args.src
    out_tiff = args.out_tiff or (src / "Star_Trail.tiff")
    out_jpg = args.out_jpg or (src / "Star_Trail.jpg")
    if not out_tiff.is_absolute():
        out_tiff = here / out_tiff
    if not out_jpg.is_absolute():
        out_jpg = here / out_jpg

    files = sorted(
        p for p in src.glob("*.tiff") if not p.name.startswith("._")
    )
    if args.max_seq is not None:
        files = [p for p in files if seq_of(p) <= args.max_seq]
    if not files:
        raise SystemExit(f"no TIFFs in {src}")

    selected = select_frames(files, args.dawn_seq, args.bright_mean)
    if not selected:
        raise SystemExit("no frames selected after dawn/brightness filters")

    print(f"selected {len(selected)} / {len(files)} frames", flush=True)
    print(f"first: {selected[0].name}", flush=True)
    print(f"last:  {selected[-1].name}", flush=True)

    acc = stack_lighten(selected)

    out_tiff.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_tiff), acc)
    cv2.imwrite(
        str(out_jpg),
        acc,
        [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality],
    )
    print(
        f"wrote {out_tiff} ({out_tiff.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    print(
        f"wrote {out_jpg} ({out_jpg.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )
    print(f"acc mean={acc.mean():.2f} max={acc.max()}", flush=True)


if __name__ == "__main__":
    main()
