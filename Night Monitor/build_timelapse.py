#!/usr/bin/env python3
"""Build timelapse MP4 from a night folder of TIFFs."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2


def seq_of(path: Path) -> int:
    m = re.search(r"seq(\d+)of", path.name)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build timelapse from TIFF sequence")
    ap.add_argument("--src", type=Path, required=True, help="Night folder (e.g. 8.3)")
    ap.add_argument("--out", type=Path, default=None, help="Default: <src>/Night_Monitor_timelapse.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--max-seq", type=int, default=None, help="Include frames with seq <= this")
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else here / args.src
    out = args.out or (src / "Night_Monitor_timelapse.mp4")
    if not out.is_absolute():
        out = here / out

    frames = sorted(
        (p for p in src.glob("*.tiff") if not p.name.startswith("._")),
        key=seq_of,
    )
    if args.max_seq is not None:
        frames = [p for p in frames if seq_of(p) <= args.max_seq]
    if not frames:
        raise SystemExit(f"no TIFFs in {src}")

    print(f"{len(frames)} frames", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="timelapse_"))
    try:
        for i, p in enumerate(frames):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"read fail: {p}")
            if args.scale != 1.0:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(tmp / "%05d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(args.crf),
            "-movflags", "+faststart",
            str(out),
        ]
        print("encoding...", flush=True)
        subprocess.run(cmd, check=True)
        print(f"DONE -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
