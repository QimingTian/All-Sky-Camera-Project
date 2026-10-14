#!/usr/bin/env python3
"""Timelapse of the project's clock-time frames with an analemma trail.

Each video frame is that day's real Half Hour image; every sun position from earlier
days (auto-detected or clicked in the app) stays drawn on top, so the analemma builds
up as the video plays.

    .venv/bin/python timelapse.py                 # output/Analemma_timelapse.mp4
    .venv/bin/python timelapse.py --fps 6 --sun-radius 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from library import HALF_HOUR_DIR, read_image
from render import screen_add, sun_sprite

APP_DIR = Path(__file__).resolve().parent


def put_label(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    for color, thick in (((0, 0, 0), 5), ((235, 235, 235), 2)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", type=Path, default=APP_DIR / "projects" / "current.json")
    ap.add_argument("--out", type=Path, default=APP_DIR / "output" / "Analemma_timelapse.mp4")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--sun-radius", type=float, default=None, help="default: project sunRadius")
    args = ap.parse_args()

    project = json.loads(args.project.read_text())
    frames = [f for f in project["frames"] if (HALF_HOUR_DIR / f["file"]).exists()]
    if not frames:
        raise SystemExit("project has no frames; run setup in the app first")
    hhmm = project["targetTime"]
    clock = f"{hhmm[:2]}:{hhmm[2:]} {project.get('timeBasis', 'EST')}"
    light, half = sun_sprite(args.sun_radius or float(project.get("sunRadius", 6)))

    trail: list[tuple[float, float]] = []
    tmp = Path(tempfile.mkdtemp(prefix="analemma_tl_"))
    try:
        for i, f in enumerate(frames):
            img = read_image(HALF_HOUR_DIR / f["file"]).astype(np.float32)
            for x, y in trail:
                screen_add(img, light, x, y, half)
            out = np.clip(img, 0, 255).astype(np.uint8)
            put_label(out, f"{f['date']}  {clock}", (28, 56), 1.3)
            put_label(out, f"{len(trail)} suns", (28, 100), 0.9)
            if args.scale != 1.0:
                out = cv2.resize(out, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if f.get("done") and not f.get("skipped") and f.get("x") is not None:
                trail.append((float(f["x"]), float(f["y"])))

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", str(tmp / "%05d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(args.crf),
            "-movflags", "+faststart",
            str(args.out),
        ]
        subprocess.run(cmd, check=True)
        print(f"{len(frames)} frames, {len(trail)} suns -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
