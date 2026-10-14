#!/usr/bin/env python3
"""
Progressive star-trail video: frame 1 = first exposure, last frame = full trail.

Each video frame is the lighten (cv2.max) stack of all source frames up to that
point — stars visibly grow into arcs over the night.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from stack_star_trail import select_frames, seq_of


def build_accumulation_frames(frames: list[Path], scale: float = 1.0) -> list[np.ndarray]:
    acc = None
    out: list[np.ndarray] = []
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"read fail: {p}")
        if scale != 1.0:
            img = cv2.resize(
                img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        acc = img.copy() if acc is None else cv2.max(acc, img)
        out.append(acc.copy())
        if (i + 1) % 25 == 0 or i + 1 == len(frames):
            print(f"  built {i + 1}/{len(frames)} accumulation frames", flush=True)
    return out


def encode_video(
    frames: list[np.ndarray],
    out_path: Path,
    fps: float,
    crf: int,
) -> None:
    if not frames:
        raise RuntimeError("no frames to encode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp = Path(tempfile.mkdtemp(prefix="star_trail_vid_"))
    try:
        for i, img in enumerate(frames):
            cv2.imwrite(str(tmp / f"{i:05d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp / "%05d.jpg"),
            "-vf",
            f"scale={w}:{h}:flags=lanczos",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        print("encoding...", flush=True)
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build progressive star-trail video")
    ap.add_argument("--src", type=Path, required=True, help="Night folder (e.g. 7.19)")
    ap.add_argument("--dawn-seq", type=int, default=172)
    ap.add_argument("--bright-mean", type=float, default=48.0)
    ap.add_argument("--out", type=Path, default=None, help="Default: <src>/Star_Trail_Build.mp4")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--hold-last", type=float, default=1.0, help="Extra seconds on final frame")
    ap.add_argument("--scale", type=float, default=0.5, help="Output scale (0.5 = half res)")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better)")
    ap.add_argument("--max-seq", type=int, default=None, help="Include frames with seq <= this")
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else here / args.src
    out = args.out or (src / "Star_Trail_Build.mp4")
    if not out.is_absolute():
        out = here / out

    files = sorted(p for p in src.glob("*.tiff") if not p.name.startswith("._"))
    if args.max_seq is not None:
        files = [p for p in files if seq_of(p) <= args.max_seq]
    selected = select_frames(files, args.dawn_seq, args.bright_mean)
    if not selected:
        raise SystemExit("no frames selected")

    print(f"selected {len(selected)} frames", flush=True)
    print(f"first: {selected[0].name}", flush=True)
    print(f"last:  {selected[-1].name}", flush=True)

    acc_frames = build_accumulation_frames(selected, scale=args.scale)
    if args.hold_last > 0:
        hold_n = max(1, int(round(args.hold_last * args.fps)))
        acc_frames.extend([acc_frames[-1]] * hold_n)
    encode_video(acc_frames, out, args.fps, crf=args.crf)
    print(f"DONE -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
