"""Half Hour image library helpers for Analemma."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np

HALF_HOUR_DIR = Path(__file__).resolve().parent.parent
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4})\.tiff$")

# Base plate window: 18:00–21:00 local (user picks visually; no auto-ranking).
TWILIGHT_TIMES = ("1800", "1830", "1900", "1930", "2000", "2030", "2100")
TWILIGHT_TIME_LABEL = "18:00–21:00"


@dataclass(frozen=True)
class FrameInfo:
    date: str
    time: str
    filename: str
    path: Path

    @property
    def time_label(self) -> str:
        return f"{self.time[:2]}:{self.time[2:]}"


def parse_filename(name: str) -> tuple[str, str] | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def list_frames(data_dir: Path | None = None) -> list[FrameInfo]:
    root = data_dir or HALF_HOUR_DIR
    out: list[FrameInfo] = []
    for path in sorted(root.glob("*.tiff")):
        if path.name.startswith("._"):
            continue
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        date, time = parsed
        out.append(FrameInfo(date=date, time=time, filename=path.name, path=path))
    return out


def frames_at_standard_time(est_hhmm: str, data_dir: Path | None = None) -> list[FrameInfo]:
    """One frame per day at a fixed EST clock time (the filename is an hour later under DST)."""
    from sun import standard_to_filename

    est_hhmm = est_hhmm.replace(":", "")
    frames = list_frames(data_dir)
    by_name = {f.filename: f for f in frames}
    out = []
    for day in sorted({f.date for f in frames}):
        f = by_name.get(standard_to_filename(date.fromisoformat(day), est_hhmm))
        if f:
            out.append(f)
    return out


def twilight_candidates(data_dir: Path | None = None, limit: int | None = None) -> list[dict]:
    """All frames in the base-plate time window — no scoring, user chooses."""
    root = data_dir or HALF_HOUR_DIR
    out: list[dict] = []
    for path in sorted(root.glob("*.tiff")):
        if path.name.startswith("._"):
            continue
        parsed = parse_filename(path.name)
        if not parsed:
            continue
        date, time = parsed
        if time not in TWILIGHT_TIMES:
            continue
        out.append(
            {
                "filename": path.name,
                "date": date,
                "time": time,
                "timeLabel": f"{time[:2]}:{time[2:]}",
            }
        )
    if limit is not None:
        return out[:limit]
    return out


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def encode_preview(img: np.ndarray, max_width: int = 1280, quality: int = 85) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def preview_scale(path: Path, max_width: int = 1280) -> float:
    img = read_image(path)
    w = img.shape[1]
    if w <= max_width:
        return 1.0
    return max_width / w
