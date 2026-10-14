"""Draw a sun at each annotated position on a twilight base plate.

The frames only supply positions. The glare in the daytime frames is hundreds of
pixels wide, so pasting it would smear neighbouring days together; a clean drawn
sun keeps every day distinct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from library import HALF_HOUR_DIR, read_image

CORE_BGR = np.array([225.0, 248.0, 255.0], dtype=np.float32)
GLOW_BGR = np.array([60.0, 185.0, 255.0], dtype=np.float32)


def sun_sprite(radius: float) -> tuple[np.ndarray, int]:
    """Premultiplied BGR light (0-255 float) for one sun, and its half size."""
    half = int(np.ceil(radius * 4))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    d = np.hypot(xx, yy)
    core = np.clip(radius + 0.5 - d, 0.0, 1.0)
    limb = 1.0 - 0.25 * np.clip(d / max(radius, 1e-6), 0.0, 1.0) ** 2
    glow = 0.55 * np.exp(-0.5 * (d / (radius * 1.4)) ** 2)
    light = core[..., None] * limb[..., None] * CORE_BGR + (1.0 - core[..., None]) * glow[..., None] * GLOW_BGR
    return light, half


def screen_add(base: np.ndarray, light: np.ndarray, x: float, y: float, half: int) -> None:
    h, w = base.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + light.shape[1], y0 + light.shape[0]
    bx0, by0, bx1, by1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if bx0 >= bx1 or by0 >= by1:
        return
    patch = light[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0]
    region = base[by0:by1, bx0:bx1]
    base[by0:by1, bx0:bx1] = 255.0 - (255.0 - region) * (255.0 - patch) / 255.0


def render_analemma(
    base_filename: str,
    annotations: list[dict[str, Any]],
    sun_radius: float = 6.0,
    data_dir: Path | None = None,
    preview_max_width: int | None = None,
) -> np.ndarray:
    root = data_dir or HALF_HOUR_DIR
    base = read_image(root / base_filename).astype(np.float32)
    light, half = sun_sprite(sun_radius)

    for ann in annotations:
        if ann.get("skipped") or ann.get("x") is None or ann.get("y") is None:
            continue
        screen_add(base, light, float(ann["x"]), float(ann["y"]), half)

    out = np.clip(base, 0, 255).astype(np.uint8)
    if preview_max_width and out.shape[1] > preview_max_width:
        scale = preview_max_width / out.shape[1]
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


def save_render(
    img: np.ndarray,
    out_dir: Path,
    stem: str = "Analemma",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tiff = out_dir / f"{stem}.tiff"
    jpg = out_dir / f"{stem}.jpg"
    cv2.imwrite(str(tiff), img)
    cv2.imwrite(str(jpg), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return tiff, jpg
