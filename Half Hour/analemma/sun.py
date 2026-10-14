"""Sun ephemeris, standard-time frame lookup, and sun detection for Half Hour frames.

Clock times for an analemma are fixed in standard time (EST, UTC-5). Half Hour filenames
are local wall-clock time, so during DST the matching file is one hour later
(09:30 EST -> YYYY-MM-DD_1030.tiff).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

HALF_HOUR_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HALF_HOUR_DIR.parent / "star_labeler"))
from asc_wcs import AscWCS  # noqa: E402

EST = timezone(timedelta(hours=-5))
LOCAL = ZoneInfo("America/New_York")


def standard_to_filename(day: date, est_hhmm: str) -> str:
    hh, mm = int(est_hhmm[:2]), int(est_hhmm[2:])
    local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=EST).astimezone(LOCAL)
    return f"{local:%Y-%m-%d_%H%M}.tiff"


@lru_cache(maxsize=1)
def wcs() -> AscWCS:
    return AscWCS()


def wcs_altaz(days: list[date], est_hhmm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ephemeris sun pixel (x, y), altitude and azimuth for each day at a fixed EST clock time."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    from astropy.time import Time

    hh, mm = int(est_hhmm[:2]), int(est_hhmm[2:])
    w = wcs()
    t = Time([datetime(d.year, d.month, d.day, hh, mm, tzinfo=EST).astimezone(timezone.utc) for d in days])
    loc = EarthLocation(lat=w.lat * u.deg, lon=w.lon * u.deg, height=180 * u.m)
    aa = get_sun(t).transform_to(AltAz(obstime=t, location=loc))
    x, y = w.altaz_to_pixel(aa.alt.deg, aa.az.deg)
    return x, y, aa.alt.deg, aa.az.deg


def predict_sun(days: list[date], est_hhmm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, alt, _ = wcs_altaz(days, est_hhmm)
    return x, y, alt


def _circle_from_3(p: np.ndarray) -> tuple[float, float, float] | None:
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return cx, cy, float(np.hypot(x1 - cx, y1 - cy))


def _fit_circle(pts: np.ndarray) -> tuple[float, float, float]:
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(c + cx * cx + cy * cy))


def detect_sun(
    img: np.ndarray,
    guess: tuple[float, float],
    search: int = 90,
    sat_level: int = 245,
    r_range: tuple[float, float] = (30.0, 220.0),
    tol: float = 4.0,
    min_coverage: float = 0.4,
    iters: int = 600,
    seed: int = 0,
) -> dict | None:
    """Find the sun as a robust circle fitted to the edge of its saturated glare.

    `guess` (the ephemeris position) only picks which saturated blob to look at and
    rejects detections that land implausibly far away; the returned centre comes from
    the image. Returns None when no clean sun disk is visible (cloud, overcast, glare
    merged with lit cloud).
    """
    h, w = img.shape[:2]
    gx, gy = guess
    if not (-search <= gx < w + search and -search <= gy < h + search):
        return None
    sat = (img.min(axis=2) >= sat_level).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sat)
    if n <= 1:
        return None
    ix, iy = int(np.clip(round(gx), 0, w - 1)), int(np.clip(round(gy), 0, h - 1))
    k = lab[iy, ix]
    if k == 0:
        ys, xs = np.nonzero(lab)
        dd = np.hypot(xs - gx, ys - gy)
        j = int(dd.argmin())
        if dd[j] > search:
            return None
        k = lab[ys[j], xs[j]]
    blob = (lab == k).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack([c[:, 0, :] for c in contours]).astype(np.float64)
    on_edge = (pts[:, 0] <= 0) | (pts[:, 0] >= w - 1) | (pts[:, 1] <= 0) | (pts[:, 1] >= h - 1)
    pts = pts[~on_edge]
    if len(pts) < 30:
        return None

    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        c = _circle_from_3(pts[rng.choice(len(pts), 3, replace=False)])
        if c is None or not (r_range[0] <= c[2] <= r_range[1]):
            continue
        if np.hypot(c[0] - gx, c[1] - gy) > search:
            continue
        inl = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2]) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = c, inl
    if best is None:
        return None
    cx, cy, r = _fit_circle(pts[best_in])
    inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol
    cx, cy, r = _fit_circle(pts[inl])

    ang = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
    coverage = len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 72))) / 72
    offset = float(np.hypot(cx - gx, cy - gy))
    if coverage < min_coverage or offset > search or not (r_range[0] <= r <= r_range[1]):
        return None
    return {"x": cx, "y": cy, "r": r, "coverage": coverage, "offset": offset}


def reject_outliers(dates: list[date], dxy: np.ndarray, window_days: int = 10, max_dev: float = 20.0) -> np.ndarray:
    """Flag detections whose offset from the ephemeris departs from their neighbours'.

    The glare centre sits a smooth, slowly drifting 10-20 px from the ephemeris
    position; a detection that breaks away from that local trend is almost always
    glare merged with lit cloud.
    """
    ords = np.array([d.toordinal() for d in dates])
    keep = np.ones(len(dates), dtype=bool)
    glob = np.median(dxy, axis=0) if len(dxy) else np.zeros(2)
    for i in range(len(dates)):
        near = (np.abs(ords - ords[i]) <= window_days) & (np.arange(len(dates)) != i)
        ref = np.median(dxy[near], axis=0) if near.sum() >= 3 else glob
        keep[i] = np.hypot(*(dxy[i] - ref)) <= max_dev
    return keep


def autodetect(frames: list[dict], est_hhmm: str, overwrite_manual: bool = False) -> dict:
    """Fill x/y for project frames from the images. Manual annotations are left alone."""
    todo = [f for f in frames if overwrite_manual or f.get("source") != "manual"]
    if not todo:
        return {"detected": 0, "rejected": 0, "none": 0}
    days = [date.fromisoformat(f["date"]) for f in todo]
    px, py, alt = predict_sun(days, est_hhmm)

    hits = []
    for f, x, y, a in zip(todo, px, py, alt):
        f["predX"], f["predY"] = float(x), float(y)
        det = None
        if a > 0:
            img = cv2.imread(str(HALF_HOUR_DIR / f["file"]), cv2.IMREAD_COLOR)
            if img is not None:
                det = detect_sun(img, (x, y))
        if det is None:
            f.update({"x": None, "y": None, "skipped": True, "done": True, "source": "auto", "reason": "no clear sun"})
        else:
            f.update({"x": det["x"], "y": det["y"], "skipped": False, "done": True, "source": "auto", "reason": None})
            hits.append((f, det["x"] - x, det["y"] - y))

    rejected = 0
    if hits:
        keep = reject_outliers([date.fromisoformat(h[0]["date"]) for h in hits], np.array([h[1:] for h in hits]))
        for (f, _, _), k in zip(hits, keep):
            if not k:
                f.update({"skipped": True, "reason": "glare merged with cloud"})
                rejected += 1
    detected = len(hits) - rejected
    return {"detected": detected, "rejected": rejected, "none": len(todo) - len(hits)}
