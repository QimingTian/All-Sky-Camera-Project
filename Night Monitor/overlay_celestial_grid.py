#!/usr/bin/env python3
"""
Plate-match an all-sky ASC frame and overlay equatorial (RA/Dec) grid lines.

Assumes equidistant fisheye projection with zenith near the optical axis.
Location default: Pomfret School, CT.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, linear_sum_assignment

# Pomfret School, CT (approx campus observatory / ASC site)
POMFRET = EarthLocation(lat=41.8992 * u.deg, lon=-71.9628 * u.deg, height=180 * u.m)
TZ = ZoneInfo("America/New_York")

# Bright stars (V < ~2.5) — name, RA hours, Dec deg
BRIGHT_STARS = [
    ("Polaris", 2.530301028, 89.264109444),
    ("Vega", 18.615648686, 38.783688960),
    ("Deneb", 20.690532782, 45.280338889),
    ("Altair", 19.846289858, 8.868321944),
    ("Arcturus", 14.261027778, 19.182409167),
    ("Capella", 5.278155556, 45.997991667),
    ("Rigel", 5.242297222, -8.201638889),
    ("Procyon", 7.655027778, 5.224988889),
    ("Betelgeuse", 5.919529167, 7.407063889),
    ("Aldebaran", 4.598678472, 16.509302500),
    ("Spica", 13.419883611, -11.161322222),
    ("Antares", 16.490127778, -26.432002778),
    ("Fomalhaut", 22.960830833, -29.622236111),
    ("Regulus", 10.139555556, 11.967208333),
    ("Pollux", 7.755377778, 28.026199722),
    ("Castor", 7.576640833, 31.888316667),
    ("Bellatrix", 5.418851667, 6.349703333),
    ("Alnilam", 5.603750556, -1.201919444),
    ("Alnitak", 5.679287500, -1.942574444),
    ("Mintaka", 5.533390278, -0.299095833),
    ("Saiph", 5.795916667, -9.669605556),
    ("Mirfak", 3.405375278, 49.861243056),
    ("Schedar", 0.675107500, 56.537331667),
    ("Caph", 0.152666944, 59.149781111),
    ("Dubhe", 11.062108333, 61.750833056),
    ("Merak", 11.030666667, 56.382430556),
    ("Alioth", 12.900416667, 55.959822222),
    ("Mizar", 13.398058333, 54.925372222),
    ("Alkaid", 13.792083333, 49.313055556),
    ("Kochab", 14.845084722, 74.155503889),
    ("Alpheratz", 0.139788889, 29.090431111),
    ("Mirach", 1.162062500, 35.620558333),
    ("Algol", 3.136166667, 40.955648611),
    ("Enif", 21.736416667, 9.875056944),
    ("Scheat", 23.062833333, 28.082675000),
    ("Markab", 23.079416667, 15.205250000),
    ("Algenib", 0.220216667, 15.183577778),
    ("Hamal", 2.119350000, 23.462416667),
    ("Menkar", 3.038116667, 4.089733333),
    ("Alcyone", 3.791416667, 24.105138889),
    ("Atlas", 3.870250000, 24.105111111),
    ("Electra", 3.747000000, 24.113055556),
    ("Maia", 3.871083333, 24.136583333),
    ("Merope", 3.779000000, 23.948333333),
    ("Sirius", 6.752481389, -16.716115833),
    ("Adhara", 6.977098611, -28.972088889),
    ("Wezen", 7.139855556, -26.393200000),
    ("Gacrux", 12.519416667, -57.113166667),
    ("Acrux", 12.443333333, -63.099166667),
    ("Hadar", 14.063733333, -60.373033333),
    ("Shaula", 17.560083333, -37.103833333),
    ("Avior", 8.375216667, -59.509583333),
    ("Alnair", 22.137416667, -46.961111111),
    ("Peacock", 20.427555556, -56.735083333),
    ("Achernar", 1.628558333, -57.236750000),
]


def parse_obs_time(path: Path) -> Time:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", path.name)
    if not m:
        raise ValueError(f"Cannot parse time from {path.name}")
    date_s, hh, mm, ss = m.groups()
    local = datetime.strptime(f"{date_s} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")
    local = local.replace(tzinfo=TZ)
    utc = local.astimezone(timezone.utc)
    return Time(utc.strftime("%Y-%m-%d %H:%M:%S"), format="iso", scale="utc")


def detect_stars(gray: np.ndarray, max_stars: int = 60) -> np.ndarray:
    """Return Nx2 array of star centroids (x, y), brightest first."""
    g = gray.astype(np.float32)
    p995 = np.percentile(g, 99.7)
    p60 = np.percentile(g, 60)
    norm = np.clip((g - p60) / max(p995 - p60, 1), 0, 1)
    blur = gaussian_filter(norm, 1.0)

    # Top-hat style local maxima on downsampled grid for speed
    k = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(blur, kernel)
    peaks = (blur > 0.42) & (blur >= dil - 1e-6)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return np.empty((0, 2))

    flux = blur[ys, xs]
    order = np.argsort(flux)[::-1]
    xs, ys = xs[order].astype(np.float64), ys[order].astype(np.float64)

    chosen: list[tuple[float, float]] = []
    min_dist = 18.0
    for x, y in zip(xs, ys):
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_dist**2 for cx, cy in chosen):
            # sub-pixel centroid in 5x5 patch
            xi, yi = int(x), int(y)
            patch = blur[max(yi - 2, 0) : yi + 3, max(xi - 2, 0) : xi + 3]
            if patch.size == 0:
                continue
            py, px = np.indices(patch.shape)
            w = patch / max(patch.sum(), 1e-6)
            cx = xi - 2 + (px * w).sum()
            cy = yi - 2 + (py * w).sum()
            chosen.append((float(cx), float(cy)))
        if len(chosen) >= max_stars:
            break
    return np.array(chosen, dtype=np.float64)


def assign_stars(
    detected: np.ndarray,
    pred: np.ndarray,
    max_dist: float,
) -> list[tuple[int, int, float]]:
    """One-to-one assignment detected->predicted within max_dist."""
    if len(detected) == 0 or len(pred) == 0:
        return []
    cost = np.linalg.norm(detected[:, None, :] - pred[None, :, :], axis=2)
    big = max_dist * 10
    cost[cost > max_dist] = big
    r, c = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(r, c):
        d = float(cost[i, j])
        if d <= max_dist:
            pairs.append((i, j, d))
    return pairs


def catalog_altaz(obstime: Time, min_alt: float = 8.0) -> list[tuple[str, float, float]]:
    ras = [s[1] for s in BRIGHT_STARS]
    decs = [s[2] for s in BRIGHT_STARS]
    names = [s[0] for s in BRIGHT_STARS]
    sc = SkyCoord(ra=ras * u.hourangle, dec=decs * u.deg)
    aa = sc.transform_to(AltAz(obstime=obstime, location=POMFRET))
    out = []
    for name, alt, az in zip(names, aa.alt.deg, aa.az.deg):
        if alt >= min_alt:
            out.append((name, alt, az))
    return out


def altaz_to_pixel(
    alt: np.ndarray | float,
    az: np.ndarray | float,
    cx: float,
    cy: float,
    f: float,
    rot_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    zeta = np.deg2rad(90.0 - np.asarray(alt))
    phi = np.deg2rad(np.asarray(az) - rot_deg)
    r = f * zeta
    x = cx + r * np.sin(phi)
    y = cy - r * np.cos(phi)
    return x, y


LAT_DEG = POMFRET.lat.deg
ZETA_NCP = np.deg2rad(90.0 - LAT_DEG)


def detect_ncp_from_star_trail(star_trail_path: Path) -> tuple[float, float]:
    """Find North Celestial Pole as center of tightest star-trail arcs (upper sky)."""
    cal_path = star_trail_path.parent / "asc_calibration.json"
    if cal_path.exists():
        import json

        data = json.loads(cal_path.read_text())
        if "ncp_xy" in data:
            return float(data["ncp_xy"][0]), float(data["ncp_xy"][1])

    img = cv2.imread(str(star_trail_path))
    if img is None:
        raise FileNotFoundError(star_trail_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    norm = np.clip(
        (g - np.percentile(g, 30)) / max(np.percentile(g, 99.8) - np.percentile(g, 30), 1),
        0,
        1,
    )
    blur = cv2.GaussianBlur((norm * 255).astype(np.uint8), (3, 3), 0)
    edges = cv2.Canny(blur, 30, 90)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=4,
        param1=50,
        param2=6,
        minRadius=2,
        maxRadius=25,
    )
    if circles is None:
        raise RuntimeError("Could not detect star-trail arcs")
    arcs = np.array([(cx, cy, r) for cx, cy, r in circles[0] if r <= 12])

    best = (-1.0, w * 0.62, h * 0.11)
    for px in np.arange(w * 0.48, w * 0.75, 3):
        for py in np.arange(40, h * 0.22, 3):
            dists = np.linalg.norm(arcs[:, :2] - (px, py), axis=1)
            err = np.abs(dists - arcs[:, 2])
            wts = 1.0 / np.maximum(arcs[:, 2], 1.0)
            score = float(np.sum(wts * np.exp(-((err / 6.0) ** 2))))
            if score > best[0]:
                best = (score, px, py)
    return float(best[1]), float(best[2])


def zenith_from_ncp(ncp_x: float, ncp_y: float, f: float, rot_deg: float) -> tuple[float, float]:
    rot = np.deg2rad(rot_deg)
    cx = ncp_x - f * ZETA_NCP * np.sin(-rot)
    cy = ncp_y + f * ZETA_NCP * np.cos(-rot)
    return float(cx), float(cy)


def fit_projection(
    detected: np.ndarray,
    catalog: list[tuple[str, float, float]],
    img_shape: tuple[int, int],
    ncp_xy: tuple[float, float],
) -> dict:
    h, w = img_shape
    ncp_x, ncp_y = ncp_xy
    alts = np.array([c[1] for c in catalog])
    azs = np.array([c[2] for c in catalog])
    names = [c[0] for c in catalog]

    best = {"inliers": 0}
    for rot in np.linspace(0, 360, 180, endpoint=False):
        for f in np.linspace(280, 680, 41):
            cx, cy = zenith_from_ncp(ncp_x, ncp_y, f, rot)
            px, py = altaz_to_pixel(alts, azs, cx, cy, f, rot)
            pred = np.column_stack([px, py])
            pairs = assign_stars(detected, pred, 45)
            if len(pairs) > best["inliers"]:
                best = {"inliers": len(pairs), "cx": cx, "cy": cy, "f": f, "rot": rot}

    if best["inliers"] < 4:
        raise RuntimeError(f"Plate match failed (only {best['inliers']} inliers)")

    p0 = np.array([best["f"], best["rot"]])

    def residuals(p):
        f, rot = p
        cx, cy = zenith_from_ncp(ncp_x, ncp_y, f, rot)
        px, py = altaz_to_pixel(alts, azs, cx, cy, f, rot)
        pred = np.column_stack([px, py])
        pairs = assign_stars(detected, pred, 45)
        res = [0.0, 0.0]
        ncp_px, ncp_py = altaz_to_pixel(LAT_DEG, 0.0, cx, cy, f, rot)
        res[0] = 5.0 * (ncp_px - ncp_x)
        res[1] = 5.0 * (ncp_py - ncp_y)
        star_res = np.full(len(detected), 40.0)
        for i, j, d in pairs:
            star_res[i] = d
        return np.concatenate([res, star_res])

    sol = least_squares(residuals, p0, method="lm", max_nfev=300)
    f, rot = sol.x
    cx, cy = zenith_from_ncp(ncp_x, ncp_y, f, rot)

    px, py = altaz_to_pixel(alts, azs, cx, cy, f, rot)
    pred = np.column_stack([px, py])
    pairs = assign_stars(detected, pred, 35)
    matched = []
    used = set()
    for i, j, d in pairs:
        if j in used:
            continue
        used.add(j)
        matched.append(
            {
                "detected": detected[i],
                "predicted": pred[j],
                "name": names[j],
                "dist_px": d,
            }
        )

    ncp_px, ncp_py = altaz_to_pixel(LAT_DEG, 0.0, cx, cy, f, rot)
    return {
        "cx": float(cx),
        "cy": float(cy),
        "f": float(f),
        "rot_deg": float(rot) % 360,
        "ncp_xy": (float(ncp_x), float(ncp_y)),
        "ncp_err_px": float(((ncp_px - ncp_x) ** 2 + (ncp_py - ncp_y) ** 2) ** 0.5),
        "matched": matched,
        "rms_px": float(np.sqrt(np.mean([m["dist_px"] ** 2 for m in matched]))) if matched else 999.0,
    }


def dec_to_ra_points(dec_deg: float, obstime: Time, n: int = 360) -> SkyCoord:
    ras = np.linspace(0, 360, n, endpoint=False)
    return SkyCoord(ra=ras * u.deg, dec=dec_deg * u.deg)


def ra_to_dec_points(ra_hour: float, obstime: Time, n: int = 180) -> SkyCoord:
    decs = np.linspace(-90, 90, n)
    return SkyCoord(ra=ra_hour * u.hourangle, dec=decs * u.deg)


def sc_to_pixels(sc: SkyCoord, fit: dict, obstime: Time) -> np.ndarray:
    aa = sc.transform_to(AltAz(obstime=obstime, location=POMFRET))
    alt, az = aa.alt.deg, aa.az.deg
    x, y = altaz_to_pixel(alt, az, fit["cx"], fit["cy"], fit["f"], fit["rot_deg"])
    pts = np.column_stack([x, y])
    visible = alt >= 0
    return pts, visible


def draw_grid(
    canvas: np.ndarray,
    fit: dict,
    obstime: Time,
    dec_step: int = 15,
    ra_step_hours: int = 2,
) -> None:
    h, w = canvas.shape[:2]

    # Dec parallels
    for dec in range(-60, 90, dec_step):
        sc = dec_to_ra_points(dec, obstime)
        pts, vis = sc_to_pixels(sc, fit, obstime)
        pts = pts[vis]
        if len(pts) < 4:
            continue
        draw_polyline(canvas, pts, (0, 255, 120), 1)
        # label near eastern horizon
        mid = pts[len(pts) // 4]
        if 0 <= mid[0] < w and 0 <= mid[1] < h:
            cv2.putText(
                canvas,
                f"Dec {dec:+d}",
                (int(mid[0]) + 4, int(mid[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )

    # RA meridians
    for ra_h in range(0, 24, ra_step_hours):
        sc = ra_to_dec_points(ra_h, obstime)
        pts, vis = sc_to_pixels(sc, fit, obstime)
        pts = pts[vis]
        if len(pts) < 4:
            continue
        draw_polyline(canvas, pts, (255, 180, 0), 1)
        top = pts[np.argmin(pts[:, 1])]
        if 0 <= top[0] < w and 0 <= top[1] < h:
            cv2.putText(
                canvas,
                f"RA {ra_h}h",
                (int(top[0]) + 4, int(top[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    # Zenith marker
    cx, cy = int(round(fit["cx"])), int(round(fit["cy"]))
    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)


def draw_polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int) -> None:
    h, w = img.shape[:2]
    prev = None
    for x, y in pts:
        if not (0 <= x < w and 0 <= y < h):
            prev = None
            continue
        cur = (int(round(x)), int(round(y)))
        if prev is not None:
            cv2.line(img, prev, cur, color, thickness, cv2.LINE_AA)
        prev = cur


def annotate_bright_stars(
    canvas: np.ndarray,
    fit: dict,
    catalog: list[tuple[str, float, float]],
) -> None:
    """Label plate-solved bright stars, snapped to detected centroids when matched."""
    h, w = canvas.shape[:2]
    matched = {m["name"]: m for m in fit["matched"]}
    alts = np.array([c[1] for c in catalog])
    azs = np.array([c[2] for c in catalog])
    names = [c[0] for c in catalog]
    px, py = altaz_to_pixel(alts, azs, fit["cx"], fit["cy"], fit["f"], fit["rot_deg"])

    for name, x, y, alt in zip(names, px, py, alts):
        if name in matched:
            x, y = matched[name]["detected"]
        elif not (8 <= x < w - 8 and 8 <= y < h - 8):
            continue
        else:
            continue  # only label stars confirmed in the frame
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(canvas, (ix, iy), 10, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (ix, iy), 2, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            name,
            (ix + 12, iy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            name,
            (ix + 12, iy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )

    # mark NCP from star trails
    ncp_x, ncp_y = fit["ncp_xy"]
    if 0 <= ncp_x < w and 0 <= ncp_y < h:
        cv2.drawMarker(
            canvas,
            (int(ncp_x), int(ncp_y)),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            22,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "NCP",
            (int(ncp_x) + 8, int(ncp_y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Label bright stars on ASC frame")
    ap.add_argument("frame", type=Path, help="Input TIFF frame")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--grid", action="store_true", help="Also draw RA/Dec grid")
    ap.add_argument(
        "--star-trail",
        type=Path,
        default=None,
        help="Star_Trail.jpg in same folder (for NCP calibration)",
    )
    args = ap.parse_args()

    img = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read {args.frame}")

    trail = args.star_trail or args.frame.parent / "Star_Trail.jpg"
    ncp_xy = detect_ncp_from_star_trail(trail)
    print(f"NCP from star trails: ({ncp_xy[0]:.1f}, {ncp_xy[1]:.1f})")

    obstime = parse_obs_time(args.frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected = detect_stars(gray)
    catalog = catalog_altaz(obstime)
    fit = fit_projection(detected, catalog, gray.shape, ncp_xy)

    out = img.copy()
    if args.grid:
        draw_grid(out, fit, obstime)
    annotate_bright_stars(out, fit, catalog)

    out_path = args.output or args.frame.with_name(args.frame.stem + "_bright_stars.jpg")
    cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(out_path)
    print(f"matched {len(fit['matched'])} stars for plate fit, RMS={fit['rms_px']:.1f}px")
    for m in sorted(fit["matched"], key=lambda x: x["dist_px"]):
        print(f"  {m['name']:10s}  err={m['dist_px']:.1f}px")


if __name__ == "__main__":
    main()
