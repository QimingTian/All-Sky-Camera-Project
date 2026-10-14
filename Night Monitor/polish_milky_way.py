#!/usr/bin/env python3
"""
Polish Milky_Way into a presentation master:
  auto-crop → ABE → elongated-star rebuild → sky denoise → gentle finish
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def auto_crop(bgr: np.ndarray, margin: int = 24, cov: float = 0.96) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (g > 14).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    row_cov = mask.mean(axis=1) / 255.0
    col_cov = mask.mean(axis=0) / 255.0
    good_rows = np.where(row_cov > cov)[0]
    good_cols = np.where(col_cov > cov)[0]
    if len(good_rows) < 80 or len(good_cols) < 80:
        ys, xs = np.where(mask > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
    else:
        y0, y1 = int(good_rows.min()), int(good_rows.max())
        x0, x1 = int(good_cols.min()), int(good_cols.max())
    y0 += margin
    x0 += margin
    y1 -= margin
    x1 -= margin
    if y1 - y0 < 200 or x1 - x0 < 200:
        return bgr
    return bgr[y0:y1, x0:x1].copy()


def star_mask(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 1.6)
    thr = max(float(np.percentile(g, 97.0)), float(np.median(g) + 12))
    peaks = (g > blur + 5.0) & (g > thr)
    m = peaks.astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    return m


def extract_background(bgr: np.ndarray, stars: np.ndarray) -> np.ndarray:
    filled = cv2.inpaint(bgr, stars, 6, cv2.INPAINT_TELEA)
    # multi-scale haze / cast
    bg = cv2.GaussianBlur(filled, (0, 0), 35)
    bg = 0.55 * bg + 0.45 * cv2.GaussianBlur(filled, (0, 0), 80)
    return bg.astype(np.float32)


def subtract_background(bgr: np.ndarray, bg: np.ndarray, strength: float = 0.96) -> np.ndarray:
    img = bgr.astype(np.float32)
    gbg = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    pedestal = float(np.percentile(gbg, 8))
    out = img - strength * (bg - pedestal)
    # channel balance: push darkest sky toward neutral
    sky = out.reshape(-1, 3)
    # sample dark pixels
    lum = out.mean(axis=2)
    dark = lum < np.percentile(lum, 40)
    if dark.sum() > 500:
        med = np.median(out[dark], axis=0)
        target = float(np.median(med))
        out = out - (med - target)[None, None, :] * 0.85
    return np.clip(out, 0, 255)


def _gaussian_patch(h: int, w: int, cy: float, cx: float, sy: float, sx: float, amp: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-0.5 * (((yy - cy) / max(sy, 0.4)) ** 2 + ((xx - cx) / max(sx, 0.4)) ** 2))
    return (amp * g).astype(np.float32)


def rebuild_elongated_stars(bgr: np.ndarray, min_area: int = 4, aspect_cut: float = 1.35) -> np.ndarray:
    """Replace elongated bright blobs with round Gaussians of similar flux."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 1.4)
    residual = g - blur
    thr = max(4.0, float(np.percentile(residual, 99.2)))
    mask = (residual > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = bgr.astype(np.float32)
    fixed = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > 180:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # pad ROI
        pad = 4
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(out.shape[1], x + w + pad)
        y1 = min(out.shape[0], y + h + pad)
        roi_lab = labels[y0:y1, x0:x1] == i
        if roi_lab.sum() < min_area:
            continue
        ys, xs = np.where(roi_lab)
        cy = float(ys.mean())
        cx = float(xs.mean())
        # covariance for elongation
        yy = ys - cy
        xx = xs - cx
        cov_xx = float((xx * xx).mean()) + 1e-3
        cov_yy = float((yy * yy).mean()) + 1e-3
        cov_xy = float((xx * yy).mean())
        # eigenvalues
        tr = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        tmp = np.sqrt(max(tr * tr / 4 - det, 0.0))
        l1 = tr / 2 + tmp
        l2 = tr / 2 - tmp
        aspect = np.sqrt(max(l1, 1e-6) / max(l2, 1e-6))
        if aspect < aspect_cut and max(w, h) <= 6:
            continue  # already round-ish small star
        # flux / color from original residual peaks
        roi = out[y0:y1, x0:x1]
        g_roi = g[y0:y1, x0:x1]
        local_bg = float(np.median(g_roi[~roi_lab])) if (~roi_lab).any() else float(np.median(g_roi))
        peak = float(g_roi[roi_lab].max() - local_bg)
        if peak < 3:
            continue
        # remove old blob by inpainting-like fill with local background color
        bg_color = np.median(roi[~roi_lab].reshape(-1, 3), axis=0) if (~roi_lab).any() else np.median(roi.reshape(-1, 3), axis=0)
        # expand removal a bit for streaks
        rem = cv2.dilate(roi_lab.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1).astype(bool)
        for c in range(3):
            ch = roi[:, :, c]
            ch[rem] = bg_color[c]
            roi[:, :, c] = ch
        # round sigma from geometric mean of ellipse axes
        sig = float(np.sqrt(np.sqrt(max(l1, 0.2) * max(l2, 0.2))))
        sig = float(np.clip(sig * 0.85, 0.55, 2.8))
        # color amp from original peak pixels
        peak_pix = roi_lab & (g_roi >= np.percentile(g_roi[roi_lab], 70))
        if peak_pix.any():
            amp_bgr = np.percentile(bgr[y0:y1, x0:x1][peak_pix], 90, axis=0).astype(np.float32) - bg_color
        else:
            amp_bgr = np.array([peak, peak, peak], dtype=np.float32)
        amp_bgr = np.clip(amp_bgr, 0, 255)
        patch = _gaussian_patch(y1 - y0, x1 - x0, cy, cx, sig, sig, 1.0)
        for c in range(3):
            roi[:, :, c] = np.clip(roi[:, :, c] + patch * amp_bgr[c], 0, 255)
        out[y0:y1, x0:x1] = roi
        fixed += 1
    print(f"  rebuilt {fixed} elongated stars", flush=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise_sky(bgr: np.ndarray, stars: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 8, 8, 7, 21)
    den = cv2.bilateralFilter(den, 9, 35, 9)
    # extra low-frequency blotch smooth on non-star areas
    soft = cv2.GaussianBlur(den, (0, 0), 2.2)
    star_w = cv2.GaussianBlur(stars, (0, 0), 2.0).astype(np.float32) / 255.0
    star_w = np.clip(star_w * 1.4, 0, 1)[..., None]
    # protect MW structure a bit: keep midtones less smoothed
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mw = (g > np.percentile(g, 55)).astype(np.float32)
    mw = cv2.GaussianBlur(mw, (0, 0), 6)[..., None]
    protect = np.clip(star_w + 0.45 * mw, 0, 1)
    src = bgr.astype(np.float32)
    base = den.astype(np.float32)
    blotch = soft.astype(np.float32)
    sky = blotch * (1 - 0.55) + base * 0.55
    out = src * protect + sky * (1 - protect)
    return np.clip(out, 0, 255).astype(np.uint8)


def finish(bgr: np.ndarray) -> np.ndarray:
    img = bgr.astype(np.float32) / 255.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lo = float(np.percentile(g, 1.5))
    hi = float(np.percentile(g, 99.5))
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    # soft S-curve: darker sky, brighter MW
    x = img
    img = np.clip(1.05 * (x - 0.08) / (1.0 + 0.9 * (x - 0.08)), 0, 1)
    img = np.power(np.clip(img, 0, 1), 0.93)

    u8 = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # mild local contrast only on L
    clahe = cv2.createCLAHE(clipLimit=1.05, tileGridSize=(12, 12))
    L = clahe.apply(L)
    A = np.clip(128 + (A.astype(np.float32) - 128) * 0.78, 0, 255).astype(np.uint8)
    B = np.clip(128 + (B.astype(np.float32) - 128) * 0.72, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    # cool/neutral deepest sky
    g2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sky = (g2 < np.percentile(g2, 45)).astype(np.float32)
    sky = cv2.GaussianBlur(sky, (0, 0), 10)[..., None]
    f = out.astype(np.float32)
    f[:, :, 0] = np.clip(f[:, :, 0] + 4.0 * sky[:, :, 0], 0, 255)
    f[:, :, 1] = np.clip(f[:, :, 1] - 3.0 * sky[:, :, 0], 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] - 2.0 * sky[:, :, 0], 0, 255)

    # light sharpen of star cores only
    blur = cv2.GaussianBlur(f, (0, 0), 1.0)
    sharp = np.clip(f * 1.18 - blur * 0.18, 0, 255)
    return sharp.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Milky_Way.tiff",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Milky_Way_final.jpg",
    )
    args = ap.parse_args()

    src = args.input
    if not src.exists():
        src = args.input.with_suffix(".jpg")
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {src}")
    print(f"input {src.name} {bgr.shape}", flush=True)

    print("auto-crop...", flush=True)
    cropped = auto_crop(bgr, margin=28, cov=0.965)
    print(f"  -> {cropped.shape}", flush=True)

    stars = star_mask(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))
    print(f"  star mask {100 * stars.mean() / 255:.2f}%", flush=True)

    print("background extraction...", flush=True)
    bg = extract_background(cropped, stars)
    flat = subtract_background(cropped, bg, strength=0.95)
    flat_u8 = flat.astype(np.uint8)

    print("rebuild elongated stars...", flush=True)
    starred = rebuild_elongated_stars(flat_u8)

    print("denoise sky...", flush=True)
    stars2 = star_mask(cv2.cvtColor(starred, cv2.COLOR_BGR2GRAY))
    den = denoise_sky(starred, stars2)

    print("finish...", flush=True)
    final = finish(den)
    final = final[6:-6, 6:-6]

    out_jpg = args.output
    out_tiff = args.output.with_suffix(".tiff")
    cv2.imwrite(str(out_jpg), final, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
    cv2.imwrite(str(out_tiff), final)
    cv2.imwrite("/tmp/mw_final_preview.jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    diag = args.output.parent / "diagnostics_final"
    diag.mkdir(exist_ok=True)
    cv2.imwrite(str(diag / "crop.jpg"), cropped)
    cv2.imwrite(str(diag / "background.jpg"), np.clip(bg, 0, 255).astype(np.uint8))
    cv2.imwrite(str(diag / "flat.jpg"), flat_u8)
    cv2.imwrite(str(diag / "stars_rebuilt.jpg"), starred)
    # detail crop
    H, W = final.shape[:2]
    detail = final[H // 3 : H // 3 + 360, W // 3 : W // 3 + 540]
    cv2.imwrite("/tmp/mw_final_detail.jpg", detail)
    print(f"DONE -> {out_jpg} ({final.shape[1]}x{final.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
