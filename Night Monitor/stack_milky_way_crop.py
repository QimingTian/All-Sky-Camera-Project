#!/usr/bin/env python3
"""
Fisheye-safe Milky Way crop stack.

Do NOT re-center the crop on each frame (that stacks different sky).
Instead (Siril-like):
  1) Detect MW ROI once on the reference frame (fixed pixel box)
  2) Globally register each full frame -> reference (asterism + Homography)
  3) Warp, then crop the SAME box
  4) Sigma-clip mean + stretch

Only frames that successfully register are kept — for a fisheye night,
use a limited seq window so the MW band stays in that box for most frames.
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import astroalign as aa
import cv2
import numpy as np


def seq_of(p: Path) -> int:
    m = re.search(r"seq(\d+)of", p.name)
    return int(m.group(1)) if m else 0


def list_frames(folder: Path, lo: int, hi: int, max_mean: float = 48.0) -> list[Path]:
    out = []
    for p in sorted(folder.glob("*.tiff")):
        if p.name.startswith("._"):
            continue
        s = seq_of(p)
        if s < lo or s > hi:
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or float(g.mean()) >= max_mean:
            continue
        out.append(p)
    return out


def mw_roi(gray_u8: np.ndarray, cw: int, ch: int) -> tuple[int, int, int, int]:
    """Return x0,y0,x1,y1 fixed crop covering MW on reference."""
    g = gray_u8.astype(np.float32)
    h, w = g.shape
    starsup = cv2.medianBlur(gray_u8, 5).astype(np.float32)
    smooth = cv2.GaussianBlur(starsup, (0, 0), 12)
    bg = cv2.GaussianBlur(starsup, (0, 0), 80)
    ext = np.clip(smooth - bg, 0, None)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.48) ** 2)
    fov = r < min(h, w) * 0.42
    ext2 = ext.copy()
    ext2[~fov] = 0
    ext2[:, : int(w * 0.12)] = 0
    ext2[:40, :] = 0
    thr = np.percentile(ext2[fov], 91)
    mask = (ext2 > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        cx, cy = w // 2, h // 2
    else:
        wts = ext2[ys, xs]
        cx = int(np.average(xs, weights=wts))
        cy = int(np.average(ys, weights=wts))
    x0 = max(0, min(cx - cw // 2, w - cw))
    y0 = max(0, min(cy - ch // 2, h - ch))
    return x0, y0, x0 + cw, y0 + ch


def detect_gray(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g[:, : int(g.shape[1] * 0.10)] = 0
    return cv2.createCLAHE(2.5, (8, 8)).apply(g).astype(np.float32)


def find_H(mov_g: np.ndarray, ref_g: np.ndarray):
    try:
        _, (src, dst) = aa.find_transform(
            mov_g, ref_g, max_control_points=100, detection_sigma=3.5
        )
    except Exception:
        return None, 0
    if src is None or len(src) < 4:
        return None, 0
    src = np.asarray(src, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, 0
    nin = int(mask.ravel().sum())
    if nin < 4:
        return None, nin
    det = abs(np.linalg.det(H[:2, :2]))
    if det < 0.25 or det > 4.0:
        return None, nin
    return H.astype(np.float64), nin


def warp(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    # dst(x)=src(H·x) — correct for findHomography(mov→ref)
    return cv2.warpPerspective(
        bgr, H.astype(np.float32), (w, h), flags=cv2.INTER_LANCZOS4, borderValue=0
    )


def enhance(bgr_f32: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(np.clip(bgr_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    black = float(np.percentile(g, 4))
    x = np.clip(bgr_f32 - black * 0.9, 0, None)
    white = float(np.percentile(x, 99.85)) + 1e-6
    x = np.clip(x / white, 0, 1)
    lum = 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]
    stretch = np.arcsinh(lum * 8.0) / np.arcsinh(8.0)
    y = np.clip(x * (stretch / (lum + 1e-6))[..., None], 0, 1)
    y = np.power(y, 0.88)
    u8 = (y * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(1.8, (8, 8)).apply(L)
    A = np.clip(128 + (A.astype(np.float32) - 128) * 1.2, 0, 255).astype(np.uint8)
    B = np.clip(128 + (B.astype(np.float32) - 128) * 1.15, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, 5, 16, 5)
    blur = cv2.GaussianBlur(out, (0, 0), 1.1)
    return cv2.addWeighted(out, 1.3, blur, -0.3, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "raw_copy")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--ref-seq", type=int, default=25)
    ap.add_argument("--min-seq", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=50)
    ap.add_argument("--crop-w", type=int, default=900)
    ap.add_argument("--crop-h", type=int, default=700)
    args = ap.parse_args()

    paths = list_frames(args.input, args.min_seq, args.max_seq)
    ref_idx = int(np.argmin([abs(seq_of(p) - args.ref_seq) for p in paths]))
    ref_path = paths[ref_idx]
    print(f"window {args.min_seq}-{args.max_seq}: {len(paths)} frames", flush=True)
    print(f"reference: {ref_path.name}", flush=True)

    ref_bgr = cv2.imread(str(ref_path))
    x0, y0, x1, y1 = mw_roi(
        cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY), args.crop_w, args.crop_h
    )
    print(f"fixed crop box: ({x0},{y0})-({x1},{y1})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dbg = ref_bgr.copy()
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.imwrite(
        str(args.output_dir / "Milky_Way_crop_roi.jpg"),
        cv2.resize(dbg, None, fx=0.4, fy=0.4),
    )
    ref_crop = ref_bgr[y0:y1, x0:x1].copy()
    cv2.imwrite(str(args.output_dir / "Milky_Way_crop_ref.jpg"), ref_crop)

    ref_g = detect_gray(ref_bgr)
    crops = [ref_crop]
    ok = 1
    fail = 0

    print("register full frame -> ref, then crop fixed MW box...", flush=True)
    for p in paths:
        if p == ref_path:
            continue
        mov = cv2.imread(str(p))
        if mov is None:
            fail += 1
            continue
        H, nin = find_H(detect_gray(mov), ref_g)
        if H is None:
            print(f"  FAIL {p.name}", flush=True)
            fail += 1
            continue
        warped = warp(mov, H)
        crop = warped[y0:y1, x0:x1]
        # require most of crop covered
        cov = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 0))
        if cov < 0.85:
            print(f"  SKIP low coverage {p.name} cov={cov:.2f} nin={nin}", flush=True)
            fail += 1
            continue
        crops.append(crop)
        ok += 1
        print(f"  OK {p.name} inliers={nin} cov={cov:.2f}", flush=True)

    print(f"crops ready: {len(crops)}  fail/skip: {fail}", flush=True)
    if len(crops) < 8:
        raise SystemExit("too few crops")

    # normalize + sigma clip
    print("stacking...", flush=True)
    arr = np.stack([c.astype(np.float32) for c in crops], 0)
    ref = arr[0]
    rg = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    rmed, rmad = float(np.median(rg)), float(np.median(np.abs(rg - np.median(rg)))) + 1e-3
    normed = []
    for im in arr:
        g = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        med = float(np.median(g))
        mad = float(np.median(np.abs(g - med))) + 1e-3
        scale = rmad / mad
        normed.append(im * scale + (rmed - med * scale))
    arr = np.stack(normed, 0)
    med = np.median(arr, 0)
    mad = np.median(np.abs(arr - med), 0) + 1e-3
    keep = np.abs(arr - med) <= (2.5 * 1.4826 * mad)
    keep = np.repeat((keep.mean(3) > 0.5)[..., None], 3, 3)
    filled = np.where(keep, arr, np.nan)
    with np.errstate(all="ignore"):
        stack = np.nanmean(filled, 0)
    stack = np.where(np.isnan(stack), med, stack).astype(np.float32)

    raw = np.clip(stack, 0, 255).astype(np.uint8)
    lap = float(cv2.Laplacian(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    print(f"stack laplacian: {lap:.2f}", flush=True)

    pretty = enhance(stack)
    out = args.output_dir
    cv2.imwrite(str(out / "Milky_Way_crop_stack_raw.tiff"), raw)
    cv2.imwrite(str(out / "Milky_Way_crop_stack_raw.jpg"), raw, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(out / "Milky_Way_crop.tiff"), pretty)
    cv2.imwrite(str(out / "Milky_Way_crop.jpg"), pretty, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite("/tmp/mw_crop_preview.jpg", pretty, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print(f"DONE -> {out / 'Milky_Way_crop.jpg'}  (n={len(crops)})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
