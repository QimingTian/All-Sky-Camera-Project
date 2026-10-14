#!/usr/bin/env python3
"""
Milky Way stack for All Sky Camera frames — Siril/WBPP-inspired pipeline.

Siril Global Registration (docs + src/registration):
  1) Detect stars (PSF / source extraction)
  2) Triangle asterism match to a single reference (Valdes / atpmatch)
  3) Fit transform (Homography preferred for wide field) with RANSAC
  4) Warp each frame directly onto the reference (NOT chained neighbors)
  5) Average stack with rejection + mild normalization
  6) Presentation stretch

We use astroalign for asterism matching (same triangle-matching family as Siril),
then refine with OpenCV findHomography for wide-field ASC data.
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import astroalign as aa
import cv2
import numpy as np


def list_night_frames(folder: Path, max_seq: int = 171, max_mean: float = 48.0) -> list[Path]:
    files = sorted(p for p in folder.glob("*.tiff") if not p.name.startswith("._") and p.is_file())
    out: list[Path] = []
    for p in files:
        m = re.search(r"seq(\d+)of", p.name)
        seq = int(m.group(1)) if m else 10**9
        if seq > max_seq:
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if float(g.mean()) >= max_mean:
            continue
        out.append(p)
    return out


def seq_of(p: Path) -> int:
    m = re.search(r"seq(\d+)of", p.name)
    return int(m.group(1)) if m else 0


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    return img


def to_detect_gray(bgr: np.ndarray) -> np.ndarray:
    """Luminance for star detection; suppress left tree/lights strip."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[:, : int(g.shape[1] * 0.10)] = 0
    # soft vignette corners are ok; CLAHE helps detection on bright MW
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(g8).astype(np.float32)


def find_homography_to_ref(
    mov_gray: np.ndarray,
    ref_gray: np.ndarray,
    max_control_points: int = 80,
    detection_sigma: float = 4.0,
) -> tuple[np.ndarray | None, int]:
    """
    Asterism match (astroalign / triangle) then Homography via RANSAC.
    Returns 3x3 H mapping mov -> ref (for warpPerspective with WARP_INVERSE_MAP),
    and number of inlier star pairs.
    """
    try:
        _transf, (src_pts, dst_pts) = aa.find_transform(
            mov_gray,
            ref_gray,
            max_control_points=max_control_points,
            detection_sigma=detection_sigma,
        )
    except aa.MaxIterError:
        return None, 0
    except Exception:
        return None, 0

    if src_pts is None or len(src_pts) < 4:
        return None, 0

    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None or mask is None:
        return None, 0
    inliers = int(mask.ravel().sum())
    if inliers < 4:
        return None, inliers
    # reject insane warps (all-sky should be mostly rotation/scale ~1)
    if abs(np.linalg.det(H[:2, :2])) < 0.2 or abs(np.linalg.det(H[:2, :2])) > 5.0:
        return None, inliers
    return H.astype(np.float64), inliers


def warp_to_ref(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp moving frame into reference coordinates.

    findHomography(mov_pts, ref_pts) yields H with ref ≈ H·mov.
    OpenCV's default warpPerspective (without WARP_INVERSE_MAP) uses
    dst(x)=src(H·x), which is the correct sampling for that H.
    """
    h, w = bgr.shape[:2]
    return cv2.warpPerspective(
        bgr,
        H.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def additive_scale_norm(img: np.ndarray, ref_med: float, ref_mad: float) -> np.ndarray:
    """Siril-like additive+scaling normalisation against reference sky stats."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = g > 1
    if valid.sum() < 1000:
        return img.astype(np.float32)
    med = float(np.median(g[valid]))
    mad = float(np.median(np.abs(g[valid] - med))) + 1e-3
    scale = ref_mad / mad
    out = img.astype(np.float32) * scale
    # additive background match on luminance
    g2 = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    med2 = float(np.median(g2[g2 > 1])) if np.any(g2 > 1) else med
    out = out + (ref_med - med2)
    return out


def sigma_clip_mean(
    paths: list[Path],
    Hs: list[np.ndarray],
    ref_bgr: np.ndarray,
    sigma: float = 2.5,
) -> np.ndarray:
    """
    Two-pass mean with sigma clipping (WBPP/Siril-style rejection lite).
    Pass1: mean; Pass2: keep pixels within sigma of that mean.
    """
    h, w = ref_bgr.shape[:2]
    ref_g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_med = float(np.median(ref_g[ref_g > 1]))
    ref_mad = float(np.median(np.abs(ref_g[ref_g > 1] - ref_med))) + 1e-3

    acc = np.zeros((h, w, 3), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    for i, (p, H) in enumerate(zip(paths, Hs)):
        warped = ref_bgr if i == 0 else warp_to_ref(load_bgr(p), H)
        normed = additive_scale_norm(warped, ref_med, ref_mad)
        valid = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.float64)
        acc += normed
        cnt += valid
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f"  pass1 {i+1}/{len(paths)}", flush=True)

    mean = acc / np.maximum(cnt, 1.0)[..., None]

    acc2 = np.zeros_like(acc)
    cnt2 = np.zeros_like(cnt)
    thr = sigma * (ref_mad * 3.0 + 1.0)
    for i, (p, H) in enumerate(zip(paths, Hs)):
        warped = ref_bgr if i == 0 else warp_to_ref(load_bgr(p), H)
        normed = additive_scale_norm(warped, ref_med, ref_mad)
        valid = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        keep = valid & (np.abs(normed - mean).mean(axis=2) < thr)
        acc2 += np.where(keep[..., None], normed, 0.0)
        cnt2 += keep.astype(np.float64)
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f"  pass2 {i+1}/{len(paths)}", flush=True)

    return (acc2 / np.maximum(cnt2, 1.0)[..., None]).astype(np.float32)


def asinh_stretch(x: np.ndarray, soft: float) -> np.ndarray:
    return np.arcsinh(x * soft) / np.arcsinh(soft)


def enhance(bgr_f32: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(np.clip(bgr_f32, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    core = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2) < min(h, w) * 0.42
    black = float(np.percentile(g[core], 5))
    x = np.clip(bgr_f32 - black * 0.9, 0, None)
    white = float(np.percentile(x[core], 99.9)) + 1e-6
    x = np.clip(x / white, 0, 1)
    lum = 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]
    y = np.clip(x * (asinh_stretch(lum, 7.0) / (lum + 1e-6))[..., None], 0, 1)
    y = np.power(y, 0.88)
    u8 = (y * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(L)
    A = np.clip(128 + (A.astype(np.float32) - 128) * 1.18, 0, 255).astype(np.uint8)
    B = np.clip(128 + (B.astype(np.float32) - 128) * 1.14, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, 5, 18, 5)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.28, blur, -0.28, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "raw_copy")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--ref-seq", type=int, default=25)
    ap.add_argument("--min-seq", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=100, help="keep MW-rich earlier night; drop late dawnward")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-control-points", type=int, default=80)
    args = ap.parse_args()

    all_night = list_night_frames(args.input, max_seq=max(args.max_seq, 171))
    paths = [p for p in all_night if args.min_seq <= seq_of(p) <= args.max_seq]
    paths = paths[:: max(1, args.step)]
    if len(paths) < 5:
        raise SystemExit(f"not enough frames: {len(paths)}")

    ref_idx = int(np.argmin([abs(seq_of(p) - args.ref_seq) for p in paths]))
    # put reference first for clearer logging; keep separate index
    ref_path = paths[ref_idx]
    print(f"frames: {len(paths)}  reference: {ref_path.name}", flush=True)

    ref_bgr = load_bgr(ref_path)
    ref_gray = to_detect_gray(ref_bgr)

    print("matching each frame -> reference (Siril-style global registration)...", flush=True)
    Hs: list[np.ndarray | None] = []
    ok = 0
    ordered_paths: list[Path] = []
    ordered_Hs: list[np.ndarray | None] = []

    # reference first with identity
    ordered_paths.append(ref_path)
    ordered_Hs.append(np.eye(3, dtype=np.float64))

    for i, p in enumerate(paths):
        if p == ref_path:
            continue
        mov = load_bgr(p)
        mov_gray = to_detect_gray(mov)
        H, nin = find_homography_to_ref(
            mov_gray, ref_gray, max_control_points=args.max_control_points
        )
        if H is None:
            print(f"  FAIL {p.name}", flush=True)
            continue
        print(f"  OK   {p.name}  inliers={nin}", flush=True)
        ordered_paths.append(p)
        ordered_Hs.append(H)
        ok += 1

    print(f"aligned: {ok + 1}/{len(paths)} (incl. reference)", flush=True)
    if ok < 4:
        raise SystemExit("too few alignments succeeded")

    print("sigma-clip mean stacking...", flush=True)
    stack = sigma_clip_mean(ordered_paths, ordered_Hs, ref_bgr, sigma=2.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_u8 = np.clip(stack, 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.output_dir / "Milky_Way_stack_raw.tiff"), raw_u8)
    cv2.imwrite(
        str(args.output_dir / "Milky_Way_stack_raw.jpg"),
        raw_u8,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )

    # sharpness check
    lap = float(
        cv2.Laplacian(cv2.cvtColor(raw_u8, cv2.COLOR_BGR2GRAY)[270:810, 480:1440], cv2.CV_64F).var()
    )
    print(f"stack laplacian var (center): {lap:.2f}", flush=True)

    print("enhancing...", flush=True)
    pretty = enhance(stack)
    cv2.imwrite(str(args.output_dir / "Milky_Way.tiff"), pretty)
    cv2.imwrite(
        str(args.output_dir / "Milky_Way.jpg"),
        pretty,
        [int(cv2.IMWRITE_JPEG_QUALITY), 96],
    )
    cv2.imwrite(
        "/tmp/milky_way_preview.jpg",
        cv2.resize(pretty, None, fx=0.4, fy=0.4),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    print(f"DONE -> {args.output_dir / 'Milky_Way.jpg'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
