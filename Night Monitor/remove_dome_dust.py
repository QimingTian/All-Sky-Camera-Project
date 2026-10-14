#!/usr/bin/env python3
"""
Remove stationary acrylic-dome dust from All Sky Camera frames.

Dust is fixed in image coordinates across the sequence; stars/sky move.
We estimate a static mask from a temporal median, then remove dust by
combining (1) multiplicative flat-field lift, (2) subtracting a static
RGB residual, and (3) temporal median of annular sky samples — real sky
that rotates behind each dust spot across the night. No spatial inpainting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def list_tiffs(folder: Path) -> list[Path]:
    files = [
        p
        for p in folder.glob("*.tiff")
        if not p.name.startswith("._") and p.is_file()
    ]
    files.sort(key=lambda p: p.name)
    return files


def seq_num(path: Path) -> int | None:
    m = re.search(r"seq(\d+)of", path.name)
    return int(m.group(1)) if m else None


def load_bgr_u8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def temporal_median_gray(paths: list[Path], max_frames: int = 80) -> np.ndarray:
    """Median luminance stack (subsample if many frames)."""
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]

    sample = load_bgr_u8(paths[0])
    h, w = sample.shape[:2]
    stack = np.empty((len(paths), h, w), dtype=np.uint8)
    for i, p in enumerate(paths):
        bgr = load_bgr_u8(p) if i else sample
        stack[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            print(f"  stacked {i + 1}/{len(paths)}", flush=True)
    return np.median(stack, axis=0).astype(np.float32)


def build_dust_mask(
    median_gray: np.ndarray,
    dog_sigma_small: float = 3.0,
    dog_sigma_large: float = 28.0,
    abs_thresh: float = 4.5,
    min_area: int = 40,
    max_area: int = 18000,
    border_tree_exclude: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect soft mid-scale stationary blobs via Difference-of-Gaussians on the
    temporal median. Large/elongated edge structure (trees) is rejected.
    """
    g_small = cv2.GaussianBlur(median_gray, (0, 0), dog_sigma_small)
    g_large = cv2.GaussianBlur(median_gray, (0, 0), dog_sigma_large)
    dog = g_small - g_large  # soft dome dust residual

    mad = np.median(np.abs(dog - np.median(dog))) + 1e-6
    score = np.abs(dog) / (1.4826 * mad)
    raw = (score > abs_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    h, w = median_gray.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    mask = np.zeros((h, w), dtype=np.uint8)
    kept = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(1, min(bw, bh))
        # circularity proxy: area vs bounding box
        fill = area / float(max(1, bw * bh))
        touches_left = x <= 2
        if border_tree_exclude and touches_left and (area > 1500 or aspect > 3.5):
            continue
        if aspect > 6.0 and area > 2000:
            continue
        # prefer blob-like over thin streaks
        if fill < 0.2 and area > 800:
            continue
        mask[labels == i] = 255
        kept += 1

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 1)
    print(f"  dust components kept: {kept}", flush=True)
    return mask, dog, score


def build_static_residual_bgr(paths: list[Path], max_frames: int = 60) -> np.ndarray:
    """Temporal median RGB used to estimate fixed dust color residual."""
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]
    sample = load_bgr_u8(paths[0]).astype(np.float32)
    acc = sample.copy()
    for i, p in enumerate(paths[1:], start=2):
        acc += load_bgr_u8(p).astype(np.float32)
        # running mean is enough for residual estimate and cheaper than median RGB
    mean = acc / float(len(paths))
    # mid-scale residual relative to large-scale sky/vignette
    blur = cv2.GaussianBlur(mean, (0, 0), 30)
    residual = mean - blur
    return residual.astype(np.float32)


def _annulus_kernel(inner: int, outer: int) -> np.ndarray:
    yy, xx = np.ogrid[-outer : outer + 1, -outer : outer + 1]
    r2 = xx * xx + yy * yy
    return ((r2 >= inner * inner) & (r2 <= outer * outer)).astype(np.float32)


def prepare_ring_source(img: np.ndarray, mask: np.ndarray, blur_sigma: float = 18.0) -> np.ndarray:
    """Replace dust (and a small halo) with blurred sky so ring stats aren't biased."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    m = cv2.dilate((mask > 0).astype(np.uint8), k, iterations=1)
    blur = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    out = img.copy()
    out[m > 0] = blur[m > 0]
    return out


def annular_mean(img: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Per-pixel weighted mean in an annular neighborhood (float32 BGR)."""
    ring_use = ring.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]
    result = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        acc = cv2.filter2D(img[..., c], -1, ring_use, borderType=cv2.BORDER_REFLECT)
        denom = cv2.filter2D(np.ones(img.shape[:2], np.float32), -1, ring_use, borderType=cv2.BORDER_REFLECT)
        result[..., c] = acc / np.maximum(denom, 1e-3)
    return result


def frame_sky_fill(bgr: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Per-frame annular sky estimate (rotates with the night sky)."""
    return annular_mean(prepare_ring_source(bgr.astype(np.float32), mask), ring)


def build_flat_field(
    paths: list[Path],
    max_frames: int = 80,
    blur_sigma: float = 35.0,
) -> tuple[np.ndarray, float]:
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).astype(int)
        paths = [paths[i] for i in idx]
    sample = load_bgr_u8(paths[0]).astype(np.float32)
    acc = sample.copy()
    for p in paths[1:]:
        acc += load_bgr_u8(p).astype(np.float32)
    mean = acc / float(len(paths))
    gray = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    flat = gray / np.maximum(smooth, 8.0)
    target = float(np.median(flat))
    return flat, target


def clean_frame(
    bgr: np.ndarray,
    mask: np.ndarray,
    static_residual: np.ndarray | None = None,
    ring: np.ndarray | None = None,
    flat_gray: np.ndarray | None = None,
    flat_target: float = 1.0,
    residual_gain: float = 1.0,
    flat_strength: float = 0.75,
) -> np.ndarray:
    if not np.any(mask):
        return bgr
    out = bgr.astype(np.float32)
    m = mask > 0

    if flat_gray is not None:
        lift = (flat_target / np.maximum(flat_gray, 0.12)) ** flat_strength
        for c in range(3):
            out[..., c][m] = np.clip(out[..., c][m] * lift[m], 0, 255)

    if static_residual is not None:
        out[m] = out[m] - residual_gain * static_residual[m]

    if ring is not None:
        sky = frame_sky_fill(bgr, mask, ring)
        out[m] = sky[m]

    return np.clip(out, 0, 255).astype(np.uint8)


def save_diagnostics(
    out_dir: Path,
    median_gray: np.ndarray,
    dog: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    sample_bgr: np.ndarray,
    sample_clean: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm_u8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        if hi <= lo:
            hi = lo + 1
        y = np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return y

    cv2.imwrite(str(out_dir / "01_temporal_median.jpg"), norm_u8(median_gray))
    cv2.imwrite(str(out_dir / "02_dog_residual.jpg"), norm_u8(dog))
    cv2.imwrite(str(out_dir / "03_dust_score.jpg"), norm_u8(score))
    cv2.imwrite(str(out_dir / "04_dust_mask.png"), mask)

    overlay = sample_bgr.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.4 + np.array([0, 0, 255], dtype=np.float32) * 0.6
    ).astype(np.uint8)
    cv2.imwrite(str(out_dir / "05_mask_overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / "06_sample_before.jpg"), sample_bgr)
    cv2.imwrite(str(out_dir / "07_sample_after.jpg"), sample_clean)

    # side-by-side
    gap = np.zeros((sample_bgr.shape[0], 16, 3), dtype=np.uint8)
    side = np.concatenate([sample_bgr, gap, sample_clean], axis=1)
    cv2.imwrite(str(out_dir / "08_before_after.jpg"), side)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "raw_copy",
        help="folder of source TIFFs (use raw_copy to protect originals)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "cleaned",
        help="folder for cleaned TIFFs",
    )
    ap.add_argument(
        "--diagnostics",
        type=Path,
        default=Path(__file__).resolve().parent / "diagnostics",
    )
    ap.add_argument("--max-stack", type=int, default=80)
    ap.add_argument("--abs-thresh", type=float, default=4.5)
    ap.add_argument("--limit", type=int, default=0, help="process only first N (0=all)")
    args = ap.parse_args()

    paths = list_tiffs(args.input)
    if not paths:
        raise SystemExit(f"no tiffs in {args.input}")
    if args.limit:
        paths = paths[: args.limit]

    print(f"input: {args.input} ({len(list_tiffs(args.input))} tiffs)")
    print(f"processing: {len(paths)} frames")
    args.output.mkdir(parents=True, exist_ok=True)

    print("building temporal median...", flush=True)
    # Use all available raw_copy frames for median (not just process list)
    all_for_stack = list_tiffs(args.input)
    median = temporal_median_gray(all_for_stack, max_frames=args.max_stack)

    print("detecting dust mask...", flush=True)
    mask, dog, score = build_dust_mask(median, abs_thresh=args.abs_thresh)
    coverage = 100.0 * float(np.count_nonzero(mask)) / mask.size
    print(f"  mask coverage: {coverage:.3f}%", flush=True)

    print("estimating static RGB residual...", flush=True)
    static_residual = build_static_residual_bgr(all_for_stack, max_frames=min(60, args.max_stack))

    print("building flat field...", flush=True)
    flat_gray, flat_target = build_flat_field(all_for_stack, max_frames=args.max_stack)
    ring = _annulus_kernel(10, 28)

    sample = load_bgr_u8(paths[min(10, len(paths) - 1)])
    sample_clean = clean_frame(
        sample,
        mask,
        static_residual=static_residual,
        ring=ring,
        flat_gray=flat_gray,
        flat_target=flat_target,
    )
    save_diagnostics(args.diagnostics, median, dog, score, mask, sample, sample_clean)
    print(f"diagnostics -> {args.diagnostics}", flush=True)

    print("cleaning frames...", flush=True)
    for i, p in enumerate(paths):
        bgr = load_bgr_u8(p)
        out = clean_frame(
            bgr,
            mask,
            static_residual=static_residual,
            ring=ring,
            flat_gray=flat_gray,
            flat_target=flat_target,
        )
        ok = cv2.imwrite(str(args.output / p.name), out)
        if not ok:
            raise RuntimeError(f"failed to write {args.output / p.name}")
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            print(f"  cleaned {i + 1}/{len(paths)}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
