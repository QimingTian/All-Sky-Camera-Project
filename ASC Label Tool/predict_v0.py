#!/usr/bin/env python3
"""Quick demo: run ASC v0 on a few labeled or unlabeled frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
from train_v0 import AscMultiHead, DATA_DIR, OUT_DIR, resolve_image, MARKED_DIR  # noqa: E402

TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(paths: list[Path], ckpt_path: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AscMultiHead().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("ckpt metrics:", ckpt.get("metrics"))
    print("val_dates:", ckpt.get("val_dates"))
    print("-" * 60)
    for path in paths:
        with Image.open(path) as im:
            x = TF(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            sky_logits, rain_logits = model(x)
            sky_p = torch.softmax(sky_logits, 1)[0]
            rain_p = torch.softmax(rain_logits, 1)[0]
        sky = "clear" if sky_p[0] >= sky_p[1] else "cloudy"
        rain = "true" if rain_p[1] >= rain_p[0] else "false"
        nogo = rain == "true" or sky == "cloudy"
        print(
            f"{path.name}\n"
            f"  sky={sky} (clear={sky_p[0]:.2f} cloudy={sky_p[1]:.2f})  "
            f"rain={rain} (no={rain_p[0]:.2f} yes={rain_p[1]:.2f})  "
            f"gate={'NO-GO' if nogo else 'OK'}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image filenames or paths")
    ap.add_argument("--ckpt", type=Path, default=OUT_DIR / "best.pt")
    args = ap.parse_args()
    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {args.ckpt}")
    if args.images:
        paths = []
        for a in args.images:
            p = Path(a)
            if p.is_file():
                paths.append(p)
            else:
                resolved = resolve_image(a)
                if resolved is None:
                    raise SystemExit(f"missing image: {a}")
                paths.append(resolved)
    else:
        # a few defaults from training pool
        names = sorted(p.name for p in DATA_DIR.glob("2026-06-0*.tiff"))[:8]
        if not names:
            marked = MARKED_DIR
            names = sorted(p.name for p in marked.glob("2026-06-0*.tiff"))[:8]
        paths = [resolve_image(n) for n in names]
        paths = [p for p in paths if p is not None]
    predict(paths, args.ckpt)


if __name__ == "__main__":
    main()
