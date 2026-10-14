#!/usr/bin/env python3
"""ASC AI v0 — multi-head Clear/Cloudy + rain (no snow until winter data)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

TOOL_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Volumes/James' SSD/ASC Training Data")
MARKED_DIR = DATA_DIR / "Marked For Training"
TEST_DIR = DATA_DIR / "Test"
CSV_PATH = TOOL_DIR / "labels.csv"
OUT_DIR = TOOL_DIR / "runs" / "v0"  # default; override with --out
BENCHMARK_MANIFEST = TOOL_DIR / "benchmark" / "manifest.json"
SOLAR_DIR = Path("/Users/tianqiming/Desktop/Pomfret Astro/website/observatory")

sys.path.insert(0, str(SOLAR_DIR))
from observatory_solar import is_asc_model_daytime  # noqa: E402
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")
_hh_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})\.(tiff|tif|jpg)$", re.I)
_nm_re = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})_.*\.(tiff|tif|jpg)$", re.I
)


def capture_local_dt(filename: str) -> datetime | None:
    m = _hh_re.match(filename)
    if m:
        return datetime(
            int(m.group(1)[:4]),
            int(m.group(1)[5:7]),
            int(m.group(1)[8:10]),
            int(m.group(2)),
            int(m.group(3)),
            tzinfo=TZ,
        )
    m = _nm_re.match(filename)
    if m:
        return datetime(
            int(m.group(1)[:4]),
            int(m.group(1)[5:7]),
            int(m.group(1)[8:10]),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            tzinfo=TZ,
        )
    return None


def nautical_day_night(filename: str) -> str:
    """Match production ASC: nautical dawn→dusk = day."""
    dt = capture_local_dt(filename)
    if dt is None:
        return ""
    return "day" if is_asc_model_daytime(dt.astimezone()) else "night"


@dataclass
class Sample:
    path: Path
    sky: int  # 0 clear, 1 cloudy
    rain: int  # 0/1
    day_night: str
    date: str


def load_frozen_test_names() -> set[str]:
    """Test filenames must never enter train/val."""
    names: set[str] = set()
    if BENCHMARK_MANIFEST.is_file():
        import json

        data = json.loads(BENCHMARK_MANIFEST.read_text(encoding="utf-8"))
        for row in data.get("test_files", []):
            if isinstance(row, dict) and row.get("filename"):
                names.add(row["filename"])
            elif isinstance(row, str):
                names.add(row)
    if TEST_DIR.is_dir():
        names.update(
            p.name
            for p in TEST_DIR.iterdir()
            if p.is_file() and not p.name.startswith("._")
        )
    return names


def resolve_image(name: str) -> Path | None:
    """Training images come only from Marked For Training. Never Test/ or unlabeled root."""
    if not MARKED_DIR.is_dir():
        return None
    path = MARKED_DIR / name
    if path.is_file():
        return path
    return None


def load_samples() -> list[Sample]:
    frozen_test = load_frozen_test_names()
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("sky") or not r.get("rain"):
                continue
            name = r["filename"]
            if name in frozen_test:
                continue  # never train on Test
            path = resolve_image(name)
            if path is None:
                continue
            sky = 0 if r["sky"].strip().lower() == "clear" else 1
            rain = 1 if r["rain"].strip().lower() == "true" else 0
            dn = nautical_day_night(name) or r.get("day_night") or ""
            rows.append(
                Sample(
                    path=path,
                    sky=sky,
                    rain=rain,
                    day_night=dn,
                    date=name[:10],
                )
            )
    return rows


def split_by_date(samples: list[Sample], val_dates: set[str] | None = None):
    dates = sorted({s.date for s in samples})
    if val_dates is None:
        # hold out the last ~25% of calendar days
        n_val = max(1, len(dates) // 4)
        val_dates = set(dates[-n_val:])
    train = [s for s in samples if s.date not in val_dates]
    val = [s for s in samples if s.date in val_dates]
    return train, val, sorted(val_dates)


class AscDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.samples = samples
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s.path) as im:
            rgb = im.convert("RGB")
        x = self.tf(rgb)
        return x, torch.tensor(s.sky), torch.tensor(s.rain)


class AscMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        in_f = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.sky_head = nn.Linear(in_f, 2)
        self.rain_head = nn.Linear(in_f, 2)

    def forward(self, x):
        f = self.backbone(x)
        return self.sky_head(f), self.rain_head(f)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sky_correct = rain_correct = n = 0
    sky_tp = sky_fp = sky_fn = 0  # cloudy as positive
    rain_tp = rain_fp = rain_fn = 0
    for x, sky, rain in loader:
        x = x.to(device)
        sky = sky.to(device)
        rain = rain.to(device)
        sky_logits, rain_logits = model(x)
        sky_pred = sky_logits.argmax(1)
        rain_pred = rain_logits.argmax(1)
        n += sky.size(0)
        sky_correct += (sky_pred == sky).sum().item()
        rain_correct += (rain_pred == rain).sum().item()
        for yt, yp in zip(sky.tolist(), sky_pred.tolist()):
            if yt == 1 and yp == 1:
                sky_tp += 1
            if yt == 0 and yp == 1:
                sky_fp += 1
            if yt == 1 and yp == 0:
                sky_fn += 1
        for yt, yp in zip(rain.tolist(), rain_pred.tolist()):
            if yt == 1 and yp == 1:
                rain_tp += 1
            if yt == 0 and yp == 1:
                rain_fp += 1
            if yt == 1 and yp == 0:
                rain_fn += 1

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    return {
        "n": n,
        "sky_acc": sky_correct / n if n else 0,
        "rain_acc": rain_correct / n if n else 0,
        "sky_cloudy_f1": f1(sky_tp, sky_fp, sky_fn),
        "rain_f1": f1(rain_tp, rain_fp, rain_fn),
        "sky_tp_fp_fn": [sky_tp, sky_fp, sky_fn],
        "rain_tp_fp_fn": [rain_tp, rain_fp, rain_fn],
    }


def class_weights(samples: list[Sample], attr: str):
    ys = [getattr(s, attr) for s in samples]
    counts = np.bincount(ys, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)


def train(args):
    global OUT_DIR
    OUT_DIR = Path(args.out)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_samples()
    if len(samples) < 20:
        raise SystemExit(f"Need more labels, only {len(samples)}")

    fixed_val_dates = None
    if args.val_dates_from:
        source = json.loads(Path(args.val_dates_from).read_text(encoding="utf-8"))
        fixed_val_dates = set(source["val_dates"])
        missing_dates = fixed_val_dates - {s.date for s in samples}
        if missing_dates:
            raise SystemExit(f"Validation dates absent from samples: {sorted(missing_dates)}")
    train_s, val_s, val_dates = split_by_date(samples, fixed_val_dates)
    print(f"samples={len(samples)} train={len(train_s)} val={len(val_s)} val_dates={val_dates}")
    print(
        "train sky",
        {0: sum(s.sky == 0 for s in train_s), 1: sum(s.sky == 1 for s in train_s)},
        "rain",
        {0: sum(s.rain == 0 for s in train_s), 1: sum(s.rain == 1 for s in train_s)},
    )
    print(
        "val sky",
        {0: sum(s.sky == 0 for s in val_s), 1: sum(s.sky == 1 for s in val_s)},
        "rain",
        {0: sum(s.rain == 0 for s in val_s), 1: sum(s.rain == 1 for s in val_s)},
    )
    dn = {}
    for s in samples:
        dn[s.day_night] = dn.get(s.day_night, 0) + 1
    print("nautical day_night counts:", dn)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    train_loader = DataLoader(
        AscDataset(train_s, train=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        AscDataset(val_s, train=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )

    model = AscMultiHead().to(device)
    sky_w = class_weights(train_s, "sky").to(device)
    rain_w = class_weights(train_s, "rain").to(device)
    print("loss weights sky", sky_w.tolist(), "rain", rain_w.tolist())
    crit_sky = nn.CrossEntropyLoss(weight=sky_w)
    crit_rain = nn.CrossEntropyLoss(weight=rain_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, sky, rain in pbar:
            x, sky, rain = x.to(device), sky.to(device), rain.to(device)
            opt.zero_grad()
            sky_logits, rain_logits = model(x)
            loss = crit_sky(sky_logits, sky) + crit_rain(rain_logits, rain)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        metrics = evaluate(model, val_loader, device)
        # score: prioritize sky acc + rain F1 (rare class)
        score = metrics["sky_acc"] + metrics["rain_f1"]
        history.append({"epoch": epoch, "train_loss": total_loss / max(n, 1), **metrics})
        print(
            f"epoch {epoch}: loss={total_loss/max(n,1):.3f} "
            f"sky_acc={metrics['sky_acc']:.3f} rain_acc={metrics['rain_acc']:.3f} "
            f"rain_f1={metrics['rain_f1']:.3f} score={score:.3f}"
        )
        if score >= best:
            best = score
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "val_dates": val_dates,
                "n_train": len(train_s),
                "n_val": len(val_s),
                "n_total_labeled": len(samples),
                "note": args.note,
                "config": {
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "seed": args.seed,
                    "val_dates_from": args.val_dates_from,
                },
            }
            torch.save(ckpt, OUT_DIR / "best.pt")
            with (OUT_DIR / "best_metrics.json").open("w") as f:
                json.dump(ckpt["metrics"] | {"epoch": epoch, "val_dates": val_dates}, f, indent=2)

    with (OUT_DIR / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"best score={best:.3f} → {OUT_DIR / 'best.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-dates-from", type=str, default=None)
    p.add_argument("--out", type=str, default=str(TOOL_DIR / "runs" / "v0"))
    p.add_argument(
        "--note",
        type=str,
        default="ASC multi-head Clear/Cloudy + rain; no snow head; nautical day/night metadata only; train on Marked For Training only",
    )
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
