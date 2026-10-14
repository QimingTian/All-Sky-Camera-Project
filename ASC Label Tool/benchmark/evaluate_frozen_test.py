#!/usr/bin/env python3
"""Evaluate ASC sky/rain checkpoints on the unchanged official Test manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR))
from train_v0 import AscMultiHead, TEST_DIR  # noqa: E402

MANIFEST = Path(__file__).with_name("manifest.json")
TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FrozenTest(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(TEST_DIR / row["filename"]) as im:
            x = TF(im.convert("RGB"))
        return x, i


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def metrics(predictions: list[dict]) -> dict:
    n = len(predictions)
    sky_tp = sky_fp = sky_fn = rain_tp = rain_fp = rain_fn = 0
    gate_tp = gate_fp = gate_fn = gate_tn = 0
    sky_correct = rain_correct = 0
    for p in predictions:
        sky_true = p["sky_true"] == "cloudy"
        sky_pred = p["sky_pred"] == "cloudy"
        rain_true = p["rain_true"] == "true"
        rain_pred = p["rain_pred"] == "true"
        sky_correct += sky_true == sky_pred
        rain_correct += rain_true == rain_pred
        sky_tp += sky_true and sky_pred
        sky_fp += not sky_true and sky_pred
        sky_fn += sky_true and not sky_pred
        rain_tp += rain_true and rain_pred
        rain_fp += not rain_true and rain_pred
        rain_fn += rain_true and not rain_pred
        gate_true = sky_true or rain_true  # positive = no-go
        gate_pred = sky_pred or rain_pred
        gate_tp += gate_true and gate_pred
        gate_fp += not gate_true and gate_pred
        gate_fn += gate_true and not gate_pred
        gate_tn += not gate_true and not gate_pred
    return {
        "n": n,
        "sky_acc": sky_correct / n,
        "rain_acc": rain_correct / n,
        "sky_cloudy_f1": f1(sky_tp, sky_fp, sky_fn),
        "rain_f1": f1(rain_tp, rain_fp, rain_fn),
        "rain_recall": rain_tp / (rain_tp + rain_fn),
        "rain_precision": rain_tp / (rain_tp + rain_fp),
        "gate_acc": (gate_tp + gate_tn) / n,
        "false_ok_rate": gate_fn / n,
        "false_nogo_rate": gate_fp / n,
        "false_ok_count": gate_fn,
        "false_nogo_count": gate_fp,
        "sky_tp_fp_fn": [sky_tp, sky_fp, sky_fn],
        "rain_tp_fp_fn": [rain_tp, rain_fp, rain_fn],
        "gate_tp_fp_fn_tn": [gate_tp, gate_fp, gate_fn, gate_tn],
    }


def evaluate(name: str, path: Path, rows: list[dict], batch: int) -> tuple[dict, list[dict]]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AscMultiHead().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(FrozenTest(rows), batch_size=batch, shuffle=False, num_workers=0)
    predictions: list[dict] = [None] * len(rows)
    with torch.no_grad():
        for x, indices in loader:
            sky_logits, rain_logits = model(x.to(device))
            sky_probs = torch.softmax(sky_logits, dim=1).cpu().tolist()
            rain_probs = torch.softmax(rain_logits, dim=1).cpu().tolist()
            for i, sky_prob, rain_prob in zip(indices.tolist(), sky_probs, rain_probs):
                row = rows[i]
                predictions[i] = {
                    "filename": row["filename"],
                    "sky_true": row["sky"],
                    "rain_true": row["rain"],
                    "sky_pred": "cloudy" if sky_prob[1] > sky_prob[0] else "clear",
                    "rain_pred": "true" if rain_prob[1] > rain_prob[0] else "false",
                    "cloudy_probability": sky_prob[1],
                    "rain_probability": rain_prob[1],
                }
    result = {"model": name, "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metrics(predictions)}
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "CHECKPOINT"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["test_files"]
    names = [row["filename"] for row in rows]
    actual = {p.name for p in TEST_DIR.iterdir() if p.is_file() and not p.name.startswith("._")}
    if len(names) != len(set(names)) or set(names) != actual or len(names) != manifest["n_test"]:
        raise SystemExit("Frozen Test folder does not match benchmark manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, checkpoint in args.model:
        result, predictions = evaluate(name, Path(checkpoint), rows, args.batch)
        results.append(result)
        with (args.out / f"test_predictions_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(name, json.dumps(result, indent=2), flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_n": len(rows),
        "test_filenames_sha256": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
        "models": results,
    }
    (args.out / "test_benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
