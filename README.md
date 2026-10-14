# All-Sky Camera Project

A long-running all-sky camera archive and the tools used to collect, inspect, label, calibrate, and process its imagery. The camera records a fixed view of the sky from Pomfret Center, Connecticut.

## Project goals

The central goal is to build a useful, sustainable record of the sky without retaining a full-resolution frame every minute. The routine archive cadence is **one frame every half hour**. That interval keeps the archive manageable while still supporting long-term studies of sky conditions, seasonal changes, solar paths, stars, and other events. Separate higher-cadence observing runs can be made for specific events such as meteor showers.

The half-hour archive is general-purpose source data. **Analemma is one downstream analysis**, not the purpose of the archive. The same frames can support future visualizations, calibration checks, weather and sky-condition analysis, and other projects.

## Repository and data policy

This repository contains source code, documentation, small configuration/catalog files, and selected lightweight benchmark artifacts. The full-resolution TIFF archive and training image corpus are large datasets and should remain outside Git (for example, on local/external storage or a dedicated data service). Do not commit raw `.tiff`/`.tif` camera frames, bulk image collections, virtual environments, or generated video/render output unless a small artifact is explicitly selected for documentation.

Expected local data locations:

```text
Half Hour/                         # YYYY-MM-DD_HHMM.tiff archive (local data)
ASC Training Data/                 # local labeled image corpus (separate folder)
ASC Label Tool/                    # local labeling/training app (separate folder)
```

The dataset is not necessarily complete at every half-hour mark. Missing frames reflect capture or storage gaps; filenames use local wall-clock time (`America/New_York`).

## Project areas

### `Half Hour/` — general archive and analysis tools

The directory contains the all-sky archive, normally named `YYYY-MM-DD_HHMM.tiff`. Captures are taken every 30 minutes. Keep the camera fixed if you want the astrometric calibration to remain valid.

`Half Hour/analemma/` is a local web application that uses the archive to build a solar analemma. It lets you choose a twilight base plate, select a clock time, detect/review solar positions, and render a composite. Its project annotations are stored in `projects/current.json`; render outputs are in `output/`.

Run it locally:

```bash
cd "Half Hour/analemma"
./run.sh
```

Then open <http://127.0.0.1:5757>.

More detail: [`Half Hour/analemma/README.md`](Half%20Hour/analemma/README.md).

### `Night Monitor/` — event-specific high-cadence observing

This area holds clear-night observing runs, including the Perseid monitoring work begun in July 2026. Unlike the general archive, these runs use back-to-back 60-second exposures (gain 200) to capture time-varying events. Each night is stored in a month.day folder such as `7.19/`.

Typical processing:

```bash
cd "Night Monitor"
python3 build_timelapse.py --src 7.19
python3 stack_star_trail.py --src 7.19
python3 build_star_trail_video.py --src 7.19
```

The star-trail stack uses a classic lighten composite (per-pixel maximum) and filters dawn/bright frames. The scripts expose options to adjust those thresholds. Check `--help` before processing a new night. Raw frames and generated videos remain data products, not Git source files.

### `star_labeler/` — fixed-camera sky coordinates

A small local browser tool and Python utilities for working with star labels, visualizing sky coordinates, and calibrating the camera’s fisheye mapping. The calibration maps pixels to altitude/azimuth without a timestamp; equatorial coordinates also require the frame’s UTC time. It assumes the camera has not moved.

Start the label viewer with:

```bash
cd star_labeler
python3 server.py
```

Read [`star_labeler/CALIBRATION.md`](star_labeler/CALIBRATION.md) and [`star_labeler/CARDINAL_DIRECTIONS.md`](star_labeler/CARDINAL_DIRECTIONS.md) before using or changing the coordinate model. In particular, the all-sky view is mirrored: east appears on the left side of the image.

### `ASC_Benchmark/` — selected model evaluation artifacts

Contains benchmark manifests/results and compact plots/previews related to the ASC image-classification work. Keep manifests and reproducibility metadata with the code; large source images belong in the separate dataset storage.

The [v1.1 report](ASC_Benchmark/v1.1/BENCHMARK.md) compares v1 and v1.1 on the original frozen 178-frame Test set. The [v1.2 report](ASC_Benchmark/v1.2/BENCHMARK.md) evaluates all three versions on that same set plus a separate 47-frame September 5 holdout. On the 225-frame combined set, v1.2 ties v1 on GO/NO-GO accuracy (92.89%), while on the original 178 frames v1 remains slightly ahead (92.13% vs 91.57%). v1 remains the preferred GO/NO-GO model pending more independent test dates. Full checkpoints remain local under `ASC Label Tool/model_backups/`, outside Git.

### `ASC Label Tool/` — labeling and model workflow

The repository includes the local labeling UI, its `labels.csv`, training/prediction scripts, and compact benchmark metadata. The image corpus remains outside Git in a separate `ASC Training Data/` folder with sets such as `Marked For Training/` and `Test/`.

The tool currently points at `/Volumes/James' SSD/ASC Training Data`. See [`ASC Label Tool/README.md`](ASC%20Label%20Tool/README.md) for the UI workflow. This workstation-specific path may need adjustment on another machine. Virtual environments, image corpora, personal label backups, and model run outputs remain outside the repository.

## Camera and calibration notes

- Camera: fixed ASC fisheye all-sky camera; typical frame is 1920 × 1080.
- Site recorded by the calibration: Pomfret Center, CT (latitude +41.879°, longitude −71.966°).
- The sensor view is a crop of the fisheye image circle; it does not include the full horizon in every direction.
- Calibration was fit from one frame and evaluated on held-out stars. See `star_labeler/CALIBRATION.md` for the model, measured error, and assumptions.
- If the camera, lens, sensor, or mount changes, re-check or redo the calibration.

## Dependencies

There is no single environment for every subproject. Install dependencies only for the tool you need:

- Analemma: see `Half Hour/analemma/requirements.txt`; it also uses the local `star_labeler` calibration module.
- Star labeling/calibration: Python packages imported by scripts in `star_labeler/` (notably Astropy, NumPy, and Pillow/OpenCV as applicable).
- Night Monitor: Python OpenCV and NumPy; video encoding scripts also require `ffmpeg` on `PATH`.
- ASC Label Tool: see its Python imports and local `.venv`; image/model dependencies are specific to that separate workflow.

Avoid committing virtual environments. A future cleanup can add pinned, per-tool requirements files where needed.

## Working with the archive

- Preserve the timestamp filename convention: `YYYY-MM-DD_HHMM.tiff` (local New York wall time).
- Keep raw archive frames separate from generated previews and analysis outputs.
- Record camera setting changes and gaps when known; fixed settings and pointing make longitudinal comparisons more meaningful.
- For large transfers, verify the newest timestamp and file counts after copying. Avoid re-downloading files already present.
- Treat any calibration values as specific to the current physical camera installation.

## License and data access

No license or redistribution terms are declared yet. Before publishing raw imagery or labels, decide what data is intended to be public and document its license/source. This repository is intended for project code and reproducibility materials; the bulk archive is managed separately.
