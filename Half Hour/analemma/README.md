# Analemma Tool

Local web app for building a solar analemma from `Half Hour/` frames.

## Workflow

1. **Pick a twilight base plate** — browse all frames from **18:00–21:00** and choose one clear sky yourself
2. **Set a clock time in EST (standard time)** — default **09:30 EST**, the only half-hour slot whose
   whole-year figure-8 stays in frame (see `plan_analemma_time.py`). During DST the matching file
   is one hour later (`YYYY-MM-DD_1030.tiff`); the tool picks it automatically, so the analemma
   does not split in two at the DST changes.
3. **Auto-detect** — fits a circle to the edge of the saturated sun glare in each frame. Days with
   no clean sun disk, or whose detection breaks away from the neighbouring days' trend (glare
   merged with lit cloud), are skipped. Review frame by frame; a click overrides the detection,
   and manual annotations are never overwritten by a re-run.
4. **Render** — draws a clean sun at each position on the base plate (the real glare is hundreds
   of pixels wide and would smear the days together). Sun size is adjustable.

Progress saves automatically to `projects/current.json`.

## Run

```bash
cd "Half Hour/analemma"
./run.sh
```

Open **http://127.0.0.1:5757**

## Output

Rendered files go to `Half Hour/analemma/output/`:

- `Analemma.tiff`
- `Analemma.jpg`

## Keys (annotate step)

- **Click** — set sun center (auto-advances)
- **S** — skip frame
- **← / →** — previous / next frame

The cyan cross is the ephemeris position (astropy sun + `star_labeler/calibration.json`). The
detected glare centre typically sits 10–20 px from it; that offset is smooth across the season.

## Planning a clock time

```bash
.venv/bin/python plan_analemma_time.py
.venv/bin/python plan_analemma_time.py --margin 0 --overlay 2026-08-20_1200.tiff --times 0900 0930 1000
```

## Requirements

- Fixed camera all year (same pointing as the star calibration)
- Half Hour filenames: `YYYY-MM-DD_HHMM.tiff` in local wall-clock time
