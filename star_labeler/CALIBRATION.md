# ASC Fisheye Astrometric Calibration

Plate solution for the all-sky camera, derived from 31 hand-labeled stars in a
single frame on 2026-09-15. Machine-readable versions live in
[`calibration.json`](calibration.json) (use this one) and
[`wcs.json`](wcs.json) (raw fit output, frame-specific).

**Accuracy: 0.28 px RMS on the 31 fitted stars, 0.38 px median on 175
independent stars that were not used in the fit.** Plate scale is 13.4 px/deg
near the axis, so that is roughly 1.7 arcmin.

---

## 1. Why this is worth keeping

The camera is fixed, and that is exactly what makes one calibration reusable
forever. The solution splits into three pieces with very different lifetimes:

| Piece | What it is | Changes when |
| --- | --- | --- |
| Lens intrinsics | optical centre, `r(theta)` polynomial, tilt term | never (barring a lens/sensor change) |
| Mount orientation | matrix `M`, i.e. 1.71° tilt + azimuth roll | only if the camera is bumped or remounted |
| Sidereal time | where the sky has rotated to | every frame, but it is fully determined by the timestamp |

So **alt/az ↔ pixel is a permanent, time-independent mapping**, and equatorial
coordinates need nothing but a UTC timestamp. Every frame in the archive —
past and future — can be plate-solved instantly without ever fitting stars
again.

What that unlocks: meteor tracks converted to equatorial coordinates and tested
against the Perseid radiant; per-pixel airmass maps for photometry and
transparency/cloud statistics; knowing exactly which part of the sky any pixel
is looking at, including which region is blocked by the trees.

---

## 2. Projection model

```
u          unit vector of the target, mean equator/equinox of date
v = R u    rotated into the camera frame (+z = optical axis)
theta      = acos(v_z)              angle from the optical axis
phi        = atan2(v_y, v_x)        azimuth within the camera frame
r(theta)   = f1*t + f3*t^3 + f5*t^5 + t^2*(a*cos(phi) + b*sin(phi))
x          = x0 + r*cos(phi)
y          = y0 + parity*r*sin(phi)
```

Ten free parameters: 3 for `R`, 2 for the centre, 3 radial, 2 for the tilt term.

The `a`/`b` term is a first-order correction for the lens axis not being quite
perpendicular to the sensor. It breaks the radial symmetry and grows with field
angle. It is not cosmetic: in a controlled comparison against stars outside the fit,
adding it improved the blind median from 0.57 px to 0.38 px and the region
beyond theta = 50° from about 2.5 px to about 2.0 px RMS. Because the gain
shows up on held-out stars, it is a real optical effect and not overfitting.

`parity = +1` was selected by brute force; `-1` fits 40x worse (10.8 px), which
confirms the image-axis handedness.

### Fitted values

```
x0, y0   = 968.8438, 624.3120        px   (frame centre is 960, 540)
f1,f3,f5 = 765.092973, 3.880508, 0.533616
a, b     = -2.042895, +1.813501      px/rad^2
parity   = +1
```

`M`, the camera frame as seen from local horizontal `(east, north, up)`:

```
-0.853456431  +0.520675139  +0.022572551
-0.521083810  -0.853284728  -0.019412261
+0.009153332  -0.028329710  +0.999556724
```

The radial part is very nearly equidistant (`r = f*theta`): the cubic and
quintic terms together contribute only 20 px at theta = 90°. Useful corollaries:

- `r(90°) = 1222 px`, so a full 180° image circle would be **2444 px across**
- the sensor is 1920x1080, i.e. **the frame is a centre crop of the image
  circle — the horizon is not in the field of view at all**
- scale near the axis: 13.36 px/deg = 4.49 arcmin/px

---

## 3. Geometry of this installation

| Quantity | Value |
| --- | --- |
| Site | Pomfret Center, CT — lat +41.879°, lon −71.966° |
| Optical axis tilt from vertical | **1.71°** toward azimuth 162.1° (roughly south) |
| Image +x axis roll | 319.3° |
| Zenith pixel | (986.1, 609.5) |
| Optical centre | (968.8, 624.3) |

### Field coverage

The horizon is outside the crop. Lowest altitude actually visible, by corner:

| Corner | theta from axis | altitude |
| --- | --- | --- |
| top-left | 85.1° | ~4.9° |
| top-right | 84.0° | ~6.0° |
| bottom-left | 79.2° | ~10.8° |
| bottom-right | 78.1° | ~11.9° |

Cardinal directions, at the lowest altitude still inside the frame:

| Direction | lowest visible altitude | pixel | where in frame |
| --- | --- | --- | --- |
| N | 37° | (1354, 3) | top, centre-right |
| E | 6° | (5, 32) | top-left corner |
| S | 49° | (699, 1078) | bottom, centre-left |
| W | 24.5° | (1732, 1077) | bottom-right |

So north is up and slightly right, east is to the left, south is down-left,
west is down-right. **Anything below 49° altitude in the south is not
recorded** — relevant when planning meteor or planetary coverage.

---

## 4. The tilt / clock ambiguity

Measured against the true zenith at Pomfret Center:

- Declination offset **−1.63°**. A clock error rotates the sky about the pole
  and cannot change declination, so this is unambiguously mechanical tilt.
- Right ascension offset **+0.68°**, which is equally consistent with more tilt
  or with a **+164 s clock offset**.

A single frame cannot separate the two. To resolve it, fit a second frame from
the same night several hours later: a clock error stays fixed in right
ascension while a mechanical tilt stays fixed in alt/az, so the two hypotheses
diverge as the sky rotates. If the camera clock is NTP-synced, 164 s is
implausible and the whole 1.71° is tilt.

---

## 5. Source data

- Frame: `2026-09-15_22-26-55_seq0001of0001_gain250_exp60.000s.tiff`, 1920x1080, 8-bit RGB, gain 250, 60 s
- Mid-exposure epoch: **2026-09-16 02:27:25 UTC** (local 22:26:55 EDT + 30 s), JD 2461299.602373
- Timing sensitivity: 1 s of clock error = 0.056 px of sky rotation
- Catalogue: HYG v4.0 (Hipparcos), J2000, precessed to epoch of date with the IAU 1976 matrix
- Constellation lines: Stellarium `skycultures/modern/index.json`, HIP-numbered chains
- Proper motion, aberration and refraction were not modelled. Refraction is
  under 0.5 px above 10° altitude and is partly absorbed by the radial
  polynomial; precession is a rigid rotation and is absorbed by `R` exactly.

### The 31 labeled stars

Saturn was also labeled at (168.21, 532.15) but is excluded from the fit, since
a planet needs an ephemeris rather than a catalogue position.

| label | HIP | name | Bayer | mag | x | y | theta | residual px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIP677 | 677 | Alpheratz | Alp And | 2.07 | 545.42 | 414.26 | 35.3° | 0.54 |
| HIP67301 | 67301 | Alkaid | Eta UMa | 1.85 | 1894.20 | 293.08 | 73.2° | 0.46 |
| HIP113963 | 113963 | Markab | Alp Peg | 2.49 | 527.68 | 699.71 | 33.4° | 0.42 |
| Vega | 91262 | Vega | Alp Lyr | 0.03 | 1366.40 | 775.91 | 31.8° | 0.41 |
| HIP98337 | 98337 | Gamma Sge | Gam Sge | 3.51 | 1067.26 | 981.45 | 27.7° | 0.40 |
| HIP113881 | 113881 | Scheat | Bet Peg | 2.44 | 651.76 | 575.20 | 24.0° | 0.39 |
| HIP100453 | 100453 | Sadr | Gam Cyg | 2.23 | 1107.45 | 695.78 | 11.7° | 0.38 |
| HIP75458 | 75458 | Edasich | Iot Dra | 3.29 | 1675.99 | 370.58 | 56.1° | 0.36 |
| HIP87833 | 87833 | Eltanin | Gam Dra | 2.24 | 1460.01 | 594.73 | 36.8° | 0.32 |
| HIP83895 | 83895 | Aldhibah | Zet Dra | 3.17 | 1489.02 | 375.52 | 43.2° | 0.30 |
| HIP5447 | 5447 | Mirach | Bet And | 2.07 | 548.34 | 209.66 | 44.1° | 0.29 |
| HIP65378 | 65378 | Mizar | Zet UMa | 2.23 | 1831.93 | 193.64 | 71.8° | 0.27 |
| Polaris | 11767 | Polaris | Alp UMi | 1.97 | 1308.76 | 56.62 | 49.5° | 0.26 |
| HIP107315 | 107315 | Enif | Eps Peg | 2.38 | 693.24 | 932.26 | 30.8° | 0.25 |
| Deneb | 102098 | Deneb | Alp Cyg | 1.25 | 1090.61 | 615.32 | 9.1° | 0.25 |
| HIP104732 | 104732 | Zeta Cyg | Zet Cyg | 3.21 | 923.60 | 755.51 | 10.4° | 0.24 |
| HIP109492 | 109492 | Zeta Cep | Zet Cep | 3.39 | 1026.12 | 372.38 | 19.4° | 0.22 |
| HIP112748 | 112748 | Sadalbari | Mu Peg | 3.51 | 648.26 | 639.45 | 24.0° | 0.20 |
| HIP62956 | 62956 | Alioth | Eps UMa | 1.76 | 1823.31 | 119.56 | 73.9° | 0.20 |
| HIP102488 | 102488 | Aljanah | Eps Cyg | 2.48 | 1014.46 | 746.90 | 9.8° | 0.18 |
| HIP105199 | 105199 | Alderamin | Alp Cep | 2.45 | 1132.07 | 376.70 | 22.2° | 0.17 |
| HIP87585 | 87585 | Grumium | Xi Dra | 3.73 | 1454.94 | 517.48 | 37.3° | 0.16 |
| HIP9640 | 9640 | Almach | Gam-1 And | 2.10 | 611.59 | 39.87 | 51.1° | 0.16 |
| HIP72607 | 72607 | Kochab | Bet UMi | 2.07 | 1529.58 | 177.76 | 53.6° | 0.15 |
| HIP80331 | 80331 | Athebyne | Eta Dra | 2.73 | 1576.33 | 401.41 | 48.4° | 0.15 |
| HIP112158 | 112158 | Matar | Eta Peg | 2.93 | 715.32 | 598.02 | 19.1° | 0.14 |
| HIP85670 | 85670 | Rastaban | Bet Dra | 2.79 | 1511.12 | 573.80 | 40.8° | 0.14 |
| HIP4427 | 4427 | Cih | Gam Cas | 2.15 | 901.06 | 120.59 | 38.0° | 0.14 |
| HIP746 | 746 | Caph | Bet Cas | 2.28 | 908.89 | 202.68 | 31.9° | 0.13 |
| HIP6686 | 6686 | Ruchbah | Del Cas | 2.66 | 883.80 | 75.01 | 41.6° | 0.10 |
| HIP3179 | 3179 | Schedar | Alp Cas | 2.24 | 852.49 | 164.93 | 35.4° | 0.06 |

---

## 6. Accuracy and its limits

Fitted stars span theta = 9.1° to 73.9° from the axis.

Blind tests compare the predicted pixel against the measured brightness
centroid for catalogue stars that were never used in the fit.

| Test | n | median | RMS |
| --- | --- | --- | --- |
| Fitted stars | 31 | 0.24 px | 0.28 px (max 0.54) |
| Blind, mag < 4.0 | 77 | 0.34 px | 1.25 px |
| Blind, mag < 4.5 | 175 | 0.38 px | 1.35 px |
| Blind, mag < 4.5, more than 50° off axis | 52 | 0.70 px | 1.96 px |

Residual structure worth knowing about:

- Errors are **radial, not tangential**, and grow beyond theta ≈ 50°.
- They are worst in image azimuth 120°–240° from the optical centre (the left
  and lower-left of the frame), which is the sector where the labeled stars
  only reach moderate field angles.
- Beyond theta = 74° the radial polynomial is extrapolating. It is nearly
  linear there so this is mild, but it is extrapolation.
- A few large blind-test outliers are measurement artefacts rather than model
  error: Alcor sits 12 arcmin from Mizar so the centroid window catches the
  wrong star, and stars in the vignetted tree corner have poor centroids.

**To tighten the outer field**, label 4–6 more stars at theta > 55° on the left
and lower-left (Hamal, Sheratan, Alpherg, Skat, iota Cet are all in frame and
detectable), then re-run `fit_wcs.py`. Nothing else needs changing.

---

## 7. How to use it

```python
from asc_wcs import AscWCS
w = AscWCS()

# time independent, valid for every frame this camera ever takes
x, y = w.altaz_to_pixel(45, 180)        # due south, 45 deg altitude
alt, az = w.pixel_to_altaz(960, 540)

# equatorial needs a UTC timestamp (year, month, day, hour, minute, second)
x, y = w.radec_to_pixel(20.69, 45.28, (2026, 9, 16, 2, 27, 25))   # Deneb, RA in hours
ra, dec = w.pixel_to_radec(960, 540, (2026, 9, 16, 2, 27, 25))     # RA in hours, J2000
```

Round trips are numerically exact (about 1e-12 px). `radec_*` take and return
J2000 by default; pass `j2000=False` to work in the mean frame of date.

### Regenerating things

```bash
python3 fit_wcs.py                                   # refit -> wcs.json
python3 asc_wcs.py --build --check                   # rebuild + verify calibration.json
python3 draw_sky.py                                  # constellation overlay
python3 draw_sky.py --grid altaz                     # + horizontal grid
python3 draw_sky.py --grid radec                     # + equatorial grid (J2000)
python3 draw_sky.py --raw                            # unstretched base image
python3 draw_sky.py --names hip|proper|none          # label style
python3 draw_sky.py --only Cyg Lyr Dra               # subset of constellations
python3 server.py 8777                               # the click-to-label web tool
```

### Files

| File | Contents |
| --- | --- |
| `calibration.json` | **the reusable calibration** — intrinsics + `M` + site |
| `asc_wcs.py` | `AscWCS` class, coordinate conversions both ways |
| `wcs.json` | raw fit output for this frame, including per-star residuals |
| `fit_wcs.py` | the fit itself (Kabsch initialisation + Levenberg-Marquardt) |
| `draw_sky.py` | overlay renderer |
| `labels.json` | the 31 hand-labeled stars + Saturn |
| `catalog.json` | 8872 HIP stars, trimmed from HYG v4.0 |
| `constellations.json` | 88 constellation line definitions |
| `server.py`, `static/` | the click-to-label tool |

### If the camera moves

Re-label about 20 stars with `server.py` on a fresh frame, update `UTC` in
`fit_wcs.py`, then re-run `fit_wcs.py` and `asc_wcs.py --build`. The lens
intrinsics should come out the same; only `M` should change.
