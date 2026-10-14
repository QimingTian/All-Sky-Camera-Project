# Drawing the N / E / S / W cross on the ASC frame

Frame: `2026-09-15_22-26-55_seq0001of0001_gain250_exp60.000s.tiff`, 1920x1080 px.

Every number here comes from the astrometric solution in [`CALIBRATION.md`](CALIBRATION.md) (0.28 px RMS on 31 stars). Because the camera is bolted down, **these pixel positions are the same in every frame this camera has ever taken or will take** - they do not depend on the date or time. They only change if the camera is physically moved.

---

## Read this first: the image is mirrored

The camera looks **up**, so the compass is mirrored compared to a map. On a map, north up means east is to the right. Here, **east is to the left**. Azimuth increases **counter-clockwise** in the image, not clockwise. Getting this backwards is the single easiest mistake to make.

The centre of the cross is the **zenith at pixel (986.1, 609.5)**. Note this is not the centre of the frame (960, 540): the optical axis sits at (968.8, 624.3) and the camera is tilted 1.71 deg from vertical, so the zenith lands slightly off centre. All four arms radiate from the zenith pixel.

---

## 1. The simple version: four straight arms

Draw a straight line from the zenith to each end point below. Screen angle is measured from straight up, clockwise, in the usual screen sense (x right, y down).

| Direction | Screen angle | Clock position | End point (x, y) | Arm length | Lowest altitude visible |
| --- | --- | --- | --- | --- | --- |
| **N** | 31.2° | 1 o'clock | (1355, 0) | 713 px | 36.8° |
| **NE** | 346.0° | 11:30 | (834, 0) | 628 px | 43.1° |
| **E** | 300.5° | 10 o'clock | (0, 29) | 1144 px | 5.7° |
| **SE** | 256.0° | 8:30 | (1, 855) | 1016 px | 15.0° |
| **S** | 211.5° | 7 o'clock | (698, 1080) | 551 px | 48.9° |
| **SW** | 166.7° | 5:30 | (1097, 1079) | 483 px | 54.0° |
| **W** | 122.1° | 4 o'clock | (1736, 1080) | 885 px | 24.2° |
| **NW** | 76.8° | 2:30 | (1920, 390) | 959 px | 18.6° |

So: **north is straight up tilted about 31° to the right, east is upper left, south is lower left, west is lower right.**

### How wrong is a straight line?

The arms are true great circles through the zenith. They would be exactly straight if the optical axis were the zenith, but the camera is tilted 1.71 deg, so they bow very slightly:

| Direction | Max deviation from a straight line | Worst at altitude |
| --- | --- | --- |
| N | 0.56 px | 62° |
| NE | 1.21 px | 66° |
| E | 5.13 px | 44° |
| SE | 1.80 px | 50° |
| S | 0.31 px | 69° |
| SW | 0.69 px | 72° |
| W | 2.79 px | 55° |
| NW | 1.64 px | 52° |

For a visual cross a straight line is fine everywhere except possibly the east arm. If you want it exact, use the polylines in section 2.

---

## 2. The exact version: polyline vertices

Vertices every 10° of altitude, from the lowest point still inside the frame up to the zenith. Join them with straight segments and the arm is accurate to well under a pixel. The first row of each table is the frame edge, the last row is the zenith (shared by all arms).

### N - north, azimuth 0°

| Altitude | x | y |
| --- | --- | --- |
| 36.8° (frame edge) | 1355.5 | 0.1 |
| 40° | 1332.9 | 37.7 |
| 50° | 1263.5 | 152.7 |
| 60° | 1194.2 | 267.3 |
| 70° | 1124.9 | 381.5 |
| 80° | 1055.6 | 495.5 |
| 90° | 986.1 | 609.5 |

### NE - north-east, azimuth 45°

| Altitude | x | y |
| --- | --- | --- |
| 43.1° (frame edge) | 833.8 | 0.3 |
| 50° | 857.0 | 90.3 |
| 60° | 890.0 | 220.3 |
| 70° | 922.5 | 350.1 |
| 80° | 954.5 | 479.7 |
| 90° | 986.1 | 609.5 |

### E - east, azimuth 90°

| Altitude | x | y |
| --- | --- | --- |
| 5.7° (frame edge) | 0.3 | 29.2 |
| 10° | 53.4 | 59.0 |
| 20° | 174.0 | 127.6 |
| 30° | 292.8 | 196.2 |
| 40° | 410.4 | 264.7 |
| 50° | 526.9 | 333.3 |
| 60° | 642.5 | 402.0 |
| 70° | 757.5 | 470.9 |
| 80° | 872.0 | 540.0 |
| 90° | 986.1 | 609.5 |

### SE - south-east, azimuth 135°

| Altitude | x | y |
| --- | --- | --- |
| 15.0° (frame edge) | 0.6 | 854.6 |
| 20° | 67.8 | 837.4 |
| 30° | 201.3 | 803.4 |
| 40° | 333.6 | 770.1 |
| 50° | 465.1 | 737.2 |
| 60° | 595.9 | 704.8 |
| 70° | 726.3 | 672.7 |
| 80° | 856.3 | 641.0 |
| 90° | 986.1 | 609.5 |

### S - south, azimuth 180°

| Altitude | x | y |
| --- | --- | --- |
| 48.9° (frame edge) | 697.8 | 1079.5 |
| 50° | 705.6 | 1066.9 |
| 60° | 776.3 | 952.0 |
| 70° | 846.5 | 837.6 |
| 80° | 916.5 | 723.5 |
| 90° | 986.1 | 609.5 |

### SW - south-west, azimuth 225°

| Altitude | x | y |
| --- | --- | --- |
| 54.0° (frame edge) | 1097.1 | 1079.4 |
| 60° | 1078.8 | 1000.1 |
| 70° | 1048.2 | 869.5 |
| 80° | 1017.3 | 739.3 |
| 90° | 986.1 | 609.5 |

### W - west, azimuth 270°

| Altitude | x | y |
| --- | --- | --- |
| 24.2° (frame edge) | 1736.4 | 1079.7 |
| 30° | 1669.3 | 1036.5 |
| 40° | 1555.0 | 963.5 |
| 50° | 1441.1 | 891.4 |
| 60° | 1327.3 | 820.1 |
| 70° | 1213.7 | 749.4 |
| 80° | 1100.0 | 679.2 |
| 90° | 986.1 | 609.5 |

### NW - north-west, azimuth 315°

| Altitude | x | y |
| --- | --- | --- |
| 18.6° (frame edge) | 1920.0 | 389.8 |
| 20° | 1901.3 | 394.1 |
| 30° | 1768.5 | 424.4 |
| 40° | 1636.9 | 454.9 |
| 50° | 1506.0 | 485.5 |
| 60° | 1375.7 | 516.2 |
| 70° | 1245.7 | 547.1 |
| 80° | 1115.9 | 578.2 |
| 90° | 986.1 | 609.5 |

---

## 3. Suggested label placement

Put each letter just inside the frame at the outer end of its arm, offset back along the arm so it does not get clipped. These are the points at 5° of altitude above each arm's lowest visible point:

| Direction | Label anchor (x, y) | Altitude there |
| --- | --- | --- |
| N | (1321, 58) | 41.8° |
| NE | (851, 66) | 48.1° |
| E | (61, 64) | 10.7° |
| SE | (68, 837) | 20.0° |
| S | (733, 1022) | 53.9° |
| SW | (1082, 1014) | 59.0° |
| W | (1679, 1043) | 29.2° |
| NW | (1853, 405) | 23.6° |

It is worth writing the altitude next to each letter, e.g. `S 49°`, because the four arms reach very different depths - see the warning below.

---

## 4. Things that will surprise whoever draws this

**The arms are wildly different lengths, and that is correct.** The frame is a 16:9 crop of a fisheye image circle that is about 2444 px across, so the horizon is nowhere in the picture. Each direction is cut off at a different altitude:

- **E** (east) is cut off at **5.7°** altitude
- **SE** (south-east) is cut off at **15.0°** altitude
- **NW** (north-west) is cut off at **18.6°** altitude
- **W** (west) is cut off at **24.2°** altitude
- **N** (north) is cut off at **36.8°** altitude
- **NE** (north-east) is cut off at **43.1°** altitude
- **S** (south) is cut off at **48.9°** altitude
- **SW** (south-west) is cut off at **54.0°** altitude

So the south arm is the shortest (551 px) and stops 49° above the horizon, while the east arm runs 1144 px and nearly reaches it. Anything in the southern sky below 49° altitude is simply not recorded.

**The frame border covers all 360° of azimuth.** Walking clockwise from the top-left corner:

| Border segment | Azimuth range | Directions crossed |
| --- | --- | --- |
| top edge, left to right | 0° .. 360° | E -> NE -> N -> NW |
| right edge, top to bottom | 275° .. 335° | NW -> W |
| bottom edge, right to left | 147° .. 275° | W -> SW -> S -> SE |
| left edge, bottom to top | 89° .. 147° | SE -> E |

(The top edge shows the full 0-360 range because it crosses due north, where azimuth wraps.)

**The trees block the best direction.** East reaches lowest (5.7°) but the trees and glow along the left edge and top-left corner sit at azimuth 89°-112° (east to east-south-east) at 10°-19° altitude. Example: pixel (60, 40) in the tree canopy looks at altitude 9.7°, azimuth 88.9°.

---

## 5. Sanity checks before you trust the drawing

**Polaris.** It sits at pixel (1309, 57) in this frame. Polaris is due north at an altitude equal to the site latitude, and solving that pixel gives altitude 42.13°, azimuth 0.97° against a true latitude of 41.879°. The north arm must pass essentially through Polaris. If it does not, something is wrong.

**Constellations.** For this frame (2026-09-15, 22:27 EDT) Cassiopeia is in the north-east near the top edge, the Big Dipper is low in the north-west at the upper right, Pegasus and Andromeda are rising in the east on the left side, and Hercules is setting in the west at the lower right. That is exactly right for a mid-September evening, and it would all be mirrored if east and west were swapped.

---

## 6. Computing any of this yourself

```python
from asc_wcs import AscWCS
w = AscWCS()

x, y = w.altaz_to_pixel(90, 0)        # zenith, the centre of the cross
x, y = w.altaz_to_pixel(30, 180)      # due south, 30 deg up
alt, az = w.pixel_to_altaz(1309, 57)  # what is this pixel looking at?
```

`altaz_to_pixel` and `pixel_to_altaz` need no timestamp - the camera does not move, so this mapping is permanent. Only `radec_to_pixel` / `pixel_to_radec` need a UTC time.

Regenerate this document with `python3 make_cardinal_doc.py`.
