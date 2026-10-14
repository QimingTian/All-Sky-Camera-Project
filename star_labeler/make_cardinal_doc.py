#!/usr/bin/env python3
"""Generate CARDINAL_DIRECTIONS.md - everything needed to draw the N/E/S/W
cross on an ASC frame. Regenerate after any recalibration:

    python3 make_cardinal_doc.py
"""

import os
import numpy as np
from asc_wcs import AscWCS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'CARDINAL_DIRECTIONS.md')
DIRS = [(0, 'N', 'north'), (45, 'NE', 'north-east'), (90, 'E', 'east'),
        (135, 'SE', 'south-east'), (180, 'S', 'south'), (225, 'SW', 'south-west'),
        (270, 'W', 'west'), (315, 'NW', 'north-west')]


def main():
    w = AscWCS()
    W, H = w.width, w.height
    zx, zy = (v[0] for v in w.altaz_to_pixel(90, 0))

    arms = {}
    for az, tag, _ in DIRS:
        alt = np.arange(0, 90.0001, 0.05)
        x, y = w.altaz_to_pixel(alt, az)
        i = np.flatnonzero((x >= 0) & (x <= W) & (y >= 0) & (y <= H))
        lo = i[0]
        ex, ey = x[lo], y[lo]
        d = np.array([ex - zx, ey - zy])
        L = np.hypot(*d)
        u = d / L
        perp = np.abs(-u[1] * (x[i] - zx) + u[0] * (y[i] - zy))
        arms[tag] = {
            'az': az, 'alt_min': alt[lo], 'end': (ex, ey), 'chord': L,
            'screen_angle': np.degrees(np.arctan2(d[0], -d[1])) % 360,
            'max_dev': perp.max(), 'dev_alt': alt[i][perp.argmax()],
        }

    def ticks(az, alt_min, step=10):
        rows = []
        for a in range(0, 91, step):
            if a < alt_min - 1e-9:
                continue
            x, y = w.altaz_to_pixel(float(a), float(az))
            rows.append((a, x[0], y[0]))
        x, y = w.altaz_to_pixel(float(alt_min), float(az))
        rows.insert(0, (alt_min, x[0], y[0]))
        return rows

    L = []
    L.append('# Drawing the N / E / S / W cross on the ASC frame')
    L.append('')
    L.append(f'Frame: `{w.meta["fit"]["source_frame"]}`, {W}x{H} px.')
    L.append('')
    L.append('Every number here comes from the astrometric solution in '
             '[`CALIBRATION.md`](CALIBRATION.md) (0.28 px RMS on 31 stars). Because the '
             'camera is bolted down, **these pixel positions are the same in every frame '
             'this camera has ever taken or will take** - they do not depend on the date '
             'or time. They only change if the camera is physically moved.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## Read this first: the image is mirrored')
    L.append('')
    L.append('The camera looks **up**, so the compass is mirrored compared to a map. '
             'On a map, north up means east is to the right. Here, **east is to the '
             'left**. Azimuth increases **counter-clockwise** in the image, not '
             'clockwise. Getting this backwards is the single easiest mistake to make.')
    L.append('')
    L.append(f'The centre of the cross is the **zenith at pixel ({zx:.1f}, {zy:.1f})**. '
             f'Note this is not the centre of the frame ({W//2}, {H//2}): the optical axis '
             f'sits at ({w.x0:.1f}, {w.y0:.1f}) and the camera is tilted '
             f'{w.meta["tilt_from_vertical_deg"]:.2f} deg from vertical, so the zenith lands '
             'slightly off centre. All four arms radiate from the zenith pixel.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 1. The simple version: four straight arms')
    L.append('')
    L.append('Draw a straight line from the zenith to each end point below. Screen angle '
             'is measured from straight up, clockwise, in the usual screen sense '
             '(x right, y down).')
    L.append('')
    L.append('| Direction | Screen angle | Clock position | End point (x, y) | Arm length | Lowest altitude visible |')
    L.append('| --- | --- | --- | --- | --- | --- |')
    clock = {'N': '1 o\'clock', 'NE': '11:30', 'E': '10 o\'clock', 'SE': '8:30',
             'S': '7 o\'clock', 'SW': '5:30', 'W': '4 o\'clock', 'NW': '2:30'}
    for az, tag, _ in DIRS:
        a = arms[tag]
        L.append(f'| **{tag}** | {a["screen_angle"]:.1f}° | {clock[tag]} | '
                 f'({a["end"][0]:.0f}, {a["end"][1]:.0f}) | {a["chord"]:.0f} px | {a["alt_min"]:.1f}° |')
    L.append('')
    L.append('So: **north is straight up tilted about 31° to the right, east is upper '
             'left, south is lower left, west is lower right.**')
    L.append('')
    L.append('### How wrong is a straight line?')
    L.append('')
    L.append('The arms are true great circles through the zenith. They would be exactly '
             'straight if the optical axis were the zenith, but the camera is tilted '
             f'{w.meta["tilt_from_vertical_deg"]:.2f} deg, so they bow very slightly:')
    L.append('')
    L.append('| Direction | Max deviation from a straight line | Worst at altitude |')
    L.append('| --- | --- | --- |')
    for az, tag, _ in DIRS:
        a = arms[tag]
        L.append(f'| {tag} | {a["max_dev"]:.2f} px | {a["dev_alt"]:.0f}° |')
    L.append('')
    L.append('For a visual cross a straight line is fine everywhere except possibly the '
             'east arm. If you want it exact, use the polylines in section 2.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 2. The exact version: polyline vertices')
    L.append('')
    L.append('Vertices every 10° of altitude, from the lowest point still inside the '
             'frame up to the zenith. Join them with straight segments and the arm is '
             'accurate to well under a pixel. The first row of each table is the frame '
             'edge, the last row is the zenith (shared by all arms).')
    L.append('')
    for az, tag, longname in DIRS:
        a = arms[tag]
        L.append(f'### {tag} - {longname}, azimuth {az}°')
        L.append('')
        L.append('| Altitude | x | y |')
        L.append('| --- | --- | --- |')
        for alt, x, y in ticks(az, a['alt_min']):
            note = ' (frame edge)' if abs(alt - a['alt_min']) < 1e-9 else ''
            lbl = f'{alt:.1f}°{note}' if note else f'{alt:.0f}°'
            L.append(f'| {lbl} | {x:.1f} | {y:.1f} |')
        L.append('')
    L.append('---')
    L.append('')
    L.append('## 3. Suggested label placement')
    L.append('')
    L.append('Put each letter just inside the frame at the outer end of its arm, offset '
             'back along the arm so it does not get clipped. These are the points at '
             '5° of altitude above each arm\'s lowest visible point:')
    L.append('')
    L.append('| Direction | Label anchor (x, y) | Altitude there |')
    L.append('| --- | --- | --- |')
    for az, tag, _ in DIRS:
        a = arms[tag]
        alt = a['alt_min'] + 5
        x, y = w.altaz_to_pixel(float(alt), float(az))
        L.append(f'| {tag} | ({x[0]:.0f}, {y[0]:.0f}) | {alt:.1f}° |')
    L.append('')
    L.append('It is worth writing the altitude next to each letter, e.g. `S 49°`, '
             'because the four arms reach very different depths - see the warning below.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 4. Things that will surprise whoever draws this')
    L.append('')
    L.append('**The arms are wildly different lengths, and that is correct.** The frame '
             'is a 16:9 crop of a fisheye image circle that is about 2444 px across, so '
             'the horizon is nowhere in the picture. Each direction is cut off at a '
             'different altitude:')
    L.append('')
    order = sorted(DIRS, key=lambda d: arms[d[1]]['alt_min'])
    for az, tag, longname in order:
        L.append(f'- **{tag}** ({longname}) is cut off at **{arms[tag]["alt_min"]:.1f}°** altitude')
    L.append('')
    L.append(f'So the south arm is the shortest ({arms["S"]["chord"]:.0f} px) and stops '
             f'{arms["S"]["alt_min"]:.0f}° above the horizon, while the east arm runs '
             f'{arms["E"]["chord"]:.0f} px and nearly reaches it. Anything in the southern '
             f'sky below {arms["S"]["alt_min"]:.0f}° altitude is simply not recorded.')
    L.append('')
    L.append('**The frame border covers all 360° of azimuth.** Walking clockwise from the '
             'top-left corner:')
    L.append('')
    L.append('| Border segment | Azimuth range | Directions crossed |')
    L.append('| --- | --- | --- |')
    for nm, xs, ys, desc in (
            ('top edge, left to right', np.linspace(1, W - 1, 400), np.full(400, 1.0), 'E -> NE -> N -> NW'),
            ('right edge, top to bottom', np.full(400, W - 1.0), np.linspace(1, H - 1, 400), 'NW -> W'),
            ('bottom edge, right to left', np.linspace(1, W - 1, 400), np.full(400, H - 1.0), 'W -> SW -> S -> SE'),
            ('left edge, bottom to top', np.full(400, 1.0), np.linspace(1, H - 1, 400), 'SE -> E')):
        alt, azs = w.pixel_to_altaz(xs, ys)
        L.append(f'| {nm} | {azs.min():.0f}° .. {azs.max():.0f}° | {desc} |')
    L.append('')
    L.append('(The top edge shows the full 0-360 range because it crosses due north, '
             'where azimuth wraps.)')
    L.append('')
    tx, ty = 60, 40
    talt, taz = w.pixel_to_altaz([tx], [ty])
    L.append(f'**The trees block the best direction.** East reaches lowest '
             f'({arms["E"]["alt_min"]:.1f}°) but the trees and glow along the left edge and '
             f'top-left corner sit at azimuth 89°-112° (east to east-south-east) at '
             f'10°-19° altitude. Example: pixel ({tx}, {ty}) in the tree canopy looks at '
             f'altitude {talt[0]:.1f}°, azimuth {taz[0]:.1f}°.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 5. Sanity checks before you trust the drawing')
    L.append('')
    px, py = 1308.76, 56.62
    alt, az = w.pixel_to_altaz([px], [py])
    L.append(f'**Polaris.** It sits at pixel ({px:.0f}, {py:.0f}) in this frame. Polaris is '
             f'due north at an altitude equal to the site latitude, and solving that pixel '
             f'gives altitude {alt[0]:.2f}°, azimuth {az[0]:.2f}° against a true latitude of '
             f'{w.lat:.3f}°. The north arm must pass essentially through Polaris. If it does '
             f'not, something is wrong.')
    L.append('')
    L.append('**Constellations.** For this frame (2026-09-15, 22:27 EDT) Cassiopeia is in '
             'the north-east near the top edge, the Big Dipper is low in the north-west at '
             'the upper right, Pegasus and Andromeda are rising in the east on the left '
             'side, and Hercules is setting in the west at the lower right. That is exactly '
             'right for a mid-September evening, and it would all be mirrored if east and '
             'west were swapped.')
    L.append('')
    L.append('---')
    L.append('')
    L.append('## 6. Computing any of this yourself')
    L.append('')
    L.append('```python')
    L.append('from asc_wcs import AscWCS')
    L.append('w = AscWCS()')
    L.append('')
    L.append('x, y = w.altaz_to_pixel(90, 0)        # zenith, the centre of the cross')
    L.append('x, y = w.altaz_to_pixel(30, 180)      # due south, 30 deg up')
    L.append('alt, az = w.pixel_to_altaz(1309, 57)  # what is this pixel looking at?')
    L.append('```')
    L.append('')
    L.append('`altaz_to_pixel` and `pixel_to_altaz` need no timestamp - the camera does '
             'not move, so this mapping is permanent. Only `radec_to_pixel` / '
             '`pixel_to_radec` need a UTC time.')
    L.append('')
    L.append('Regenerate this document with `python3 make_cardinal_doc.py`.')
    L.append('')

    with open(OUT, 'w') as f:
        f.write('\n'.join(L))
    print(f'wrote {OUT} ({len(L)} lines)')


if __name__ == '__main__':
    main()
