#!/usr/bin/env python3
"""Rank clock times for an analemma by how much of the year's sun path lands in frame.

Uses the star calibration in star_labeler/calibration.json (alt/az -> pixel is time
independent) and astropy's sun ephemeris. Clock times are in constant standard time
(EST, UTC-5); during DST the matching Half Hour filename is one hour later.

    python3 plan_analemma_time.py                 # ranking table
    python3 plan_analemma_time.py --overlay 2026-08-20_1300.tiff --times 1100 1130 1200
"""

import argparse
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

from sun import wcs as load_wcs, wcs_altaz

HERE = os.path.dirname(os.path.abspath(__file__))


def sun_track(wcs, hhmm, start, days):
    dates = [start + timedelta(days=i) for i in range(days)]
    x, y, alt, az = wcs_altaz(dates, f'{hhmm:04d}')
    return dates, alt, az, x, y


def in_frame(wcs, alt, x, y, margin):
    return (alt > 0) & (x >= margin) & (x < wcs.width - margin) & (y >= margin) & (y < wcs.height - margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-01-01')
    ap.add_argument('--days', type=int, default=365)
    ap.add_argument('--margin', type=int, default=40, help='px, roughly the sun glare radius')
    ap.add_argument('--overlay', help='Half Hour frame to draw candidate analemmas on')
    ap.add_argument('--times', nargs='*', type=int, help='EST HHMM values to overlay')
    a = ap.parse_args()

    wcs = load_wcs()
    start = datetime.strptime(a.start, '%Y-%m-%d')
    times = [h * 100 + m for h in range(5, 20) for m in (0, 30)]

    rows = []
    for hhmm in times:
        dates, alt, az, x, y = sun_track(wcs, hhmm, start, a.days)
        ok = in_frame(wcs, alt, x, y, a.margin)
        if not ok.any():
            continue
        out = [d.strftime('%m-%d') for d, k in zip(dates, ok) if not k]
        rows.append((ok.mean(), hhmm, alt.min(), alt.max(), az.min(), az.max(),
                     x.min(), x.max(), y.min(), y.max(), out))

    print(f'margin {a.margin}px, {a.days} days from {a.start}; times are EST (DST filename = +1h)\n')
    print(' EST   in-frame  alt range     az range      x range        y range      days out')
    for frac, hhmm, a0, a1, z0, z1, x0, x1, y0, y1, out in sorted(rows, reverse=True):
        span = f'{out[0]}..{out[-1]}' if out else ''
        print(f' {hhmm:04d}  {frac*100:5.1f}%   {a0:5.1f}-{a1:5.1f}  {z0:5.1f}-{z1:5.1f}  '
              f'{x0:6.0f}-{x1:6.0f}  {y0:6.0f}-{y1:6.0f}  {len(out):3d} {span}')

    if a.overlay:
        path = a.overlay if os.path.isabs(a.overlay) else os.path.join(HERE, '..', a.overlay)
        img = cv2.imread(path)
        colors = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0), (80, 80, 255)]
        for i, hhmm in enumerate(a.times or []):
            dates, alt, az, x, y = sun_track(wcs, hhmm, start, a.days)
            ok = in_frame(wcs, alt, x, y, 0)
            c = colors[i % len(colors)]
            for d, k, xi, yi in zip(dates, ok, x, y):
                if k:
                    cv2.circle(img, (int(xi), int(yi)), 3 if d.day == 1 else 2, c, -1, cv2.LINE_AA)
            j = int(np.argmin(y)) if ok.any() else 0
            cv2.putText(img, f'{hhmm:04d} EST', (int(np.clip(x[ok].mean() if ok.any() else 20, 10, 1800)), 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
        out = os.path.join(HERE, 'output', 'analemma_time_candidates.jpg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('\n' + out)


if __name__ == '__main__':
    main()
