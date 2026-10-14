#!/usr/bin/env python3
"""Reusable all-sky camera WCS.

The camera is bolted down, so the map between the local horizontal frame
(alt/az) and pixels never changes -- that part is stored here as a fixed
matrix `M` plus the lens intrinsics. Only equatorial coordinates need a
timestamp, because that is the one thing the sky rotation touches.

    altaz  <-> pixel      no time needed
    radec  <-> pixel      needs UTC (via local sidereal time)

Build the calibration once from wcs.json:

    python3 asc_wcs.py --build          # writes calibration.json
    python3 asc_wcs.py --check          # re-projects the labeled stars

Then use it anywhere:

    from asc_wcs import AscWCS
    w = AscWCS()
    x, y = w.altaz_to_pixel(45, 180)              # due south, 45 deg up
    x, y = w.radec_to_pixel(20.69, 45.28, (2026, 9, 16, 2, 27, 25))
    alt, az = w.pixel_to_altaz(960, 540)
    ra, dec = w.pixel_to_radec(960, 540, (2026, 9, 16, 2, 27, 25))
"""

import argparse
import json
import os
import numpy as np

from fit_wcs import (SITE, SITE_NAME, gmst_deg, julian_date, precession_matrix,
                     radec_to_vec)

ROOT = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(ROOT, 'calibration.json')


def horizontal_basis(jd, lat, lon):
    """Columns [east, north, up] expressed in the mean equatorial frame of date."""
    lst = (gmst_deg(jd) + lon) % 360
    up = radec_to_vec([lst / 15], [lat])[0]
    ncp = precession_matrix(jd) @ np.array([0, 0, 1.0])
    north = ncp - (ncp @ up) * up
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    return np.stack([east, north, up], axis=1)


class AscWCS:
    def __init__(self, path=CAL):
        c = json.load(open(path))
        self.M = np.array(c['M'])                # camera frame <- (east, north, up)
        self.x0, self.y0 = c['x0'], c['y0']
        self.coef = np.array(c['coef'])
        self.asym = np.array(c['asym'])
        self.parity = c['parity']
        self.lat, self.lon = c['site']
        self.width, self.height = c['width'], c['height']
        self.meta = c
        t = np.linspace(0, np.radians(100), 40001)
        self._t_grid, self._r_grid = t, self._r_of_theta(t)

    # --- lens model ------------------------------------------------------
    def _r_of_theta(self, theta, phi=None):
        r = self.coef[0] * theta + self.coef[1] * theta ** 3 + self.coef[2] * theta ** 5
        if phi is not None:
            r = r + theta ** 2 * (self.asym[0] * np.cos(phi) + self.asym[1] * np.sin(phi))
        return r

    def camera_to_pixel(self, v):
        """v: (...,3) unit vectors in the camera frame (+z = optical axis)."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        theta = np.arccos(np.clip(v[:, 2], -1, 1))
        phi = np.arctan2(v[:, 1], v[:, 0])
        r = self._r_of_theta(theta, phi)
        return np.stack([self.x0 + r * np.cos(phi),
                         self.y0 + self.parity * r * np.sin(phi)], axis=-1), theta

    def pixel_to_camera(self, x, y):
        dx = np.atleast_1d(np.asarray(x, dtype=float)) - self.x0
        dy = (np.atleast_1d(np.asarray(y, dtype=float)) - self.y0) * self.parity
        phi = np.arctan2(dy, dx)
        r = np.hypot(dx, dy)
        theta = np.interp(r, self._r_grid, self._t_grid)
        for _ in range(3):                       # Newton polish, the asym term is phi dependent
            f = self._r_of_theta(theta, phi) - r
            d = (self.coef[0] + 3 * self.coef[1] * theta ** 2 + 5 * self.coef[2] * theta ** 4
                 + 2 * theta * (self.asym[0] * np.cos(phi) + self.asym[1] * np.sin(phi)))
            theta = theta - f / d
        return np.stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)

    # --- horizontal (time independent) -----------------------------------
    def altaz_to_pixel(self, alt, az):
        alt = np.radians(np.atleast_1d(np.asarray(alt, dtype=float)))
        az = np.radians(np.atleast_1d(np.asarray(az, dtype=float)))
        enu = np.stack([np.cos(alt) * np.sin(az),        # east
                        np.cos(alt) * np.cos(az),        # north
                        np.sin(alt)], axis=-1)           # up
        xy, _ = self.camera_to_pixel(enu @ self.M.T)
        return xy[:, 0], xy[:, 1]

    def pixel_to_altaz(self, x, y):
        enu = self.pixel_to_camera(x, y) @ self.M      # M is orthonormal
        alt = np.degrees(np.arcsin(np.clip(enu[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360
        return alt, az

    # --- equatorial (needs a timestamp) -----------------------------------
    def _R_at(self, utc):
        """Camera frame <- mean equatorial frame of date, at this instant."""
        jd = julian_date(*utc) if not np.isscalar(utc) else utc
        return self.M @ horizontal_basis(jd, self.lat, self.lon).T, jd

    def radec_to_pixel(self, ra_hours, dec_deg, utc, j2000=True):
        R, jd = self._R_at(utc)
        u = radec_to_vec(np.atleast_1d(ra_hours), np.atleast_1d(dec_deg))
        if j2000:
            u = u @ precession_matrix(jd).T
        xy, _ = self.camera_to_pixel(u @ R.T)
        return xy[:, 0], xy[:, 1]

    def pixel_to_radec(self, x, y, utc, j2000=True):
        R, jd = self._R_at(utc)
        u = self.pixel_to_camera(x, y) @ R           # back to equatorial of date
        if j2000:
            u = u @ precession_matrix(jd)            # of date -> J2000
        ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360 / 15
        dec = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
        return ra, dec


# --- build / verify -------------------------------------------------------
def build():
    w = json.load(open(os.path.join(ROOT, 'wcs.json')))
    R = np.array(w['R'])
    B = horizontal_basis(w['jd'], *SITE)
    M = R @ B                                        # camera <- (east, north, up)
    axis_enu = M.T @ np.array([0, 0, 1.0])
    tilt = np.degrees(np.arccos(np.clip(axis_enu[2], -1, 1)))
    tilt_az = np.degrees(np.arctan2(axis_enu[0], axis_enu[1])) % 360
    roll = np.degrees(np.arctan2(M[1, 2], M[0, 2])) % 360

    cal = {
        'camera': 'ASC fisheye, 1920x1080',
        'site': list(SITE), 'site_name': SITE_NAME,
        'width': w['width'], 'height': w['height'],
        'parity': w['parity'], 'x0': w['x0'], 'y0': w['y0'],
        'coef': w['coef'], 'asym': w['asym'],
        'M': M.tolist(),
        'tilt_from_vertical_deg': float(tilt), 'tilt_azimuth_deg': float(tilt_az),
        'axis_roll_deg': float(roll),
        'fit': {'source_frame': w['image'], 'n_stars': w['n_stars'],
                'rms_px': w['rms_px'], 'utc_mid': w['utc_mid'], 'jd': w['jd'],
                'theta_max_fit_deg': w['theta_max_fit_deg']},
        'note': ('M maps local horizontal (east, north, up) into the camera frame and is '
                 'valid for as long as the camera is not moved. Equatorial conversions '
                 'only need a UTC timestamp.'),
    }
    json.dump(cal, open(CAL, 'w'), indent=2)
    print(f'wrote {CAL}')
    print(f'optical axis: {tilt:.3f} deg from vertical toward azimuth {tilt_az:.2f} deg')
    print(f'image +x axis roll: {roll:.2f} deg')
    return cal


def check():
    w = AscWCS()
    labels = json.load(open(os.path.join(ROOT, 'labels.json')))
    cat = {int(k): v for k, v in json.load(open(os.path.join(ROOT, 'catalog.json'))).items()}
    by_name = {v['proper'].lower(): k for k, v in cat.items() if v['proper']}
    utc = tuple(w.meta['fit']['utc_mid'])

    d = []
    for s in labels['stars']:
        n = s['name']
        h = int(n[3:]) if n.upper().startswith('HIP') and n[3:].isdigit() else by_name.get(n.lower())
        if h is None or h not in cat:
            continue
        px, py = w.radec_to_pixel(cat[h]['ra'], cat[h]['dec'], utc)
        d.append(np.hypot(px[0] - s['x'], py[0] - s['y']))
    d = np.array(d)
    print(f'{len(d)} labeled stars re-projected through calibration.json: '
          f'RMS {np.sqrt((d**2).mean()):.3f} px, max {d.max():.3f} px')

    # round trip
    xs = np.array([200., 700., 960., 1500., 1800.])
    ys = np.array([150., 400., 540., 800., 1000.])
    alt, az = w.pixel_to_altaz(xs, ys)
    bx, by = w.altaz_to_pixel(alt, az)
    print(f'pixel -> altaz -> pixel round trip: max error {np.abs(np.hypot(bx-xs, by-ys)).max():.2e} px')
    ra, dec = w.pixel_to_radec(xs, ys, utc)
    bx, by = w.radec_to_pixel(ra, dec, utc)
    print(f'pixel -> radec -> pixel round trip: max error {np.abs(np.hypot(bx-xs, by-ys)).max():.2e} px')
    print('\nsome reference points (time independent):')
    for nm, (a, z) in {'zenith': (90, 0), 'N 45 deg': (45, 0), 'E 45 deg': (45, 90),
                       'S 45 deg': (45, 180), 'W 45 deg': (45, 270)}.items():
        x, y = w.altaz_to_pixel(a, z)
        inside = 0 <= x[0] <= w.width and 0 <= y[0] <= w.height
        print(f'  {nm:<10} -> pixel ({x[0]:8.1f}, {y[0]:8.1f}) {"" if inside else "[outside frame]"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.build or not os.path.exists(CAL):
        build()
    if a.check:
        check()
