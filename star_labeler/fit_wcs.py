#!/usr/bin/env python3
"""Fit an all-sky (fisheye) WCS from hand-labeled stars.

Model
-----
    u        celestial unit vector of the star (mean equator/equinox of date)
    v = R u  rotation into the camera frame (+z = optical axis)
    theta    = acos(v_z)            angular distance from the optical axis
    phi      = atan2(v_y, v_x)      azimuth in the camera frame
    r(theta) = f1*t + f3*t^3 + f5*t^5 + t^2*(a*cos(phi) + b*sin(phi))
    x        = x0 + r*cos(phi)
    y        = y0 + parity * r*sin(phi)

Free parameters: R (3), x0, y0, f1, f3, f5, a, b -> 10 total.
`parity` (+-1) handles the image-axis handedness and is chosen by brute force.
The a/b term is a first-order correction for the lens axis not being exactly
perpendicular to the sensor; it breaks the radial symmetry and grows with field
angle. Cross-validation against unlabeled catalogue stars shows it generalises
(blind median offset 0.57 -> 0.38 px), so it is fitted by default.
"""

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# Mid-exposure instant of 2026-09-15_22-26-55 (60 s exposure), local time UTC-4.
UTC = (2026, 9, 16, 2, 27, 25.0)
SITE = (41.879, -71.966)          # Pomfret Center, CT
SITE_NAME = 'Pomfret Center, CT'


# --- time / frame helpers ------------------------------------------------
def julian_date(y, m, d, hh, mm, ss):
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    return jd + (hh + mm / 60 + ss / 3600) / 24


def gmst_deg(jd):
    d = jd - 2451545.0
    t = d / 36525.0
    g = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t ** 3 / 38710000.0
    return g % 360.0


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def precession_matrix(jd):
    """J2000 -> mean equator and equinox of date (IAU 1976)."""
    t = (jd - 2451545.0) / 36525.0
    sec = np.pi / 180 / 3600
    zeta = (2306.2181 * t + 0.30188 * t ** 2 + 0.017998 * t ** 3) * sec
    z = (2306.2181 * t + 1.09468 * t ** 2 + 0.018203 * t ** 3) * sec
    theta = (2004.3109 * t - 0.42665 * t ** 2 - 0.041833 * t ** 3) * sec
    return rz(-z) @ ry(theta) @ rz(-zeta)


def radec_to_vec(ra_hours, dec_deg):
    ra = np.radians(np.asarray(ra_hours) * 15.0)
    dec = np.radians(np.asarray(dec_deg))
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def rodrigues(w):
    t = np.linalg.norm(w)
    if t < 1e-12:
        return np.eye(3)
    k = w / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


# --- projection ----------------------------------------------------------
def project(u, R, x0, y0, coef, parity, asym=(0.0, 0.0)):
    v = u @ R.T
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    r = coef[0] * theta + coef[1] * theta ** 3 + coef[2] * theta ** 5
    r = r + theta ** 2 * (asym[0] * np.cos(phi) + asym[1] * np.sin(phi))
    return np.stack([x0 + r * np.cos(phi), y0 + parity * r * np.sin(phi)], axis=-1), theta


def unpack(p, R0):
    return rodrigues(p[:3]) @ R0, p[3], p[4], p[5:8], p[8:10]


def fit(u, xy, parity, R0, p0, iters=200):
    """Levenberg-Marquardt with a numerical Jacobian."""
    def resid(p):
        R, x0, y0, coef, asym = unpack(p, R0)
        pred, _ = project(u, R, x0, y0, coef, parity, asym)
        return (pred - xy).ravel()

    p = p0.copy()
    r = resid(p)
    cost = r @ r
    lam = 1e-3
    steps = np.array([1e-6, 1e-6, 1e-6] + [1e-3] * (p0.size - 3))
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            q = p.copy()
            q[k] += steps[k]
            rp = resid(q)
            q[k] -= 2 * steps[k]
            rm = resid(q)
            J[:, k] = (rp - rm) / (2 * steps[k])
        JTJ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(12):
            try:
                dp = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            rn = resid(p + dp)
            cn = rn @ rn
            if cn < cost:
                p, r, cost = p + dp, rn, cn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10
        if not improved or lam > 1e12:
            break
    return p, cost


def main():
    cat = {int(k): v for k, v in json.load(open(os.path.join(ROOT, 'catalog.json'))).items()}
    by_name = {v['proper'].lower(): (k, v) for k, v in cat.items() if v['proper']}
    labels = json.load(open(os.path.join(ROOT, 'labels.json')))

    stars, skipped = [], []
    for s in labels['stars']:
        n = s['name'].strip()
        if n.upper().startswith('HIP') and n[3:].strip().isdigit():
            h = int(n[3:]); rec = cat.get(h)
        elif n.lower() in by_name:
            h, rec = by_name[n.lower()]
        else:
            skipped.append(n); continue
        if rec is None:
            skipped.append(n); continue
        stars.append({'hip': h, 'label': n, 'x': s['x'], 'y': s['y'], **rec})

    print(f'using {len(stars)} labeled stars' + (f' (skipped: {", ".join(skipped)})' if skipped else ''))

    jd = julian_date(*UTC)
    P = precession_matrix(jd)
    u = radec_to_vec([s['ra'] for s in stars], [s['dec'] for s in stars]) @ P.T
    xy = np.array([[s['x'], s['y']] for s in stars])

    W, H = labels.get('width', 1920), labels.get('height', 1080)
    best = None
    for parity in (+1, -1):
        # initial directions from an equidistant guess centred on the frame
        x0, y0, f = W / 2, H / 2, 640.0
        d = xy - np.array([x0, y0])
        rr = np.hypot(d[:, 0], d[:, 1])
        th = rr / f
        ph = np.arctan2(parity * d[:, 1], d[:, 0])
        v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
        # Wahba / Kabsch: best rotation taking u -> v
        U, S, Vt = np.linalg.svd(v.T @ u)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R0 = U @ D @ Vt
        p0 = np.array([0, 0, 0, x0, y0, f, 0.0, 0.0, 0.0, 0.0])
        p, cost = fit(u, xy, parity, R0, p0)
        rms = np.sqrt(cost / len(stars))
        print(f'  parity {parity:+d}: RMS {rms:.3f} px')
        if best is None or cost < best[1]:
            best = (p, cost, parity, R0)

    p, cost, parity, R0 = best
    R, x0, y0, coef, asym = unpack(p, R0)
    pred, theta = project(u, R, x0, y0, coef, parity, asym)
    res = pred - xy
    dist = np.hypot(res[:, 0], res[:, 1])
    rms = np.sqrt((dist ** 2).mean())

    print(f'\nparity {parity:+d}   centre ({x0:.2f}, {y0:.2f})')
    print(f'r(theta) = {coef[0]:.3f}*t + {coef[1]:.3f}*t^3 + {coef[2]:.3f}*t^5'
          f' + t^2*({asym[0]:+.3f}*cos(phi) {asym[1]:+.3f}*sin(phi))   [px, t in rad]')
    print(f'axis/sensor tilt term: {np.hypot(*asym):.2f} px at t=1 rad, toward image azimuth '
          f'{np.degrees(np.arctan2(asym[1], asym[0])) % 360:.0f} deg')
    print(f'RMS {rms:.2f} px | median {np.median(dist):.2f} | max {dist.max():.2f} px')
    deg_per_px = 1.0 / (coef[0] * np.pi / 180)
    print(f'plate scale near axis: {1/deg_per_px:.1f} px/deg')

    # Where the optical axis points. This is the zenith only if the camera is
    # perfectly level, so compare it against the zenith of the known site.
    axis = R.T @ np.array([0, 0, 1.0])
    dec_z = np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))
    ra_z = np.degrees(np.arctan2(axis[1], axis[0])) % 360
    print(f'\noptical axis -> RA {ra_z/15:.4f} h  Dec {dec_z:+.3f} deg  (mid-exposure JD {jd:.5f})')

    lat, lon = SITE
    lst = (gmst_deg(jd) + lon) % 360
    zen_true = radec_to_vec([lst / 15], [lat])[0]
    up = zen_true
    ncp = P @ np.array([0, 0, 1.0])
    n_hat = ncp - (ncp @ up) * up
    n_hat /= np.linalg.norm(n_hat)
    e_hat = np.cross(n_hat, up)
    tilt = np.degrees(np.arccos(np.clip(axis @ zen_true, -1, 1)))
    az_t = np.degrees(np.arctan2(axis @ e_hat, axis @ n_hat)) % 360
    print(f'site {SITE_NAME}: lat {lat:+.4f}, lon {lon:+.4f} -> zenith RA {lst/15:.4f} h Dec {lat:+.3f} deg')
    print(f'camera tilt from vertical: {tilt:.2f} deg toward azimuth {az_t:.1f} deg')
    print(f'  Dec offset {dec_z - lat:+.3f} deg is tilt only (a clock error cannot move Dec)')
    print(f'  RA  offset {ra_z - lst:+.3f} deg == {(ra_z - lst)/(15/3600):+.0f} s of clock error if the mount were true')
    print(f'labeled stars span theta = {np.degrees(theta).min():.1f}..{np.degrees(theta).max():.1f} deg from axis')

    print(f'\n{"label":<12} {"name":<12} {"con":<4} {"mag":>5} {"theta":>6} {"dx":>7} {"dy":>7} {"|d|":>6}')
    order = np.argsort(-dist)
    for i in order:
        s = stars[i]
        nm = s['proper'] or f"{s['bayer']} {s['con']}"
        print(f"{s['label']:<12} {nm:<12} {s['con']:<4} {s['mag']:>5.2f} "
              f"{np.degrees(theta[i]):>6.1f} {res[i,0]:>7.2f} {res[i,1]:>7.2f} {dist[i]:>6.2f}")

    json.dump({
        'image': labels.get('image'), 'width': W, 'height': H,
        'utc_mid': list(UTC), 'jd': jd,
        'parity': parity, 'x0': x0, 'y0': y0, 'coef': list(coef), 'asym': list(asym),
        'R': R.tolist(), 'rms_px': rms, 'n_stars': len(stars),
        'axis_ra_deg': ra_z, 'axis_dec_deg': dec_z,
        'site': list(SITE), 'site_name': SITE_NAME, 'lst_deg': lst,
        'tilt_deg': float(tilt), 'tilt_az_deg': float(az_t),
        'theta_max_fit_deg': float(np.degrees(theta).max()),
        'residuals': [{'label': s['label'], 'hip': s['hip'], 'dx': float(res[i, 0]),
                       'dy': float(res[i, 1]), 'd': float(dist[i]),
                       'theta_deg': float(np.degrees(theta[i]))} for i, s in enumerate(stars)],
    }, open(os.path.join(ROOT, 'wcs.json'), 'w'), indent=2)
    print('\nwrote wcs.json')


if __name__ == '__main__':
    sys.exit(main())
