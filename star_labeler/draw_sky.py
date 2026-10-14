#!/usr/bin/env python3
"""Draw constellation figures on the all-sky frame using the fitted WCS.

Lines are sampled along great circles so they bend correctly under the
fisheye projection. Run fit_wcs.py first to produce wcs.json.
"""

import argparse
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fit_wcs import SITE, gmst_deg, precession_matrix, radec_to_vec, project

ROOT = os.path.dirname(os.path.abspath(__file__))
SS = 2                            # supersampling factor
THETA_MAX = np.radians(86.0)      # stop drawing this far from the optical axis

# Foreground obstructions: the tree line and the dark corner outside the lens
# circle, both along the top-left edge. No stars are visible here, so
# constellation figures are not drawn over them. Pixel polygons, easy to edit
# if the trees grow or the camera is re-aimed.
OBSTRUCTIONS = [
    [(0, 0), (350, 0), (170, 80), (130, 165), (70, 265), (40, 320), (40, 520), (0, 570)],
]

C_LINE = (120, 200, 255, 165)
C_LINE_FAINT = (120, 200, 255, 70)
C_CON = (255, 198, 120, 225)
C_STAR = (255, 255, 255, 150)
C_LABELED = (150, 255, 200, 235)
C_GRID = (255, 255, 255, 48)
C_CARD = (255, 120, 120, 230)
C_EQ_GRID = (200, 140, 255, 60)
C_EQ_MAIN = (220, 160, 255, 110)
C_EQ_TEXT = (225, 165, 255, 225)


def font(size):
    for p in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf',
              '/System/Library/Fonts/Supplemental/Arial.ttf',
              '/System/Library/Fonts/Helvetica.ttc'):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def base_image(path, raw=False, black=34.0, gain=2.2, gamma=0.72):
    """Load the frame, optionally applying the same stretch the labeller shows."""
    im = Image.open(path).convert('RGB')
    if raw:
        return im
    a = np.asarray(im).astype(np.float32)
    v = np.clip((a - black) / (255 - black), 0, None)
    v = np.power(np.clip(v * gain, 0, 1), gamma)
    return Image.fromarray((v * 255).astype(np.uint8))


class Sky:
    def __init__(self, wcs):
        self.R = np.array(wcs['R'])
        self.x0, self.y0 = wcs['x0'], wcs['y0']
        self.coef = np.array(wcs['coef'])
        self.asym = np.array(wcs.get('asym', (0.0, 0.0)))
        self.parity = wcs['parity']
        self.W, self.H = wcs['width'], wcs['height']

    def to_pixel(self, u):
        """u: (...,3) unit vectors, equator/equinox of date -> (xy, theta)."""
        u = np.atleast_2d(u)
        return project(u, self.R, self.x0, self.y0, self.coef, self.parity, self.asym)

    def visible(self, xy, theta, margin=40):
        return ((theta < THETA_MAX) &
                (xy[:, 0] > -margin) & (xy[:, 0] < self.W + margin) &
                (xy[:, 1] > -margin) & (xy[:, 1] < self.H + margin))


def in_obstruction(x, y):
    """Point-in-polygon test against every obstruction zone, vectorised."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    hit = np.zeros(x.shape, bool)
    for poly in OBSTRUCTIONS:
        p = np.asarray(poly, dtype=float)
        inside = np.zeros(x.shape, bool)
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            crosses = ((y1 > y) != (y2 > y))
            with np.errstate(divide='ignore', invalid='ignore'):
                xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < xint)
        hit |= inside
    return hit


def slerp(a, b, n):
    dot = np.clip(a @ b, -1, 1)
    om = np.arccos(dot)
    t = np.linspace(0, 1, n)[:, None]
    if om < 1e-9:
        return np.repeat(a[None, :], n, axis=0)
    s = np.sin(om)
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / s


def draw_runs(dr, sky, pts, color, width):
    """Project a polyline of unit vectors and stroke the visible runs."""
    xy, th = sky.to_pixel(pts)
    ok = sky.visible(xy, th, margin=200)
    run = []
    for i in range(len(xy)):
        if ok[i]:
            run.append((xy[i, 0] * SS, xy[i, 1] * SS))
        else:
            if len(run) > 1:
                dr.line(run, fill=color, width=width, joint='curve')
            run = []
    if len(run) > 1:
        dr.line(run, fill=color, width=width, joint='curve')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', choices=('none', 'altaz', 'radec'), default='none',
                    help='overlay a horizontal (alt/az) or equatorial (RA/Dec, J2000) grid')
    ap.add_argument('--raw', action='store_true',
                    help='use the untouched frame as the base instead of the stretched version')
    ap.add_argument('--only', nargs='*', help='restrict to these constellation abbreviations')
    ap.add_argument('--skip', nargs='*', help='omit these constellation abbreviations entirely')
    ap.add_argument('--names', choices=('proper', 'hip', 'none'), default='proper',
                    help='how to label the hand-labeled stars')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    wcs = json.load(open(os.path.join(ROOT, 'wcs.json')))
    cat = {int(k): v for k, v in json.load(open(os.path.join(ROOT, 'catalog.json'))).items()}
    cons = json.load(open(os.path.join(ROOT, 'constellations.json')))
    labels = json.load(open(os.path.join(ROOT, 'labels.json')))
    sky = Sky(wcs)

    P = precession_matrix(wcs['jd'])
    vec = {h: (radec_to_vec([r['ra']], [r['dec']])[0] @ P.T) for h, r in cat.items()}

    base = base_image(os.path.join(ROOT, 'static', 'image.png'), raw=args.raw)
    canvas = base.resize((sky.W * SS, sky.H * SS), Image.LANCZOS).convert('RGBA')
    over = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    f_con, f_star, f_grid = font(13 * SS), font(11 * SS), font(10 * SS)

    # --- equatorial (RA/Dec) grid, J2000 ---------------------------------
    if args.grid == 'radec':
        def radec(ra_h, dec_d):
            return radec_to_vec(ra_h, dec_d) @ P.T          # J2000 -> frame of date

        for dec in range(-30, 90, 15):
            ra = np.linspace(0, 24, 577)
            col = C_EQ_MAIN if dec == 0 else C_EQ_GRID
            draw_runs(dr, sky, radec(ra, np.full_like(ra, float(dec))), col, 2 * SS)
        for ra_h in range(0, 24):
            dec = np.linspace(-40, 89.5, 320)
            draw_runs(dr, sky, radec(np.full_like(dec, float(ra_h)), dec), C_EQ_GRID, 2 * SS)

        # label each meridian where it is highest in the frame, and the parallels
        for ra_h in range(0, 24):
            dec = np.arange(-30, 89.5, 0.5)
            pts = radec(np.full_like(dec, float(ra_h)), dec)
            xy, th = sky.to_pixel(pts)
            ok = sky.visible(xy, th, margin=-45)
            if not ok.any():
                continue
            # tag the meridian near the frame edge (largest field angle) so the
            # labels spread out instead of piling up at the pole
            i = np.flatnonzero(ok)[np.argmax(th[ok])]
            dr.text((xy[i, 0] * SS, xy[i, 1] * SS), f'{ra_h}h', fill=C_EQ_TEXT, font=f_grid,
                    anchor='mm', stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 190))
        for dec in range(-30, 90, 15):
            ra = np.arange(0, 24, 0.05)
            pts = radec(ra, np.full_like(ra, float(dec)))
            xy, th = sky.to_pixel(pts)
            ok = sky.visible(xy, th, margin=-45)
            if not ok.any():
                continue
            i = np.flatnonzero(ok)[np.argmax(xy[ok, 0])]
            dr.text((xy[i, 0] * SS, xy[i, 1] * SS), f'{dec:+d}°', fill=C_EQ_TEXT, font=f_grid,
                    anchor='mm', stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 190))
        xy, th = sky.to_pixel(P @ np.array([0, 0, 1.0]))
        if sky.visible(xy, th)[0]:
            x, y = xy[0, 0] * SS, xy[0, 1] * SS
            r = 7 * SS
            dr.line([(x - r, y), (x + r, y)], fill=C_EQ_TEXT, width=2 * SS)
            dr.line([(x, y - r), (x, y + r)], fill=C_EQ_TEXT, width=2 * SS)
            dr.text((x + r + 3 * SS, y - 7 * SS), 'NCP', fill=C_EQ_TEXT, font=f_grid,
                    stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 190))

    # --- alt/az grid ----------------------------------------------------
    if args.grid == 'altaz':
        lat, lon = SITE
        lst = (gmst_deg(wcs['jd']) + lon) % 360
        up = radec_to_vec([lst / 15], [lat])[0]
        ncp = P @ np.array([0, 0, 1.0])
        n_hat = ncp - (ncp @ up) * up
        n_hat /= np.linalg.norm(n_hat)
        e_hat = np.cross(n_hat, up)

        def altaz(alt, az):
            alt, az = np.radians(alt), np.radians(az)
            return (up * np.sin(alt)[..., None] + np.cos(alt)[..., None] *
                    (n_hat * np.cos(az)[..., None] + e_hat * np.sin(az)[..., None]))

        for alt in (10, 20, 30, 40, 50, 60, 70, 80):
            az = np.linspace(0, 360, 721)
            draw_runs(dr, sky, altaz(np.full_like(az, alt), az), C_GRID, 2 * SS)
            xy, th = sky.to_pixel(altaz(np.array([float(alt)]), np.array([125.0])))
            if sky.visible(xy, th, margin=-25)[0]:
                dr.text((xy[0, 0] * SS, xy[0, 1] * SS), f'{alt}°', fill=(255, 255, 255, 110),
                        font=f_grid, anchor='mm', stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 150))

        cardinals = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
        for az in range(0, 360, 15):
            alt = np.linspace(0, 88, 300)
            draw_runs(dr, sky, altaz(alt, np.full_like(alt, float(az))),
                      C_GRID if az not in cardinals else (255, 255, 255, 80), 2 * SS)
        # the horizon is outside this crop, so tag each cardinal meridian at its
        # lowest point that is still inside the frame
        for az, nm in cardinals.items():
            for alt in np.arange(0, 80, 0.5):
                xy, th = sky.to_pixel(altaz(np.array([alt]), np.array([float(az)])))
                if sky.visible(xy, th, margin=-38)[0]:
                    dr.text((xy[0, 0] * SS, xy[0, 1] * SS), f'{nm} {alt:.0f}°', fill=C_CARD,
                            font=f_con, anchor='mm', stroke_width=max(1, SS // 2),
                            stroke_fill=(0, 0, 0, 190))
                    break
        for vv, nm, col in ((up, 'zenith', C_CARD), (ncp, 'NCP', C_CARD)):
            xy, th = sky.to_pixel(vv)
            if sky.visible(xy, th)[0]:
                x, y = xy[0, 0] * SS, xy[0, 1] * SS
                r = 7 * SS
                dr.line([(x - r, y), (x + r, y)], fill=col, width=2 * SS)
                dr.line([(x, y - r), (x, y + r)], fill=col, width=2 * SS)
                dr.text((x + r + 3 * SS, y + 2 * SS), nm, fill=col, font=f_grid,
                        stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 190))

    # --- constellation figures ------------------------------------------
    drawn_stars, drawn_cons, skipped_obstructed = set(), [], 0
    for abbr, c in sorted(cons.items()):
        if args.only and abbr not in args.only:
            continue
        if args.skip and abbr in args.skip:
            continue
        seen = []
        for chain in c['lines']:
            for h1, h2 in zip(chain, chain[1:]):
                if h1 not in vec or h2 not in vec:
                    continue
                a, b = vec[h1], vec[h2]
                xy, th = sky.to_pixel(np.stack([a, b]))
                vis = sky.visible(xy, th)
                if not vis.any():
                    continue
                # drop the whole segment if either end sits behind the trees,
                # otherwise a stub would stick out of the canopy
                if in_obstruction(xy[:, 0], xy[:, 1]).any():
                    skipped_obstructed += 1
                    continue
                ang = np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
                pts = slerp(a, b, max(12, int(ang * 2)))
                draw_runs(dr, sky, pts, C_LINE if vis.all() else C_LINE_FAINT, 2 * SS)
                for h, v in ((h1, vis[0]), (h2, vis[1])):
                    if v:
                        drawn_stars.add(h)
                        seen.append(h)
        if len(set(seen)) >= 2:
            pos = np.array([sky.to_pixel(vec[h])[0][0] for h in set(seen)])
            keep = ~in_obstruction(pos[:, 0], pos[:, 1])
            pos = pos[keep]
            inside = pos[(pos[:, 0] > 60) & (pos[:, 0] < sky.W - 60) &
                         (pos[:, 1] > 40) & (pos[:, 1] < sky.H - 40)] if len(pos) else pos
            if len(inside):
                drawn_cons.append((abbr, c['name'], inside.mean(axis=0), len(set(seen))))

    for h in drawn_stars:
        xy, _ = sky.to_pixel(vec[h])
        if in_obstruction(xy[0, 0], xy[0, 1])[0]:
            continue
        x, y = xy[0, 0] * SS, xy[0, 1] * SS
        rad = max(1.6, 4.2 - 0.45 * cat[h]['mag']) * SS
        dr.ellipse([x - rad, y - rad, x + rad, y + rad], outline=C_STAR, width=max(1, SS))

    for abbr, name, pos, n in drawn_cons:
        dr.text((pos[0] * SS, pos[1] * SS), name.upper(), fill=C_CON, font=f_con, anchor='mm',
                stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 170))

    # --- the hand-labeled stars -----------------------------------------
    for s in labels['stars']:
        x, y = s['x'] * SS, s['y'] * SS
        r = 9 * SS
        dr.ellipse([x - r, y - r, x + r, y + r], outline=C_LABELED, width=max(1, SS))
        if args.names == 'none':
            continue
        name = s['name']
        if args.names == 'proper' and name.upper().startswith('HIP') and name[3:].isdigit():
            rec = cat.get(int(name[3:]))
            if rec:
                name = rec['proper'] or f"{rec['bayer']} {rec['con']}"
        dr.text((x + r + 3 * SS, y - 6 * SS), name, fill=C_LABELED, font=f_star,
                stroke_width=max(1, SS // 2), stroke_fill=(0, 0, 0, 170))

    out = Image.alpha_composite(canvas, over).convert('RGB')
    out = out.resize((sky.W, sky.H), Image.LANCZOS)
    default = {'none': 'sky_constellations', 'altaz': 'sky_grid_altaz', 'radec': 'sky_grid_radec'}[args.grid]
    path = args.out or os.path.join(ROOT, f'{default}{"_raw" if args.raw else ""}.png')
    out.save(path)
    print(f'wrote {path}')
    if skipped_obstructed:
        print(f'skipped {skipped_obstructed} line segments behind the tree line')
    print(f'constellations drawn ({len(drawn_cons)}): ' +
          ', '.join(f'{a}({n})' for a, _, _, n in sorted(drawn_cons, key=lambda t: -t[3])))


if __name__ == '__main__':
    main()
