"""
scripts/59_amber_native_visuals.py

Pictures for building intuition about AMBER's native HEALPix lightcone and
about the diag / near / far split used in scripts/58.

Figure 1  <prefix>_geometry.png
  row 1  a cut through the lightcone volume (plane z = 0, observer at the
         origin, shells = the annulus chi_min..chi_max) with the lattice of
         periodic box copies drawn on top: the same box content is repeated
         at x + L*m. For the box sizes of the two runs (L = 2000 and 256).
  row 2  the full-sky HEALPix map of one shell (Mollweide). HEALPix is
         only a way of cutting the sphere of directions n = x/|x| into equal-
         area pixels; directions are in the SIMULATION BOX axes (the pole is
         the box z axis), not RA/Dec.
  row 3  a "lightcone slice" like the 21cmFAST box-vs-redshift image: the
         pixels along the equator ring (azimuth, vertical) against shell
         distance (horizontal).
  Default content is a TOY periodic field (sum of a few box modes), clearly
  labelled: it is NOT kSZ and NOT AMBER output, only a field that is exactly
  periodic so the geometry is easy to see. With --run <dir> (a run with the
  0002 maps) the sky panels and the slice are the real shell maps.

Figure 2  <prefix>_decomposition.png
  how diag / near / far are defined: for a reference shell, which other shells
  are 'near' (|dchi| < L/2) or 'far' (>= L/2), and the shell-pair matrix
  C_ij coloured by class, for both box sizes.

Usage
-----
    python scripts/59_amber_native_visuals.py                 # toy, both L
    python scripts/59_amber_native_visuals.py --run runs/native_shells_n512_L2000_nside256
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

CHI_MIN, CHI_MAX = 5265.0, 7422.0          # z = 4.5-18.5, chen2022, Mpc/h
C_H0 = 2997.92458                          # c / (100 km/s/Mpc) in Mpc/h


def chi_of_z(z, Om=0.3):
    zg = np.linspace(0, 20, 20001)
    E = np.sqrt(Om * (1 + zg)**3 + 1 - Om)
    cum = np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zg))])
    return C_H0 * np.interp(z, zg, cum)


def shell_geometry(zmin=4.5, zmax=18.5, dz=0.1):
    """Edges and midpoints of the 140 shells (czdel = 0.1), Mpc/h."""
    ze = np.arange(zmin, zmax + 1e-9, dz)
    ce = chi_of_z(ze)                        # distance increases with z
    return ce, 0.5 * (ce[1:] + ce[:-1])


def toy_field(L, nmin=1, nmax=3, seed=3, slope=1.5):
    """Exactly L-periodic broadband toy field (sum of lattice plane waves)."""
    rng = np.random.default_rng(seed)
    ks, amps, phs = [], [], []
    for n in np.ndindex(*(2 * nmax + 1,) * 3):
        n = np.array(n) - nmax
        n2 = int((n**2).sum())
        if nmin**2 <= n2 <= nmax**2:
            ks.append(2 * np.pi * n / L)
            amps.append(rng.normal() * n2**(-slope / 2))
            phs.append(rng.uniform(0, 2 * np.pi))
    ks, amps, phs = map(np.array, (ks, amps, phs))
    return lambda x: (amps * np.cos(x @ ks.T + phs)).sum(axis=-1)


def cut_panel(ax, L, f, npx=420):
    """Cut z = 0. Big box: whole annulus. Small box (L < 500): zoom on a
    6 x 6 box patch at the outer edge, otherwise the grid is unreadable."""
    zoom = L < 500
    if zoom:
        h = 3.0 * L
        x0, y0 = 0.0, CHI_MAX - 1.2 * h
        xl, yl = (x0 - h, x0 + h), (y0 - h, y0 + h)
    else:
        xl = yl = (-7600.0, 7600.0)
    gx = np.linspace(*xl, npx)
    gy = np.linspace(*yl, npx)
    X, Y = np.meshgrid(gx, gy)
    r = np.hypot(X, Y)
    val = f(np.stack([X, Y, np.zeros_like(X)], axis=-1).reshape(-1, 3)).reshape(X.shape)
    val = np.where((r >= CHI_MIN) & (r <= CHI_MAX), val, np.nan)
    sd = np.nanstd(val)
    ax.imshow(val, extent=[*xl, *yl], origin='lower', cmap='RdBu_r',
              vmin=-2.5 * sd, vmax=2.5 * sd)
    for v in np.arange(np.ceil(xl[0] / L), np.floor(xl[1] / L) + 1) * L:
        ax.axvline(v, color='k', lw=0.8, alpha=0.5)
    for v in np.arange(np.ceil(yl[0] / L), np.floor(yl[1] / L) + 1) * L:
        ax.axhline(v, color='k', lw=0.8, alpha=0.5)
    for rr in (CHI_MIN, CHI_MAX):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color='0.2', lw=1))
    if not zoom:
        ax.plot(0, 0, 'k*', ms=10)
        p = CHI_MIN * 1.18 * np.array([np.cos(np.radians(35)), np.sin(np.radians(35))])
        ax.annotate('', xy=p + [L, 0], xytext=p, arrowprops=dict(arrowstyle='->', lw=1.8))
        ax.plot(*p, 'ko', ms=5)
        ax.plot(*(p + [L, 0]), 'ko', ms=5, mfc='w')
        ax.text(p[0] + L / 2, p[1] + 200, 'shift by L', ha='center', fontsize=8)
    ax.set(xlim=xl, ylim=yl, aspect='equal', xlabel='x [Mpc/h]', ylabel='y [Mpc/h]',
           title=(f'L = {L:g} Mpc/h: cut z=0, ZOOM on 6x6 box copies at the outer edge'
                  if zoom else f'L = {L:g} Mpc/h: cut z=0; lines = box copies; '
                                f'star = observer; arrow = a lattice image'))


def strip_image(ax, chi_edges, maps_eq, phi_deg, title):
    d = phi_deg[1] - phi_deg[0]
    pe = np.linspace(phi_deg[0] - d / 2, phi_deg[-1] + d / 2, len(phi_deg) + 1)
    im = ax.pcolormesh(chi_edges, pe, maps_eq.T, cmap='RdBu_r', shading='flat',
                       vmin=-2.5, vmax=2.5)
    ax.set(xlabel=r'shell distance $\chi$ [Mpc/h]  (z=4.5 left ... 18.5 right: '
                  r'$\chi$ increases to high z)', ylabel='azimuth along equator [deg]',
           title=title)
    return im


def axis_ring(L, chi_mid, nphi=360):
    """Azimuths (rad) of a strip of rays around the +x box axis on the
    equator, +-1.5 tile angles (tile angle = L / chi_mid)."""
    w = 1.5 * L / chi_mid
    return np.linspace(-w, w, nphi)


def zscore(m):
    m = np.asarray(m, float)
    return (m - m.mean(axis=-1, keepdims=True)) / m.std(axis=-1, keepdims=True)


def sky_panel(fig, sub, m, title, zoom_deg=None):
    """Mollweide of the whole sphere, or (zoom_deg) a gnomonic zoom centred
    on the +x box axis (theta = 90 deg, phi = 0)."""
    import healpy as hp
    if zoom_deg is None:
        hp.mollview(m, fig=fig.number, sub=sub, title=title, cmap='RdBu_r', min=-2.5,
                    max=2.5, cbar=False, notext=True)
        hp.graticule(dpar=30, dmer=30, alpha=0.3)
    else:
        xs = 500
        hp.gnomview(m, fig=fig.number, sub=sub, rot=(0, 0), xsize=xs,
                    reso=zoom_deg * 60.0 / xs, title=title, cmap='RdBu_r', min=-2.5,
                    max=2.5, cbar=False, notext=True)
        hp.graticule(dpar=5, dmer=5, alpha=0.3)


def figure1(prefix, Ls, nside, real=None):
    import healpy as hp
    fig = plt.figure(figsize=(15, 16))
    edges, mids = shell_geometry()
    for col, L in enumerate(Ls):
        nc = len(Ls)
        i = len(mids) // 2
        if real is None:
            f = toy_field(L)
            cut_panel(fig.add_subplot(3, nc, 1 + col), L, f)
            ns = nside * (2 if L < 500 else 1)
            n = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns)))).T
            sky = zscore(f(mids[i] * n))
            phi = axis_ring(L, mids[i])
            nvec = np.stack([np.cos(phi), np.sin(phi), 0 * phi], axis=-1)
            strip = zscore(np.array([f(c * nvec) for c in mids]))
            ce = edges
            tag = 'TOY periodic field (not kSZ, not AMBER)'
        else:
            maps, ce, mids = real['maps'], real['edges'], real['mids']
            ns = hp.npix2nside(maps.shape[1])
            i = len(mids) // 2
            sky = zscore(maps[i])
            phi = axis_ring(L, mids[i])
            idx = hp.ang2pix(ns, np.pi / 2, np.mod(phi, 2 * np.pi))
            strip = np.array([zscore(m)[idx] for m in maps])
            ax = fig.add_subplot(3, nc, 1 + col)
            ax.axis('off')
            ax.text(0.5, 0.5, f'L = {L:g} Mpc/h\n(no 3D cut for real maps)',
                    ha='center', va='center')
            tag = 'AMBER native kSZ shell map'
        zoom = 17.0 if L < 500 else None
        sky_panel(fig, (3, nc, nc + 1 + col), sky,
                  f'{tag}\n' + (f'ZOOM {zoom:g} deg around the +x box axis' if zoom
                                else 'full sky') +
                  f', ONE shell (chi = {mids[i]:.0f} Mpc/h), Nside = {ns};\n'
                  f'box repeats every ~{np.degrees(L / mids[i]):.1f} deg', zoom_deg=zoom)
        ax = fig.add_subplot(3, nc, 2 * nc + 1 + col)
        strip_image(ax, ce, strip, np.degrees(phi),
                    f'"lightcone slice": rays around the +x box axis (equator), '
                    f'+-{np.degrees(phi[-1]):.1f} deg, vs distance;\n'
                    f'on the axis the pattern repeats every L = {L:g} Mpc/h in chi')
        ax.set_ylabel('azimuth offset from the box x axis [deg]')
    fig.suptitle('AMBER native HEALPix lightcone: geometry (observer at origin, full sky)',
                 fontsize=13)
    fig.subplots_adjust(hspace=0.75, top=0.93, bottom=0.05, left=0.07, right=0.97)
    out = f'{prefix}_geometry.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def figure2(prefix, Ls, geom=None):
    edges, mids = geom if geom is not None else shell_geometry()
    cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#d62728'])
    fig, axs = plt.subplots(2, len(Ls), figsize=(7.5 * len(Ls), 9),
                            gridspec_kw=dict(height_ratios=[1, 3]))
    axs = np.atleast_2d(axs.T).T
    for col, L in enumerate(Ls):
        d = np.abs(mids[:, None] - mids[None, :])
        cls = np.where(d == 0, 0, np.where(d < 0.5 * L, 1, 2))
        ref = len(mids) // 2
        ax = axs[0, col]
        ax.pcolormesh(edges, [0, 1], cls[ref][None, :], cmap=cmap, vmin=0, vmax=2,
                      shading='flat')
        ax.set(yticks=[], xlabel=r'shell distance $\chi$ [Mpc/h]',
               title=f'L = {L:g}: shells paired with the reference shell '
                     f'(chi = {mids[ref]:.0f})')
        ax = axs[1, col]
        ax.pcolormesh(edges, edges, cls, cmap=cmap, vmin=0, vmax=2, shading='flat')
        ax.set_aspect('equal')
        ax.set(xlabel=r'$\chi_j$ [Mpc/h]', ylabel=r'$\chi_i$ [Mpc/h]',
               title=r'pair classes of $C_\ell^{ij}$ (140 x 140): '
                     f'near = |$\\Delta\\chi$| < L/2 = {L / 2:g}')
        frac = [(cls == k).mean() for k in range(3)]
        ax.text(0.02, 0.98, f'diag {frac[0] * 100:.1f}% of pairs\nnear {frac[1] * 100:.0f}%\n'
                            f'far {frac[2] * 100:.0f}%', transform=ax.transAxes, va='top',
                bbox=dict(fc='w', alpha=0.8), fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#1f77b4', label=r'diag: $i = j$ (a shell with itself)'),
                        Patch(color='#ff7f0e', label=r'near: $i \neq j$, $|\Delta\chi| < L/2$'),
                        Patch(color='#d62728', label=r'far: $|\Delta\chi| \geq L/2$')],
               loc='lower center', ncol=3)
    fig.suptitle(r'$C_\ell^{\rm total} = \sum_{i,j} C_\ell^{ij}$ = diag + near + far'
                 '   (cross-spectrum C_ij of shell maps i and j; can be NEGATIVE)', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = f'{prefix}_decomposition.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='run dir with output/cmb/mapksz_*.dat')
    ap.add_argument('--L', default='2000,256', help='box sizes for the toy panels [Mpc/h]')
    ap.add_argument('--nside', type=int, default=128, help='toy-map Nside')
    ap.add_argument('--prefix', default=None)
    a = ap.parse_args(argv)
    real = None
    Ls = [float(x) for x in a.L.split(',')]
    if a.run:
        from ksz_pipeline.amber.shell_maps import load_shell_maps
        meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
        Ls = [float(meta['L'])]
        sm = load_shell_maps(os.path.join(a.run, 'output', 'cmb'))
        edges = np.concatenate([sm['chi_lo'], sm['chi_hi'][-1:]])
        real = dict(maps=sm['maps'], edges=edges, mids=sm['chi'])
        name = os.path.basename(os.path.normpath(a.run))
    else:
        name = 'toy'
    prefix = a.prefix or os.path.join('data', 'plots', f'native_visuals_{name}')
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    print(figure1(prefix, Ls, a.nside, real))
    print(figure2(prefix, Ls, None if real is None else (real['edges'], real['mids'])))


if __name__ == '__main__':
    main()
