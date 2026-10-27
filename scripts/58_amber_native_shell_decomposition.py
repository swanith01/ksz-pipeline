"""
scripts/58_amber_native_shell_decomposition.py

Is AMBER's native HEALPix kSZ lightcone periodic, and how much of its power
is coherence between shells (P_off) that Limber cannot have? Needs an AMBER
run built with external/amber_patch/0002 and Map = make/write, which writes
output/cmb/mapksz_<z>.dat after every CMB shell (see
ksz_pipeline.amber.shell_maps for the method).

Outputs (in --run and data/plots/):
  native_shells.npz                 everything below, for re-plotting
  native_shells_<run>.png           three panels:
    (a) D_l: native total (sum over shell pairs) vs AMBER's own
        cl_ksz_healpix.txt and its Limber cl_ksz.txt; diag, near-off
        (|dchi| < L/2) and far-off (>= L/2) parts
    (b) shell-pair correlation r(dchi/L) in ell bands; lattice returns
        expected at 1, sqrt2, sqrt3, 2
    (c) lattice-image correlation per lattice vector, all sky and per
        direction class of the pixel, with half-lattice controls

Sanity checks printed first: the shells sum back to AMBER's
map_ksz_nside=*.fits (float32), and the summed spectrum to
cl_ksz_healpix.txt.

Reading the results: the lattice-image correlation is the definitive
periodicity test (an exactly periodic field gives ~1 for every lattice
vector, the half-lattice control ~0). The all-sky r(dchi) is weak by
construction -- only near-axis rays repeat radially, so a sky average
largely washes it out even for perfect tiling. The question "is P_off
zero / how much of the native power is cross-shell coherence" is answered
by the far part of the decomposition and by (diag+near)/Limber.

Usage
-----
    PYTHONPATH=src python -u scripts/58_amber_native_shell_decomposition.py \
        --run runs/native_shells_n128_nside256
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from ksz_pipeline.amber import io
from ksz_pipeline.amber.shell_maps import (
    load_shell_maps, shell_alms, cross_spectra, decompose, r_vs_dchi,
    lattice_vectors, direction_class, lattice_image_correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--lmax', type=int, default=None, help='default 3*Nside-1')
    ap.add_argument('--half-angle', type=float, default=10.0,
                    help='direction-class cap radius [deg]')
    ap.add_argument('--bands', default='20,100;100,300;300,700',
                    help='ell bands for r(dchi), "lo,hi;lo,hi;..."')
    a = ap.parse_args()
    import healpy as hp

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    L = float(meta['L'])                                   # Mpc/h, as AMBER's r
    Tcmb_uK = meta['amber_params'].get('Tcmb', 2.725) * 1e6
    cmb = os.path.join(a.run, 'output', 'cmb')
    run_name = os.path.basename(os.path.normpath(a.run))
    t0 = time.time()

    sm = load_shell_maps(cmb)
    nside = sm['nside']
    lmax = a.lmax or 3 * nside - 1
    print(f"{run_name}: {len(sm['chi'])} shells, Nside={nside}, L={L:g} Mpc/h, "
          f"chi={sm['chi'][0]:.0f}-{sm['chi'][-1]:.0f} Mpc/h "
          f"({(sm['chi'][-1] - sm['chi'][0]) / L:.1f} box lengths), "
          f"shell width {np.min(sm['chi_hi'] - sm['chi_lo']):.0f}-"
          f"{np.max(sm['chi_hi'] - sm['chi_lo']):.0f} Mpc/h", flush=True)

    # ---- sanity: shells reproduce AMBER's own map and spectrum ----------
    fits = glob.glob(os.path.join(cmb, 'map_ksz_nside=*.fits'))
    if fits:
        m_fits = hp.read_map(fits[0])
        d = np.abs(sm['total'] - m_fits).max() / sm['total'].std()
        print(f"  sum of shells vs {os.path.basename(fits[0])}: "
              f"max|diff|/rms = {d:.2e} (float32 FITS: expect ~1e-6)", flush=True)
    else:
        print("  (no map_ksz FITS to check against -- Map = make?)")

    alms = shell_alms(sm['maps'], lmax)
    C = cross_spectra(alms, lmax)
    del alms
    dec = decompose(C, sm['chi'], L)
    ell = np.arange(lmax + 1)
    toD = ell * (ell + 1) / (2 * np.pi) * Tcmb_uK**2
    D = {k: v * toD for k, v in dec.items()}
    print(f"  cross-spectra: {time.time() - t0:.0f}s", flush=True)

    hpx = os.path.join(cmb, 'cl_ksz_healpix.txt')
    D_hpx = None
    if os.path.exists(hpx):
        lh, Ch = io.read_amber_cl(hpx)
        D_hpx = (lh, lh * (lh + 1) / (2 * np.pi) * Ch * Tcmb_uK**2)
        sel = (lh >= 20) & (lh <= lmax)
        ratio = np.interp(lh[sel], ell, D['total']) / D_hpx[1][sel]
        print(f"  summed-shell D_l vs cl_ksz_healpix.txt (l=20-{lmax}): "
              f"median ratio {np.median(ratio):.4f}, range "
              f"{ratio.min():.3f}-{ratio.max():.3f}", flush=True)
    la, Ca = io.read_amber_cl(os.path.join(cmb, 'cl_ksz.txt'))
    D_lim = (la, la * (la + 1) / (2 * np.pi) * Ca * Tcmb_uK**2)

    print("\n   ell    total     diag     near      far   far/total   Limber   (diag+near)/Limber")
    for lq in (30, 60, 100, 150, 200, 300, 500, 700):
        if lq > lmax:
            continue
        i = lq
        print(f"  {lq:4d}  {D['total'][i]:7.3f}  {D['diag'][i]:7.3f}  "
              f"{D['near'][i]:7.3f}  {D['far'][i]:7.3f}   {D['far'][i] / D['total'][i]:7.3f}"
              f"   {np.interp(lq, *D_lim):6.3f}   "
              f"{(D['diag'][i] + D['near'][i]) / np.interp(lq, *D_lim):6.3f}")

    # ---- r(dchi) ----------------------------------------------------------
    bands = [tuple(int(x) for x in b.split(',')) for b in a.bands.split(';')]
    edges = np.arange(0.0, 3.0001, 0.05)
    R = {b: r_vs_dchi(C, sm['chi'], L, b, edges) for b in bands if b[1] <= lmax + 1}
    print("\n  r(dchi) at lattice lags (band: dchi/L = 0.5, 1, sqrt2, sqrt3, 2):")
    for b, r in R.items():
        at = lambda x: r[min(int(x / 0.05), len(r) - 1)]
        print(f"    l={b[0]}-{b[1]}:  " + "  ".join(
            f"{at(x):+.3f}" for x in (0.5, 1.0, np.sqrt(2), np.sqrt(3), 2.0)))

    # ---- lattice-image correlation -------------------------------------
    cls = direction_class(nside, a.half_angle)
    groups = {'all': None, 'axis-px': cls == 1, 'face-px': cls == 2, 'body-px': cls == 3}
    vecs = lattice_vectors(3)
    img = np.full((len(vecs), len(groups), 2), np.nan)
    npairs = np.zeros((len(vecs), len(groups)), int)
    print(f"\n  lattice-image correlation (control = half lattice), "
          f"direction caps {a.half_angle:g} deg:")
    print("    m           |m|    " + "  ".join(f"{g:>16s}" for g in groups))
    for vi, m in enumerate(vecs):
        row = []
        for gi, (g, mask) in enumerate(groups.items()):
            rho, n = lattice_image_correlation(sm, L, m, 1.0, mask)
            ctl, _ = lattice_image_correlation(sm, L, m, 0.5, mask)
            img[vi, gi] = rho, ctl
            npairs[vi, gi] = n
            row.append(f"{rho:+.3f} ({ctl:+.3f})")
        print(f"    {tuple(int(x) for x in m)!s:12s} {np.linalg.norm(m):.3f}  "
              + "  ".join(f"{c:>16s}" for c in row), flush=True)

    np.savez(os.path.join(a.run, 'native_shells.npz'),
             ell=ell, D_total=D['total'], D_diag=D['diag'], D_near=D['near'],
             D_far=D['far'], ell_limber=D_lim[0], D_limber=D_lim[1],
             ell_hpx=D_hpx[0] if D_hpx else np.array([]),
             D_hpx=D_hpx[1] if D_hpx else np.array([]),
             chi=sm['chi'], chi_lo=sm['chi_lo'], chi_hi=sm['chi_hi'], z=sm['z'],
             L=L, nside=nside, r_edges=edges,
             r_bands=np.array(list(R.keys())), r_vals=np.array(list(R.values())),
             lattice_vectors=np.array(vecs), lattice_groups=np.array(list(groups)),
             lattice_rho=img[..., 0], lattice_control=img[..., 1], lattice_npairs=npairs)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sl = slice(2, lmax + 1)
    for k, sty in (('total', 'k-'), ('diag', 'C0-'), ('near', 'C1-'), ('far', 'C3-')):
        ax[0].plot(ell[sl], np.abs(D[k][sl]), sty, lw=1.5,
                   label=f'{k}' + (' (|.|)' if k in ('near', 'far') else ''))
    if D_hpx:
        ax[0].plot(*D_hpx, 'k:', lw=1, label='AMBER cl_ksz_healpix.txt')
    ax[0].plot(*D_lim, '--', color='0.5', lw=1.5, label='AMBER Limber cl_ksz.txt')
    ax[0].set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel=r'$D_\ell\ [\mu K^2]$',
              xlim=(10, lmax), title='native map: diag / off(near, far)')
    ax[0].legend(fontsize=8)
    xc = 0.5 * (edges[1:] + edges[:-1])
    for b, r in R.items():
        ax[1].plot(xc, r, '.-', label=fr'$\ell$={b[0]}-{b[1]}')
    for x in (1, np.sqrt(2), np.sqrt(3), 2):
        ax[1].axvline(x, color='0.7', ls=':')
    ax[1].axhline(0, color='0.7', lw=0.5)
    ax[1].set(xlabel=r'$\Delta\chi / L$', ylabel='shell-pair correlation',
              title='returns at L, sqrt2 L, sqrt3 L, 2L?')
    ax[1].legend(fontsize=8)
    norms = np.array([np.linalg.norm(v) for v in vecs])
    for gi, g in enumerate(groups):
        jit = (gi - 1.5) * 0.02
        ax[2].plot(norms + jit, img[:, gi, 0], 'o', ms=4, label=g)
        ax[2].plot(norms + jit, img[:, gi, 1], 'x', ms=4, color='0.6')
    ax[2].axhline(0, color='0.7', lw=0.5)
    ax[2].set(xlabel='|m| (lattice vector, units of L)', ylabel='lattice-image correlation',
              title='o: shift L m    x: control L m / 2')
    ax[2].legend(fontsize=8)
    fig.suptitle(f"{run_name}: AMBER native kSZ lightcone, {len(sm['chi'])} shells, "
                 f"Nside={nside}, L={L:g} Mpc/h", fontsize=10)
    fig.tight_layout()
    out = os.path.join('data', 'plots', f'native_shells_{run_name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out} and {os.path.join(a.run, 'native_shells.npz')} "
          f"({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
