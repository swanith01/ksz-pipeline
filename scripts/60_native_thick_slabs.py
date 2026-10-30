"""
scripts/60_native_thick_slabs.py

Is P_off zero, and does that mean Limber is valid? -- the thick-slab version
of scripts/58, on the native HEALPix shell maps (needs only the mapksz_*.dat
of an AMBER run built with patch 0002; no new AMBER run).

Shell maps add linearly, so merging neighbouring shells into slabs of
thickness T only means summing blocks of the shell cross-spectrum matrix
C_ij. The total is unchanged; what changes is how it splits into
D_diag(T) (each slab with itself) and D_off(T) = total - diag.
    * thin slabs: diag >> total (neighbours cancel; off is large and negative)
    * thick slabs, T >> coherence length of the field along the LOS: slabs are
      independent, off -> 0, diag(T) -> total.
So diag(T)/total vs T shows whether and where P_off -> 0. Whether the
resulting total equals Limber is then a separate comparison (printed).

Also: cumulative band-mean D(|dchi| <= dmax), i.e. what the pairs closer than
dmax add up to, relative to the total (panel c).

Outputs: <run>/native_thick_slabs.npz and data/plots/native_thick_slabs_<run>.png
(three panels: a) binned D_ell: total, diag(T) for several T, Limber;
 b) diag(T)/total vs T in ell bands; c) cumulative / total vs dmax).

Usage (compute node; cross-spectra take ~20 s, the rest is quick):
    PYTHONPATH=src python -u scripts/60_native_thick_slabs.py --run runs/native_shells_n512_L2000_nside256
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from ksz_pipeline.amber import io
from ksz_pipeline.amber.shell_maps import (
    load_shell_maps, shell_alms, cross_spectra, slab_groups, merge_blocks,
    band_mean, cumulative_vs_separation)


def log_bins(lo, hi, n):
    e = np.unique(np.round(np.geomspace(lo, hi, n + 1)).astype(int))
    return list(zip(e[:-1], e[1:]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--thickness', default='0,50,100,250,500,1000',
                    help='slab thicknesses [Mpc/h]; 0 = the individual shells')
    ap.add_argument('--bands', default='100,200;200,300;300,500;500,767')
    a = ap.parse_args(argv)
    import healpy as hp                                   # noqa: F401

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    L = float(meta['L'])
    Tcmb_uK = meta['amber_params'].get('Tcmb', 2.725) * 1e6
    cmb = os.path.join(a.run, 'output', 'cmb')
    name = os.path.basename(os.path.normpath(a.run))
    t0 = time.time()
    sm = load_shell_maps(cmb)
    nside = sm['nside']
    lmax = 3 * nside - 1
    chi, lo, hi = sm['chi'], sm['chi_lo'], sm['chi_hi']
    C = cross_spectra(shell_alms(sm['maps'], lmax), lmax)
    ell = np.arange(lmax + 1)
    toD = ell * (ell + 1) / (2 * np.pi) * Tcmb_uK**2
    print(f"{name}: {len(chi)} shells, L = {L:g} Mpc/h, window {hi[-1] - lo[0]:.0f} Mpc/h "
          f"({(hi[-1] - lo[0]) / L:.2f} L); cross-spectra {time.time() - t0:.0f}s", flush=True)

    la, Ca = io.read_amber_cl(os.path.join(cmb, 'cl_ksz.txt'))
    Dlim_all = np.full(lmax + 1, np.nan)
    ok = (ell >= la.min()) & (ell <= la.max())
    Dlim_all[ok] = np.interp(ell[ok], la, la * (la + 1) / (2 * np.pi) * Ca * Tcmb_uK**2)

    Ts = [float(x) for x in a.thickness.split(',')]
    bands = [tuple(int(x) for x in b.split(',')) for b in a.bands.split(';')]
    total = C.sum(axis=(0, 1)) * toD
    Dtot = total
    diag = {}
    nslab = {}
    for T in Ts:
        groups = [np.array([i]) for i in range(len(chi))] if T <= 0 else slab_groups(lo, hi, T)
        Cs = merge_blocks(C, groups)
        diag[T] = np.einsum('iil->l', Cs) * toD
        nslab[T] = len(groups)
        assert np.allclose(Cs.sum(axis=(0, 1)) * toD, Dtot, rtol=1e-8, atol=1e-14)

    print("\nband-mean D_ell [uK^2]: total, Limber, and diag(T) for slab thickness T")
    head = "  ell band      total   Limber  total/Lim | " + "  ".join(
        f"T={T:g}({nslab[T]})" for T in Ts)
    print(head)
    ratio = np.zeros((len(bands), len(Ts)))
    for bi, b in enumerate(bands):
        tot, lim = band_mean(Dtot, b), band_mean(Dlim_all, b)
        row = f"  {b[0]:4d}-{b[1]:<4d} {tot:9.3f} {lim:8.3f} {tot / lim:9.2f} | "
        for ti, T in enumerate(Ts):
            ratio[bi, ti] = band_mean(diag[T], b) / tot
            row += f"{ratio[bi, ti]:11.2f}"
        print(row)
    print("  (entries after '|' are diag(T) / total: -> 1 when P_off -> 0)")

    dmax = np.linspace(0, hi[-1] - lo[0], 60)
    cum = np.array([cumulative_vs_separation(C, chi, toD, b, dmax) /
                    band_mean(Dtot, b) for b in bands])
    print("\ncumulative D(|dchi| <= dmax) / total, per ell band:")
    for dm in (50, 100, 250, 500, 1000, 1500, 2000):
        if dm <= dmax[-1]:
            j = np.argmin(np.abs(dmax - dm))
            print(f"  dmax {dmax[j]:6.0f} Mpc/h: " +
                  "  ".join(f"{b[0]}-{b[1]}: {cum[bi, j]:7.2f}" for bi, b in enumerate(bands)))

    np.savez(os.path.join(a.run, 'native_thick_slabs.npz'), ell=ell, D_total=Dtot,
             D_limber=Dlim_all, thickness=np.array(Ts), nslab=np.array([nslab[T] for T in Ts]),
             D_diag=np.array([diag[T] for T in Ts]), bands=np.array(bands), ratio=ratio,
             dmax=dmax, cumulative=cum, L=L)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    bins = log_bins(20, lmax, 18)
    bc = np.array([np.sqrt(b[0] * b[1]) for b in bins])
    def bm(D):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN Limber bins
            return np.array([np.nanmean(D[b[0]:b[1]]) for b in bins])
    ax[0].plot(bc, bm(Dtot), 'k-o', ms=3, lw=2, label='total (all pairs)')
    for k, T in enumerate(Ts):
        lab = 'diag, single shells' if T <= 0 else f'diag, slabs of {T:g} Mpc/h ({nslab[T]})'
        ax[0].plot(bc, bm(diag[T]), color=plt.cm.viridis(k / max(len(Ts) - 1, 1)), lw=1.3, label=lab)
    ax[0].plot(bc, bm(Dlim_all), '--', color='0.4', lw=1.5, label='AMBER Limber (cl_ksz.txt)')
    ax[0].set(xscale='log', yscale='log', xlabel=r'$\ell$ (binned)',
              ylabel=r'$D_\ell\ [\mu K^2]$', title='diag(T) -> total as slabs get thicker?')
    ax[0].legend(fontsize=7)
    xs = [max(T, 0.5 * (hi - lo).min()) for T in Ts]
    for bi, b in enumerate(bands):
        ax[1].plot(xs, ratio[bi], 'o-', label=fr'$\ell$ = {b[0]}-{b[1]}')
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set(xscale='log', yscale='log', xlabel='slab thickness T [Mpc/h] (shells at left)',
              ylabel='diag(T) / total',
              title='P_off -> 0 where the curve reaches 1')
    ax[1].legend(fontsize=8)
    for bi, b in enumerate(bands):
        ax[2].plot(dmax, cum[bi], label=fr'$\ell$ = {b[0]}-{b[1]}')
    ax[2].axhline(1, color='k', lw=0.8)
    ax[2].axvline(L / 2, color='0.6', ls=':')
    ax[2].set(xlabel=r'$\Delta\chi_{max}$ [Mpc/h]: pairs with $|\Delta\chi| \leq \Delta\chi_{max}$',
              ylabel='cumulative D / total',
              title='where the pair sum converges (dotted: L/2)')
    ax[2].set_yscale('symlog', linthresh=1)
    ax[2].legend(fontsize=8)
    fig.suptitle(f"{name}: thick slabs of the AMBER native kSZ lightcone, L = {L:g} Mpc/h",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join('data', 'plots', f'native_thick_slabs_{name}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
