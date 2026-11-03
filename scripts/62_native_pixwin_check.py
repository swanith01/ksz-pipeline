"""
scripts/62_native_pixwin_check.py

Pixel-window correction of the native HEALPix D_ell and the final comparison
with the stitched flat-sky D_ell and AMBER's Limber curve.

The native map is a pixel average, so its spectrum is C_l * w_l^2 (w_l the
HEALPix pixel window of its Nside); Limber and the stitched curve are not.
Dividing by w_l^2 should therefore bring runs of different Nside onto the
same curve (checked below, same realization) and, if the native lightcone is
right, onto the stitched curve at ell >~ 300.

Light job: reads native_shells.npz of each run (scripts/58) and the stitched
npz of scripts/55; seconds, can run on the login node.

    PYTHONPATH=src python scripts/62_native_pixwin_check.py \
        --native-run runs/native2gpc_s1_nside512 runs/native2gpc_s1_fields \
        --stitched-npz runs/native2gpc_s1_fields/stitched_streaming_momentum_norot.npz \
        --pixwin-dir ~/Healpix_3.83/data

The thick-slab ratios diag(T)/total of scripts/60 are unaffected: the window
is the same for every shell pair at a given ell.
"""
import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ksz_pipeline.amber.shell_maps import pixel_window


def log_bins(lo, hi, n):
    e = np.unique(np.round(np.geomspace(lo, hi, n + 1)).astype(int))
    return list(zip(e[:-1], e[1:]))


def _mean_or_nan(x):
    x = np.asarray(x, float)
    return np.nan if x.size == 0 or np.all(np.isnan(x)) else float(np.nanmean(x))


def binned(ell, D, bins):
    return np.array([_mean_or_nan(D[(ell >= b[0]) & (ell < b[1])]) for b in bins])


def loglog_interp(x, xp, fp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(np.clip(fp, 1e-30, None)),
                            left=np.nan, right=np.nan))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-run', nargs='+', required=True)
    ap.add_argument('--stitched-npz', default=None)
    ap.add_argument('--pixwin-dir', default=None)
    ap.add_argument('--bands', default='100,200;200,300;300,500;500,767;767,1100;1100,1535')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    runs = []
    for r in a.native_run:
        d = np.load(os.path.join(r, 'native_shells.npz'))
        nside = int(d['nside'])
        ell = d['ell']
        w = pixel_window(nside, int(ell.max()), a.pixwin_dir)
        runs.append(dict(name=os.path.basename(os.path.normpath(r)), nside=nside,
                         ell=ell, D=d['D_total'], w2=w**2, Dc=d['D_total'] / w**2,
                         ll=d['ell_limber'], Dl=d['D_limber'], L=float(d['L'])))
    st = None
    if a.stitched_npz:
        s = np.load(a.stitched_npz)
        st = (s['ell_total'], s['D_total'], s['Derr_total'])
    ref = max(runs, key=lambda x: x['nside'])                  # highest Nside
    bands = [tuple(int(x) for x in b.split(',')) for b in a.bands.split(';')]

    print("pixel window w_l^2 at band centres:")
    for n in runs:
        print(f"  Nside {n['nside']:4d}: " + "  ".join(
            f"{(b[0] + b[1]) // 2}:{n['w2'][min((b[0] + b[1]) // 2, len(n['w2']) - 1)]:.3f}"
            for b in bands if b[0] < len(n['w2'])))

    lim = ref['ll'], ref['Dl']
    print("\nband-mean D_ell [uK^2]: raw -> corrected (D / w^2), and ratios")
    for n in runs:
        print(f"\n  native {n['name']} (Nside {n['nside']}):")
        print("   ell band       raw  corrected   Limber  stitched  corr/Limber  corr/stitched")
        for b in bands:
            raw = _mean_or_nan(n['D'][b[0]:b[1]])
            cor = _mean_or_nan(n['Dc'][b[0]:b[1]])
            lm = (lim[0] >= b[0]) & (lim[0] < b[1])
            Dl = float(np.mean(lim[1][lm])) if lm.any() else np.nan
            if st is not None:
                m = (st[0] >= b[0]) & (st[0] < b[1])
                Ds = float(np.nanmean(st[1][m])) if m.any() else np.nan
            else:
                Ds = np.nan
            print(f"   {b[0]:4d}-{b[1]:<5d} {raw:9.3f} {cor:10.3f} {Dl:8.3f} {Ds:9.3f}"
                  f" {cor / Dl:12.3f} {cor / Ds:13.3f}")

    others = [n for n in runs if n is not ref]
    lmax_lo = min([int(n['ell'].max()) for n in others], default=None)
    if others:
        print(f"\ncorrected / raw consistency vs the highest-Nside run "
              f"(Nside {ref['nside']}), per ell bin up to the lower lmax:")
        bins = log_bins(20, lmax_lo, 12)
        bc = np.array([np.sqrt(b[0] * b[1]) for b in bins])
        rb = binned(ref['ell'], ref['D'], bins)
        rcb = binned(ref['ell'], ref['Dc'], bins)
        for n in others:
            nb, ncb = binned(n['ell'], n['D'], bins), binned(n['ell'], n['Dc'], bins)
            print(f"  Nside {n['nside']} / Nside {ref['nside']}")
            print("    ell          " + "  ".join(f"{c:6.0f}" for c in bc))
            print("    raw          " + "  ".join(f"{v:6.3f}" for v in nb / rb))
            print("    corrected    " + "  ".join(f"{v:6.3f}" for v in ncb / rcb))
            sel = bc >= 100
            if not sel.any():                     # tiny synthetic runs
                sel = np.ones_like(bc, bool)
            dev = np.nanmax(np.abs(ncb / rcb - 1)[sel])
            print(f"    max |corrected ratio - 1| for ell >= {bc[sel].min():.0f}: {dev:.3f}")

    # ---- plot -----------------------------------------------------------
    lmax = max(int(n['ell'].max()) for n in runs)
    bins = log_bins(20, lmax, 18)
    bc = np.array([np.sqrt(b[0] * b[1]) for b in bins])
    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.5]))
    for k, n in enumerate(runs):
        ax[0].plot(bc, binned(n['ell'], n['Dc'], bins), 'o-', ms=4, color=f'C{k}',
                   label=f"native Nside {n['nside']}, corrected")
        ax[0].plot(bc, binned(n['ell'], n['D'], bins), ':', color=f'C{k}', lw=1,
                   label=f"native Nside {n['nside']}, raw")
    if st is not None:
        ax[0].errorbar(st[0], st[1], st[2], color='k', marker='.', ms=3, lw=1,
                       label='stitched flat-sky (scripts/55)')
    ax[0].plot(lim[0], lim[1], '--', color='0.4', lw=1.5, label='AMBER Limber')
    ax[0].set(xscale='log', yscale='log', ylabel=r'$D_\ell\ [\mu K^2]$',
              title=f"pixel-window-corrected native vs stitched vs Limber, L = {ref['L']:g} Mpc/h")
    ax[0].set_xlim(10, 12000)
    ax[0].legend(fontsize=8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k, n in enumerate(runs):
            y = binned(n['ell'], n['Dc'], bins)
            if st is not None:
                ax[1].plot(bc, y / loglog_interp(bc, st[0], st[1]), 'o-', color=f'C{k}', ms=4,
                           label=f"corrected Nside {n['nside']} / stitched")
            lm = loglog_interp(bc, lim[0], lim[1])
            ax[1].plot(bc, y / lm, '^--', color=f'C{k}', ms=3, lw=0.8, alpha=0.7,
                       label=f"corrected Nside {n['nside']} / Limber")
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel='ratio', ylim=(0.4, 3))
    ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = a.out or os.path.join('data', 'plots', 'native_pixwin_corrected_' + ref['name'] + '.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
