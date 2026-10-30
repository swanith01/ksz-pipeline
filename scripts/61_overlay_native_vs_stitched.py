"""
scripts/61_overlay_native_vs_stitched.py

Native all-sky D_ell (scripts/58 output, one or more seeds) against the
stitched flat-sky D_ell of scripts/55 and AMBER's Limber curve.

  * with --stitched-npz from the SAME run (same initial conditions: AMBER rerun
    with the field dumps kept, scripts/55 --mode momentum --no-rotation
    --zmin 4.55 --zmax 18.45): decides whether the low-ell excess over Limber
    is a feature of the native map-making / tiling or is present in a lightcone
    without line-of-sight tiling.
  * with two or more --native-run (different seeds): the ell-by-ell scatter at
    low ell (cosmic variance of one realization).

Native D_ell is averaged in log ell bins (single-ell values oscillate by
up to 10x); stitched points are interpolated (log-log) to the bin centres.

Usage
-----
    PYTHONPATH=src python scripts/61_overlay_native_vs_stitched.py \
        --native-run runs/native2gpc_s1_fields runs/native2gpc_s2 \
        --stitched-npz runs/native2gpc_s1_fields/stitched_streaming_momentum_norot.npz
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_bins(lo, hi, n):
    e = np.unique(np.round(np.geomspace(lo, hi, n + 1)).astype(int))
    return list(zip(e[:-1], e[1:]))


def binned(ell, D, bins):
    return np.array([np.nanmean(D[(ell >= b[0]) & (ell < b[1])]) for b in bins])


def loglog_interp(x, xp, fp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(np.clip(fp, 1e-30, None)),
                            left=np.nan, right=np.nan))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-run', nargs='+', required=True)
    ap.add_argument('--stitched-npz', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--bands', default='100,200;200,300;300,500;500,767')
    a = ap.parse_args(argv)

    nat = []
    for r in a.native_run:
        d = np.load(os.path.join(r, 'native_shells.npz'))
        nat.append(dict(name=os.path.basename(os.path.normpath(r)), ell=d['ell'],
                        D=d['D_total'], ll=d['ell_limber'], Dl=d['D_limber'], L=float(d['L'])))
    lmax = int(nat[0]['ell'].max())
    bins = log_bins(20, lmax, 16)
    bc = np.array([np.sqrt(b[0] * b[1]) for b in bins])
    nb = [binned(n['ell'], n['D'], bins) for n in nat]
    st = None
    if a.stitched_npz:
        s = np.load(a.stitched_npz)
        st = (s['ell_total'], s['D_total'], s['Derr_total'])
        st_at = loglog_interp(bc, st[0], st[1])

    bands = [tuple(int(x) for x in b.split(',')) for b in a.bands.split(';')]
    print("band-mean D_ell [uK^2]")
    head = "  ell band   " + "  ".join(f"native[{n['name'][:18]}]" for n in nat)
    head += "   stitched    Limber"
    print(head)
    for b in bands:
        row = f"  {b[0]:4d}-{b[1]:<4d}"
        for n in nat:
            row += f"  {np.nanmean(n['D'][b[0]:b[1]]):24.3f}"
        if st is not None:
            m = (st[0] >= b[0]) & (st[0] < b[1])
            row += f"  {np.nanmean(st[1][m]) if m.any() else float('nan'):9.3f}"
        else:
            row += "        n/a"
        lm = (nat[0]['ll'] >= b[0]) & (nat[0]['ll'] < b[1])
        row += f"  {np.mean(nat[0]['Dl'][lm]) if lm.any() else float('nan'):8.3f}"
        print(row)
    if len(nat) > 1:
        r = nb[1] / nb[0]
        print("\nbinned native ratio seed2/seed1 per ell bin (cosmic variance at low ell):")
        print("  " + "  ".join(f"{c:.0f}:{v:.2f}" for c, v in zip(bc, r)))

    fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.4]))
    for k, (n, y) in enumerate(zip(nat, nb)):
        ax[0].plot(bc, y, 'o-', ms=4, color=f'C{k}', label=f"native, {n['name']} (binned)")
        ax[0].plot(n['ell'], n['D'], '-', color=f'C{k}', lw=0.4, alpha=0.4)
    if st is not None:
        ax[0].errorbar(st[0], st[1], st[2], color='k', marker='.', ms=3, lw=1,
                       label='stitched flat-sky (scripts/55)')
    ax[0].plot(nat[0]['ll'], nat[0]['Dl'], '--', color='0.4', lw=1.5, label='AMBER Limber')
    ax[0].set(xscale='log', yscale='log', ylabel=r'$D_\ell\ [\mu K^2]$',
              title=f"native vs stitched vs Limber, L = {nat[0]['L']:g} Mpc/h")
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(10, 12000)
    if st is not None:
        for k, y in enumerate(nb):
            ax[1].plot(bc, y / st_at, 'o-', color=f'C{k}', ms=4, label=f'native {k + 1} / stitched')
    if len(nat) > 1:
        ax[1].plot(bc, nb[1] / nb[0], 's--', color='C2', ms=4, label='native 2 / native 1')
    ax[1].axhline(1, color='k', lw=0.8)
    ax[1].set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel='ratio')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = a.out or os.path.join('data', 'plots', 'native_vs_stitched_' + nat[0]['name'] + '.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
