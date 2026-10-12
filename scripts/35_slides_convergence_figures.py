#!/usr/bin/env python
"""
Script 35: two-panel slide figures for the convergence tests.

Each figure: LEFT = kSZ power D_ell(ell), RIGHT = reionisation history
<xH>(z), one curve per configuration, same colour on both panels. The point
of the right panel: the history stays essentially the same while the box size /
resolution / snapshot count changes, so any change on the left is not a
reionisation-history effect.

  --which coeval   : coeval-direct (Cain) -- box size, resolution, dz.
                     D_ell from data/products/convergence_{coeval_boxsize,
                     coeval_resolution,dz_coeval}_full.npz (scripts 04/06);
                     <xH>(z) from data/products/convergence_xH_histories.npz
                     (script 34, reads the sweep pickle caches).
  --which stitched : wrap-cycle stitched lightcones -- box size, resolution,
                     dz. D_ell and the stitched lightcone's own <xH>(z) from
                     convergence_stitched_{boxsize,resolution}_wrapcycle<seed>.npz
                     (script 05 --wrap-cycle-seed) and
                     <plot_dir>/28_stitched_dz_wrapcycle<seed>/stitched_dz_convergence.npz
                     (script 28 --wrap-cycle-seed). Reference: closure_test.npz
                     direct curve (black dashed) in the box-size/resolution panels.

Missing inputs are reported and that figure is skipped. No py21cmfast.

Usage:
    python scripts/35_slides_convergence_figures.py --which coeval
    python scripts/35_slides_convergence_figures.py --which stitched --seed 123
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# categorical slots 1-3 of the reference palette (blue, orange, aqua): used for
# the dz variants (3 unordered-ish categories); ordered sweeps use one hue,
# light -> dark with the swept quantity (box size, HII_DIM).
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#6250d6']
INK, MUTED = '#0b0b0b', '#52514e'

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': False,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})


def ramp(n):
    return [plt.cm.Blues(v) for v in np.linspace(0.38, 0.95, n)]


def d3000(ell, Dl):
    """D_ell at ell=3000, or NaN if 3000 lies outside the measured ell range
    (np.interp would silently return the end-point value there, e.g. for
    HII_DIM=32 whose Nyquist limit stops near ell~1700)."""
    ell = np.asarray(ell)
    if len(ell) == 0 or not (ell.min() <= 3000 <= ell.max()):
        return float('nan')
    return float(np.interp(3000, ell, Dl))


def d3000_label(v):
    return r"$D_{3000}$ = " + (f"{v:.2f}" if np.isfinite(v) else r"n/a ($\ell_{max}<3000$)")


def style_power(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'$D_\ell\ [\mu\mathrm{K}^2]$')
    ax.axvline(3000, color=MUTED, ls=':', lw=1.0)
    ax.text(3000 * 1.06, 0.02, r'$\ell=3000$', transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=10, va='bottom', ha='left')
    ax.set_title('kSZ power')


def style_hist(ax, zmax=12.0):
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=':')
    ax.set_xlim(4, zmax); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('redshift z'); ax.set_ylabel(r'$\langle x_{\rm HI}\rangle$')
    ax.set_title('reionisation history')


def finish(fig, axL, handles, out, suptitle):
    # legend below the panels, one row per ~4 entries, so it never covers data
    ncol = min(len(handles), 4) if len(handles) > 5 else len(handles)
    fig.legend(handles=handles, loc='lower center', ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, y=1.0, fontsize=15)
    fig.tight_layout(rect=(0, 0.04 if len(handles) <= 5 else 0.12, 1, 0.96))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {out}")


def _exists(p):
    if os.path.exists(p):
        return True
    print(f"SKIP: {p} not found")
    return False


# ------------------------------------------------------------------ coeval
def coeval_figure(kind, prod, out_dir):
    full = {'boxsize': 'convergence_coeval_boxsize_full.npz',
            'resolution': 'convergence_coeval_resolution_full.npz',
            'dz': 'convergence_dz_coeval_full.npz'}[kind]
    hp = f"{prod}/convergence_xH_histories.npz"
    if not (_exists(f"{prod}/{full}") and _exists(hp)):
        return
    d = np.load(f"{prod}/{full}", allow_pickle=True)
    H = np.load(hp)
    tags = [str(t) for t in d['tags']]
    cols = CAT[:len(tags)] if kind == 'dz' else ramp(len(tags))
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    handles = []
    for tag, c in zip(tags, cols):
        ell, Dl = d[f'ell_{tag}'], d[f'Dl_{tag}']
        v = d3000(ell, Dl)
        if kind == 'boxsize':
            L, N = tag[3:].split('_N'); lab = f"L = {L} Mpc ({N}³)"
        elif kind == 'resolution':
            lab = f"HII_DIM = {tag[3:]}"
        else:
            nz = int(d['x_values'][tags.index(tag)]); lab = f"{nz} snapshots"
        lab += "   " + d3000_label(v)
        aL.plot(ell, Dl, color=c, lw=2.2)
        key = f"{kind}__{tag}"
        if f"{key}__z" in H.files:
            aR.plot(H[f"{key}__z"], H[f"{key}__xH"], color=c, lw=2.2,
                    marker='o', ms=3.5 if kind != 'dz' else 4.5)
        else:
            lab += '  (no history)'
        handles.append(Line2D([0], [0], color=c, lw=2.4, label=lab))
    style_power(aL); style_hist(aR)
    sub = {'boxsize': 'box size, cell size fixed at 6.25 Mpc',
           'resolution': 'resolution, box fixed at 800 Mpc',
           'dz': 'number of redshift snapshots, 800 Mpc / 512³'}[kind]
    finish(fig, aL, handles, f"{out_dir}/slides_coeval_{kind}.png",
           f"Coeval-direct kSZ: convergence in {sub}")


# ---------------------------------------------------------------- stitched
def stitched_sweep_figure(kind, prod, out_dir, seed):
    f = f"{prod}/convergence_stitched_{kind}_wrapcycle{seed}.npz"
    if not _exists(f):
        return
    d = np.load(f, allow_pickle=True)
    tags = [str(t) for t in d['tags']]
    cols = ramp(len(tags))
    # same-box, same-resolution coeval-direct counterparts (already computed by
    # the coeval sweeps; tags are identical: box<L>_N<N> / res<N>)
    cf = f"{prod}/convergence_coeval_{kind}_full.npz"
    C = np.load(cf, allow_pickle=True) if os.path.exists(cf) else None
    if C is None:
        print(f"note: {cf} not found -> no coeval-direct dashed curves")
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    handles = []
    for i, (tag, col) in enumerate(zip(tags, cols)):
        ell, Dl = d[f'ell_{i}'], d[f'Dl_{i}']
        if kind == 'boxsize':
            L, N = tag[3:].split('_N'); lab = f"L = {L} Mpc ({N}\u00b3)"
        else:
            lab = f"HII_DIM = {tag[3:]}"
        v_st = d3000(ell, Dl)
        lab += "   stitched " + d3000_label(v_st).replace(r"$D_{3000}$ = ", "")
        aL.plot(ell, Dl, color=col, lw=2.2)
        if C is not None and f'ell_{tag}' in C.files:
            v_di = d3000(C[f'ell_{tag}'], C[f'Dl_{tag}'])
            aL.plot(C[f'ell_{tag}'], C[f'Dl_{tag}'], color=col, lw=1.6, ls='--')
            lab += " | direct " + d3000_label(v_di).replace(r"$D_{3000}$ = ", "")
        if f'z_hist_{i}' in d.files:
            aR.plot(d[f'z_hist_{i}'], d[f'xH_hist_{i}'], color=col, lw=2.2)
        handles.append(Line2D([0], [0], color=col, lw=2.4, label=lab))
    handles += [Line2D([0], [0], color=MUTED, lw=2.2, label='solid: wrap-cycle stitched'),
                Line2D([0], [0], color=MUTED, lw=1.6, ls='--', label='dashed: coeval-direct, same box and resolution')]
    style_power(aL); style_hist(aR)
    sub = {'boxsize': 'box size, cell size fixed at 6.25 Mpc',
           'resolution': 'resolution, box fixed at 800 Mpc'}[kind]
    finish(fig, aL, handles, f"{out_dir}/slides_wrapcycle_{kind}.png",
           f"Wrap-cycle stitched lightcone (seed {seed}): convergence in {sub}")


def stitched_dz_figure(path, out_dir, seed):
    if not _exists(path):
        return
    d = np.load(path)
    ms = [int(m) for m in d['dz_multiples']]
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    handles = []
    for m, c in zip(ms, CAT):
        aL.plot(d[f'ell_direct_dz_x{m}'], d[f'Dl_direct_dz_x{m}'], '-', lw=2.8, color=c)
        aL.plot(d[f'ell_dz_x{m}'], np.abs(d[f'Dl_total_dz_x{m}']), '-', lw=1.0, color=c)
        aL.plot(d[f'ell_dz_x{m}'], d[f'Dl_diag_dz_x{m}'], ':', lw=2.2, color=c)
        n_snap = len(d[f'snap_z_dz_x{m}'])
        aR.plot(d[f'hist_z_dz_x{m}'], d[f'hist_xH_dz_x{m}'], '-', lw=2.0, color=c)
        aR.plot(d[f'snap_z_dz_x{m}'], d[f'snap_xH_dz_x{m}'], 'o', ms=5.5, mfc='white',
                mec=c, mew=1.5)
        handles.append(Line2D([0], [0], color=c, lw=2.6, label=f"{n_snap} snapshots"))
    handles += [Line2D([0], [0], color=MUTED, lw=2.8, label='P_direct'),
                Line2D([0], [0], color=MUTED, lw=1.0, label='P_total (stitched)'),
                Line2D([0], [0], color=MUTED, lw=2.2, ls=':', label='P_diag (stitched)')]
    style_power(aL); style_hist(aR)
    aR.set_title('reionisation history (line: stitched, circles: snapshots)')
    box = int(d['BOX_LEN']) if 'BOX_LEN' in d.files else 800
    hd = int(d['HII_DIM']) if 'HII_DIM' in d.files else 512
    finish(fig, aL, handles, f"{out_dir}/slides_wrapcycle_dz.png",
           f"Wrap-cycle stitched lightcone (seed {seed}): convergence in number of snapshots, {box} Mpc / {hd}³")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['coeval', 'stitched', 'all'], default='all')
    ap.add_argument('--products', default='data/products')
    ap.add_argument('--plots', default='data/plots')
    ap.add_argument('--out', default='data/plots/slides')
    ap.add_argument('--seed', type=int, default=123, help='wrap_cycle_seed of the stitched runs')
    ap.add_argument('--stitched-dz-npz', default=None)
    a = ap.parse_args()
    if a.which in ('coeval', 'all'):
        for k in ('boxsize', 'resolution', 'dz'):
            coeval_figure(k, a.products, a.out)
    if a.which in ('stitched', 'all'):
        for k in ('boxsize', 'resolution'):
            stitched_sweep_figure(k, a.products, a.out, a.seed)
        stitched_dz_figure(a.stitched_dz_npz or
                           f"{a.plots}/28_stitched_dz_wrapcycle{a.seed}/stitched_dz_convergence.npz",
                           a.out, a.seed)


if __name__ == '__main__':
    main()
