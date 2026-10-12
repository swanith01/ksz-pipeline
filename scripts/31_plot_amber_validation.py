"""
scripts/31_plot_amber_validation.py

Plots the gate's saved comparison: this pipeline's compute_cell(D_ell)
vs AMBER's own Limber sum, both from the same AMBER fields. Reads
<run>/amber_gate.npz, written by scripts/30_amber_validation_gate.py --
run that first if this file doesn't exist yet.

Light (matplotlib on a few thousand saved numbers) -- fine on a login
node, no PBS needed.

Usage:
  python scripts/31_plot_amber_validation.py --run runs/amber_q01
  # -> data/plots/<run_name>_dell_validation.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None,
                    help='output PNG path (default: data/plots/<run>_dell_validation.png)')
    a = ap.parse_args()

    npz_path = os.path.join(a.run, 'amber_gate.npz')
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found -- run "
                         f"scripts/30_amber_validation_gate.py --run {a.run} first")
    d = np.load(npz_path)
    ells, D_cc, D_win = d['ells'], d['D_compute_cell'], d['D_amber_window']
    ell_a, D_full = d['ell_amber'], d['D_amber_full']
    zlo, zhi = d['window']
    frac_out = float(d['frac_outside'])

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))

    top.loglog(ell_a, D_full, color='0.6', lw=1.5,
              label='AMBER (native), P$_{q\\perp}$, full z range \u2014 includes post-reionization kSZ')
    top.loglog(ells, D_win, 'o', ms=4, mfc='none', mec='C1', mew=1.3,
              label=f'AMBER (native), P$_{{q\\perp}}$, same window (z={zlo:.2f}\u2013{zhi:.2f})')
    top.loglog(ells, D_cc, '-', color='C0', lw=1.8,
              label='Our pipeline, P$_{q\\perp}$ (on AMBER\u2019s fields)')
    top.set_ylabel(r'$D_\ell$  [$\mu$K$^2$]')
    top.legend(frameon=False, fontsize=9, loc='lower left')
    top.set_title('Our P$_{q\\perp}$+Limber pipeline vs AMBER\u2019s own \u2014 same AMBER fields')

    ratio = D_cc / np.interp(ells, ell_a, D_full)
    ratio_win = D_cc / D_win
    bot.axhline(1.0, color='0.7', lw=1)
    bot.semilogx(ells, ratio_win, 'o-', ms=4, color='C0',
                 label='our P$_{q\\perp}$ / AMBER (native) P$_{q\\perp}$, same window')
    bot.set_ylabel('ratio')
    bot.set_xlabel(r'$\ell$')
    bot.set_ylim(0.8, 1.2)
    bot.legend(frameon=False, fontsize=8, loc='upper left')

    fig.text(0.99, 0.01,
             f'{frac_out:.1%} of AMBER\u2019s (native, full-range) D$_{{3000}}$ comes from '
             f'below the patchy window (post-reionization kSZ, excluded here by design)',
             ha='right', va='bottom', fontsize=7.5, color='0.4', style='italic')

    os.makedirs('data/plots', exist_ok=True)
    run_name = os.path.basename(os.path.normpath(a.run))
    out = a.out or f'data/plots/{run_name}_dell_validation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
