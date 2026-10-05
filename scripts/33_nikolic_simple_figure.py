#!/usr/bin/env python
"""
Script 33: simple two-panel summary of the Nikolic+23 replication.

  Left : reionization history, <xHI>(z): ours vs Nikolic+23 (digitized Fig. 1).
  Right: D_3000 histogram: ours (fixed and unfixed stitching) vs the Nikolic-
         implied 500 Mpc distribution.

Reads (no py21cmfast needed):
  data/products/nikolic_closing_summary.npz   (from script 32)
  data/external/nikolic_reionisation_history.csv  (columns x=z, y=xHI; digitized)

Usage:  python scripts/33_nikolic_simple_figure.py
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def main(summary, hist_csv, out):
    s = np.load(summary)
    fx, uf = s['fixed'][:, 0], s['unfixed'][:, 0]
    imp_m, imp_s = float(s['implied_mean']), float(s['implied_std'])
    z, xH = s['hist_z'], s['hist_xH']
    nik = np.loadtxt(hist_csv, delimiter=',', skiprows=1)
    nik = nik[np.argsort(nik[:, 0])]
    n = len(fx)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- left: reionization history
    a.plot(nik[:, 0], nik[:, 1], 's--', color='tab:orange', ms=5, lw=1.5,
           label='Nikolic+23 (digitized, sparse)')
    a.plot(z, xH, 'o-', color='tab:blue', ms=4, lw=1.8, label='this work')
    a.axhline(0.5, color='gray', lw=0.6)
    a.set_xlim(4, 14); a.set_ylim(-0.02, 1.02)
    a.set_xlabel('redshift z'); a.set_ylabel(r'neutral fraction $\langle x_{HI}\rangle$')
    a.set_title('reionization history')
    a.legend(loc='lower right', frameon=False)

    # ---- right: histogram
    edges = np.linspace(0.2, 2.2, 21)
    bw = edges[1] - edges[0]
    b.hist(fx, bins=edges, color='tab:blue', alpha=0.7, edgecolor='k',
           label=f'this work, rotated box (mean {fx.mean():.2f})')
    b.hist(uf, bins=edges, color='tab:red', alpha=0.6, edgecolor='k',
           label=f'this work, plain box reuse (mean {uf.mean():.2f})')
    xs = np.linspace(edges[0], edges[-1], 400)
    b.plot(xs, n * bw * gauss(xs, imp_m, imp_s), color='tab:orange', lw=2.5,
           label=f'Nikolic+23 ({imp_m:.2f}$\\pm${imp_s:.2f})')
    b.set_xlabel(r'$D_{3000}$ [$\mu$K$^2$]'); b.set_ylabel(f'realizations (N={n} seeds)')
    b.set_title(r'kSZ $D_{3000}$, 500 Mpc box')
    b.set_ylim(0, b.get_ylim()[1] * 1.45)
    b.legend(loc='upper right', frameon=False, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/products/nikolic_closing_summary.npz")
    ap.add_argument("--history-csv", default="data/external/nikolic_reionisation_history.csv")
    ap.add_argument("--out", default="data/plots/nikolic_simple_figure.png")
    a = ap.parse_args()
    main(a.summary, a.history_csv, a.out)
