#!/usr/bin/env python
"""
Script 30: histogram of D_3000 across the Nikolic-replication ensemble,
in the style of Nikolic et al. (2023) Appendix A / Fig. A1.

Companion to script 29 (which draws a scatter+band summary). Two panels:

  left  : histogram of our per-seed D_3000 (P_total), with Nikolic's
          calibration-implied 500 Mpc distribution (their Fig.1 value at
          1.5 Gpc divided by f=1.27+/-0.19) overlaid as a Gaussian scaled to
          the same counts.
  right : the same draws expressed as f_i = D_3000(1.5 Gpc) / D_3000_i, the
          quantity Nikolic quote (f=1.27+/-0.19, N=20), so our f can be read
          directly against theirs.

No py21cmfast dependency (numpy/matplotlib/yaml only). Needs the per-seed
.npz products from script 17 wherever it runs, plus
data/external/nikolic_ksz_power.csv.

FILE SELECTION -- READ THIS BEFORE RUNNING
------------------------------------------
configs/nikolic_mesinger.yaml now has PHOTON_CONS: true, so the plain file
coherence_decomposition_wrapcycle101_nikolic.npz is the PHOTON_CONS run of
seed 101 (the original no-photon-cons seed-101 output was renamed to
..._nikolic_nophotoncons.npz). Script 29 does not know this and would
silently mix the two. This script therefore tries, for every seed,
  1. coherence_decomposition_wrapcycle{seed}_nikolic{--baseline-suffix}.npz
  2. coherence_decomposition_wrapcycle{seed}_nikolic.npz
in that order, and PRINTS which file each seed used. The histogram is built
from the baseline (no-photon-cons) draws; seeds passed via --pc-seeds are
read from the plain file name and drawn as separate dashed markers, never
included in the histogram, mean or std.

Usage
-----
    python scripts/30_nikolic_histogram.py --pc-seeds 101
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Nikolic et al. (2023), Appendix A / Fig. A1 -- same constants as script 29.
NIKOLIC_F_MEAN  = 1.27
NIKOLIC_F_STD   = 0.19
NIKOLIC_N_SEEDS = 20


def _d3000(path):
    d = np.load(path)
    return float(np.interp(3000, d['ell_dec'], d['Dl_total']))


def _load_baseline(out_dir, seed, suffix):
    """Return (D_3000, path used). Prefers the suffixed (backup) file."""
    candidates = [f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic{suffix}.npz",
                  f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic.npz"]
    for path in candidates:
        if os.path.exists(path):
            return _d3000(path), path
    raise FileNotFoundError(
        f"neither {candidates[0]} nor {candidates[1]} found -- run script 17 for seed {seed} "
        f"(--config configs/nikolic_mesinger.yaml --skip-direct --random-seed {seed} "
        f"--wrap-cycle-seed {seed}) first.")


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def main(config_path, seeds, suffix, pc_seeds, bins):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')

    d3000, used = [], []
    for seed in seeds:
        val, path = _load_baseline(out_dir, seed, suffix)
        d3000.append(val)
        used.append(path)
        print(f"  seed {seed}: D_3000 = {val:.4f}  <- {os.path.basename(path)}")
    d3000 = np.array(d3000)
    n = len(d3000)

    pc_vals = {}
    for seed in pc_seeds:
        path = f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"--pc-seeds {seed}: {path} not found")
        pc_vals[seed] = _d3000(path)
        print(f"  seed {seed} (PHOTON_CONS overlay): D_3000 = {pc_vals[seed]:.4f}  <- "
              f"{os.path.basename(path)}")

    mid = np.loadtxt('data/external/nikolic_ksz_power.csv', delimiter=',', skiprows=1)
    d3000_1p5 = float(np.interp(3000, mid[:, 0], mid[:, 1]))
    implied_mean = d3000_1p5 / NIKOLIC_F_MEAN
    implied_std  = implied_mean * (NIKOLIC_F_STD / NIKOLIC_F_MEAN)

    mean, std = d3000.mean(), d3000.std(ddof=1)
    f_ours = d3000_1p5 / d3000
    f_mean, f_std = f_ours.mean(), f_ours.std(ddof=1)

    print(f"\nBaseline ensemble: N={n}, seeds {seeds}")
    print(f"  D_3000: mean={mean:.4f}, std={std:.4f} ({100*std/mean:.1f}% relative)")
    print(f"  Nikolic-implied 500 Mpc: {implied_mean:.4f} +/- {implied_std:.4f} "
          f"({100*implied_std/implied_mean:.1f}% relative, N={NIKOLIC_N_SEEDS})")
    gap = abs(mean - implied_mean)
    print(f"  gap = {gap:.4f} uK^2 = {gap/(std/np.sqrt(n)):.2f} sigma on OUR mean's standard error, "
          f"{gap/implied_std:.2f} sigma on their single-draw scatter")
    print(f"  f = D_3000(1.5 Gpc)/D_3000(500 Mpc): ours {f_mean:.2f} +/- {f_std:.2f}  "
          f"vs Nikolic {NIKOLIC_F_MEAN} +/- {NIKOLIC_F_STD}")
    if n < 10:
        print(f"  NOTE: N={n} is thin for a histogram (Nikolic use {NIKOLIC_N_SEEDS}); "
              f"read the shape loosely, the mean/std are the meaningful numbers.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---- left: D_3000 histogram
    extras = list(pc_vals.values())
    lo = min([d3000.min(), implied_mean - 3 * implied_std] + extras)
    hi = max([d3000.max(), implied_mean + 3 * implied_std] + extras)
    edges = np.histogram_bin_edges(d3000, bins=bins if bins else 'auto')
    bw = edges[1] - edges[0]
    ax1.hist(d3000, bins=edges, color='tab:red', alpha=0.65, edgecolor='k',
             label=f'our draws (N={n}, 500 Mpc/256$^3$)')
    xs = np.linspace(lo - 0.1, hi + 0.1, 400)
    ax1.plot(xs, n * bw * _gauss(xs, implied_mean, implied_std), color='tab:orange', lw=2,
             label=f'Nikolic+23-implied\n({implied_mean:.2f}$\\pm${implied_std:.2f}, scaled to N={n})')
    ax1.axvline(mean, color='k', lw=1.5, label=f'our mean {mean:.2f}$\\pm${std:.2f}')
    ax1.axvspan(mean - std, mean + std, color='k', alpha=0.08)
    for seed, v in pc_vals.items():
        ax1.axvline(v, color='tab:blue', ls='--', lw=1.5,
                    label=f'seed {seed} with PHOTON_CONS ({v:.2f})')
    ax1.axvline(d3000_1p5, color='tab:green', ls=':', lw=1.5,
                label=f'Nikolic+23 Fig.1, 1.5 Gpc ({d3000_1p5:.2f}, illustrative)')
    ax1.set_xlabel(r'$D_{3000}$ [$\mu$K$^2$]')
    ax1.set_ylabel('realizations')
    ax1.set_title('per-seed $D_{3000}$ (500 Mpc)')
    ax1.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.24), borderaxespad=0)

    # ---- right: f = D_3000(1.5 Gpc) / D_3000_i
    fedges = np.histogram_bin_edges(f_ours, bins='auto')
    ax2.hist(f_ours, bins=fedges, color='tab:red', alpha=0.65, edgecolor='k',
             label=f'our $f_i$ (N={n})')
    ax2.axvspan(NIKOLIC_F_MEAN - NIKOLIC_F_STD, NIKOLIC_F_MEAN + NIKOLIC_F_STD,
                color='tab:orange', alpha=0.25,
                label=f'Nikolic+23 $f$ = {NIKOLIC_F_MEAN}$\\pm${NIKOLIC_F_STD} (N={NIKOLIC_N_SEEDS})')
    ax2.axvline(NIKOLIC_F_MEAN, color='tab:orange', lw=1.5)
    ax2.axvline(f_mean, color='k', lw=1.5, label=f'our mean $f$ = {f_mean:.2f}$\\pm${f_std:.2f}')
    ax2.set_xlabel(r'$f = D_{3000}(1.5\,\mathrm{Gpc})\,/\,D_{3000}(500\,\mathrm{Mpc})$')
    ax2.set_ylabel('realizations')
    ax2.set_title('sample-variance factor $f$')
    ax2.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.24), borderaxespad=0)

    fig.suptitle('kSZ $D_{3000}$: our Nikolic-replication ensemble vs Nikolic+23 (Fig. A1 style)')
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{plot_dir}/nikolic_d3000_histogram.png"
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/nikolic_d3000_histogram.npz",
             seeds=np.array(seeds), d3000=d3000, f_ours=f_ours,
             mean=mean, std=std, f_mean=f_mean, f_std=f_std,
             files_used=np.array([os.path.basename(p) for p in used]),
             pc_seeds=np.array(list(pc_vals.keys()), dtype=int),
             pc_d3000=np.array(list(pc_vals.values()), dtype=float),
             d3000_nikolic_1p5gpc=d3000_1p5,
             implied_mean=implied_mean, implied_std=implied_std,
             nikolic_f_mean=NIKOLIC_F_MEAN, nikolic_f_std=NIKOLIC_F_STD)
    print(f"Saved -> {out_dir}/nikolic_d3000_histogram.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nikolic_mesinger.yaml")
    parser.add_argument("--seeds", type=int, nargs='+', default=[101, 102, 103, 104, 105, 106])
    parser.add_argument("--baseline-suffix", default="_nophotoncons",
                        help="tried first when locating each seed's file (see module docstring)")
    parser.add_argument("--pc-seeds", type=int, nargs='*', default=[],
                        help="seeds whose plain _nikolic.npz is a PHOTON_CONS run; drawn as "
                             "separate markers, excluded from the histogram/mean/std")
    parser.add_argument("--bins", type=int, default=None, help="override automatic binning")
    args = parser.parse_args()
    main(args.config, args.seeds, args.baseline_suffix, args.pc_seeds, args.bins)
