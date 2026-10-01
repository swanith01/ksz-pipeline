#!/usr/bin/env python
"""
Script 29: Nikolic/Mesinger/Gorce ensemble comparison.

Builds our own distribution of D_3000 across the 6-realization Nikolic-
replication ensemble (configs/nikolic_mesinger.yaml, seeds 101-106,
scripts/17_coherence_decomposition_fiducial.py --skip-direct), and
compares it against Nikolic et al. (2023)'s own reported calibration: a
single 500 Mpc/256^3 realization's D_3000 undershoots their 1.5 Gpc/1050^3
"truth" by a factor f=1.27+/-0.19 (N=20 seeds, their Appendix A / Fig. A1)
-- purely sample variance from missing large-scale power at 500 Mpc.

Rationale: a SINGLE realization can't be read against their curve with any
confidence given this much realization-to-realization scatter (see script
17's module docstring) -- this builds our own small ensemble instead of
leaning on their calibration factor alone, since f was derived under
THEIR (undisclosed-in-detail) rotation/stitching method, not ours.

Reads D_3000 directly from each ensemble member's saved .npz via
np.interp on the saved ell_dec/Dl_total/Dl_diag arrays -- not transcribed
from log text -- consistent with how every other D_3000 number in this
pipeline is computed.

FIRST RESULT (1 Oct 2026, seeds 101-106): our ensemble mean D_3000(P_total)
= 0.50 +/- 0.08 uK^2 (16.7% relative scatter -- close to Nikolic's own
~15%, a nice independent cross-check that 500 Mpc sample variance is a
robust feature of the box size, not a pipeline artifact). But the MEAN
sits well below their calibration-implied expectation (0.84 +/- 0.13
uK^2) -- ~4 sigma using our own ensemble's scatter, with ZERO overlap
between the two distributions. This is flagged as a genuine, sizeable gap
worth investigating, NOT yet attributed to either pipeline -- the most
likely causes (astro_params/flag_options not actually taking effect in
21cmFAST despite being passed; our reionization history not actually
landing near their stated z_r=6.1/z_end=4.9; our xH_mean-threshold patchy
window vs their hard z>=5 cutoff; cosmology differences) have NOT yet
been individually ruled out. See the accompanying verification check
(coeval.astro_params.F_STAR10 / coeval.flag_options.USE_MASS_DEPENDENT_ZETA
actually returned by a real run_coeval() call, and mean xH at z=6.1) before
treating this gap as a finding about either this pipeline or their paper.

Usage
-----
    python scripts/29_nikolic_ensemble_comparison.py --config configs/nikolic_mesinger.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Nikolic et al. (2023), Appendix A / Fig. A1: histogram of D_3000 from 20
# independent 500 Mpc/256^3 realizations (varying only the cosmic initial
# seed), calibrated against their single 1.5 Gpc/1050^3 box. f is defined
# such that D_3000(1.5 Gpc) ~= f * D_3000(500 Mpc, single realization).
NIKOLIC_F_MEAN  = 1.27
NIKOLIC_F_STD   = 0.19
NIKOLIC_N_SEEDS = 20


def main(config_path, seeds):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')

    d3000_total, d3000_diag = [], []
    for seed in seeds:
        path = f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run script 17 for seed {seed} first "
                                     f"(--config configs/nikolic_mesinger.yaml --skip-direct "
                                     f"--random-seed {seed} --wrap-cycle-seed {seed}).")
        d = np.load(path)
        d3000_total.append(float(np.interp(3000, d['ell_dec'], d['Dl_total'])))
        d3000_diag.append(float(np.interp(3000, d['ell_dec'], d['Dl_diag'])))
    d3000_total = np.array(d3000_total)
    d3000_diag  = np.array(d3000_diag)

    mean_total, std_total = d3000_total.mean(), d3000_total.std(ddof=1)
    mean_diag,  std_diag  = d3000_diag.mean(),  d3000_diag.std(ddof=1)
    print(f"Our ensemble ({len(seeds)} realizations, seeds {seeds}):")
    print(f"  D_3000 P_total: mean={mean_total:.4f}, std={std_total:.4f} "
          f"({100*std_total/mean_total:.1f}% relative)")
    print(f"  D_3000 P_diag : mean={mean_diag:.4f}, std={std_diag:.4f} "
          f"({100*std_diag/mean_diag:.1f}% relative)")

    nikolic_dir = 'data/external'
    mid = np.loadtxt(f'{nikolic_dir}/nikolic_ksz_power.csv', delimiter=',', skiprows=1)
    d3000_nikolic_1p5gpc = float(np.interp(3000, mid[:, 0], mid[:, 1]))
    print(f"\nNikolic+23 Fig.1 (1.5 Gpc/1050^3): D_3000 = {d3000_nikolic_1p5gpc:.4f} uK^2")

    implied_mean = d3000_nikolic_1p5gpc / NIKOLIC_F_MEAN
    implied_std  = implied_mean * (NIKOLIC_F_STD / NIKOLIC_F_MEAN)
    print(f"Nikolic+23's own calibration (f={NIKOLIC_F_MEAN}+/-{NIKOLIC_F_STD}, "
          f"N={NIKOLIC_N_SEEDS}) implies a single 500 Mpc realization should land near "
          f"D_3000 = {implied_mean:.4f} +/- {implied_std:.4f} uK^2\n")

    print(f"{'':30s} {'mean':>10s} {'std':>10s} {'rel.std':>10s}")
    print(f"{'our ensemble (P_total)':30s} {mean_total:>10.4f} {std_total:>10.4f} "
          f"{100*std_total/mean_total:>9.1f}%")
    print(f"{'Nikolic-implied (500 Mpc)':30s} {implied_mean:>10.4f} {implied_std:>10.4f} "
          f"{100*implied_std/implied_mean:>9.1f}%")

    gap = abs(mean_total - implied_mean)
    print(f"\n|our mean - Nikolic-implied mean| = {gap:.4f} uK^2 "
          f"(ours is {100*mean_total/implied_mean:.1f}% of the implied value)")
    print(f"  = {gap/std_total:.2f} sigma using OUR ensemble's own scatter")
    print(f"  = {gap/implied_std:.2f} sigma using Nikolic's quoted (f) scatter")
    print(">>> Flags WHETHER there's a gap worth investigating -- does NOT by itself say "
          "which pipeline (or which assumption: astro_params/flag_options actually taking "
          "effect, reionization-window convention, cosmology) is responsible. See module "
          "docstring for the recommended verification check before reading further into this.")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rng = np.random.default_rng(0)
    x_jitter = rng.uniform(-0.08, 0.08, size=len(seeds))
    ax.scatter(1 + x_jitter, d3000_total, color='tab:red', zorder=4, s=50,
               label=f'our ensemble draws ({len(seeds)} seeds, 500 Mpc/256$^3$)')
    for s, x, y in zip(seeds, 1 + x_jitter, d3000_total):
        ax.annotate(str(s), (x, y), fontsize=7, xytext=(4, 2), textcoords='offset points')
    ax.errorbar([1], [mean_total], yerr=[std_total], fmt='_', color='k',
                capsize=6, markersize=24, elinewidth=1.5, zorder=5,
                label='our ensemble mean $\\pm$1$\\sigma$')
    ax.axhspan(implied_mean - implied_std, implied_mean + implied_std,
               color='tab:orange', alpha=0.22, zorder=1,
               label=f'Nikolic+23-implied 500 Mpc\n(f={NIKOLIC_F_MEAN}$\\pm${NIKOLIC_F_STD}, N={NIKOLIC_N_SEEDS} seeds)')
    ax.axhline(implied_mean, color='tab:orange', lw=1.3, zorder=2)
    ax.axhline(d3000_nikolic_1p5gpc, color='tab:green', lw=1.5, ls='--', zorder=2,
               label='Nikolic+23 Fig.1 (1.5 Gpc/1050$^3$, illustrative)')
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1]); ax.set_xticklabels(['500 Mpc / 256$^3$\n(this work, --skip-direct)'])
    ax.set_ylabel(r'$D_{3000}$ [$\mu$K$^2$]')
    ax.set_title(f'kSZ $D_{{3000}}$: our {len(seeds)}-realization ensemble vs Nikolic+23')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = f"{plot_dir}/nikolic_ensemble_comparison.png"
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/nikolic_ensemble_comparison.npz",
             seeds=np.array(seeds), d3000_total=d3000_total, d3000_diag=d3000_diag,
             mean_total=mean_total, std_total=std_total,
             mean_diag=mean_diag, std_diag=std_diag,
             d3000_nikolic_1p5gpc=d3000_nikolic_1p5gpc,
             nikolic_f_mean=NIKOLIC_F_MEAN, nikolic_f_std=NIKOLIC_F_STD,
             implied_mean=implied_mean, implied_std=implied_std)
    print(f"Saved -> {out_dir}/nikolic_ensemble_comparison.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nikolic_mesinger.yaml")
    parser.add_argument("--seeds", type=int, nargs='+', default=[101, 102, 103, 104, 105, 106])
    args = parser.parse_args()
    main(args.config, args.seeds)
