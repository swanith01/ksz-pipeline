#!/usr/bin/env python
"""
Script 32: closing figure for the Nikolic+23 replication.

Pulls together, for seeds 101-106 of configs/nikolic_mesinger_nophotoncons.yaml
(500 Mpc / 256^3, Nikolic Fig.1 astro params, USE_MASS_DEPENDENT_ZETA, NO
PHOTON_CONS -- see README for why photon-cons is not part of this):

  FIXED    : wrap-cycle-decorrelated stitching (script 17 --wrap-cycle-seed s)
             coherence_decomposition_wrapcycle{s}_nikolic[_nophotoncons].npz
  UNFIXED  : plain periodic box reuse, same seeds / same initial conditions
             (script 17 without --wrap-cycle-seed, --random-seed s)
             coherence_decomposition_seed{s}_nikolic.npz
  HISTORY  : data/products/reion_history_seed101.npz (script 31)

Panels: (A) our xH(z) against Nikolic's stated midpoint / end; (B) D_3000
histograms, fixed vs unfixed, with Nikolic's calibration-implied 500 Mpc
distribution; (C) per-seed P_diag + P_off stacked bars, fixed vs unfixed.

File lookup for FIXED tries `..._nikolic{--baseline-suffix}.npz` first and
falls back to `..._nikolic.npz`, printing which file each seed used (seed 101's
plain file name holds the invalid single-redshift PHOTON_CONS run; the valid
baseline was renamed to `_nophotoncons`).

No py21cmfast dependency. Usage:
    python scripts/32_nikolic_closing_figure.py
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NIKOLIC_F_MEAN, NIKOLIC_F_STD, NIKOLIC_N_SEEDS = 1.27, 0.19, 20
NIKOLIC_ZR, NIKOLIC_ZEND, NIKOLIC_DZ, NIKOLIC_TAU = 6.1, 4.9, 0.76, 0.042


def d3000_parts(path):
    d = np.load(path)
    f = lambda k: float(np.interp(3000, d['ell_dec'], d[k]))
    return f('Dl_total'), f('Dl_diag'), f('Dl_off')


def load_fixed(out_dir, seed, suffix):
    for p in (f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic{suffix}.npz",
              f"{out_dir}/coherence_decomposition_wrapcycle{seed}_nikolic.npz"):
        if os.path.exists(p):
            return d3000_parts(p), p
    raise FileNotFoundError(f"no fixed-stitching file for seed {seed}")


def load_unfixed(out_dir, seed):
    p = f"{out_dir}/coherence_decomposition_seed{seed}_nikolic.npz"
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found -- run script 17 without --wrap-cycle-seed "
                                f"(--random-seed {seed}) on the nophotoncons config")
    return d3000_parts(p), p


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def main(config_path, seeds, suffix, history):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out_dir, plot_dir = cfg['data']['output_dir'].rstrip('/'), cfg['data']['plot_dir'].rstrip('/')

    fx, uf = [], []
    print(f"{'seed':>5s} | {'FIXED total':>11s} {'diag':>7s} {'off':>7s} | "
          f"{'UNFIXED total':>13s} {'diag':>7s} {'off':>7s} | ratio")
    for s in seeds:
        (ft, fd, fo), pf = load_fixed(out_dir, s, suffix)
        (ut, ud, uo), pu = load_unfixed(out_dir, s)
        fx.append((ft, fd, fo)); uf.append((ut, ud, uo))
        print(f"{s:>5d} | {ft:>11.4f} {fd:>7.4f} {fo:>7.4f} | {ut:>13.4f} {ud:>7.4f} {uo:>7.4f} | "
              f"{ut/ft:5.2f}   [{os.path.basename(pf)}]")
    fx, uf = np.array(fx), np.array(uf)
    n = len(seeds)
    mu = lambda a: (a.mean(0), a.std(0, ddof=1))
    (fm, fs), (um, us) = mu(fx), mu(uf)
    print(f"\n FIXED   mean total {fm[0]:.3f} +/- {fs[0]:.3f}   P_diag {fm[1]:.3f}   P_off {fm[2]:+.3f}")
    print(f" UNFIXED mean total {um[0]:.3f} +/- {us[0]:.3f}   P_diag {um[1]:.3f}   P_off {um[2]:+.3f}")
    print(f" unfixed/fixed total = {um[0]/fm[0]:.2f}; P_diag unfixed/fixed = {um[1]/fm[1]:.2f} "
          f"(~1 means the whole difference is the off-diagonal term)")

    mid = np.loadtxt('data/external/nikolic_ksz_power.csv', delimiter=',', skiprows=1)
    d1p5 = float(np.interp(3000, mid[:, 0], mid[:, 1]))
    imp_m = d1p5 / NIKOLIC_F_MEAN
    imp_s = imp_m * NIKOLIC_F_STD / NIKOLIC_F_MEAN
    frac = (imp_m - fm[0]) / (um[0] - fm[0])
    print(f" Nikolic-implied 500 Mpc: {imp_m:.3f} +/- {imp_s:.3f}  (their 1.5 Gpc Fig.1: {d1p5:.3f})")
    print(f" -> it sits {100*frac:.0f}% of the way from our FIXED mean to our UNFIXED mean "
          f"(0% = fully decorrelated, 100% = plain box reuse)")

    h = np.load(history)
    z, xH = h['z'], h['xH_mean']
    print(f"\n reionization history ({os.path.basename(history)}): midpoint {float(h['z_r']):.2f} "
          f"(Nikolic {NIKOLIC_ZR}), dz {float(h['dz']):.2f} ({NIKOLIC_DZ}), "
          f"z(xH=.05) {float(h['z_xH05']):.2f} (z_end {NIKOLIC_ZEND}), tau~{float(h['tau']):.4f} ({NIKOLIC_TAU})")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 5.8))
    # A: history
    a.plot(z, xH, 'o-', color='tab:red', ms=4, label='our $\\langle x_{HI}\\rangle(z)$ (no photon-cons)')
    a.axvline(NIKOLIC_ZR, color='tab:orange', lw=1.8, label=f'Nikolic+23 midpoint $z_r$={NIKOLIC_ZR}')
    a.axvline(NIKOLIC_ZEND, color='tab:orange', ls='--', lw=1.5, label=f'Nikolic+23 $z_{{end}}$={NIKOLIC_ZEND}')
    a.axhline(0.5, color='gray', lw=0.6)
    a.set_xlim(4, 10); a.set_ylim(-0.02, 1.02)
    a.set_xlabel('z'); a.set_ylabel(r'$\langle x_{HI}\rangle$'); a.set_title('reionization history')
    a.text(0.97, 0.30, f"ours: $z_r$={float(h['z_r']):.2f}, $\\Delta z$={float(h['dz']):.2f}, "
           f"$\\tau$~{float(h['tau']):.3f}\ntheirs: $z_r$={NIKOLIC_ZR}, $\\Delta z$={NIKOLIC_DZ}, $\\tau$={NIKOLIC_TAU}",
           transform=a.transAxes, ha='right', va='center', fontsize=8)
    a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), borderaxespad=0)
    # B: histograms
    lo = min(fx[:, 0].min(), imp_m - 3 * imp_s)
    hi = max(uf[:, 0].max(), imp_m + 3 * imp_s)
    edges = np.linspace(lo - 0.05, hi + 0.05, 14)
    bw = edges[1] - edges[0]
    b.hist(fx[:, 0], bins=edges, color='tab:blue', alpha=0.65, edgecolor='k',
           label=f'FIXED (wrap-cycle), mean {fm[0]:.2f}$\\pm${fs[0]:.2f}')
    b.hist(uf[:, 0], bins=edges, color='tab:red', alpha=0.55, edgecolor='k',
           label=f'UNFIXED (plain box reuse), mean {um[0]:.2f}$\\pm${us[0]:.2f}')
    xs = np.linspace(edges[0], edges[-1], 400)
    b.plot(xs, n * bw * gauss(xs, imp_m, imp_s), color='tab:orange', lw=2,
           label=f'Nikolic+23-implied 500 Mpc\n({imp_m:.2f}$\\pm${imp_s:.2f}, scaled to N={n})')
    b.axvline(d1p5, color='tab:green', ls=':', lw=1.5, label=f'Nikolic+23 Fig.1, 1.5 Gpc ({d1p5:.2f})')
    b.set_xlabel(r'$D_{3000}$ [$\mu$K$^2$]'); b.set_ylabel('realizations')
    b.set_title('same seeds, same boxes, stitching on/off')
    b.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), borderaxespad=0)
    # C: stacked bars per seed
    x = np.arange(n); w = 0.38
    for k, (dat, off_x, lab) in enumerate([(fx, -w / 2, 'FIXED'), (uf, w / 2, 'UNFIXED')]):
        c.bar(x + off_x, dat[:, 1], w, color='tab:gray', edgecolor='k',
              label='$P_{diag}$' if k == 0 else None)
        c.bar(x + off_x, dat[:, 2], w, bottom=dat[:, 1], color=('tab:blue' if k == 0 else 'tab:red'),
              edgecolor='k', alpha=0.8, label=f'$P_{{off}}$ ({lab})')
    c.axhspan(imp_m - imp_s, imp_m + imp_s, color='tab:orange', alpha=0.25,
              label='Nikolic+23-implied 500 Mpc')
    c.set_xticks(x); c.set_xticklabels([str(s) for s in seeds])
    c.set_xlabel('seed (left bar FIXED, right bar UNFIXED)'); c.set_ylabel(r'$D_{3000}$ [$\mu$K$^2$]')
    c.set_title('where the difference lives: $P_{off}$')
    c.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17), borderaxespad=0)

    fig.suptitle('Nikolic+23 replication: matched box/resolution/astro params, stitching on vs off')
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    pp = f"{plot_dir}/nikolic_closing_figure.png"
    fig.savefig(pp, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {pp}")
    np.savez(f"{out_dir}/nikolic_closing_summary.npz", seeds=np.array(seeds), fixed=fx, unfixed=uf,
             implied_mean=imp_m, implied_std=imp_s, d3000_1p5gpc=d1p5, frac_between=frac,
             hist_z=z, hist_xH=xH)
    print(f"Saved -> {out_dir}/nikolic_closing_summary.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nikolic_mesinger_nophotoncons.yaml")
    ap.add_argument("--seeds", type=int, nargs='+', default=[101, 102, 103, 104, 105, 106])
    ap.add_argument("--baseline-suffix", default="_nophotoncons")
    ap.add_argument("--history", default="data/products/reion_history_seed101.npz")
    a = ap.parse_args()
    main(a.config, a.seeds, a.baseline_suffix, a.history)
