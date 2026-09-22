#!/usr/bin/env python
"""
make_summary_figure.py -- Gorce-Fig.3-style figure: x_e(z) and P_ee(k) for the
history-matched (zeta, log10 Tvir, R_BUBBLE_MAX) solutions from solve_history*.py.

Recomputes everything from py21cmfast directly (does NOT reuse check_solutions.py's
numbers, which ran at 8 threads -- a different realisation from the 16-thread solver).
Requires morph_compare.py in the same directory (imports load_sets, power_spectrum).

Usage (p21c_v3 env):
  python make_summary_figure.py --json solutions_400_64.json --z-pee 8.0

Match --box-len/--hii-dim/--seed/--threads to the solver run (defaults already do).
"""
import argparse
import os
import time

import numpy as np

from morph_compare import load_sets, power_spectrum

# ── Plot style, matched to 29May2026_Bubble_parameter_play.py ───────────────
PDF_STYLE = dict(
    **{"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
       "mathtext.fontset": "cm", "font.size": 20, "axes.labelsize": 20,
       "axes.titlesize": 20, "xtick.labelsize": 17, "ytick.labelsize": 17,
       "legend.fontsize": 14, "xtick.direction": "in", "ytick.direction": "in",
       "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
       "ytick.minor.visible": True, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
       "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05}
)
PNG_STYLE = dict(PDF_STYLE, **{"font.size": 13, "legend.fontsize": 11})


def astro(p21c, zeta, tvir, rmax):
    kw = dict(HII_EFF_FACTOR=float(zeta), ION_Tvir_MIN=float(tvir))
    if rmax is not None:
        kw["R_BUBBLE_MAX"] = float(rmax)
    return p21c.AstroParams(**kw)


def sweep(p21c, ap_, zs, pf, ic, fl, pee_idx):
    """Full x_e(z) over the grid; also returns the xH_box at zs[pee_idx]."""
    xe = np.empty(len(zs))
    xH_at = None
    for i, z in enumerate(zs):
        ib = p21c.ionize_box(redshift=z, init_boxes=ic, perturbed_field=pf[z],
                             astro_params=ap_, flag_options=fl, write=False)
        xe[i] = 1.0 - float(np.mean(ib.xH_box))
        if i == pee_idx:
            xH_at = np.asarray(ib.xH_box, dtype=float)
    return xe, xH_at


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", default=["solutions_400_64.json"])
    ap.add_argument("--max-resid", type=float, default=0.02,
                    help="keep only solutions whose solver max|dx_e| is below this")
    ap.add_argument("--target-zeta", type=float, default=30.0,
                    help="fiducial zeta the target history was solved against")
    ap.add_argument("--z-pee", type=float, default=8.0,
                    help="redshift for the P_ee panel; snapped to the nearest grid point")
    ap.add_argument("--box-len", type=float, default=400.0)
    ap.add_argument("--hii-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
    ap.add_argument("--outdir", default="summary_out")
    a = ap.parse_args()

    import py21cmfast as p21c

    sets = load_sets(a.json, a.max_resid)
    if not sets:
        raise SystemExit(f"No solutions with max_abs_dx <= {a.max_resid} in {a.json}")
    print("Parameter sets (solver-reported residual):")
    for lab, s in sets.items():
        print(f"  {lab:20s} zeta={s['zeta']:.3f}  log10Tvir={s['tvir']:.4f}  "
              f"resid={s['resid']:.4f}")

    zs = np.arange(20.0, 4.99, -0.5)          # must match solve_history.py's grid
    pee_idx = int(np.argmin(np.abs(zs - a.z_pee)))
    print(f"P_ee panel at z={zs[pee_idx]:.1f} (nearest grid point to requested "
          f"{a.z_pee})")

    d0 = p21c.AstroParams()
    tv0 = float(d0.ION_Tvir_MIN)              # already log10(K); see solve_history.py

    os.makedirs(a.direc, exist_ok=True)
    up = p21c.UserParams(HII_DIM=a.hii_dim, BOX_LEN=a.box_len, N_THREADS=a.threads)
    ic = p21c.initial_conditions(user_params=up, random_seed=a.seed, direc=a.direc)
    fl = p21c.FlagOptions()
    pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=a.direc) for z in zs}
    delta_pee = np.asarray(pf[zs[pee_idx]].density, dtype=float)   # shared across sets

    t0 = time.time()
    target_xe, _ = sweep(p21c, astro(p21c, a.target_zeta, tv0, None), zs, pf, ic, fl, -1)
    print(f"target history: {time.time() - t0:.1f}s")

    results = {}
    for lab, s in sets.items():
        t0 = time.time()
        xe, xH_at = sweep(p21c, astro(p21c, s["zeta"], s["tvir"], s["rmax"]),
                          zs, pf, ic, fl, pee_idx)
        x = 1.0 - xH_at
        e = x * (1.0 + delta_pee)
        k, Pee, _ = power_spectrum(e / e.mean() - 1.0, a.box_len)
        results[lab] = dict(xe=xe, xHII_pee=float(x.mean()), k=k, Pee=Pee)
        print(f"  {lab:20s} done in {time.time() - t0:5.1f}s  "
              f"(xHII at z_pee = {x.mean():.3f})", flush=True)

    os.makedirs(a.outdir, exist_ok=True)
    np.savez(os.path.join(a.outdir, "summary_data.npz"),
             zs=zs, target_xe=target_xe, z_pee=zs[pee_idx],
             **{f"{lab}__xe": r["xe"] for lab, r in results.items()},
             **{f"{lab}__k": r["k"] for lab, r in results.items()},
             **{f"{lab}__Pee": r["Pee"] for lab, r in results.items()})

    make_plot(zs, target_xe, results, zs[pee_idx], a.outdir, a.box_len, a.hii_dim)


def make_plot(zs, target_xe, results, z_pee, outdir, box_len, hii_dim):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    labs = list(results)
    cmap = plt.get_cmap("viridis")
    col = {l: cmap(i / max(1, len(labs) - 1)) for i, l in enumerate(labs)}
    dx = box_len / hii_dim

    def draw(fig):
        ax0, ax1 = fig.subplots(1, 2)

        ax0.plot(zs, target_xe, color="gray", ls="--", lw=1.6, zorder=1,
                label="target history")
        for lab in labs:
            ax0.plot(zs, results[lab]["xe"], color=col[lab], label=lab, zorder=2)
        ax0.axvline(z_pee, color="k", lw=0.8, ls=":", alpha=0.6)
        ax0.set_xlabel(r"Redshift $z$")
        ax0.set_ylabel(r"$\bar{x}_e(z)$")
        ax0.set_xlim(20, 5)
        ax0.set_ylim(-0.02, 1.05)
        ax0.legend(fontsize=10, loc="upper left")
        ax0.set_title("Reionisation history")

        for lab in labs:
            r = results[lab]
            ax1.loglog(r["k"], r["Pee"], color=col[lab], label=lab)
        ax1.axvline(2 * np.pi / box_len, color="gray", lw=0.8, ls=":",
                    label=r"box fundamental $k_f$")
        ax1.axvline(np.pi / dx, color="gray", lw=0.8, ls="-.",
                    label="Nyquist")
        ax1.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$P_{ee}(k)\ [{\rm Mpc}^3]$")
        ax1.set_title(rf"$z={z_pee:g}$")
        ax1.legend(fontsize=9, loc="lower left")

    with mpl.rc_context(PDF_STYLE):
        fig = plt.figure(figsize=(15, 6), constrained_layout=True)
        draw(fig)
        fig.savefig(os.path.join(outdir, "history_and_pee.pdf"))
        plt.close(fig)
    with mpl.rc_context(PNG_STYLE):
        fig = plt.figure(figsize=(15, 6), constrained_layout=True)
        draw(fig)
        fig.suptitle("History-matched, morphology-varied 21cmFAST scenarios",
                    fontweight="bold")
        fig.savefig(os.path.join(outdir, "history_and_pee.png"))
        plt.close(fig)
    print(f"Saved {outdir}/history_and_pee.{{pdf,png}}")


if __name__ == "__main__":
    main()
