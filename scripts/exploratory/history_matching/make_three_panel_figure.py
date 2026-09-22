#!/usr/bin/env python
"""
make_three_panel_figure.py -- combined figure: reionisation history | P_ee(k) |
D_ell^kSZ, for the history-matched, morphology-varied 21cmFAST parameter sets.

Reads:
  summary_out/summary_data.npz  (written by make_summary_figure.py)
  dl_out/dl_<label>.npz         (written by compute_dl_matched_sets.py)
and the solutions JSON(s), to recover each set's (zeta, log10 Tvir,
R_BUBBLE_MAX) for the legend -- same load_sets() as the other two scripts,
so labels and colors match exactly across all figures.

Both input scripts must have been run first; this one does no 21cmFAST
compute of its own.

Usage:
  python make_three_panel_figure.py --json solutions_400_64.json solutions_400_64_v2.json
"""
import argparse
import os

import numpy as np

from morph_compare import load_sets


def dl_filename(label):
    """Must match compute_dl_matched_sets.py's own sanitizing exactly."""
    safe = label.replace(" ", "_").replace("(", "").replace(")", "")
    return f"dl_{safe}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", default=["solutions_400_64.json"])
    ap.add_argument("--max-resid", type=float, default=0.02)
    ap.add_argument("--summary-npz", default="summary_out/summary_data.npz")
    ap.add_argument("--dl-dir", default="dl_out")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    sets = load_sets(a.json, a.max_resid)
    if not sets:
        raise SystemExit(f"No solutions with max_abs_dx <= {a.max_resid} in {a.json}")

    summ = np.load(a.summary_npz)
    zs = summ["zs"]
    target_xe = summ["target_xe"]
    z_pee = float(summ["z_pee"])

    data = {}
    for lab, s in sets.items():
        if f"{lab}__xe" not in summ.files:
            print(f"  SKIP {lab}: not in {a.summary_npz} (run make_summary_figure.py "
                  f"with this set included)")
            continue
        dlf = os.path.join(a.dl_dir, dl_filename(lab))
        if not os.path.exists(dlf):
            print(f"  SKIP {lab}: {dlf} not found (run compute_dl_matched_sets.py "
                  f"with this set included)")
            continue
        dl = np.load(dlf)
        data[lab] = dict(
            zeta=s["zeta"], tvir=s["tvir"], rmax=s["rmax"],
            xe=summ[f"{lab}__xe"], k=summ[f"{lab}__k"], Pee=summ[f"{lab}__Pee"],
            ell=dl["ell"], Dl=dl["Dl"], sigma=dl["sigma_Dl"],
        )
    if not data:
        raise SystemExit("No parameter set had both summary and D_ell data present.")
    print(f"Plotting {len(data)} set(s): {list(data)}")

    make_plot(zs, target_xe, z_pee, data, a.outdir)


def make_plot(zs, target_xe, z_pee, data, outdir):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    labs = list(data)
    rmax_vals = [15.0 if data[l]["rmax"] is None else data[l]["rmax"] for l in labs]
    norm = mpl.colors.Normalize(vmin=min(rmax_vals), vmax=max(rmax_vals))
    cmap = mpl.colormaps["viridis"]

    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "cm", "font.size": 15, "axes.labelsize": 16,
             "axes.titlesize": 16, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True, "legend.fontsize": 9,
             "savefig.dpi": 300, "savefig.bbox": "tight"}

    with mpl.rc_context(style):
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(19, 5.5),
                                            constrained_layout=True)

        ax0.plot(zs, target_xe, color="gray", ls="--", lw=1.4, zorder=1,
                label="target history")
        for lab, rmax in zip(labs, rmax_vals):
            r = data[lab]
            color = cmap(norm(rmax))
            leg = (rf"$R_{{\rm max}}$={rmax:g} Mpc, $\zeta$={r['zeta']:.1f}, "
                  rf"$\log_{{10}}T_{{\rm vir}}$={r['tvir']:.2f}")
            ax0.plot(zs, r["xe"], color=color, lw=2.0, zorder=2, label=leg)
            ax1.loglog(r["k"], r["Pee"], color=color, lw=2.0)
            ax2.errorbar(r["ell"], r["Dl"], yerr=r["sigma"], color=color,
                        lw=1.8, capsize=2, elinewidth=0.8)

        ax0.axvline(z_pee, color="k", lw=0.8, ls=":", alpha=0.6)
        ax0.set_xlim(20, 5)
        ax0.set_ylim(-0.02, 1.05)
        ax0.set_xlabel(r"Redshift $z$")
        ax0.set_ylabel(r"$\bar{x}_e(z)$")
        ax0.set_title("Reionisation history")
        ax0.legend(fontsize=8.5, loc="upper left")

        ax1.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
        ax1.set_ylabel(r"$P_{ee}(k)\ [{\rm Mpc}^3]$")
        ax1.set_title(rf"$P_{{ee}}$ at $z={z_pee:g}$")

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"Multipole $\ell$")
        ax2.set_ylabel(r"$D_\ell^{\rm kSZ}\ [\mu{\rm K}^2]$")
        ax2.axvline(3000, color="gray", ls=":", lw=1)
        ax2.set_title("kSZ power (direct/coeval + Limber)")

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, pad=0.02)
        cbar.set_label(r"$R_{\rm BUBBLE\_MAX}$ [Mpc]")

        for ext in ("pdf", "png"):
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(os.path.join(outdir, f"three_panel_summary.{ext}"))
        plt.close(fig)
    print(f"Saved {outdir}/three_panel_summary.{{pdf,png}}")


if __name__ == "__main__":
    main()
