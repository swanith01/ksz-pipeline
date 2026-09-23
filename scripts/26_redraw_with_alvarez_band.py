#!/usr/bin/env python
"""
scripts/26_redraw_with_alvarez_band.py

Redraws EITHER dl_off_corrected.png (the P_off comparison) OR
pdiag_formula_vs_native.png (the P_diag comparison), from their already-
saved .npz files, adding a shaded band at Alvarez et al. 2016's own stated
Doppler-component peak, ell ~ 20-30 (arXiv:1511.02846) -- NOT a re-derivation
of their Appendix A analytic threshold (which needs delta_z, itself
requiring how the source term evolves with z in OUR simulation; not
attempted here), but their own paper's stated result for where the
q_parallel/Doppler term's power peaks in a full non-Limber treatment --
precisely the regime their Appendix A shows a q_perp-only Limber
calculation (this pipeline's own 'formula'/compute_cell) cannot capture.

IMPORTANT CAVEAT, stated in the plot itself: ell~20-30 is BELOW this box's
own fundamental mode (ell_min_box, ~60-85 depending on the run) -- the band
will sit at or past the left edge of the plotted range. This is not hidden;
the x-axis is extended down to show it, with the box's own resolvable floor
marked separately so the two limits aren't confused with each other.

No new compute -- reads existing saved arrays only.

Usage
-----
    python scripts/26_redraw_with_alvarez_band.py --which poff \
        --npz data/plots/23_offdiag/dl_off_corrected.npz \
        --stitched data/products/coherence_decomposition_fiducial.npz

    python scripts/26_redraw_with_alvarez_band.py --which pdiag \
        --npz data/plots/24_pdiag_check/pdiag_formula_vs_native.npz
"""
import argparse
import logging

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger("redraw_alvarez")

ALVAREZ_ELL_LO, ALVAREZ_ELL_HI = 20.0, 30.0
ALVAREZ_LABEL = "Alvarez+16 Doppler peak (ell~20-30, arXiv:1511.02846)"


def add_alvarez_band(ax, ell_min_box=None, x_low_extend=15.0):
    """Shades ALVAREZ_ELL_LO-HI and, if given, marks this box's own
    ell_min_box separately, on a log-x axis -- extends xlim down to show
    the band even when it sits below the plotted data's own range."""
    ax.axvspan(ALVAREZ_ELL_LO, ALVAREZ_ELL_HI, color="tab:orange", alpha=0.18,
               label=ALVAREZ_LABEL, zorder=0)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(min(xmin, x_low_extend), xmax)
    if ell_min_box is not None:
        ax.axvline(ell_min_box, color="grey", ls="--", lw=1,
                   label=f"this box's own ell_min ({ell_min_box:.0f}) -- "
                         f"NOT the same limit as the band above")


def redraw_poff(npz_path, stitched_path, out_path):
    z = np.load(npz_path)
    ell, dl = z["ell"], z["dl"]
    ell_min_box = float(z["ell_min_box"])
    caveat = str(z["caveat"]) if "caveat" in z.files and z["caveat"] else None

    dl_stitched = None
    if stitched_path:
        s = np.load(stitched_path)
        ell_key = "ell_dec" if "ell_dec" in s.files else "ell_off"
        dl_stitched = np.interp(ell, s[ell_key], s["Dl_off"])

    fig, ax = plt.subplots(figsize=(8, 5))
    band = ell > ell_min_box
    ax.plot(ell[band], dl[band], "g.-", label="direct, corrected (Limber failure)")
    if dl_stitched is not None:
        ax.plot(ell, dl_stitched, "s-", c="tab:blue", label=r"stitched $P_{\rm off}$")
        frac = dl_stitched - dl
        ax.plot(ell[band], frac[band], "r--", label="difference = periodicity")
    ax.axhline(0, ls=":", c="k")
    ax.axvline(ell_min_box, ls="--", c="grey")
    ax.set_xscale("log")
    add_alvarez_band(ax, ell_min_box=ell_min_box)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax.legend(fontsize=7)
    if caveat:
        fig.text(0.5, 0.01, caveat, ha="center", va="bottom", fontsize=7,
                 style="italic", wrap=True,
                 bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))
        fig.subplots_adjust(bottom=0.2)
    fig.tight_layout(rect=[0, 0.08, 1, 1] if caveat else None)
    fig.savefig(out_path, dpi=140)
    log.info("wrote %s", out_path)


def redraw_pdiag(npz_path, out_path):
    z = np.load(npz_path)
    ell_formula, Dl_formula = z["ell_formula"], z["Dl_formula"]
    ell_native, Dl_native = z["ell_native"], z["Dl_native"]
    has_ours = "ell_ours" in z.files and len(z["ell_ours"]) > 0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ell_formula, Dl_formula, "o-", ms=3,
            label="P_diag from formula (compute_cell / Limber)")
    ax.plot(ell_native, Dl_native, "s-", ms=3, label="native P_diag (stitched map)")
    if has_ours:
        ax.plot(z["ell_ours"], z["Dl_ours"], "^--", ms=4, c="red",
                label="our new estimator's diagonal")
    ax.set_xscale("log")
    add_alvarez_band(ax)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    log.info("wrote %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["poff", "pdiag"], required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--stitched", default=None,
                   help="(poff only) coherence_decomposition npz for the blue curve")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib not available")

    if args.which == "poff":
        out = args.out or "dl_off_corrected_with_alvarez.png"
        redraw_poff(args.npz, args.stitched, out)
    else:
        out = args.out or "pdiag_formula_vs_native_with_alvarez.png"
        redraw_pdiag(args.npz, out)


if __name__ == "__main__":
    main()
