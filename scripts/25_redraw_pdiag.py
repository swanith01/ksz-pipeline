#!/usr/bin/env python
"""
scripts/25_redraw_pdiag.py

One-off fix: the 19 Sep run of scripts/24_pdiag_formula_vs_native.py wrote
"FAILS gate" / "UNRESOLVED, not a validated result" onto
pdiag_formula_vs_native.png unconditionally -- hardcoded before we knew
whether hii_dim=512 would pass. It did pass (median 1.027), but the image's
own text never got updated, so the figure contradicts its own caption
(which correctly prints the passing number right next to text saying it
failed). script 24 itself is fixed (see its git history) so this won't
recur on a future run; this script fixes the EXISTING image cheaply, from
already-saved data, without repeating the ~2.5 hour computation.

Usage
-----
    python scripts/25_redraw_pdiag.py \
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

log = logging.getLogger("redraw_pdiag")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/plots/24_pdiag_check/pdiag_formula_vs_native.npz")
    p.add_argument("--out", default=None, help="default: same dir as --npz")
    p.add_argument("--hii-dim-hint", type=int, default=None,
                   help="the npz from before this field was saved has no "
                        "record of what hii_dim 'ours' was computed at -- "
                        "supply it here if known (e.g. 512)")
    args = p.parse_args()

    z = np.load(args.npz)
    ell_formula, Dl_formula = z["ell_formula"], z["Dl_formula"]
    ell_native, Dl_native = z["ell_native"], z["Dl_native"]
    ell_cmp, ratio, median_ratio = z["ell_cmp"], z["ratio"], float(z["median_ratio"])
    finite_n = int(np.isfinite(ratio).sum())

    has_ours = "ell_ours" in z.files and len(z["ell_ours"]) > 0
    ell_ours = Dl_ours = ratio_ours = None
    med_ours = ours_hii_dim = ours_passed = None
    if has_ours:
        ell_ours, Dl_ours = z["ell_ours"], z["Dl_ours"]
        if "ours_hii_dim" in z.files and int(z["ours_hii_dim"]) > 0:
            ours_hii_dim = int(z["ours_hii_dim"])
        elif args.hii_dim_hint is not None:
            ours_hii_dim = args.hii_dim_hint
            log.info("ours_hii_dim not in npz -- using --hii-dim-hint=%d",
                     ours_hii_dim)
        else:
            ours_hii_dim = "unrecorded"
            log.warning("ours_hii_dim not in npz and no --hii-dim-hint given "
                       "-- label will say 'unrecorded'. Pass --hii-dim-hint "
                       "512 if you know this was the 512 run.")
        # median_ratio_ours wasn't saved by the 19 Sep run that produced this
        # npz (that's the bug being fixed) -- recompute it here, exactly, from
        # the same two arrays the original plot already had.
        if "median_ratio_ours" in z.files and np.isfinite(z["median_ratio_ours"]):
            med_ours = float(z["median_ratio_ours"])
        else:
            Dl_formula_at_ours = np.interp(ell_ours, ell_formula, Dl_formula)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_ours = Dl_ours / Dl_formula_at_ours
            med_ours = float(np.nanmedian(ratio_ours))
            log.info("median_ratio_ours not in npz -- recomputed from saved "
                     "ell_ours/Dl_ours/ell_formula/Dl_formula: %.4f "
                     "(should match the original log's printed value)",
                     med_ours)
        if ratio_ours is None:
            Dl_formula_at_ours = np.interp(ell_ours, ell_formula, Dl_formula)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_ours = Dl_ours / Dl_formula_at_ours
        ours_passed = 0.95 <= med_ours <= 1.05
        log.info("ours/formula median = %.4f -> %s", med_ours,
                 "PASS" if ours_passed else "FAIL")

    out = args.out or args.npz.rsplit("/", 1)[0]

    if plt is None:
        raise RuntimeError("matplotlib not available")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(ell_formula, Dl_formula, "o-", ms=3,
             label="P_diag from formula (compute_cell / Limber)")
    ax1.plot(ell_native, Dl_native, "s-", ms=3,
             label="native P_diag (stitched map, Dl_diag)")
    if has_ours:
        status = "PASSES gate" if ours_passed else "FAILS gate"
        ax1.plot(ell_ours, Dl_ours, "^--", ms=4, c="red",
                 label=f"OUR new estimator's diagonal (hii_dim={ours_hii_dim}, "
                       f"{status})")
    ax1.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax1.set_xscale("log")
    ax1.legend(fontsize=8)
    ax1.set_title("Limber formula vs stitched map's native diagonal "
                  "vs our new estimator's own diagonal")

    ax2.plot(ell_cmp, ratio, "k.-", ms=3, label="formula/native")
    if has_ours:
        ax2.plot(ell_ours, ratio_ours, "r^--", ms=4, label="ours/formula")
        ax2.legend(fontsize=8)
    ax2.axhline(1.0, ls="--", c="grey")
    ax2.fill_between(ell_cmp, 0.95, 1.05, color="grey", alpha=0.2)
    ax2.set_ylabel("ratio")
    ax2.set_xlabel(r"$\ell$")
    ax2.set_xscale("log")

    caption = (f"formula/native median = {median_ratio:.3f} over "
              f"{finite_n} points (grey band = within 5%)")
    if has_ours:
        verdict_text = ("VALIDATED -- within 5% of the Limber formula"
                        if ours_passed else
                        "NOT VALIDATED -- outside 5% of the Limber formula, "
                        "treat with caution")
        caption += (f"  |  ours/formula median = {med_ours:.3f} at "
                   f"hii_dim={ours_hii_dim} ({verdict_text}). Low-ell points "
                   f"near the box fundamental mode carry large sample "
                   f"variance for any estimator.")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8, style="italic",
             wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    outpath = f"{out}/pdiag_formula_vs_native.png"
    fig.savefig(outpath, dpi=140)
    log.info("wrote %s", outpath)


if __name__ == "__main__":
    main()
