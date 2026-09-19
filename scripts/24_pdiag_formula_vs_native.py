#!/usr/bin/env python
"""
scripts/24_pdiag_formula_vs_native.py

Compares two ALREADY-VALIDATED, INDEPENDENT pipelines against each other:

  "formula"  -- limber.compute_cell's D_ell, the Limber-projection prediction
                built from the cached per-snapshot P_qperp(k) in
                data/cache/qperp_power.pkl (momentum.qperp_power's output).
  "native"   -- Dl_diag in data/products/coherence_decomposition_fiducial.npz,
                the diagonal (i==j) term measured DIRECTLY from the real
                stitched 2D map by decompose_p_total_diag_off -- an entirely
                separate code path (real-space map construction + auto-power),
                never touching qperp_power or momentum.py at all.

Per docs/validation_table.md / the AMBER handoff: "The P_diag/direct
comparison is robust across radial-grouping choices" -- i.e. this specific
comparison has a track record of agreeing. Reproducing it here, independent
of anything written for the 23_offdiag_projection.py investigation, tells us
where the 17 Sep gate failure (~1.56x, a_power=-2) sits:

  - If formula and native agree here -> compute_cell, tau, ne0, and the
    window logic are all fine. The 1.56x is specific to
    offdiag_projection.py's own dl_off implementation, or to running the
    gate check at the hii-dim=128 test override rather than the true
    fiducial resolution. Look there next.
  - If they DISAGREE by a similar factor -> the problem is upstream of
    everything written for that investigation: something in compute_cell's
    inputs (the cached qperp_power.pkl) or in how the two pipelines'
    conventions (tau, ne0, window) relate to each other at the ACTUAL
    fiducial resolution, not the 128 override.

Deliberately does NOT import offdiag_projection, run_coeval_fields, or
anything that touches py21cmfast -- this only reads two files already on
disk, so it is cheap enough to run without a PBS job.

Usage
-----
    python scripts/24_pdiag_formula_vs_native.py \
        --config configs/fiducial.yaml \
        --stitched data/products/coherence_decomposition_fiducial.npz
"""
import argparse
import logging
import os
import pickle

import numpy as np
import yaml

from ksz_pipeline.coeval.limber import compute_cell

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger("pdiag_check")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fiducial.yaml")
    p.add_argument("--stitched", default="data/products/coherence_decomposition_fiducial.npz")
    p.add_argument("--stitched-ell-key", default="ell_dec")
    p.add_argument("--stitched-diag-key", default="Dl_diag")
    p.add_argument("--outdir", default=None,
                   help="default: <plot_dir>/24_pdiag_check")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    outdir = args.outdir or os.path.join(cfg["data"]["plot_dir"].rstrip("/"),
                                         "24_pdiag_check")
    os.makedirs(outdir, exist_ok=True)

    pkl_path = os.path.join(cfg["data"]["cache_dir"], "qperp_power.pkl")
    with open(pkl_path, "rb") as f:
        results_all = pickle.load(f)
    log.info("loaded %d cached snapshots from %s", len(results_all), pkl_path)

    # No pre-filtering here -- compute_cell applies its own patchy window
    # internally, exactly as it would in normal use. This deliberately does
    # NOT go through build_reference_results/self_consistent_window from the
    # 23_offdiag_projection.py investigation -- the whole point is to test
    # compute_cell in isolation, on its native cached input.
    ell_formula, Dl_formula, sigma_D, *_ = compute_cell(results_all)
    log.info("compute_cell valid ell range: [%.0f, %.0f]",
             ell_formula.min(), ell_formula.max())

    if not os.path.exists(args.stitched):
        raise FileNotFoundError(
            f"{args.stitched} not found -- this script only reads existing "
            f"products, it does not generate them. Check the path.")
    npz = np.load(args.stitched, allow_pickle=True)
    if args.stitched_ell_key not in npz.files or args.stitched_diag_key not in npz.files:
        raise KeyError(
            f"expected keys {args.stitched_ell_key!r}/{args.stitched_diag_key!r} "
            f"not both in {args.stitched}. Keys present: {sorted(npz.files)}")
    ell_native = np.asarray(npz[args.stitched_ell_key], dtype=float)
    Dl_native = np.asarray(npz[args.stitched_diag_key], dtype=float)
    log.info("native P_diag from %s [%s, %s]: %d points, ell range [%.0f, %.0f]",
             args.stitched, args.stitched_ell_key, args.stitched_diag_key,
             len(ell_native), ell_native.min(), ell_native.max())

    # compare only where BOTH are valid -- no silent clamping either direction
    overlap = (ell_native >= ell_formula.min()) & (ell_native <= ell_formula.max())
    n_dropped = (~overlap).sum()
    if n_dropped:
        log.warning("%d/%d native ell points fall outside compute_cell's valid "
                   "range and are excluded from the ratio (not extrapolated).",
                   n_dropped, len(ell_native))
    ell_cmp = ell_native[overlap]
    Dl_native_cmp = Dl_native[overlap]
    Dl_formula_cmp = np.interp(ell_cmp, ell_formula, Dl_formula)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = Dl_formula_cmp / Dl_native_cmp
    finite = np.isfinite(ratio) & (Dl_native_cmp != 0)
    median_ratio = float(np.nanmedian(ratio[finite])) if finite.any() else np.nan
    log.info("median ratio (formula / native) over %d overlapping points: %.4f",
             finite.sum(), median_ratio)
    if 0.95 <= median_ratio <= 1.05:
        log.info("VERDICT: formula and native P_diag AGREE within 5%%. "
                 "compute_cell/tau/ne0/window are fine -- the 17 Sep gate "
                 "failure is specific to offdiag_projection.py or the "
                 "hii-dim=128 test override, not to this reference plumbing.")
    else:
        log.warning("VERDICT: formula and native P_diag DISAGREE by %.2fx. "
                   "This predates anything written for the off-diagonal "
                   "estimator -- look at compute_cell's inputs/conventions "
                   "at the TRUE fiducial resolution, not the 128 override.",
                   median_ratio)

    np.savez(f"{outdir}/pdiag_formula_vs_native.npz",
             ell_formula=ell_formula, Dl_formula=Dl_formula,
             ell_native=ell_native, Dl_native=Dl_native,
             ell_cmp=ell_cmp, ratio=ratio, median_ratio=median_ratio)

    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(ell_formula, Dl_formula, "o-", ms=3,
                 label="P_diag from formula (compute_cell / Limber)")
        ax1.plot(ell_native, Dl_native, "s-", ms=3,
                 label="native P_diag (stitched map, Dl_diag)")
        ax1.set_ylabel(r"$D_\ell\ [\mu K^2]$")
        ax1.set_xscale("log")
        ax1.legend()
        ax1.set_title("Independent cross-check: Limber formula vs the "
                      "stitched map's own diagonal power")

        ax2.plot(ell_cmp, ratio, "k.-", ms=3)
        ax2.axhline(1.0, ls="--", c="grey")
        ax2.fill_between(ell_cmp, 0.95, 1.05, color="grey", alpha=0.2)
        ax2.set_ylabel("formula / native")
        ax2.set_xlabel(r"$\ell$")
        ax2.set_xscale("log")

        fig.text(0.5, 0.005,
                 f"median ratio = {median_ratio:.3f} over {finite.sum()} "
                 f"overlapping points (shaded band = within 5%)",
                 ha="center", fontsize=9, style="italic")
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(f"{outdir}/pdiag_formula_vs_native.png", dpi=140)
        log.info("wrote %s/pdiag_formula_vs_native.png", outdir)


if __name__ == "__main__":
    main()
