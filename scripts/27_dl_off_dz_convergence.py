#!/usr/bin/env python
"""
Script 27: does D_ell^off (the corrected off-diagonal estimator, Girish's
eq 14) depend on how many snapshots feed it?

This is the off-diagonal analogue of script 20's P_diag-vs-n_groups
convergence check, but for a DIFFERENT and more specific reason than
generic due diligence: build_layers() (offdiag_projection.py) sets each
layer's dchi from np.gradient(chi) -- the LOCAL spacing to its neighbours
in whatever snapshot list is passed in. Coarsen the sampling and every
surviving layer's dchi widens automatically, which widens
window_overlap()'s trapezoid -- the thing that sets how xi(Delta chi) gets
weighted when building C_l^ij. If the green curve drifts with snapshot
count, that's a real sign the radial sampling is too coarse to trust the
resulting periodicity split; if it doesn't, that's a genuine, meaningful
piece of evidence closing this line of doubt.

Reuses build_dz_subsets from convergence/dz_sweep.py -- the SAME
subsampling convention already used for the coeval-direct and stitched
dz-convergence tests (every Nth point of the patchy-filtered window,
descending in z) -- applied here to ZS_win, the window
scripts/23_offdiag_projection.py's --stage run actually consumes, not the
raw config z_snapshots list.

CACHE NOTE: if scripts/23_offdiag_projection.py --stage run has already
completed at this resolution (populating py21cmfast's own cache for every
snapshot in the window), every dz_multiple below is a SUBSET of already-
cached boxes -- no new simulation, only the pair-sum math reruns. Same
efficiency principle as dz_sweep.py's own sweeps, via py21cmfast's cache
layer directly rather than coeval_sweep.run_one_config's results dict.

Loads scripts/23_offdiag_projection.py by file path (module names can't
start with a digit for a plain `import`, and scripts/ is not a package)
and reuses its functions directly, matching the pattern already used and
tested in scripts/24_pdiag_formula_vs_native.py's load_ours_diagonal().

Usage
-----
    python scripts/27_dl_off_dz_convergence.py --config configs/fiducial.yaml
    python scripts/27_dl_off_dz_convergence.py --config configs/fiducial.yaml \
        --dz-multiples 1 2 3 4 --hii-dim 64   # fast interactive shakedown
"""
import argparse
import logging
import os
import pickle

import numpy as np
import yaml

from ksz_pipeline.convergence.dz_sweep import build_dz_subsets
from ksz_pipeline.utils.constants import ne0_cgs

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger("dl_off_dz")


def _load_s23():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "s23", os.path.join(here, "23_offdiag_projection.py"))
    s23 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s23)
    return s23


def run_one_multiple(s23, cfg, zs_m, results_all, ell, L, weights):
    """One dz_multiple's worth of layers -> dl_off, reusing script 23's
    own build_layers/limber_tau_history/make_loader/dl_off unchanged."""
    tau_arr = s23.limber_tau_history(zs_m, results_all)
    tau_of_z = dict(zip(zs_m, tau_arr)).__getitem__
    layers = s23.build_layers(zs_m, tau_of_z)
    z_by_label = {l.label: z for l, z in zip(layers, zs_m)}
    load_q = s23.make_loader(cfg, z_by_label)
    dl, rep = s23.dl_off(ell, layers, load_q, L, weights, progress=False)
    return dl, rep


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fiducial.yaml")
    p.add_argument("--dz-multiples", type=int, nargs="+", default=None,
                   help="default: cfg['convergence']['dz_multiples'], "
                        "e.g. [1, 2, 4] -- the same convention already "
                        "used for the coeval/stitched dz sweeps")
    p.add_argument("--hii-dim", type=int, default=None,
                   help="override HII_DIM_coeval for fast interactive testing")
    p.add_argument("--box-len", type=float, default=None,
                   help="override BOX_LEN for fast interactive testing")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    s23 = _load_s23()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.hii_dim is not None:
        cfg["21cmfast"]["HII_DIM_coeval"] = args.hii_dim
    if args.box_len is not None:
        cfg["21cmfast"]["BOX_LEN"] = args.box_len
    L = cfg["21cmfast"]["BOX_LEN"]

    dz_multiples = args.dz_multiples or cfg.get("convergence", {}).get(
        "dz_multiples", [1, 2, 4])
    log.info("dz_multiples: %s (BOX_LEN=%.0f, HII_DIM=%d)",
             dz_multiples, L, cfg["21cmfast"]["HII_DIM_coeval"])

    ps = cfg["power_spectrum"]
    ell = np.geomspace(ps["ell_min"], ps["ell_max"], ps["n_ell_bins"])

    ZS = sorted(cfg["coeval_ksz"]["z_snapshots"])
    with open(os.path.join(cfg["data"]["cache_dir"], "qperp_power.pkl"), "rb") as f:
        results_all = pickle.load(f)
    ZS_win = s23.patchy_window(ZS, results_all)
    log.info("full patchy window: %d snapshots", len(ZS_win))

    subsets = build_dz_subsets(ZS_win, dz_multiples)
    weights = s23.LayerWeights(nbar_e0_cgs=ne0_cgs(), a_power=-2.0)

    curves, summary = {}, []
    for m in sorted(dz_multiples):
        zs_m = subsets[m]
        log.info("dz_x%d: %d snapshots", m, len(zs_m))
        dl, rep = run_one_multiple(s23, cfg, zs_m, results_all, ell, L, weights)
        curves[m] = dl
        summary.append(dict(m=m, n_snap=len(zs_m), n_pairs=rep["n_pairs"],
                            n_pairs_skipped=rep["n_pairs_skipped"],
                            ell_min_box=rep["ell_min_box"]))
        log.info("  %d pairs kept (%d skipped), ell_min_box=%.0f",
                 rep["n_pairs"], rep["n_pairs_skipped"], rep["ell_min_box"])

    # common valid range across all multiples, so the overlay compares
    # apples to apples rather than each curve's own (different) band
    ell_min_common = max(s["ell_min_box"] for s in summary)
    band = ell > ell_min_common

    # stability check at a few representative ell (D_off is not single-
    # peaked like P_diag, so one D_3000-style number would hide structure
    # a full-curve overlay shows directly -- report both)
    finest = min(dz_multiples)
    ref = curves[finest]
    log.info("\n%-8s %8s %8s %10s %10s %10s", "dz_x", "n_snap", "n_pairs",
             "D(ell=300)", "D(ell=1300)", "D(ell=5000)")
    for m in sorted(dz_multiples):
        s = next(x for x in summary if x["m"] == m)
        d1 = np.interp(300, ell, curves[m])
        d2 = np.interp(1300, ell, curves[m])
        d3 = np.interp(5000, ell, curves[m])
        log.info("%-8s %8d %8d %10.3g %10.3g %10.3g",
                 f"dz_x{m}", s["n_snap"], s["n_pairs"], d1, d2, d3)

    with np.errstate(invalid="ignore", divide="ignore"):
        max_frac_dev = {}
        for m in sorted(dz_multiples):
            if m == finest:
                continue
            ok = band & np.isfinite(ref) & np.isfinite(curves[m]) & (np.abs(ref) > 1e-30)
            frac = np.abs(curves[m][ok] - ref[ok]) / np.abs(ref[ok])
            max_frac_dev[m] = float(np.nanmax(frac)) if ok.any() else np.nan
    log.info("\nmax fractional deviation from dz_x%d, over the common valid "
             "ell range:", finest)
    log.info("CAVEAT: unlike P_diag, D_off crosses zero (around ell~800-1000 "
             "on real data) -- fractional deviation near a zero-crossing can "
             "look enormous without meaning much. Read the overlay PLOT as "
             "the primary evidence; use these numbers as a rough guide only.")
    for m, f in max_frac_dev.items():
        log.info("  dz_x%d: %.1f%%", m, f * 100)

    outdir = args.outdir or os.path.join(cfg["data"]["plot_dir"].rstrip("/"),
                                         "27_dl_off_dz")
    os.makedirs(outdir, exist_ok=True)

    save_dict = dict(ell=ell, ell_min_common=ell_min_common,
                     dz_multiples=sorted(dz_multiples))
    for m in dz_multiples:
        save_dict[f"dl_dz_x{m}"] = curves[m]
        save_dict[f"n_snap_dz_x{m}"] = next(s["n_snap"] for s in summary if s["m"] == m)
        save_dict[f"n_pairs_dz_x{m}"] = next(s["n_pairs"] for s in summary if s["m"] == m)
    np.savez(f"{outdir}/dl_off_dz_convergence.npz", **save_dict)
    log.info("Saved -> %s/dl_off_dz_convergence.npz", outdir)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        for m in sorted(dz_multiples):
            n_snap = next(s["n_snap"] for s in summary if s["m"] == m)
            ax.plot(ell[band], curves[m][band], "o-", ms=3,
                    label=f"dz_x{m} ({n_snap} snapshots)")
        ax.axhline(0, ls=":", c="k")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell^{\rm off}\ [\mu K^2]$")
        ax.set_title("D_ell^off vs snapshot-sampling density (dz convergence)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = f"{outdir}/dl_off_dz_convergence.png"
        fig.savefig(plot_path, dpi=140)
        log.info("Saved -> %s", plot_path)


if __name__ == "__main__":
    main()
