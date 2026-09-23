#!/usr/bin/env python
"""
Script 28: does the STITCHED lightcone's P_total/P_diag/P_off depend on
how many coeval snapshots it's interpolated from?

Different question from script 20's regrouping sweep (Image 1) and script
27's snapshot-count sweep for the off-diagonal estimator -- neither of
those touches the STITCHING interpolation itself:

  - Script 20 re-bins an ALREADY-BUILT stitched map into different numbers
    of groups. P_total is grouping-invariant BY CONSTRUCTION there (same
    pixels, different partition) -- not an open question.
  - Script 27 sums pairs of INDEPENDENT coeval boxes directly, no
    interpolation between them at all.
  - THIS script subsets which snapshots stitch_lightcone_from_coeval
    interpolates BETWEEN (stitch_field's interp1d across snap_z) --
    fewer input snapshots means coarser interpolation, which genuinely
    changes the resulting continuous field's values. P_total is NOT
    guaranteed invariant here; it's an open empirical question, reported
    neutrally rather than checked against a known-correct answer.

Answers two things directly: (1) does P_total drift with snapshot count,
and (2) does P_diag's match to coeval-direct (already shown to be robust
to REGROUPING by script 20) hold up when the STITCH ITSELF is built from
fewer snapshots -- if it doesn't, script 20's convergence result was
checking the wrong axis.

Reuses build_dz_subsets from convergence/dz_sweep.py, applied to the FULL
cfg['coeval_ksz']['z_snapshots'] list (the same z_fine that function's own
docstring names as its typical input) -- NOT the patchy-filtered ZS_win
scripts 20/23/27 use, since stitch_lightcone_from_coeval interpolates
across the full z_min..z_max range before any windowing happens.

Does NOT import from scripts/17_coherence_decomposition_fiducial.py --
that script doesn't expose its steps as separate functions (everything is
inline in main()), so refactoring it to be reusable here would mean
editing an already-deployed, already-validated file. Instead this
replicates its exact sequence of LIBRARY calls (stitch_from_coeval.py,
optical_depth.py, coherence_decomposition.py) directly, at each dz
multiple -- same already-tested functions script 17 itself calls, script
17 stays completely untouched.

COST WARNING, read before submitting as a job: unlike script 27, this
reruns the FULL ~20-50 GB per-slice compute_ksz_map_per_slice step at
EVERY dz_multiple (interpolating from fewer snapshots is a genuinely
different field, not just a different sum over the same independent
boxes) -- roughly sum(dz_multiples) times the cost of a single script 17
run, not len(dz_multiples) times a cheap step. Test small first.

Usage
-----
    python scripts/28_stitched_dz_convergence.py --config configs/fiducial.yaml
    python scripts/28_stitched_dz_convergence.py --config configs/fiducial.yaml \
        --dz-multiples 1 2 4 --hii-dim 64   # fast interactive shakedown
"""
import argparse
import gc
import logging
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.dz_sweep import build_dz_subsets
from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.coherence_decomposition import (compute_ksz_map_per_slice,
                                                       decompose_p_total_diag_off,
                                                       group_slices_by_snapshot)
from ksz_pipeline.utils.constants import ne0_cgs, MPC_CM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger("stitched_dz")


def run_one_multiple(zs_m, cfg, chi_eff, z_lo, z_hi):
    """One dz_multiple's worth of: stitch -> tau/visibility -> window ->
    per-slice theta -> group-by-snapshot -> decompose. Exactly script 17's
    own sequence, called directly rather than imported from it."""
    sim_cfg = cfg["21cmfast"]
    BOX_LEN, HII_DIM = sim_cfg["BOX_LEN"], sim_cfg["HII_DIM_coeval"]
    z_min, z_max = sim_cfg["z_min"], sim_cfg["z_max"]
    cache_dir = cfg["data"]["cache_dir"]

    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    stitched = stitch_lightcone_from_coeval(
        z_snapshots=zs_m, z_arr=z_arr, HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
        cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg["N_THREADS"], random_seed=sim_cfg["random_seed"])
    density_1plus = 1.0 + stitched["density"]
    x_HII_field = 1.0 - stitched["xH_box"]
    v_los_Mpc_s = stitched["velocity_z"] / MPC_CM
    x_e_interp = 1.0 - stitched["xH_box"].mean(axis=(0, 1))
    pos_axis = stitched["pos_axis"]

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    i0 = np.searchsorted(z_arr, z_lo)
    i1 = np.searchsorted(z_arr, z_hi)
    density_1plus_w = density_1plus[:, :, i0:i1]
    x_HII_field_w = x_HII_field[:, :, i0:i1]
    v_los_Mpc_s_w = v_los_Mpc_s[:, :, i0:i1]
    z_arr_w = z_arr[i0:i1]
    ds_w = ds[i0:i1 - 1]
    visibility_3D_w = visibility_3D[:, :, i0:i1]
    patchy_mask_3D_w = patchy_mask_3D[:, :, i0:i1]
    del density_1plus, x_HII_field, v_los_Mpc_s, visibility_3D, patchy_mask_3D

    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)
    del density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, visibility_3D_w, patchy_mask_3D_w
    gc.collect()

    # group at the SAME granularity the stitch was built from -- zs_m,
    # not the full snapshot list, matching script 17's own convention
    theta_grouped, chi_grouped = group_slices_by_snapshot(theta_slices, chi_mid_mpc, zs_m)
    del theta_slices
    gc.collect()

    ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
        theta_grouped, BOX_LEN, chi_eff)
    return ell_dec, Dl_total, Dl_diag, Dl_off, theta_grouped.shape[-1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fiducial.yaml")
    p.add_argument("--dz-multiples", type=int, nargs="+", default=None,
                   help="default: cfg['convergence']['dz_multiples']")
    p.add_argument("--hii-dim", type=int, default=None)
    p.add_argument("--box-len", type=float, default=None)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.hii_dim is not None:
        cfg["21cmfast"]["HII_DIM_coeval"] = args.hii_dim
    if args.box_len is not None:
        cfg["21cmfast"]["BOX_LEN"] = args.box_len

    dz_multiples = args.dz_multiples or cfg.get("convergence", {}).get(
        "dz_multiples", [1, 2, 4])
    log.info("dz_multiples: %s (BOX_LEN=%.0f, HII_DIM=%d)",
             dz_multiples, cfg["21cmfast"]["BOX_LEN"], cfg["21cmfast"]["HII_DIM_coeval"])

    out_dir = cfg["data"]["output_dir"].rstrip("/")
    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(f"{closure_path} not found -- run script 14 first.")
    closure = np.load(closure_path)
    chi_eff = float(closure["chi_eff"])
    z_lo, z_hi = float(closure["z_lo"]), float(closure["z_hi"])
    ell_direct, Dl_direct = closure["ell_direct"], closure["Dl_direct"]
    d3000_direct = float(np.interp(3000, ell_direct, Dl_direct))
    log.info("chi_eff=%.1f Mpc, window z=[%.2f,%.2f], direct D_3000=%.4g uK^2",
             chi_eff, z_lo, z_hi, d3000_direct)

    z_snapshots = sorted(cfg["coeval_ksz"]["z_snapshots"])
    subsets = build_dz_subsets(z_snapshots, dz_multiples)

    curves, rows = {}, []
    for m in sorted(dz_multiples):
        zs_m = subsets[m]
        log.info("dz_x%d: stitching from %d snapshots...", m, len(zs_m))
        ell_dec, Dl_total, Dl_diag, Dl_off, n_groups = run_one_multiple(
            zs_m, cfg, chi_eff, z_lo, z_hi)
        d3000_total = float(np.interp(3000, ell_dec, Dl_total))
        d3000_diag = float(np.interp(3000, ell_dec, Dl_diag))
        d3000_off = float(np.interp(3000, ell_dec, Dl_off))
        frac_diff = abs(d3000_diag - d3000_direct) / d3000_direct
        curves[m] = (ell_dec.copy(), Dl_total.copy(), Dl_diag.copy(), Dl_off.copy())
        rows.append(dict(m=m, n_snap=len(zs_m), n_groups=n_groups,
                         d3000_total=d3000_total, d3000_diag=d3000_diag,
                         d3000_off=d3000_off, frac_diff=frac_diff))
        log.info("  n_groups=%d  D_3000: total=%.4g diag=%.4g off=%.4g  "
                 "|diag-direct|/direct=%.1f%%",
                 n_groups, d3000_total, d3000_diag, d3000_off, frac_diff * 100)

    total_vals = np.array([r["d3000_total"] for r in rows])
    total_spread = (total_vals.max() - total_vals.min()) / np.mean(np.abs(total_vals))
    log.info("\nP_total spread across dz_multiples = %.1f%% -- unlike script 20's "
             "regrouping test, this is NOT guaranteed to be small a priori; "
             "reported as-is, not checked against a known-correct answer.",
             total_spread * 100)

    diag_vals = np.array([r["d3000_diag"] for r in rows])
    diag_spread = (diag_vals.max() - diag_vals.min()) / np.mean(np.abs(diag_vals))
    log.info("P_diag spread across dz_multiples = %.1f%% (all still compared "
             "against the SAME direct D_3000=%.4g)", diag_spread * 100, d3000_direct)

    outdir = args.outdir or os.path.join(cfg["data"]["plot_dir"].rstrip("/"),
                                         "28_stitched_dz")
    os.makedirs(outdir, exist_ok=True)

    save_dict = dict(dz_multiples=sorted(dz_multiples), d3000_direct=d3000_direct,
                     chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi)
    for m in dz_multiples:
        ell_dec, Dl_total, Dl_diag, Dl_off = curves[m]
        save_dict[f"ell_dz_x{m}"] = ell_dec
        save_dict[f"Dl_total_dz_x{m}"] = Dl_total
        save_dict[f"Dl_diag_dz_x{m}"] = Dl_diag
        save_dict[f"Dl_off_dz_x{m}"] = Dl_off
    np.savez(f"{outdir}/stitched_dz_convergence.npz", **save_dict)
    log.info("Saved -> %s/stitched_dz_convergence.npz", outdir)

    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        for m in sorted(dz_multiples):
            r = next(x for x in rows if x["m"] == m)
            ell_dec, Dl_total, Dl_diag, Dl_off = curves[m]
            ax1.plot(ell_dec, Dl_diag, "--", lw=1.5,
                    label=f"P_diag, dz_x{m} ({r['n_snap']} snapshots)")
        ax1.axhline(d3000_direct, color="k", ls="-", lw=2, label="coeval-direct D_3000")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel(r"$\ell$"); ax1.set_ylabel(r"$D_\ell$ [$\mu K^2$]")
        ax1.set_title("P_diag vs direct, across snapshot count")
        ax1.legend(fontsize=7)

        ms = [r["m"] for r in rows]
        ax2.plot(ms, [r["d3000_total"] for r in rows], "o-", label="P_total")
        ax2.plot(ms, [r["d3000_diag"] for r in rows], "s-", label="P_diag")
        ax2.plot(ms, [r["d3000_off"] for r in rows], "^-", label="P_off")
        ax2.axhline(d3000_direct, color="k", ls="--", label="coeval-direct")
        ax2.set_xlabel("dz_multiple (coarser ->)")
        ax2.set_ylabel(r"$D_{3000}$ [$\mu K^2$]")
        ax2.set_title("D_3000 vs stitching snapshot count")
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plot_path = f"{outdir}/stitched_dz_convergence.png"
        fig.savefig(plot_path, dpi=140)
        log.info("Saved -> %s", plot_path)


if __name__ == "__main__":
    main()
