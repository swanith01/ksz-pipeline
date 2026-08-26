#!/usr/bin/env python
"""
Script 20: does P_diag depend on the specific choice of 26 groups?

Girish's request (2026-08-18): "repeat the P_diag calculation using
progressively thinner radial groups... P_diag should approach a stable
result as the radial grouping is refined. If it does not converge, its
comparison with the direct calculation is not yet well defined."

IMPORTANT REFRAME, worth understanding before reading results -- see
group_slices_uniform's own docstring for the full reasoning: P_diag is
NOT expected to converge to a single value as n_groups grows without
bound. P_total = P_diag + P_off is grouping-invariant (same map, just
summed in a different order), so finer grouping structurally moves real
correlation OUT of P_diag and INTO P_off, by construction. P_diag should
DECREASE monotonically as n_groups grows, toward the already-measured
ungrouped floor (well below direct). The well-posed question is
therefore NOT "does P_diag converge as n_groups->infinity" but "is
there a PLATEAU in a reasonable range around n_groups=26" -- if P_diag
is roughly stable for n_groups in, say, [15,40], the match to direct at
26 is robust, not a coincidence tied to that exact number. If P_diag
keeps changing meaningfully even close to 26, the comparison to direct
is fragile.

BUILT-IN SANITY CHECK: P_total should come back numerically IDENTICAL
across every n_groups tested (grouping-invariant by construction, unlike
P_diag). If it drifts, that is a bug in this script or
group_slices_uniform, not physics -- checked and printed automatically.

Reuses the SAME theta_slices construction as script 17 (matched window,
chi_eff, ne0_cgs), built ONCE, then swept cheaply across n_groups
values -- the expensive ~20-50 GB per-slice step only happens once.

Usage
-----
    python scripts/20_pdiag_grouping_convergence.py --config configs/fiducial.yaml
"""
import argparse
import gc
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map
from ksz_pipeline.ksz.coherence_decomposition import (compute_ksz_map_per_slice,
                                                       decompose_p_total_diag_off,
                                                       group_slices_uniform)
from ksz_pipeline.utils.constants import ne0_cgs, MPC_CM

N_GROUPS_SWEEP = [5, 8, 13, 20, 26, 35, 50, 75, 110, 165, 250, 400]


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    BOX_LEN = sim_cfg['BOX_LEN']
    HII_DIM = sim_cfg['HII_DIM_coeval']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots = sorted(cfg['coeval_ksz']['z_snapshots'])
    cache_dir = cfg['data']['cache_dir']
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(f"{closure_path} not found -- run script 14 first.")
    closure = np.load(closure_path)
    chi_eff = float(closure['chi_eff'])
    z_lo, z_hi = float(closure['z_lo']), float(closure['z_hi'])
    ell_direct, Dl_direct = closure['ell_direct'], closure['Dl_direct']
    d3000_direct = float(np.interp(3000, ell_direct, Dl_direct))
    print(f"chi_eff={chi_eff:.1f} Mpc, window z=[{z_lo:.2f},{z_hi:.2f}], "
          f"direct D_3000={d3000_direct:.4g} uK^2\n")

    print("Building stitched lightcone (fiducial scale, cache-hit expected)...")
    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    stitched = stitch_lightcone_from_coeval(
        z_snapshots=z_snapshots, z_arr=z_arr, HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
        cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])
    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM
    x_e_interp    = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis      = stitched['pos_axis']

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    i0 = np.searchsorted(z_arr, z_lo)
    i1 = np.searchsorted(z_arr, z_hi)
    density_1plus_w  = density_1plus[:, :, i0:i1]
    x_HII_field_w    = x_HII_field[:, :, i0:i1]
    v_los_Mpc_s_w    = v_los_Mpc_s[:, :, i0:i1]
    z_arr_w          = z_arr[i0:i1]
    ds_w             = ds[i0:i1 - 1]
    visibility_3D_w  = visibility_3D[:, :, i0:i1]
    patchy_mask_3D_w = patchy_mask_3D[:, :, i0:i1]
    del density_1plus, x_HII_field, v_los_Mpc_s, visibility_3D, patchy_mask_3D

    print("Computing per-slice theta (the ~20-50 GB peak step, done ONCE)...")
    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)
    del density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, visibility_3D_w, patchy_mask_3D_w
    gc.collect()
    print(f"  {theta_slices.shape[-1]} native thin LOS pixels available\n")

    n_groups_list = [n for n in N_GROUPS_SWEEP if n <= theta_slices.shape[-1]]

    d3000_total_vals, d3000_diag_vals = [], []
    print(f"{'n_groups':>10} {'mean thickness [Mpc]':>22} {'D_3000 total':>14} {'D_3000 diag':>13}")
    for n_groups in n_groups_list:
        theta_grouped, chi_grouped = group_slices_uniform(theta_slices, chi_mid_mpc, n_groups)
        ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
            theta_grouped, BOX_LEN, chi_eff)
        d3000_total = float(np.interp(3000, ell_dec, Dl_total))
        d3000_diag  = float(np.interp(3000, ell_dec, Dl_diag))
        d3000_total_vals.append(d3000_total)
        d3000_diag_vals.append(d3000_diag)
        mean_thickness = (float(chi_mid_mpc.max()) - float(chi_mid_mpc.min())) / n_groups
        print(f"{n_groups:>10} {mean_thickness:>22.2f} {d3000_total:>14.4g} {d3000_diag:>13.4g}")
        del theta_grouped
        gc.collect()

    del theta_slices
    gc.collect()

    # ---- sanity check: P_total should be numerically identical throughout ----
    total_spread = (max(d3000_total_vals) - min(d3000_total_vals)) / np.mean(d3000_total_vals)
    print(f"\nSANITY CHECK: P_total spread across all n_groups = {total_spread:.2%} "
          f"(should be ~0 -- grouping-invariant by construction)")
    if total_spread > 0.01:
        print("  *** WARNING: P_total is NOT grouping-invariant -- bug in "
              "group_slices_uniform or this script, investigate before trusting "
              "the P_diag trend below. ***")
    else:
        print("  OK.")

    # ---- the actual question: is there a plateau near n_groups=26? ----
    d3000_diag_vals = np.array(d3000_diag_vals)
    idx_26 = n_groups_list.index(26) if 26 in n_groups_list else None
    if idx_26 is not None:
        neighbors = [i for i in range(len(n_groups_list))
                     if 15 <= n_groups_list[i] <= 40]
        plateau_vals = d3000_diag_vals[neighbors]
        plateau_spread = (plateau_vals.max() - plateau_vals.min()) / plateau_vals.mean()
        print(f"\nP_diag spread across n_groups in [15,40] (around the n=26 choice): "
              f"{plateau_spread:.1%}")
        if plateau_spread < 0.15:
            print(">>> Reasonably stable plateau near n_groups=26 -- the match to "
                  "direct is not sensitively tied to that exact choice.")
        else:
            print(">>> P_diag still changing meaningfully even near n_groups=26 -- "
                  "the comparison to direct may be fragile, not yet well-defined.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_groups_list, d3000_diag_vals, 'o-', color='tab:blue', label='P_diag D_3000')
    ax.axhline(d3000_direct, color='k', ls='--', lw=1.5, label=f'direct D_3000={d3000_direct:.3g}')
    ax.axvspan(15, 40, color='gray', alpha=0.15, label='range around n=26')
    ax.axvline(26, color='tab:orange', lw=1, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('number of radial groups')
    ax.set_ylabel(r'$D_{3000}$ [$\mu$K$^2$]')
    ax.set_title('P_diag vs radial grouping: is n=26 a robust choice?')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = f"{plot_dir}/pdiag_grouping_convergence.png"
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/pdiag_grouping_convergence.npz",
              n_groups=n_groups_list, d3000_total=d3000_total_vals,
              d3000_diag=d3000_diag_vals, d3000_direct=d3000_direct,
              chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi)
    print(f"Saved -> {out_dir}/pdiag_grouping_convergence.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
