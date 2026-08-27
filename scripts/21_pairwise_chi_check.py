#!/usr/bin/env python
"""
Script 21: single-chi P_off vs pairwise-chi_ij P_off.

Tests whether using ONE shared chi_eff to convert every slice-pair's
k-space cross-term to ell (the existing approach, script 17) meaningfully
differs from converting each pair through its OWN chi_ij = sqrt(chi_i*chi_j)
(decompose_off_pairwise_chi). See conversation notes for why this stays
RECTILINEAR/flat-sky (not the full spherical/Bessel-function non-Limber
treatment) and why that's an honest, deliberate choice for now.

Rebuilds theta_grouped (n=26, matched to script 17/coherence_decomposition_
fiducial.npz) ONCE, then computes both versions from the same slices --
so any difference is purely the chi-per-pair effect, nothing else changed.

Usage
-----
    python scripts/21_pairwise_chi_check.py --config configs/fiducial.yaml
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
from ksz_pipeline.ksz.coherence_decomposition import (compute_ksz_map_per_slice,
                                                       decompose_p_total_diag_off,
                                                       decompose_off_pairwise_chi,
                                                       group_slices_by_snapshot)
from ksz_pipeline.utils.constants import ne0_cgs, MPC_CM


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
    print(f"chi_eff (single-chi reference) = {chi_eff:.1f} Mpc, window z=[{z_lo:.2f},{z_hi:.2f}]\n")

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

    print("Computing per-slice theta (the ~20-50 GB peak step)...")
    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)
    del density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, visibility_3D_w, patchy_mask_3D_w
    gc.collect()

    print("Grouping into 26 snapshot-matched slices (same as script 17)...")
    theta_grouped, chi_grouped = group_slices_by_snapshot(theta_slices, chi_mid_mpc, z_snapshots)
    del theta_slices
    gc.collect()
    print(f"  {theta_grouped.shape[-1]} groups\n")

    print("Computing SINGLE-CHI decomposition (existing method, script 17)...")
    ell_single, Dl_total_s, Dl_diag_s, Dl_off_single = decompose_p_total_diag_off(
        theta_grouped, BOX_LEN, chi_eff)
    d3000_off_single = float(np.interp(3000, ell_single, Dl_off_single))
    print(f"  D_3000(P_off, single-chi) = {d3000_off_single:.4g} uK^2\n")

    print("Computing PAIRWISE-chi_ij decomposition (new)...")
    ell_pair, Dl_off_pairwise, n_pairs_used = decompose_off_pairwise_chi(
        theta_grouped, chi_grouped, BOX_LEN)
    n_pairs_expected = theta_grouped.shape[-1] * (theta_grouped.shape[-1] - 1) // 2
    print(f"  {n_pairs_used}/{n_pairs_expected} pairs contributed")
    d3000_off_pairwise = float(np.interp(3000, ell_pair, Dl_off_pairwise))
    print(f"  D_3000(P_off, pairwise-chi) = {d3000_off_pairwise:.4g} uK^2\n")

    frac_diff = abs(d3000_off_pairwise - d3000_off_single) / abs(d3000_off_single)
    print(f"Fractional difference at ell=3000: {frac_diff:.1%}")
    if frac_diff < 0.10:
        print(">>> Small difference -- the single-chi approximation was reasonable "
              "for this window's actual range of chi_i values.")
    else:
        print(">>> Meaningful difference -- pair-specific chi matters here; the "
              "single-chi P_off numbers reported so far should be treated as "
              "approximate, not final.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ell_single, Dl_off_single, 'o-', color='tab:blue', ms=4, label='P_off, single chi_eff (existing)')
    ax.plot(ell_pair, Dl_off_pairwise, 's-', color='tab:red', ms=4, label='P_off, pairwise chi_ij (new)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$] (P_off, can be negative)')
    ax.set_title('Single shared chi vs pair-specific chi_ij -- rectilinear approximation only')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot_path = f"{plot_dir}/pairwise_chi_check.png"
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/pairwise_chi_check.npz",
              ell_single=ell_single, Dl_off_single=Dl_off_single,
              ell_pair=ell_pair, Dl_off_pairwise=Dl_off_pairwise,
              n_pairs_used=n_pairs_used, n_pairs_expected=n_pairs_expected,
              chi_eff=chi_eff, chi_grouped=chi_grouped)
    print(f"Saved -> {out_dir}/pairwise_chi_check.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
