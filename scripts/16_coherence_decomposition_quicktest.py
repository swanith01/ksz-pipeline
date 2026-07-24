#!/usr/bin/env python
"""
Script 16: coherence decomposition quick test (P_total / P_diag / P_off).

Implements the advisor's 2026-07-23 suggestion at SMALL scale, as a fast
desktop sanity check before ever running this at fiducial resolution on
the cluster (see coherence_decomposition.py's memory warning -- full
resolution needs a different, not-yet-written implementation).

Decision logic (exactly as specified):
  1. Does P_diag match coeval-direct? If NOT, stop -- there's still an
     unreconciled normalization/geometry/FFT-convention issue, and
     nothing below means anything until that's fixed.
  2. If P_diag matches, examine P_off vs Delta-chi. Smooth physical
     decay -> evidence for genuine lightcone cross-correlation. Spikes
     tied to box size / periodicity -> stitching artifact, not physics.

SANITY CHECKS (run automatically, printed): P_total reconstructed from
the per-slice decomposition should match ksz_map_to_Dl applied to the
ordinarily-summed map, to numerical precision. If these DON'T match,
there is a bug in coherence_decomposition.py itself -- stop and debug
that before reading anything else this script prints.

THIS IS A LOGIC/PIPELINE TEST, NOT A SCIENTIFIC RESULT. Small BOX_LEN/
HII_DIM by design, for speed -- do not read the actual D_3000/P_off
NUMBERS here as meaningful; only whether P_diag~direct, and the
qualitative shape of P_off(Delta-chi).

Usage
-----
    python scripts/16_coherence_decomposition_quicktest.py --config configs/fiducial.yaml
    python scripts/16_coherence_decomposition_quicktest.py --box-len 200 --hii-dim 24
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ksz_pipeline.convergence.coeval_sweep import run_one_config as run_coeval_one_config
from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ksz_pipeline.ksz.coherence_decomposition import (compute_ksz_map_per_slice,
                                                       decompose_p_total_diag_off,
                                                       cross_power_by_dchi)
from ksz_pipeline.utils.constants import ne0_cgs


def main(config_path, box_len, hii_dim, n_z_subset):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots_full = sorted(cfg['coeval_ksz']['z_snapshots'])
    # Subsample for speed if requested -- every Nth snapshot, always
    # keeping the endpoints so the window stays roughly representative.
    if n_z_subset and n_z_subset < len(z_snapshots_full):
        step = max(1, len(z_snapshots_full) // n_z_subset)
        z_snapshots = z_snapshots_full[::step]
    else:
        z_snapshots = z_snapshots_full

    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    cache_dir = os.path.join(cfg['data']['cache_dir'], "quicktest_coherence")

    print(f"QUICK TEST -- BOX_LEN={box_len} Mpc, HII_DIM={hii_dim}, "
          f"{len(z_snapshots)} z_snapshots (of {len(z_snapshots_full)} full). "
          f"Logic test only, NOT a scientific result -- see script docstring.\n")

    # ================================================================
    # coeval-direct reference, at the SAME small box/resolution --
    # reuses the existing trusted coeval_sweep machinery unmodified.
    # ================================================================
    print("Computing coeval-direct reference...")
    direct = run_coeval_one_config(box_len, hii_dim, z_snapshots, cache_dir,
                                    tag="quicktest_direct",
                                    N_THREADS=sim_cfg['N_THREADS'],
                                    random_seed=sim_cfg['random_seed'])
    ells_direct, Dl_direct = direct['ells_direct'], direct['Dl_direct']
    d3000_direct = float(np.interp(3000, ells_direct, Dl_direct)) if len(ells_direct) else float('nan')
    print(f"  direct D_3000 = {d3000_direct:.4g} uK^2\n")

    # ================================================================
    # stitched fields -- same construction as script 14, small scale
    # ================================================================
    print("Building stitched lightcone...")
    cell_size = box_len / hii_dim
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    stitched = stitch_lightcone_from_coeval(
        z_snapshots=z_snapshots, z_arr=z_arr, HII_DIM=hii_dim, BOX_LEN=box_len,
        cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])
    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    from ksz_pipeline.utils.constants import MPC_CM
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM
    x_e_interp    = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis      = stitched['pos_axis']

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    print(f"  n_lc_pix = {len(z_arr)}\n")

    # ================================================================
    # SANITY CHECK 1: per-slice sum reproduces the trusted compute_ksz_map
    # ================================================================
    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus, x_HII_field, v_los_Mpc_s, z_arr, ds, visibility_3D,
        ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D)

    ksz_map_reference = compute_ksz_map(
        density_1plus, x_HII_field, v_los_Mpc_s, z_arr, ds, visibility_3D,
        ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D)
    ksz_map_from_slices = theta_slices.sum(axis=-1)

    max_abs_diff = np.max(np.abs(ksz_map_reference - ksz_map_from_slices))
    print(f"SANITY CHECK 1 (per-slice sum vs compute_ksz_map): "
          f"max|diff| = {max_abs_diff:.3e} "
          f"(should be ~0, i.e. floating-point noise only)")
    if max_abs_diff > 1e-10 * np.max(np.abs(ksz_map_reference)):
        print("  *** WARNING: mismatch larger than floating-point noise -- "
              "bug in compute_ksz_map_per_slice, stop here. ***\n")
    else:
        print("  OK -- per-slice decomposition faithfully reproduces the map.\n")

    ell_ref, Dl_ref, _ = ksz_map_to_Dl(ksz_map_reference, box_len)

    # ================================================================
    # decomposition: P_total / P_diag / P_off
    # ================================================================
    # chi_Mpc here: for a quick test, use the midpoint of the window's
    # own chi range as a simple reference distance -- NOT chi_eff from
    # the fiducial closure test, which doesn't apply at this box/z config.
    chi_Mpc_quicktest = float(np.median(chi_mid_mpc))
    ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
        theta_slices, box_len, chi_Mpc_quicktest)

    d3000_total = float(np.interp(3000, ell_dec, Dl_total)) if len(ell_dec) else float('nan')
    d3000_ref   = float(np.interp(3000, ell_ref, Dl_ref)) if len(ell_ref) else float('nan')
    print(f"SANITY CHECK 2 (D_3000, total from decomposition vs ksz_map_to_Dl "
          f"directly on the summed map): {d3000_total:.4g} vs {d3000_ref:.4g} uK^2 "
          f"(should closely agree; small differences OK from independent "
          f"radial binning/chi choice, large differences are a bug)\n")

    d3000_diag = float(np.interp(3000, ell_dec, Dl_diag)) if len(ell_dec) else float('nan')
    d3000_off  = float(np.interp(3000, ell_dec, Dl_off)) if len(ell_dec) else float('nan')

    print(f"{'':20s} {'D_3000 [uK^2]':>15s}")
    print(f"{'coeval-direct':20s} {d3000_direct:>15.4g}")
    print(f"{'stitched P_total':20s} {d3000_total:>15.4g}")
    print(f"{'stitched P_diag':20s} {d3000_diag:>15.4g}")
    print(f"{'stitched P_off':20s} {d3000_off:>15.4g}")

    # ================================================================
    # THE DECISION: does P_diag match direct?
    # ================================================================
    frac_diff = abs(d3000_diag - d3000_direct) / d3000_direct if d3000_direct else float('nan')
    print(f"\n|P_diag - direct| / direct = {frac_diff:.1%}")
    if frac_diff > 0.3:
        print(">>> P_diag does NOT closely match coeval-direct. Per the advisor's "
              "logic: STOP HERE. There is still an unreconciled normalization, "
              "geometry, q_parallel/q_perp, or FFT-convention issue -- the "
              "Delta-chi test below is not yet meaningful. Reconcile this first, "
              "at fiducial resolution on the cluster, before trusting anything "
              "past this point.")
    else:
        print(">>> P_diag reasonably matches coeval-direct. Per the advisor's "
              "logic: proceed to the Delta-chi decay test below.")

    # ================================================================
    # Delta-chi binned cross-power (only meaningful if the above passed)
    # ================================================================
    print("\nComputing Delta-chi binned cross-power (P_off decomposed by "
          "radial separation)...")
    dchi_centers, cross_mean, cross_std, n_pairs = cross_power_by_dchi(
        theta_slices, chi_mid_mpc, box_len)

    print(f"{'dchi [Mpc]':>12} {'mean cross-power':>18} {'n_pairs':>10}")
    for i in range(len(dchi_centers)):
        if n_pairs[i] > 0:
            print(f"{dchi_centers[i]:>12.1f} {cross_mean[i]:>18.4e} {n_pairs[i]:>10d}")

    # ================================================================
    # plot + save
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(ells_direct, Dl_direct, 'k-', lw=2, label='coeval-direct')
    ax1.plot(ell_dec, Dl_diag, color='tab:blue', lw=2, ls='--', label='stitched P_diag')
    ax1.plot(ell_dec, Dl_total, color='tab:red', lw=1.5, label='stitched P_total')
    ax1.plot(ell_dec, np.abs(Dl_off), color='tab:orange', lw=1.5, ls=':',
              label='stitched |P_off| (can be negative)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax1.set_title('P_diag vs direct (Stage 1 test)')
    ax1.legend(fontsize=8)

    valid_bins = n_pairs > 0
    ax2.errorbar(dchi_centers[valid_bins], cross_mean[valid_bins],
                 yerr=cross_std[valid_bins] / np.sqrt(np.maximum(n_pairs[valid_bins], 1)),
                 fmt='o-', color='tab:purple', capsize=3)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel(r'$|\Delta\chi|$ [Mpc]')
    ax2.set_ylabel('mean pairwise cross-power')
    ax2.set_title('P_off vs radial separation (Stage 2 test)')

    plt.tight_layout()
    plot_path = f"{plot_dir}/coherence_decomposition_quicktest.png"
    fig.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/coherence_decomposition_quicktest.npz",
              box_len=box_len, hii_dim=hii_dim, n_z_snapshots=len(z_snapshots),
              ells_direct=ells_direct, Dl_direct=Dl_direct,
              ell_dec=ell_dec, Dl_total=Dl_total, Dl_diag=Dl_diag, Dl_off=Dl_off,
              d3000_direct=d3000_direct, d3000_total=d3000_total,
              d3000_diag=d3000_diag, d3000_off=d3000_off,
              dchi_centers=dchi_centers, cross_mean=cross_mean,
              cross_std=cross_std, n_pairs=n_pairs)
    print(f"Saved -> {out_dir}/coherence_decomposition_quicktest.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--box-len", type=float, default=200.0,
                         help="Small box for speed -- NOT fiducial's 800 Mpc")
    parser.add_argument("--hii-dim", type=int, default=24,
                         help="Small resolution for speed -- NOT fiducial's 512")
    parser.add_argument("--n-z-subset", type=int, default=10,
                         help="Subsample z_snapshots to roughly this many, for speed")
    args = parser.parse_args()
    main(args.config, args.box_len, args.hii_dim, args.n_z_subset)
