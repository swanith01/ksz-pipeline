#!/usr/bin/env python
"""
Script 16: coherence decomposition quick test (P_total / P_diag / P_off).

UPDATED to mirror script 17's logic exactly (snapshot grouping +
periodicity shift control), at toy scale -- this is now a cheap,
fast way to catch bugs in group_slices_by_snapshot/random_shift_slices
BEFORE spending real cluster time on the fiducial-resolution run. If
this passes cleanly, script 17 is very likely to run without a logic
error (though obviously not immune to scale-dependent issues like
memory or single-threading).

ONE THING THIS DOES NOT TEST: script 17's closure_test.npz loading
(chi_eff, z_lo, z_hi, Dl_direct) -- there's no small-scale equivalent of
that file, so this script computes its own small-scale coeval-direct
reference and its own chi convention instead, same as the original
quicktest did. That file-loading code in 17 is simple I/O, low risk,
and separately checkable with a one-line python snippet on the cluster
once closure_test.npz exists there.

Decision logic (exactly as specified by the advisor):
  1. Does P_diag (now snapshot-grouped, matching coeval-direct's own
     "diagonal" granularity) match coeval-direct? If NOT, stop -- there
     is still an unreconciled normalization/geometry/FFT-convention
     issue, and nothing below means anything until that's fixed.
  2. If P_diag matches, examine P_off vs Delta-chi, unshifted vs the
     periodicity-shift control. Smooth physical decay, and a real drop
     under shifting -> evidence for genuine lightcone cross-correlation
     (though shifting alone can't distinguish real physics from a
     stitching artifact -- see script 17's docstring).

SANITY CHECKS (run automatically, printed): P_total reconstructed from
the per-slice decomposition should match ksz_map_to_Dl applied to the
ordinarily-summed map, to numerical precision. If these DON'T match,
there is a bug in coherence_decomposition.py itself -- stop and debug
that before reading anything else this script prints. The shift
control's own consistency (P_diag unshifted vs shifted should closely
agree -- shifting preserves per-slice power exactly) is ALSO checked
automatically.

THIS IS A LOGIC/PIPELINE TEST, NOT A SCIENTIFIC RESULT. Small BOX_LEN/
HII_DIM by design, for speed -- do not read the actual D_3000/P_off
NUMBERS here as meaningful; only whether the sanity checks pass, and
the qualitative shape of things.

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
                                                       group_slices_by_snapshot,
                                                       random_shift_slices,
                                                       cross_power_by_dchi)
from ksz_pipeline.utils.constants import ne0_cgs, MPC_CM


def main(config_path, box_len, hii_dim, n_z_subset, shift_seed):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots_full = sorted(cfg['coeval_ksz']['z_snapshots'])
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
          f"Logic test for script 17's grouping+shift-control code, NOT a "
          f"scientific result.\n")

    # ================================================================
    # coeval-direct reference, at the SAME small box/resolution
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
    # stitched fields
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
    max_abs_diff = np.max(np.abs(ksz_map_reference - theta_slices.sum(axis=-1)))
    print(f"SANITY CHECK 1 (per-slice sum vs compute_ksz_map): "
          f"max|diff| = {max_abs_diff:.3e} (should be ~0)")
    if max_abs_diff > 1e-10 * np.max(np.abs(ksz_map_reference)):
        print("  *** WARNING: mismatch larger than floating-point noise -- "
              "bug in compute_ksz_map_per_slice, stop here. ***\n")
    else:
        print("  OK.\n")

    # ================================================================
    # NEW: SNAPSHOT-LEVEL GROUPING -- the thing this update actually tests
    # ================================================================
    print("Grouping thin LOS pixels into per-snapshot slices "
          "(testing group_slices_by_snapshot)...")
    theta_grouped, chi_grouped = group_slices_by_snapshot(theta_slices, chi_mid_mpc, z_snapshots)
    print(f"  {theta_slices.shape[-1]} thin LOS pixels -> "
          f"{theta_grouped.shape[-1]} snapshot groups (expected ~{len(z_snapshots)})\n")

    chi_Mpc_quicktest = float(np.median(chi_grouped))
    ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
        theta_grouped, box_len, chi_Mpc_quicktest)

    d3000_total = float(np.interp(3000, ell_dec, Dl_total)) if len(ell_dec) else float('nan')
    d3000_diag  = float(np.interp(3000, ell_dec, Dl_diag)) if len(ell_dec) else float('nan')
    d3000_off   = float(np.interp(3000, ell_dec, Dl_off)) if len(ell_dec) else float('nan')

    print(f"{'':26s} {'D_3000 [uK^2]':>15s}")
    print(f"{'coeval-direct':26s} {d3000_direct:>15.4g}")
    print(f"{'stitched P_total':26s} {d3000_total:>15.4g}")
    print(f"{'stitched P_diag (grouped)':26s} {d3000_diag:>15.4g}")
    print(f"{'stitched P_off':26s} {d3000_off:>15.4g}")

    frac_diff = abs(d3000_diag - d3000_direct) / d3000_direct if d3000_direct else float('nan')
    print(f"\n|P_diag - direct| / direct = {frac_diff:.1%}")
    if frac_diff > 0.3:
        print(">>> Still substantial mismatch even with grouping. Per the theory "
              "note, P_diag was never guaranteed to equal direct exactly -- but "
              "check this against script 17's fiducial-scale result before "
              "assuming it's just physics; small-box sample variance is large.")
    else:
        print(">>> P_diag reasonably matches coeval-direct at proper granularity.")

    # ================================================================
    # NEW: PERIODICITY CONTROL -- testing random_shift_slices
    # ================================================================
    print(f"\nTesting periodicity control (random_shift_slices, seed={shift_seed})...")
    theta_shifted = random_shift_slices(theta_grouped, seed=shift_seed)
    ell_shift, Dl_total_shift, Dl_diag_shift, Dl_off_shift = decompose_p_total_diag_off(
        theta_shifted, box_len, chi_Mpc_quicktest)

    d3000_diag_shift = float(np.interp(3000, ell_shift, Dl_diag_shift)) if len(ell_shift) else float('nan')
    d3000_off_shift  = float(np.interp(3000, ell_shift, Dl_off_shift)) if len(ell_shift) else float('nan')

    diag_shift_frac = abs(d3000_diag_shift - d3000_diag) / d3000_diag if d3000_diag else float('nan')
    print(f"  P_diag unshifted vs shifted: {d3000_diag:.4g} vs {d3000_diag_shift:.4g} "
          f"({diag_shift_frac:.1%} difference)")
    if diag_shift_frac > 0.05:
        print("  *** WARNING: shift should preserve per-slice power almost exactly "
              "-- a >5% difference suggests a bug in random_shift_slices or the "
              "decomposition, not sampling noise. Investigate before trusting "
              "the shift control on the cluster. ***")
    else:
        print("  OK -- shift control preserves P_diag as expected.")
    print(f"  P_off unshifted vs shifted: {d3000_off:.4g} vs {d3000_off_shift:.4g}")

    # ================================================================
    # Delta-chi binned cross-power, unshifted vs shifted
    # ================================================================
    print("\nComputing Delta-chi binned cross-power (unshifted and shifted)...")
    dchi_c, cross_mean, cross_std, n_pairs = cross_power_by_dchi(theta_grouped, chi_grouped, box_len)
    dchi_c_s, cross_mean_s, cross_std_s, n_pairs_s = cross_power_by_dchi(theta_shifted, chi_grouped, box_len)

    print(f"{'dchi [Mpc]':>12} {'unshifted':>16} {'shifted':>16} {'n_pairs':>10}")
    for i in range(len(dchi_c)):
        if n_pairs[i] > 0:
            print(f"{dchi_c[i]:>12.1f} {cross_mean[i]:>16.4e} {cross_mean_s[i]:>16.4e} {n_pairs[i]:>10d}")

    # ================================================================
    # plot + save
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(ells_direct, Dl_direct, 'k-', lw=2, label='coeval-direct')
    ax1.plot(ell_dec, Dl_diag, color='tab:blue', lw=2, ls='--', label='stitched P_diag (grouped)')
    ax1.plot(ell_dec, Dl_total, color='tab:red', lw=1.5, label='stitched P_total')
    ax1.plot(ell_shift, Dl_total_shift, color='tab:green', lw=1.5, ls=':', label='P_total (shifted)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax1.set_title('P_diag vs direct (grouped, Stage 1 test)')
    ax1.legend(fontsize=8)

    valid = n_pairs > 0
    ax2.errorbar(dchi_c[valid], cross_mean[valid],
                 yerr=cross_std[valid] / np.sqrt(np.maximum(n_pairs[valid], 1)),
                 fmt='o-', color='tab:purple', capsize=3, label='unshifted')
    ax2.errorbar(dchi_c_s[valid], cross_mean_s[valid],
                 yerr=cross_std_s[valid] / np.sqrt(np.maximum(n_pairs_s[valid], 1)),
                 fmt='s--', color='gray', capsize=3, label='shifted control')
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_xlabel(r'$|\Delta\chi|$ [Mpc]')
    ax2.set_ylabel('mean pairwise cross-power')
    ax2.set_title('P_off vs radial separation: real vs shifted')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plot_path = f"{plot_dir}/coherence_decomposition_quicktest.png"
    fig.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/coherence_decomposition_quicktest.npz",
              box_len=box_len, hii_dim=hii_dim, n_z_snapshots=len(z_snapshots),
              ells_direct=ells_direct, Dl_direct=Dl_direct,
              ell_dec=ell_dec, Dl_total=Dl_total, Dl_diag=Dl_diag, Dl_off=Dl_off,
              ell_shift=ell_shift, Dl_total_shift=Dl_total_shift,
              Dl_diag_shift=Dl_diag_shift, Dl_off_shift=Dl_off_shift,
              d3000_direct=d3000_direct, d3000_total=d3000_total,
              d3000_diag=d3000_diag, d3000_off=d3000_off,
              dchi_centers=dchi_c, cross_mean=cross_mean, cross_std=cross_std, n_pairs=n_pairs,
              cross_mean_shifted=cross_mean_s, cross_std_shifted=cross_std_s)
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
    parser.add_argument("--shift-seed", type=int, default=42)
    args = parser.parse_args()
    main(args.config, args.box_len, args.hii_dim, args.n_z_subset, args.shift_seed)
