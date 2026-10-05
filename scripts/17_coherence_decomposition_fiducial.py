#!/usr/bin/env python
"""
Script 17: coherence decomposition, FIDUCIAL resolution (with optional
box-size override for the periodicity-vs-physics test).

Scales script 16's quicktest logic up to full fiducial resolution
(800 Mpc / 512^3, all 29 z_snapshots), per Girish's 2026-08-14 note.

BOX-SIZE OVERRIDE (added 2026-08-17, after the fiducial run showed a
P_off spike at Delta-chi ~830 Mpc, suspiciously close to BOX_LEN=800):
--box-len/--hii-dim let you rerun at a DIFFERENT box size while keeping
cell size (dx) IDENTICAL to fiducial (so resolution isn't also changing
at the same time -- box size is isolated as the only varied axis). If
the Delta-chi bump moves proportionally with BOX_LEN, that's strong
evidence of a periodicity/box-reuse artifact, not real physics; if it
stays fixed in physical Mpc regardless of box size, that argues for
something real. chi_eff/z_lo/z_hi are reused from closure_test.npz
regardless of box size (these are LOS/redshift-only quantities,
independent of the transverse box size) -- but the coeval-direct
REFERENCE curve is recomputed fresh at the overridden box/resolution
(reusing coeval_sweep.run_one_config, same as script 16's quicktest),
since P_qperp genuinely depends on box size/resolution and
closure_test.npz's own Dl_direct is fiducial-specific.

THREE THINGS ADDED beyond script 16's quicktest, at fiducial defaults:
1. SNAPSHOT-LEVEL GROUPING (group_slices_by_snapshot) before the P_diag/
   P_off split -- "diagonal" now means what it means for coeval-direct
   (one unit per snapshot's full box depth), not per thin LOS pixel.
   Script 16's quicktest skipped this and got a 56.6% P_diag/direct
   mismatch that was very likely partly this granularity mismatch, not
   (only) missing physics -- see the accompanying theory note for why
   P_diag was never guaranteed to equal direct even with correct
   grouping.
2. PERIODICITY CONTROL (random_shift_slices) -- independent random
   cyclic shift per GROUPED slice. Preserves each slice's own power
   exactly (pure phase rotation), destroys any FIXED cross-slice
   alignment -- both genuine LOS q_parallel-cancellation correlation
   AND any periodicity/box-reuse artifact. A drop in P_off under
   shifting means the coherent excess IS alignment-dependent; it does
   NOT by itself distinguish real physics from a stitching artifact --
   both are alignment-dependent. ATON is the real discriminator if this
   control alone doesn't settle it.
3. MATCHED WINDOW + chi_eff, reused from closure_test.npz (script 14),
   same convention as scripts 05/10/13 -- avoids reintroducing the
   window-mismatch confound the closure test exists to remove. Also
   reuses closure_test.npz's OWN Dl_direct directly rather than
   recomputing coeval-direct fresh, so the comparison is against the
   exact trusted number.

MEMORY: peak ~20-50 GB (corrected estimate -- an earlier ~9.9 TB figure
was a units error, GB vs TB). Comfortably within the 125-515 GB node
memory seen on this cluster.

CACHING: uses the MAIN fiducial cache_dir directly (no subdirectory),
deliberately, so the already-built coeval boxes and stitched lightcone
are reused rather than resimulated. If N_THREADS is not actually
reaching py21cmfast under the hood (open, unconfirmed question from
earlier this session), this could still run single-threaded regardless
of config -- request generous walltime.

Usage
-----
    python scripts/17_coherence_decomposition_fiducial.py --config configs/fiducial.yaml
"""
import argparse
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def main(config_path, seed_for_shift, box_len_override, hii_dim_override):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg = cfg['21cmfast']
    BOX_LEN = box_len_override if box_len_override is not None else sim_cfg['BOX_LEN']
    HII_DIM = hii_dim_override if hii_dim_override is not None else sim_cfg['HII_DIM_coeval']
    is_fiducial_config = (BOX_LEN == sim_cfg['BOX_LEN'] and HII_DIM == sim_cfg['HII_DIM_coeval'])
    dx_fiducial = sim_cfg['BOX_LEN'] / sim_cfg['HII_DIM_coeval']
    dx_this_run = BOX_LEN / HII_DIM
    if abs(dx_this_run - dx_fiducial) > 1e-6:
        print(f"WARNING: this run's dx={dx_this_run:.4f} Mpc differs from "
              f"fiducial's dx={dx_fiducial:.4f} Mpc -- resolution is NOT held "
              f"fixed, so box size is not isolated as the only varied axis. "
              f"If testing the periodicity hypothesis, pick HII_DIM so dx matches.\n")
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots = sorted(cfg['coeval_ksz']['z_snapshots'])
    cache_dir = cfg['data']['cache_dir']   # SAME as fiducial -- deliberate, for cache-hit
    out_dir  = cfg['data']['output_dir'].rstrip('/')
    plot_dir = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    tag_suffix = "" if is_fiducial_config else f"_box{int(BOX_LEN)}"
    print(f"Coherence decomposition -- BOX_LEN={BOX_LEN} Mpc, "
          f"HII_DIM={HII_DIM} (dx={dx_this_run:.4f} Mpc), {len(z_snapshots)} z_snapshots. "
          f"{'FIDUCIAL config.' if is_fiducial_config else 'BOX-SIZE OVERRIDE -- periodicity test run.'}")

    # ---- load the matched window + chi_eff from the closure test (script 14) --
    # these are LOS/redshift-only quantities, reused regardless of box size. ----
    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(
            f"{closure_path} not found -- run scripts/14_closure_test.py first.")
    closure = np.load(closure_path)
    chi_eff = float(closure['chi_eff'])
    z_lo, z_hi = float(closure['z_lo']), float(closure['z_hi'])

    if is_fiducial_config:
        # Reuse script 14's own trusted direct curve directly.
        ell_direct, Dl_direct = closure['ell_direct'], closure['Dl_direct']
    else:
        # Box size differs -- coeval-direct's own P_qperp depends on box
        # size/resolution, so recompute a FRESH direct reference at THIS
        # run's own (BOX_LEN, HII_DIM), same as script 16's quicktest does.
        print(f"Computing a FRESH coeval-direct reference at BOX_LEN={BOX_LEN}, "
              f"HII_DIM={HII_DIM} (closure_test.npz's own Dl_direct is fiducial-"
              f"specific, not reusable here)...")
        from ksz_pipeline.convergence.coeval_sweep import run_one_config as run_coeval_one_config
        direct = run_coeval_one_config(BOX_LEN, HII_DIM, z_snapshots, cache_dir,
                                        tag=f"coherence_direct{tag_suffix}",
                                        N_THREADS=sim_cfg['N_THREADS'],
                                        random_seed=sim_cfg['random_seed'])
        ell_direct, Dl_direct = direct['ells_direct'], direct['Dl_direct']

    d3000_direct = float(np.interp(3000, ell_direct, Dl_direct))
    print(f"Matched window z=[{z_lo:.2f},{z_hi:.2f}], chi_eff={chi_eff:.1f} Mpc "
          f"-- direct D_3000={d3000_direct:.4g} uK^2\n")

    # ================================================================
    # stitched fields, fiducial scale, SAME cache_dir as everything
    # else -- should cache-hit the already-built lightcone.
    # ================================================================
    print("Building stitched lightcone (fiducial scale)...")
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
    print(f"  n_lc_pix (full) = {len(z_arr)}")

    # ---- truncate to the MATCHED window, same as script 14 ----
    i0 = np.searchsorted(z_arr, z_lo)
    i1 = np.searchsorted(z_arr, z_hi)
    density_1plus_w  = density_1plus[:, :, i0:i1]
    x_HII_field_w    = x_HII_field[:, :, i0:i1]
    v_los_Mpc_s_w    = v_los_Mpc_s[:, :, i0:i1]
    z_arr_w          = z_arr[i0:i1]
    ds_w             = ds[i0:i1 - 1]
    visibility_3D_w  = visibility_3D[:, :, i0:i1]
    patchy_mask_3D_w = patchy_mask_3D[:, :, i0:i1]
    print(f"  n_lc_pix (matched window) = {len(z_arr_w)}\n")

    # ================================================================
    # per-slice decomposition (thin LOS pixels), windowed
    # ================================================================
    print("Computing per-slice theta (this is the ~20-50 GB peak step)...")
    theta_slices, chi_mid_mpc = compute_ksz_map_per_slice(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)

    # SANITY CHECK: per-slice sum reproduces the trusted compute_ksz_map,
    # on the SAME windowed inputs script 14 itself used.
    ksz_map_reference = compute_ksz_map(
        density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, z_arr_w, ds_w,
        visibility_3D_w, ne0=ne0_cgs(), patchy_mask_3D=patchy_mask_3D_w)
    max_abs_diff = np.max(np.abs(ksz_map_reference - theta_slices.sum(axis=-1)))
    print(f"SANITY CHECK (per-slice sum vs compute_ksz_map): "
          f"max|diff| = {max_abs_diff:.3e} (should be ~0)")
    if max_abs_diff > 1e-8 * np.max(np.abs(ksz_map_reference)):
        print("  *** WARNING: larger than floating-point noise -- stop and debug. ***\n")
    else:
        print("  OK.\n")

    del density_1plus_w, x_HII_field_w, v_los_Mpc_s_w, visibility_3D_w, patchy_mask_3D_w
    del density_1plus, x_HII_field, v_los_Mpc_s, visibility_3D, patchy_mask_3D  # free memory

    # ================================================================
    # SNAPSHOT-LEVEL GROUPING -- "diagonal" now matches coeval-direct's
    # own definition (see theory note)
    # ================================================================
    print("Grouping thin LOS pixels into per-snapshot slices...")
    theta_grouped, chi_grouped = group_slices_by_snapshot(theta_slices, chi_mid_mpc, z_snapshots)
    print(f"  {theta_slices.shape[-1]} thin LOS pixels -> {theta_grouped.shape[-1]} snapshot groups\n")
    del theta_slices  # free the large thin-pixel array, no longer needed

    # ================================================================
    # UNSHIFTED decomposition -- the actual stage-1 test
    # ================================================================
    ell_dec, Dl_total, Dl_diag, Dl_off = decompose_p_total_diag_off(
        theta_grouped, BOX_LEN, chi_eff)
    d3000_total = float(np.interp(3000, ell_dec, Dl_total))
    d3000_diag  = float(np.interp(3000, ell_dec, Dl_diag))
    d3000_off   = float(np.interp(3000, ell_dec, Dl_off))

    print(f"{'':26s} {'D_3000 [uK^2]':>15s}")
    print(f"{'coeval-direct':26s} {d3000_direct:>15.4g}")
    print(f"{'stitched P_total':26s} {d3000_total:>15.4g}")
    print(f"{'stitched P_diag (grouped)':26s} {d3000_diag:>15.4g}")
    print(f"{'stitched P_off':26s} {d3000_off:>15.4g}")

    frac_diff = abs(d3000_diag - d3000_direct) / d3000_direct
    print(f"\n|P_diag - direct| / direct = {frac_diff:.1%}")
    if frac_diff > 0.3:
        print(">>> Still a substantial mismatch even with correct snapshot-level "
              "grouping. Per the theory note, P_diag was never GUARANTEED to equal "
              "direct exactly -- but a mismatch this large still likely means an "
              "unreconciled convention issue, not (only) the q_parallel-cancellation "
              "physics. Investigate before treating P_off below as informative.")
    else:
        print(">>> P_diag now reasonably matches coeval-direct at proper granularity. "
              "Proceed to the periodicity control and Delta-chi test below.")

    # ================================================================
    # PERIODICITY CONTROL -- random shift, preserves per-slice power,
    # destroys fixed cross-slice alignment
    # ================================================================
    print(f"\nRunning periodicity control (random shift, seed={seed_for_shift})...")
    theta_shifted = random_shift_slices(theta_grouped, seed=seed_for_shift)
    ell_shift, Dl_total_shift, Dl_diag_shift, Dl_off_shift = decompose_p_total_diag_off(
        theta_shifted, BOX_LEN, chi_eff)

    d3000_diag_shift = float(np.interp(3000, ell_shift, Dl_diag_shift))
    d3000_off_shift  = float(np.interp(3000, ell_shift, Dl_off_shift))
    print(f"  P_diag  unshifted vs shifted: {d3000_diag:.4g} vs {d3000_diag_shift:.4g} "
          f"(should closely agree -- shift preserves per-slice power exactly; "
          f"a mismatch here is a bug in random_shift_slices or the decomposition)")
    print(f"  P_off   unshifted vs shifted: {d3000_off:.4g} vs {d3000_off_shift:.4g} "
          f"(a large drop under shifting means the coherent excess IS "
          f"alignment-dependent -- consistent with either real LOS physics or a "
          f"stitching artifact, does NOT by itself distinguish the two)")

    # ================================================================
    # Delta-chi binned cross-power, unshifted vs shifted
    # ================================================================
    print("\nComputing Delta-chi binned cross-power (unshifted and shifted)...")
    dchi_c, cross_mean, cross_std, n_pairs = cross_power_by_dchi(theta_grouped, chi_grouped, BOX_LEN)
    dchi_c_s, cross_mean_s, cross_std_s, n_pairs_s = cross_power_by_dchi(theta_shifted, chi_grouped, BOX_LEN)

    print(f"{'dchi [Mpc]':>12} {'unshifted':>16} {'shifted':>16} {'n_pairs':>10}")
    for i in range(len(dchi_c)):
        if n_pairs[i] > 0:
            print(f"{dchi_c[i]:>12.1f} {cross_mean[i]:>16.4e} {cross_mean_s[i]:>16.4e} {n_pairs[i]:>10d}")

    # ================================================================
    # plot + save
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(ell_direct, Dl_direct, 'k-', lw=2, label='coeval-direct (script 14)')
    ax1.plot(ell_dec, Dl_diag, color='tab:blue', lw=2, ls='--', label='stitched P_diag (grouped, unshifted)')
    ax1.plot(ell_dec, Dl_total, color='tab:red', lw=1.5, label='stitched P_total (unshifted)')
    ax1.plot(ell_shift, Dl_total_shift, color='tab:green', lw=1.5, ls=':',
              label='stitched P_total (shifted control)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax1.set_title('P_diag vs direct, fiducial resolution')
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
    plot_path = f"{plot_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.png"
    fig.savefig(plot_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {plot_path}")

    np.savez(f"{out_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.npz",
              box_len=BOX_LEN, hii_dim=HII_DIM,
              ell_direct=ell_direct, Dl_direct=Dl_direct, d3000_direct=d3000_direct,
              ell_dec=ell_dec, Dl_total=Dl_total, Dl_diag=Dl_diag, Dl_off=Dl_off,
              ell_shift=ell_shift, Dl_total_shift=Dl_total_shift,
              Dl_diag_shift=Dl_diag_shift, Dl_off_shift=Dl_off_shift,
              dchi_centers=dchi_c, cross_mean=cross_mean, cross_std=cross_std, n_pairs=n_pairs,
              cross_mean_shifted=cross_mean_s, cross_std_shifted=cross_std_s,
              chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi, seed_for_shift=seed_for_shift)
    print(f"Saved -> {out_dir}/coherence_decomposition{tag_suffix or '_fiducial'}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--shift-seed", type=int, default=42)
    parser.add_argument("--box-len", type=float, default=None,
                         help="Override BOX_LEN for the periodicity test -- "
                              "pick --hii-dim so dx matches fiducial's own "
                              "(e.g. --box-len 400 --hii-dim 256, since "
                              "fiducial is 800/512, same dx=1.5625 Mpc)")
    parser.add_argument("--hii-dim", type=int, default=None,
                         help="Override HII_DIM -- see --box-len")
    args = parser.parse_args()
    main(args.config, args.shift_seed, args.box_len, args.hii_dim)
