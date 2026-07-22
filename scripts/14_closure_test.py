#!/usr/bin/env python
"""
Script 14: direct-vs-stitched closure test, matched patchy window.

Addresses four things explicitly requested:
  1. dD_3000/dz and cumulative D_3000(<z), same window, both methods
  2. full stitched/direct D_ell ratio over the resolved ell range
  3. how much of the excess disappears with a z>=13 hard cut
  4. chi_eff: definition, computation, and two alternative chi's for
     comparison (NOT to pick whichever minimizes the gap -- reported
     side by side as a robustness check on the single-chi approximation
     itself)

ASSUMPTIONS FLAGGED, NOT VERIFIED:
  - chi_eff is NOT found anywhere in lightcone_integral.py or script 11
    as uploaded. The function below is a fresh implementation of the
    spec (RMS/power-weighted mean comoving distance, weighted by the
    same visibility^2 * patchy_mask kernel compute_ksz_map builds) --
    it has NOT been checked against whatever ran in job 1680475.
    Before trusting this script's chi_eff numbers, diff this function's
    logic against whatever's actually in stitch_from_coeval.py or
    optical_depth.py post-commit-6934d3c.
  - The unified patchy window below uses the INTERSECTION of coeval's
    xH_mean-based window and stitched's x_e-based window (the more
    conservative choice -- both methods only get credit for epochs
    BOTH agree are patchy). This is a judgment call, not something
    the advisor specified -- flag it in the writeup.

Usage
-----
    python scripts/14_closure_test.py --config configs/fiducial.yaml
"""
import argparse, json, os, pickle
import numpy as np
import yaml
from astropy.cosmology import Planck18 as cosmo

from ksz_pipeline.coeval.fields import run_coeval_fields
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ksz_pipeline.utils.constants import (MPC_CM, SIGMA_T, C_CGS, T_CMB_K, ne0_cgs)

# reuse script 11's helpers directly rather than re-deriving them
import importlib.util
spec = importlib.util.spec_from_file_location("audit11", "scripts/11_direct_vs_stitched_audit.py")
audit11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit11)
per_z_D3000_contributions = audit11.per_z_D3000_contributions
_interp_loglog = audit11._interp_loglog

XHI_MIN_PATCHY, XHI_MAX_PATCHY = 1.0e-4, 1.0 - 1.0e-4


def unified_patchy_window(ZS, results_qperp_all, z_arr, x_e_interp):
    """
    Intersection of coeval's xH_mean window and stitched's x_e window.
    Returns (z_lo, z_hi) -- the SAME bounds to be applied to both methods.
    """
    xH_mean_coeval = np.array([results_qperp_all[z]['xH_mean'] for z in ZS])
    xe_coeval = 1.0 - xH_mean_coeval
    coeval_patchy = np.array(ZS)[(xe_coeval >= XHI_MIN_PATCHY) & (xe_coeval <= XHI_MAX_PATCHY)]

    stitched_patchy_mask = (x_e_interp >= XHI_MIN_PATCHY) & (x_e_interp <= XHI_MAX_PATCHY)
    stitched_patchy_z = z_arr[stitched_patchy_mask]

    z_lo = max(coeval_patchy.min(), stitched_patchy_z.min())
    z_hi = min(coeval_patchy.max(), stitched_patchy_z.max())
    print(f"Coeval patchy window:   z=[{coeval_patchy.min():.2f}, {coeval_patchy.max():.2f}]")
    print(f"Stitched patchy window: z=[{stitched_patchy_z.min():.2f}, {stitched_patchy_z.max():.2f}]")
    print(f"UNIFIED (intersection): z=[{z_lo:.2f}, {z_hi:.2f}]  <- applied to BOTH methods below")
    return z_lo, z_hi


def chi_eff_power_weighted(density_1plus, x_HII_field, v_los_Mpc_s, red_axis,
                            ds, visibility_3D, patchy_mask_3D, z_lo, z_hi, ne0):
    """
    chi_eff = [ integral dz w(z)^2 chi(z) ] / [ integral dz w(z)^2 ]

    where w(z) is the RMS, over map pixels, of the per-slice temperature
    contribution compute_ksz_map would sum at that slice -- i.e. the
    power-weighting the map itself uses, collapsed to a per-z scalar.

    This is the same quantity described in the earlier chi_eff commit
    message; re-derived here from the compute_ksz_map integrand rather
    than imported, since no callable chi_eff function was found in the
    two files provided. VERIFY against the actual implementation.
    """
    window_mask = (red_axis >= z_lo) & (red_axis <= z_hi)
    idx = np.where(window_mask)[0]
    if len(idx) < 2:
        raise ValueError("Unified window contains <2 LOS pixels -- check z_lo/z_hi against z_arr resolution")

    c_Mpc_s = C_CGS / MPC_CM
    integrand = density_1plus * x_HII_field * v_los_Mpc_s / c_Mpc_s * visibility_3D * patchy_mask_3D
    # per-slice RMS over pixels, in the window only
    w_z = np.sqrt(np.mean(integrand[:, :, idx]**2, axis=(0, 1)))  # shape (len(idx),)
    chi_z = np.array([cosmo.comoving_distance(z).value for z in red_axis[idx]])

    num = np.trapz(w_z**2 * chi_z, red_axis[idx])
    den = np.trapz(w_z**2, red_axis[idx])
    return num / den


def alt_chi_candidates(ZS, results_qperp_all, z_lo, z_hi):
    """Two alternatives to chi_eff, for the robustness comparison --
    NOT for picking whichever gives the best-looking gap closure."""
    xH_mean = np.array([results_qperp_all[z]['xH_mean'] for z in ZS])
    ZS_arr = np.array(ZS)
    in_window = (ZS_arr >= z_lo) & (ZS_arr <= z_hi)

    # (a) chi at "end of reionization" -- xH_mean crosses 0.5
    z_half = np.interp(0.5, xH_mean[::-1], ZS_arr[::-1])  # xH_mean decreasing with z? check sign
    chi_end_reion = cosmo.comoving_distance(z_half).value

    # (b) unweighted mean chi over the unified window
    chi_window = np.array([cosmo.comoving_distance(z).value for z in ZS_arr[in_window]])
    chi_unweighted = chi_window.mean()

    return {"chi_end_reionization": (z_half, chi_end_reion), "chi_unweighted_mean": chi_unweighted}


def full_ell_curves(results_qperp_all, ZS_kept, stitched_full_map, BOX_LEN, chi_for_stitched):
    """Direct's full D_ell(ell) from compute_cell; stitched's full
    D_ell(ell) from the FULL (untruncated) map, using the SAME chi
    passed in (so caller controls which chi variant is being tested)."""
    ell_direct, Dl_direct, *_ = compute_cell(results_qperp_all)
    ell_stitched, Dl_stitched, Dl_stitched_err = ksz_map_to_Dl(stitched_full_map, BOX_LEN, chi_Mpc=chi_for_stitched)
    return (ell_direct, Dl_direct), (ell_stitched, Dl_stitched, Dl_stitched_err)


def ratio_on_common_grid(ell_a, Dl_a, ell_b, Dl_b):
    """Ratio b/a on the OVERLAP of both ell ranges only -- do not
    extrapolate past either method's resolved range."""
    lo = max(ell_a.min(), ell_b.min())
    hi = min(ell_a.max(), ell_b.max())
    ell_common = np.logspace(np.log10(lo), np.log10(hi), 40)
    Dl_a_i = _interp_loglog(ell_common, ell_a, Dl_a)
    Dl_b_i = _interp_loglog(ell_common, ell_b, Dl_b)
    return ell_common, Dl_b_i / Dl_a_i


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    sim_cfg, coeval_cfg = cfg['21cmfast'], cfg['coeval_ksz']
    cache_dir = cfg['data']['cache_dir']
    HII_DIM, BOX_LEN = sim_cfg['HII_DIM_coeval'], sim_cfg['BOX_LEN']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    ZS = sorted(coeval_cfg['z_snapshots'])
    out_dir = cfg['data']['output_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)

    qperp_pkl = os.path.join(cache_dir, "qperp_power.pkl")
    with open(qperp_pkl, 'rb') as f:
        results_qperp_all = pickle.load(f)

    # ---- rebuild stitched lightcone once (cache-hit, per script 11 pattern) ----
    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    stitched = stitch_lightcone_from_coeval(
        z_snapshots=coeval_cfg['z_snapshots'], z_arr=z_arr, HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
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

    # ================================================================
    # STEP 0: unified window -- the thing that has to happen BEFORE
    # any of the four diagnostics mean what the advisor wants them to.
    # ================================================================
    z_lo, z_hi = unified_patchy_window(ZS, results_qperp_all, z_arr, x_e_interp)

    # ================================================================
    # 4. chi_eff -- computed over the UNIFIED window, plus alternatives
    # ================================================================
    chi_eff = chi_eff_power_weighted(density_1plus, x_HII_field, v_los_Mpc_s, z_arr,
                                      ds, visibility_3D, patchy_mask_3D, z_lo, z_hi, ne0_cgs())
    alt_chi = alt_chi_candidates(ZS, results_qperp_all, z_lo, z_hi)
    print(f"\nchi_eff (power-weighted, unified window)  = {chi_eff:.1f} Mpc")
    print(f"chi at xH_mean=0.5 crossing                = {alt_chi['chi_end_reionization'][1]:.1f} Mpc "
          f"(z={alt_chi['chi_end_reionization'][0]:.2f})")
    print(f"chi unweighted mean over window            = {alt_chi['chi_unweighted_mean']:.1f} Mpc")

    # ================================================================
    # 1. dD_3000/dz and cumulative D_3000(<z), unified window
    # ================================================================
    ZS_win = [z for z in ZS if z_lo <= z <= z_hi]
    dD3000_direct, ZS_kept = per_z_D3000_contributions(results_qperp_all, ZS_win)
    per_slice_direct = np.array([dD3000_direct.get(z, 0.0) for z in ZS_win])
    dz_direct = np.gradient(ZS_win)
    dD3000_dz_direct = per_slice_direct / dz_direct

    cumulative_stitched, per_slice_stitched = [], []
    running = 0.0
    for i, z_checkpoint in enumerate(ZS_win):
        idx = np.searchsorted(z_arr, z_checkpoint)
        if idx < 2:
            cumulative_stitched.append(0.0); per_slice_stitched.append(0.0); continue
        ksz_trunc = compute_ksz_map(density_1plus[:, :, :idx], x_HII_field[:, :, :idx],
                                     v_los_Mpc_s[:, :, :idx], z_arr[:idx], ds[:idx-1],
                                     visibility_3D[:, :, :idx], ne0=ne0_cgs(),
                                     patchy_mask_3D=patchy_mask_3D[:, :, :idx])
        ell_t, Dl_t, _ = ksz_map_to_Dl(ksz_trunc, BOX_LEN, chi_Mpc=chi_eff)
        d3000 = float(np.interp(3000, ell_t, Dl_t)) if len(ell_t) else 0.0
        step = d3000 - running
        cumulative_stitched.append(d3000); per_slice_stitched.append(step)
        running = d3000
    cumulative_stitched = np.array(cumulative_stitched)
    dD3000_dz_stitched = np.array(per_slice_stitched) / dz_direct

    cumulative_direct = np.cumsum(per_slice_direct)

    # ================================================================
    # 2. full D_ell ratio, resolved range, unified window, using chi_eff
    # ================================================================
    full_idx = np.searchsorted(z_arr, z_hi)
    full_start = np.searchsorted(z_arr, z_lo)
    ksz_full_window = compute_ksz_map(
        density_1plus[:, :, full_start:full_idx], x_HII_field[:, :, full_start:full_idx],
        v_los_Mpc_s[:, :, full_start:full_idx], z_arr[full_start:full_idx], ds[full_start:full_idx-1],
        visibility_3D[:, :, full_start:full_idx], ne0=ne0_cgs(),
        patchy_mask_3D=patchy_mask_3D[:, :, full_start:full_idx])
    (ell_d, Dl_d), (ell_s, Dl_s, Dl_s_err) = full_ell_curves(results_qperp_all, ZS_kept, ksz_full_window, BOX_LEN, chi_eff)
    ell_ratio, ratio = ratio_on_common_grid(ell_d, Dl_d, ell_s, Dl_s)

    # ================================================================
    # 3. z>=13 exclusion -- how much of the excess disappears
    # ================================================================
    z_cut = 13.0
    excess_full = cumulative_stitched[-1] - cumulative_direct[-1]
    idx_cut = np.searchsorted(np.array(ZS_win), z_cut, side='right') - 1
    if idx_cut >= 0:
        excess_below_cut = cumulative_stitched[idx_cut] - cumulative_direct[idx_cut]
        frac_from_high_z = 1.0 - (excess_below_cut / excess_full) if excess_full != 0 else np.nan
        print(f"\nExcess (stitched - direct), full window: {excess_full:.4f} uK^2")
        print(f"Excess with z<{z_cut} cut only:            {excess_below_cut:.4f} uK^2")
        print(f"Fraction of excess attributable to z>={z_cut}: {100*frac_from_high_z:.1f}%")
    else:
        print(f"\nWARNING: z_cut={z_cut} falls outside unified window [{z_lo:.2f},{z_hi:.2f}] -- no cut applied")

    # ================================================================
    # save everything
    # ================================================================
    np.savez(f"{out_dir}/closure_test.npz",
              z_window=np.array(ZS_win), dD3000_dz_direct=dD3000_dz_direct,
              dD3000_dz_stitched=dD3000_dz_stitched,
              cumulative_direct=cumulative_direct, cumulative_stitched=cumulative_stitched,
              ell_direct=ell_d, Dl_direct=Dl_d, ell_stitched=ell_s, Dl_stitched=Dl_s,
              Dl_stitched_err=Dl_s_err, ell_ratio=ell_ratio, ratio=ratio,
              chi_eff=chi_eff, chi_end_reion=alt_chi['chi_end_reionization'][1],
              chi_unweighted=alt_chi['chi_unweighted_mean'],
              z_lo=z_lo, z_hi=z_hi, z_cut=z_cut)
    print(f"\nSaved -> {out_dir}/closure_test.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
