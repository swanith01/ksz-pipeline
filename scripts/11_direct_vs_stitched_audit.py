#!/usr/bin/env python
"""
Script 11: direct-vs-stitched audit, at matched redshift checkpoints.

Per the advisor's request: compare coeval-direct and stitched using the
SAME redshift slices, via (1) per-slice weighted kSZ integrand, (2)
cumulative D_3000(<z), (3) map mean/RMS and a direct-vs-stitched
regression slope, (4) an explicit convention list. Georgiev-Eq10 is
DELIBERATELY excluded -- that's a separate audit of the Eq.10
normalization, not part of this comparison. Neither direct nor stitched
is assumed trusted going in; this is what establishes that, or doesn't.

IMPORTANT (fixed in this version): the coeval-direct cumulative
D_3000(<z) is built by computing each redshift's Limber weight ONCE on
the FULL 29-point grid, then summing already-correct per-z
contributions. An earlier version called compute_cell() repeatedly on
shrinking subsets -- np.gradient's spacing depends on neighboring points
in whatever array it's given, so that approach silently recomputed
different, not-directly-comparable weights at each checkpoint. This
version includes a self-check confirming the replicated per-z sum
matches compute_cell()'s own trusted D_3000 output when summed over all
29 points.

The two methods don't share a redshift grid natively (coeval-direct: 29
discrete z_snapshots; stitched: 2320 fine LOS pixels from
build_los_z_grid). Rather than expensively re-deriving coeval-direct at
2320 points, this records stitched's CUMULATIVE sum specifically at the
29 z_snapshots checkpoints coeval-direct already uses.

Deliberately separate from scripts 02/03 (frozen/trusted) and script 08.
Reuses run_coeval_fields, stitch_lightcone_from_coeval, compute_ksz_map,
compute_cell (for the self-check only), and the qperp_power.pkl cache
already written by script 02.

Produces
--------
data/products/audit_cumulative_d3000.npz  -- z checkpoints, cumulative
                                              D_3000(<z) for both methods
data/products/audit_per_slice.npz         -- per-slice contribution to
                                              D_3000 at each checkpoint,
                                              both methods
data/products/audit_field_regression.npz  -- flattened per-pixel arrays
                                              at one representative
                                              snapshot, plus fitted
                                              slope/intercept/r
data/products/audit_conventions.json      -- explicit convention record

Usage
-----
    python scripts/11_direct_vs_stitched_audit.py --config configs/fiducial.yaml
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
from ksz_pipeline.utils.constants import (MPC_CM, SIGMA_T, C_CGS, T_CMB_K,
                                           NE0_HYDROGEN_ONLY, ne0_cgs)


def _interp_loglog(xq, xp, fp):
    """Copied verbatim from limber.py's private helper -- not imported,
    to avoid depending on another module's internal implementation detail."""
    xp, fp = np.asarray(xp), np.asarray(fp)
    m = (xp > 0) & (fp > 0)
    lx, lf = np.log(xp[m]), np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


def per_z_D3000_contributions(results_qperp, ZS_asc, ne0=None, ell_eval=3000.0):
    """
    Mirrors limber.py's compute_cell EXACTLY (same tau0, helium ne0,
    xH_mean-based patchy filter), but returns the PER-Z additive
    contribution to D_ell at a single ell -- computed ONCE on the full
    z grid, so weights are consistent across every checkpoint.
    """
    if ne0 is None:
        ne0 = ne0_cgs()

    XHI_MIN_PATCHY, XHI_MAX_PATCHY = 1.0e-4, 1.0 - 1.0e-4
    ZS_kept = np.array(sorted([z for z in ZS_asc
                                if XHI_MIN_PATCHY <= results_qperp[z]['xH_mean'] <= XHI_MAX_PATCHY]))

    chi_mpc  = np.array([cosmo.comoving_distance(z).value for z in ZS_kept])
    dchi_mpc = np.abs(np.gradient(chi_mpc))
    dchi_cm  = dchi_mpc * MPC_CM
    xe_arr   = np.array([1.0 - results_qperp[z]['xH_mean'] for z in ZS_kept])

    tau0 = analytic_tau_below(ZS_kept.min())
    tau  = np.full_like(ZS_kept, tau0)
    for i in range(len(ZS_kept) - 1):
        zmid   = 0.5 * (ZS_kept[i] + ZS_kept[i + 1])
        xe_mid = 0.5 * (xe_arr[i]  + xe_arr[i + 1])
        tau[i + 1] = tau[i] + (SIGMA_T * ne0 * xe_mid * (1.0 + zmid) ** 2 * dchi_cm[i])

    pref  = (SIGMA_T * ne0 / C_CGS) ** 2
    prefD = ell_eval * (ell_eval + 1.0) / (2.0 * np.pi) * T_CMB_K ** 2 * 1e12

    dD3000 = {}
    for i, z in enumerate(ZS_kept):
        s_mpc = chi_mpc[i]
        a_i   = 1.0 / (1.0 + z)
        vis2  = np.exp(-2.0 * tau[i])
        w     = vis2 / (s_mpc ** 2 * a_i ** 4) * dchi_mpc[i]
        k_ell = ell_eval / s_mpc
        P_now = _interp_loglog(np.array([k_ell]), results_qperp[z]['k'], results_qperp[z]['Pqperp'])[0]
        C_contrib = pref * w * P_now * MPC_CM ** 2
        dD3000[z] = prefD * C_contrib
    return dD3000, ZS_kept


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg    = cfg['21cmfast']
    coeval_cfg = cfg['coeval_ksz']
    cache_dir  = cfg['data']['cache_dir']
    HII_DIM    = sim_cfg['HII_DIM_coeval']
    BOX_LEN    = sim_cfg['BOX_LEN']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    ZS         = sorted(coeval_cfg['z_snapshots'])
    out_dir    = cfg['data']['output_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)

    # ================================================================
    # 4. EXPLICIT CONVENTION AUDIT
    # ================================================================
    ne0_coeval_direct = ne0_cgs()
    ne0_stitched      = NE0_HYDROGEN_ONLY
    ne0_ratio_sq      = (ne0_coeval_direct / ne0_stitched) ** 2

    conventions = {
        "scale_factor": {
            "coeval_direct": "a_i = 1/(1+z), weight uses a_i^4 in denominator (limber.py compute_cell)",
            "stitched":      "a = 1/(1+red_axis), a_squared used as a2_mid (lightcone_integral.py compute_ksz_map)",
            "note": "same functional form; NOT yet numerically cross-checked for equivalent net power of (1+z)",
        },
        "dchi_discretization": {
            "coeval_direct": f"np.gradient over {len(ZS)} irregular z_snapshots (astropy Planck18 comoving_distance)",
            "stitched":      "np.diff over 2320 UNIFORM comoving-distance LOS pixels (build_los_z_grid)",
            "note": "fundamentally different grids -- checkpoint approach compares CUMULATIVE sums at matched z",
        },
        "speed_of_light": {
            "coeval_direct": "C_CGS (constants.py)", "stitched": "C_CGS (constants.py)",
            "note": "identical shared constant",
        },
        "velocity_convention": {
            "coeval_direct": "cm/s, enters momentum field q=w*v, squared in P_qperp",
            "stitched":      "cm/s -> Mpc/s, enters LINEARLY in the real-space temperature integrand (expected)",
            "note": "structurally consistent",
        },
        "electron_fraction_ne0": {
            "coeval_direct": f"ne0_cgs() [helium-inclusive] = {ne0_coeval_direct:.6e} cm^-3",
            "stitched":      f"NE0_HYDROGEN_ONLY [never overridden by script 03] = {ne0_stitched:.6e} cm^-3",
            "ratio_squared_D_ell_impact": f"{ne0_ratio_sq:.4f} (~{100*(ne0_ratio_sq-1):.1f}% systematic)",
            "note": "documented in constants.py's own docstring as the 'primary source of the baseline offset'",
        },
    }
    print("=== Convention audit ===")
    print(json.dumps(conventions, indent=2))
    with open(f"{out_dir}/audit_conventions.json", "w") as f:
        json.dump(conventions, f, indent=2)
    print(f"\nSaved -> {out_dir}/audit_conventions.json\n")

    # ================================================================
    qperp_pkl = os.path.join(cache_dir, "qperp_power.pkl")
    with open(qperp_pkl, 'rb') as f:
        results_qperp_all = pickle.load(f)
    missing = [z for z in ZS if z not in results_qperp_all]
    if missing:
        raise RuntimeError(f"qperp_power.pkl missing redshifts {missing} -- run script 02 first")

    # ================================================================
    # 1 & 2. Coeval-direct: per-z contributions computed ONCE on the
    #    full grid (the fix), then cumulative-summed.
    # ================================================================
    print("=== Coeval-direct: per-z D_3000 contributions (single-pass, full-grid weights) ===")
    dD3000_direct, ZS_kept_direct = per_z_D3000_contributions(results_qperp_all, ZS)
    for z in ZS_kept_direct:
        print(f"  z={z:5.1f}: dD_3000 = {dD3000_direct[z]:.5f} uK^2")

    # Self-check: does summing every per-z contribution match
    # compute_cell's own trusted D_3000 output?
    ells_check, D_ell_check, *_ = compute_cell(results_qperp_all)
    d3000_trusted = float(np.interp(3000, ells_check, D_ell_check))
    d3000_summed  = float(sum(dD3000_direct.values()))
    print(f"\nSelf-check: compute_cell's trusted D_3000 = {d3000_trusted:.4f} uK^2")
    print(f"            summed per-z replication        = {d3000_summed:.4f} uK^2")
    if not np.isclose(d3000_trusted, d3000_summed, rtol=1e-3):
        print("  WARNING: replication does NOT match trusted output closely -- "
              "do not trust the cumulative curve below until this is resolved.")
    else:
        print("  MATCH -- replication is faithful, cumulative curve below is trustworthy.")

    cumulative_direct = []
    running = 0.0
    for z in ZS:
        running += dD3000_direct.get(z, 0.0)  # 0 contribution if outside the patchy xH_mean window
        cumulative_direct.append(running)
    cumulative_direct = np.array(cumulative_direct)

    # ================================================================
    print("\n=== Stitched: rebuilding full lightcone (cache-hit) for cumulative truncation ===")
    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    stitched = stitch_lightcone_from_coeval(
        z_snapshots=coeval_cfg['z_snapshots'], z_arr=z_arr,
        HII_DIM=HII_DIM, BOX_LEN=BOX_LEN, cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'],
    )
    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM
    x_e_interp    = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis      = stitched['pos_axis']
    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)

    print("=== Stitched: cumulative D_3000(<z) at the SAME checkpoints ===")
    cumulative_stitched = []
    for z_checkpoint in ZS:
        idx = np.searchsorted(z_arr, z_checkpoint)
        if idx < 2:
            cumulative_stitched.append(0.0)
            continue
        ksz_map_trunc = compute_ksz_map(
            density_1plus[:, :, :idx], x_HII_field[:, :, :idx], v_los_Mpc_s[:, :, :idx],
            z_arr[:idx], ds[:idx-1], visibility_3D[:, :, :idx],
            patchy_mask_3D=patchy_mask_3D[:, :, :idx],
        )
        ell_t, Dl_t, _ = ksz_map_to_Dl(ksz_map_trunc, BOX_LEN)
        d3000 = float(np.interp(3000, ell_t, Dl_t)) if len(ell_t) else 0.0
        cumulative_stitched.append(d3000)
        print(f"  z<={z_checkpoint:5.1f} ({idx:4d}/{len(z_arr)} LOS pixels): D_3000(<z) = {d3000:.4f} uK^2")

    cumulative_stitched = np.array(cumulative_stitched)
    per_slice_direct    = np.diff(cumulative_direct,   prepend=0.0)
    per_slice_stitched  = np.diff(cumulative_stitched, prepend=0.0)

    np.savez(f"{out_dir}/audit_cumulative_d3000.npz", z=np.array(ZS),
             cumulative_direct=cumulative_direct, cumulative_stitched=cumulative_stitched)
    np.savez(f"{out_dir}/audit_per_slice.npz", z=np.array(ZS),
             per_slice_direct=per_slice_direct, per_slice_stitched=per_slice_stitched)
    print(f"\nSaved -> {out_dir}/audit_cumulative_d3000.npz")
    print(f"Saved -> {out_dir}/audit_per_slice.npz")

    # ================================================================
    # 3. Field-level regression at one representative snapshot
    # ================================================================
    z_rep = min(ZS, key=lambda z: abs(z - 7.0))
    print(f"\n=== Field-level regression at z={z_rep} (nearest available to 7.0) ===")
    print("  NOTE: 'direct' here collapses the coeval box's own 3rd axis as a "
          "thin-slice PROXY -- an approximation, not a rigorous derivation. "
          "Treat a surprising slope as worth investigating further, not as final.")

    delta_r, xH_r, vx_r, vy_r, vz_r = run_coeval_fields(
        z_rep, HII_DIM, BOX_LEN, cache_dir,
        N_THREADS=sim_cfg['N_THREADS'], random_seed=sim_cfg['random_seed'])

    # Seed-consistency self-check: this fresh call and the cached fiducial
    # run (results_qperp_all, loaded earlier from qperp_power.pkl) should
    # give BIT-IDENTICAL xH_mean at the same z if the seed is genuinely
    # consistent -- not just "should be by construction," actually verified.
    xH_mean_fresh  = float(xH_r.mean())
    xH_mean_cached = results_qperp_all[z_rep]['xH_mean']
    print(f"\nSeed-consistency check at z={z_rep}:")
    print(f"  xH_mean (fresh, this script's own call): {xH_mean_fresh:.8f}")
    print(f"  xH_mean (cached, from fiducial run):     {xH_mean_cached:.8f}")
    if abs(xH_mean_fresh - xH_mean_cached) < 1e-9:
        print("  MATCH -- seed is consistent between this script and the fiducial run.")
    else:
        print("  MISMATCH -- seed inconsistency detected. Do not trust this script's "
              "field-level comparisons until this is resolved.")

    density_1plus_r = 1.0 + delta_r
    x_HII_r         = 1.0 - xH_r
    v_los_r_Mpc_s   = vz_r / MPC_CM
    a_r             = 1.0 / (1.0 + z_rep)
    map_direct_2d = (density_1plus_r * x_HII_r * v_los_r_Mpc_s / (C_CGS / MPC_CM)) / a_r ** 2
    map_direct_2d = map_direct_2d.mean(axis=2)

    idx_rep = np.searchsorted(z_arr, z_rep)
    map_stitched_2d = density_1plus[:, :, idx_rep] * x_HII_field[:, :, idx_rep] * \
                       v_los_Mpc_s[:, :, idx_rep] / (C_CGS / MPC_CM) / a_r ** 2

    flat_direct   = map_direct_2d.ravel()
    flat_stitched = map_stitched_2d.ravel()
    slope, intercept = np.polyfit(flat_direct, flat_stitched, 1)
    r = np.corrcoef(flat_direct, flat_stitched)[0, 1]

    print(f"  direct   map: mean={map_direct_2d.mean():.4e}  rms={np.sqrt(np.mean(map_direct_2d**2)):.4e}")
    print(f"  stitched map: mean={map_stitched_2d.mean():.4e}  rms={np.sqrt(np.mean(map_stitched_2d**2)):.4e}")
    print(f"  regression (stitched vs direct): slope={slope:.4f}  intercept={intercept:.4e}  r={r:.4f}")

    np.savez(f"{out_dir}/audit_field_regression.npz",
             z_rep=z_rep, flat_direct=flat_direct, flat_stitched=flat_stitched,
             slope=slope, intercept=intercept, r=r)
    print(f"\nSaved -> {out_dir}/audit_field_regression.npz")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
