#!/usr/bin/env python
"""
Script 03: kSZ D_ell from a lightcone stitched out of coeval boxes.

Independent cross-check of 01_make_ksz_lightcone_maps.py's D_ell, using
ONLY already-validated pieces: coeval box generation + velocity handling
(coeval/fields.run_coeval_fields, confirmed twice today) stitched into a
lightcone (ksz/stitch_from_coeval.py, geometry-tested), then run through
the SAME compute_ksz_map/ksz_map_to_Dl used by script 01. Bypasses
skewed_los.py entirely.

Read before running: density_1plus = 1.0 + stitched['density'] happens
ONCE, explicitly, below -- because stitch_from_coeval's output is RAW
delta (matching run_coeval_fields' convention), unlike the native
lightcone where lightcone.density is already (1+delta) after rotation.
Getting this wrong is exactly the bug class already hit and fixed once
this session; don't "fix" the line below without re-reading
stitch_from_coeval.py's module docstring first.

Produces
--------
<output_dir>/ksz_Dl_stitched.npz   -- ell, Dl [uK^2]  (no formal error bars
                                       yet -- see note at save time below)

Usage
-----
    python scripts/03_stitched_lightcone_crosscheck.py --config configs/quicktest.yaml
"""

import argparse
import numpy as np
import yaml

from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ksz_pipeline.utils.constants import MPC_CM, ne0_cgs


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg    = cfg['21cmfast']
    coeval_cfg = cfg['coeval_ksz']
    cache_dir  = cfg['data']['cache_dir']
    out_dir    = cfg['data']['output_dir'].rstrip('/')

    HII_DIM = sim_cfg['HII_DIM_coeval']
    BOX_LEN = sim_cfg['BOX_LEN']
    z_min   = sim_cfg['z_min']
    z_max   = sim_cfg['z_max']
    z_snapshots = coeval_cfg['z_snapshots']

    # LOS grid with UNIFORM COMOVING DISTANCE spacing == cell_size, NOT
    # uniform in z (dchi/dz varies ~8.5x over z=4-20 -- see
    # build_los_z_grid's docstring for the concrete check that found this).
    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    n_lc_pix = len(z_arr)

    print(f"Stitching lightcone from {len(z_snapshots)} coeval snapshots "
          f"onto {n_lc_pix} LOS pixels, z={z_min}-{z_max}, "
          f"HII_DIM={HII_DIM}, BOX_LEN={BOX_LEN} Mpc")
    print("angle_deg=0 (unrotated) -- see stitch_from_coeval.py docstring for why")

    stitched = stitch_lightcone_from_coeval(
        z_snapshots=z_snapshots, z_arr=z_arr,
        HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
        cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg['N_THREADS'],
        random_seed=sim_cfg['random_seed'],
    )

    # density_1plus: exactly ONE "+1" -- stitched['density'] is raw delta.
    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    # velocity_z from run_coeval_fields is cm/s; compute_ksz_map wants Mpc/s.
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM

    print(f"  [check] mean(density_1plus)={density_1plus.mean():.4f}  "
          f"(want ~1, i.e. raw delta + 1 -- NOT ~2, which would mean the "
          f"'+1' above is double-counting)")
    print(f"  [check] rms(v_los_Mpc_s)={np.sqrt(np.mean(v_los_Mpc_s**2)):.4e} Mpc/s  "
          f"(want ~1e-17 to 1e-18, same target as script 01)")

    x_e_interp = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis   = stitched['pos_axis']

    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)

    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)
    z_patchy = z_arr[patchy_mask.astype(bool)]
    if z_patchy.size > 0:
        print(f"  Patchy regime: z = {z_patchy.max():.2f} -> {z_patchy.min():.2f} "
              f"({z_patchy.size}/{n_lc_pix} slices kept)")

    print("Integrating kSZ map...")
    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               z_arr, ds, visibility_3D,
                               ne0=ne0_cgs(),  # FIXED 19Jul2026: was silently defaulting to
                               # NE0_HYDROGEN_ONLY, a ~0.4% mismatch vs coeval-direct's
                               # helium-inclusive convention -- now explicit and consistent
                               patchy_mask_3D=patchy_mask_3D)
    print(f"  ksz_map shape: {ksz_map.shape}  (should be ({HII_DIM},{HII_DIM}), "
          f"no reshape needed -- unlike script 01, there's no stray "
          f"leading dim here since these are proper 3D arrays throughout)")

    # Signal-weighted effective comoving distance for the ell<->k
    # conversion, replacing the previous hardcoded chi_Mpc=7800 default
    # (confirmed too low -- see docs/validation_table.md). Weighted by
    # the RMS (not mean -- kSZ is a fluctuation field around zero;
    # mean-weighting would be near-meaningless) of each LOS slice's own
    # contribution, using the SAME tau/visibility/patchy_mask weighting
    # compute_ksz_map itself already uses -- not a separately-invented
    # weighting scheme.
    integrand_for_weight = density_1plus * x_HII_field * v_los_Mpc_s
    if patchy_mask_3D is not None:
        integrand_for_weight = integrand_for_weight * patchy_mask_3D
    integrand_for_weight = integrand_for_weight * visibility_3D
    slice_rms = np.sqrt(np.mean(integrand_for_weight**2, axis=(0, 1)))
    chi_eff = float(np.sum(slice_rms * pos_axis) / np.sum(slice_rms))
    print(f"  [chi_eff] signal-weighted effective comoving distance = "
          f"{chi_eff:.1f} Mpc (previous hardcoded default was 7800 Mpc)")

    print("Computing D_ell...")
    ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, BOX_LEN, chi_Mpc=chi_eff)

    import os
    os.makedirs(out_dir, exist_ok=True)
    # No formal error bars from this path yet (Dl_err here is just
    # ksz_map_to_Dl's own per-bin FFT scatter, same as script 01 -- not
    # an independent uncertainty estimate on the stitching itself).
    np.savez(f"{out_dir}/ksz_Dl_stitched.npz", ell=ell, Dl=Dl, Dl_err=Dl_err)

    print(f"Saved to {out_dir}/ksz_Dl_stitched.npz")
    print(f"  kSZ map RMS : {np.sqrt(np.mean(ksz_map**2)):.4e}")
    print(f"  D_3000      : {float(np.interp(3000, ell, Dl)):.4e} uK^2  "
          f"(compare against script 01's lightcone D_3000 and script 02's "
          f"coeval D_3000 for the SAME config)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
