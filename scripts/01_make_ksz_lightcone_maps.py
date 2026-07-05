#!/usr/bin/env python
"""
Script 01: Run lightcone simulation and compute kSZ map + D_ell.

Produces
--------
data/products/ksz_map_lightcone.npy         -- 2D delta_T/T map
data/products/ksz_Dl_lightcone.npz          -- ell, Dl, Dl_err [uK^2]
data/products/lightcone_reion_history.npz   -- z, xe, tau

Usage
-----
    python scripts/01_make_ksz_lightcone_maps.py
    python scripts/01_make_ksz_lightcone_maps.py --angle 0    # unrotated
    python scripts/01_make_ksz_lightcone_maps.py --patchy     # patchy tau
    python scripts/01_make_ksz_lightcone_maps.py --config configs/local.yaml
"""

import argparse, os
import yaml
import numpy as np
import py21cmfast as p21c

from ksz_pipeline.ksz.optical_depth      import compute_tau, compute_visibility
from ksz_pipeline.ksz.skewed_los         import make_los_grid, extract_skewers, RotatedLightcone
from ksz_pipeline.ksz.patchy_screening   import compute_patchy_tau
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl


def main(config_path, angle_deg=None, use_patchy=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lc_cfg  = cfg['lightcone_ksz']
    sim_cfg = cfg['21cmfast']

    if angle_deg is None:
        angle_deg  = lc_cfg['skew_angle_deg']
    if use_patchy is None:
        use_patchy = lc_cfg['use_patchy_screening']

    # Run lightcone
    print(f"Running lightcone: BOX_LEN={sim_cfg['BOX_LEN']} Mpc, "
          f"HII_DIM={sim_cfg['HII_DIM']}, "
          f"z={sim_cfg['z_min']}-{sim_cfg['z_max']}")

    user_params = p21c.UserParams(
        HII_DIM   = sim_cfg['HII_DIM'],
        BOX_LEN   = sim_cfg['BOX_LEN'],
        N_THREADS = sim_cfg['N_THREADS'],
    )
    lightcone = p21c.run_lightcone(
        redshift             = sim_cfg['z_min'],
        max_redshift         = sim_cfg['z_max'],
        lightcone_quantities = ('brightness_temp', 'density', 'xH_box', 'velocity'),
        user_params          = user_params,
        random_seed          = sim_cfg['random_seed'],
        direc                = cfg['data']['cache_dir'],
    )

    # Trim to z_max
    red_axis = lightcone.lightcone_redshifts
    pos_axis = lightcone.lightcone_distances
    ind_z    = np.where(red_axis <= sim_cfg['z_max'])[0]
    red_axis = red_axis[ind_z]
    pos_axis = pos_axis[ind_z]

    # Optical depth
    x_e_nodes  = 1.0 - lightcone.global_xH[::-1]
    z_nodes    = lightcone.node_redshifts[::-1]
    x_e_interp = np.interp(red_axis, z_nodes, x_e_nodes)
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, red_axis, pos_axis)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, red_axis, z_mid)

    # Skewed ray extraction
    Ndim      = int(user_params.HII_DIM)
    Lbox      = float(user_params.BOX_LEN)
    cell_size = Lbox / Ndim
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    pos_arr  = np.array(pos_axis, dtype=np.float64)
    s_axis   = pos_arr - pos_arr[0]
    delta_z  = float(pos_arr[1] - pos_arr[0])
    Nbins    = len(pos_arr)

    Delta_3d = np.array(lightcone.density[:, :, ind_z],  dtype=np.float32) + 1.0
    xHI_3d   = np.array(lightcone.xH_box[:, :, ind_z],   dtype=np.float32)
    vel_3d   = np.array(lightcone.velocity[:, :, ind_z],  dtype=np.float32) / 67.4

    LOS_ind = make_los_grid(lc_cfg['Nlos'], Ndim)
    lc_unrot, lc_rot = extract_skewers(
        Delta_3d, xHI_3d, vel_3d, LOS_ind,
        s_axis, delta_z, cell_size, Ndim, Nbins, cos_a, sin_a)

    lc_active = lc_rot if angle_deg > 0 else lc_unrot
    lightcone = RotatedLightcone(lightcone,
                                  lc_active['density'],
                                  lc_active['xH_box'],
                                  lc_active['velocity'])

    # Patchy screening
    if use_patchy:
        print("Computing patchy tau cube...")
        _, _, visibility_3D = compute_patchy_tau(
            lightcone.density, lightcone.xH_box, ind_z, ds, z_mid)

    # kSZ map
    density_1plus = 1.0 + lightcone.density[:, :, ind_z]
    x_HII_field   = 1.0 - lightcone.xH_box[:, :, ind_z]
    v_los_Mpc_s   = lightcone.velocity[:, :, ind_z] / 67.4

    print("Integrating kSZ map...")
    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               red_axis, ds, visibility_3D)

    print("Computing D_ell...")
    ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, Lbox)

    # Save
    os.makedirs("data/products", exist_ok=True)
    np.save("data/products/ksz_map_lightcone.npy", ksz_map)
    np.savez("data/products/ksz_Dl_lightcone.npz",
             ell=ell, Dl=Dl, Dl_err=Dl_err)
    np.savez("data/products/lightcone_reion_history.npz",
             z=red_axis, xe=x_e_interp, tau=tau_at_lc)

    print(f"Saved to data/products/")
    print(f"  kSZ map RMS : {np.sqrt(np.mean(ksz_map**2)):.4e}")
    print(f"  D_3000      : {float(np.interp(3000, ell, Dl)):.4e} uK^2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--angle",  type=float, default=None)
    parser.add_argument("--patchy", action="store_true")
    args = parser.parse_args()
    main(args.config, angle_deg=args.angle,
         use_patchy=args.patchy if args.patchy else None)
