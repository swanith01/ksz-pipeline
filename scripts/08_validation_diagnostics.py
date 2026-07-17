#!/usr/bin/env python
"""
Script 08: Validation-table diagnostics -- field slices, kSZ map, and
per-path mean/RMS stats for the advisor's compact validation table.

Deliberately SEPARATE from scripts 02/03 (which are frozen/trusted) --
this only reads already-cached py21cmfast boxes (via the same
run_coeval_fields/stitch_lightcone_from_coeval used there) to extract
diagnostics, and does not touch or modify the qperp/Limber pipeline in
scripts 02/03 at all. If this script has a bug, it cannot silently
corrupt the trusted D_ell results those scripts already produced.

Produces
--------
data/products/coeval_field_stats.npz    -- per-redshift density/velocity/xH
                                            mean+RMS from the coeval boxes
                                            (coeval-direct has no 2D map by
                                            construction -- this is the
                                            closest comparable diagnostic)
data/products/stitched_field_slice.npz  -- one spatial slice of
                                            delta/v_los/xHI vs redshift,
                                            for a density-lightcone-style plot
data/products/ksz_map_stitched.npy      -- the full 2D stitched kSZ map,
                                            LOS-integrated over the full
                                            z_min-z_max range (previously
                                            computed in script 03 but never
                                            saved to disk)
data/products/validation_run_stats.json -- map mean/RMS, git commit,
                                            config path, patchy z-range --
                                            everything the validation table needs

Usage
-----
    python scripts/08_validation_diagnostics.py --config configs/fiducial.yaml
"""
import argparse, os, json, subprocess
import numpy as np
import yaml

from ksz_pipeline.coeval.fields import run_coeval_fields
from ksz_pipeline.ksz.stitch_from_coeval import stitch_lightcone_from_coeval, build_los_z_grid
from ksz_pipeline.ksz.optical_depth import (compute_tau, compute_visibility,
                                             analytic_tau_below, compute_patchy_mask)
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map
from ksz_pipeline.utils.constants import MPC_CM


def get_commit_hash():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return "unknown"


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg    = cfg['21cmfast']
    coeval_cfg = cfg['coeval_ksz']
    cache_dir  = cfg['data']['cache_dir']
    out_dir    = cfg['data']['output_dir'].rstrip('/')
    HII_DIM    = sim_cfg['HII_DIM_coeval']
    BOX_LEN    = sim_cfg['BOX_LEN']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    z_snapshots  = coeval_cfg['z_snapshots']

    commit_hash = get_commit_hash()
    print(f"Commit: {commit_hash}")
    print(f"Config: {os.path.abspath(config_path)}")

    # ---- 1. per-redshift coeval field stats (density/xH/velocity mean+RMS) ----
    # Hits py21cmfast's own on-disk cache (same seed, same params as scripts
    # 02/03) -- should NOT trigger fresh 21cmFAST runs if the fiducial run's
    # cache is still present at cache_dir.
    print(f"\nComputing per-redshift field stats for {len(z_snapshots)} redshifts "
          f"(reading cached boxes, not recomputing)...")
    zs_sorted = sorted(z_snapshots)
    delta_mean, delta_rms   = [], []
    xH_mean_arr, xH_rms_arr = [], []
    vz_mean, vz_rms         = [], []

    for z in zs_sorted:
        delta, xH, vx, vy, vz = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=sim_cfg['N_THREADS'],
            random_seed=sim_cfg['random_seed'])
        delta_mean.append(float(delta.mean()));  delta_rms.append(float(np.sqrt(np.mean(delta**2))))
        xH_mean_arr.append(float(xH.mean()));    xH_rms_arr.append(float(np.sqrt(np.mean(xH**2))))
        vz_mean.append(float(vz.mean()));        vz_rms.append(float(np.sqrt(np.mean(vz**2))))
        print(f"  z={z:5.1f}: <delta>={delta_mean[-1]:+.4f}  <xH>={xH_mean_arr[-1]:.4f}  "
              f"rms(vz)={vz_rms[-1]:.3e} cm/s")

    np.savez(f"{out_dir}/coeval_field_stats.npz",
             z=np.array(zs_sorted),
             delta_mean=np.array(delta_mean), delta_rms=np.array(delta_rms),
             xH_mean=np.array(xH_mean_arr),   xH_rms=np.array(xH_rms_arr),
             vz_mean_cm_s=np.array(vz_mean),  vz_rms_cm_s=np.array(vz_rms))
    print(f"Saved -> {out_dir}/coeval_field_stats.npz")

    # ---- 2. stitched lightcone: field slice + full kSZ map ----
    print("\nRebuilding stitched lightcone (cache-hit, for field slice + map)...")
    cell_size = BOX_LEN / HII_DIM
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    n_lc_pix = len(z_arr)

    stitched = stitch_lightcone_from_coeval(
        z_snapshots=z_snapshots, z_arr=z_arr,
        HII_DIM=HII_DIM, BOX_LEN=BOX_LEN,
        cache_dir=cache_dir, angle_deg=0.0,
        N_THREADS=sim_cfg['N_THREADS'],
        random_seed=sim_cfg['random_seed'],
    )
    density_1plus = 1.0 + stitched['density']
    x_HII_field   = 1.0 - stitched['xH_box']
    v_los_Mpc_s   = stitched['velocity_z'] / MPC_CM

    x_e_interp = 1.0 - stitched['xH_box'].mean(axis=(0, 1))
    pos_axis   = stitched['pos_axis']
    tau0 = analytic_tau_below(z_arr.min())
    z_mid, ds, dtau, tau = compute_tau(x_e_interp, z_arr, pos_axis, tau0=tau0)
    tau_at_lc, visibility, visibility_3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp)
    z_patchy = z_arr[patchy_mask.astype(bool)]

    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                               z_arr, ds, visibility_3D,
                               patchy_mask_3D=patchy_mask_3D)

    np.save(f"{out_dir}/ksz_map_stitched.npy", ksz_map)
    print(f"Saved -> {out_dir}/ksz_map_stitched.npy  shape={ksz_map.shape}")

    # A single spatial slice through the middle of the box (not an edge --
    # avoids any periodic-boundary artifact dominating the picture), giving
    # a (y, z_arr) slab matching the density_lightcone.png reference style.
    x_slice_idx = HII_DIM // 2
    np.savez(f"{out_dir}/stitched_field_slice.npz",
             delta=stitched['density'][x_slice_idx, :, :],   # raw delta, NOT 1+delta -- see stitch_from_coeval.py docstring
             v_los_Mpc_s=v_los_Mpc_s[x_slice_idx, :, :],
             xHI=stitched['xH_box'][x_slice_idx, :, :],
             z_arr=z_arr, box_len=BOX_LEN, x_slice_idx=x_slice_idx)
    print(f"Saved -> {out_dir}/stitched_field_slice.npz")

    # ---- 3. run stats / provenance, for the validation table ----
    stats = dict(
        commit=commit_hash,
        config_path=os.path.abspath(config_path),
        HII_DIM=int(HII_DIM), BOX_LEN=float(BOX_LEN),
        z_min=float(z_min), z_max=float(z_max),
        mean_density_1plus=float(density_1plus.mean()),
        rms_v_los_Mpc_s=float(np.sqrt(np.mean(v_los_Mpc_s**2))),
        ksz_map_mean=float(ksz_map.mean()),
        ksz_map_rms=float(np.sqrt(np.mean(ksz_map**2))),
        patchy_z_min=float(z_patchy.min()) if z_patchy.size > 0 else None,
        patchy_z_max=float(z_patchy.max()) if z_patchy.size > 0 else None,
        n_patchy_slices=int(z_patchy.size),
        n_total_slices=int(n_lc_pix),
    )
    with open(f"{out_dir}/validation_run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved -> {out_dir}/validation_run_stats.json")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    args = parser.parse_args()
    main(args.config)
