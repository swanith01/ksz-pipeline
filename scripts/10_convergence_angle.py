#!/usr/bin/env python
"""
Script 10: angle-of-stitching convergence -- sweeps angle_deg at FIXED
fiducial BOX_LEN/HII_DIM/z_snapshots, to test whether the low-ell spike
seen in the stitched D_ell (ell~130-180, up to ~62x the coeval-direct
value at those ell) is a periodic-replication artifact from stitching
at angle_deg=0.0. See stitch_from_coeval.py's module docstring: 0 was
chosen deliberately for the FIRST cross-check, specifically to avoid
conflating this path's own potential issues with skewed_los.py's known
rotation bug -- this script is the follow-up that actually varies it.

Reuses stitched_sweep.run_one_config directly (no new core function
needed -- angle_deg was already a parameter there). Since BOX_LEN/HII_DIM/
z_snapshots match the fiducial config exactly, this reuses the EXISTING
fiducial py21cmfast cache -- only the stitch+map+FFT step repeats per
angle, no new box generation.

CHANGE (2026-07-23): now loads chi_eff and the unified window from
closure_test.npz (script 14), same pattern as scripts 05/13, and passes
them to run_one_config for EVERY angle. Previously used stitched's own
native window and the hardcoded chi_Mpc=7800 default -- same convention
mismatch already fixed elsewhere. NEW DEPENDENCY: requires
scripts/14_closure_test.py to have been run first.

Produces
--------
data/products/convergence_angle_stitched.npz -- angle_deg, D3000,
                                                  ksz_map_rms per angle,
                                                  plus full D_ell arrays

Usage
-----
    python scripts/10_convergence_angle.py --config configs/fiducial.yaml
    python scripts/10_convergence_angle.py --angles 0,15,30,45,60,75,90
"""
import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.stitched_sweep import run_one_config


def main(config_path, angles):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg     = cfg['21cmfast']
    z_snapshots = cfg['coeval_ksz']['z_snapshots']
    BOX_LEN     = sim_cfg['BOX_LEN']
    HII_DIM     = sim_cfg['HII_DIM_coeval']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    cache_dir   = cfg['data']['cache_dir']   # SAME as fiducial -- deliberate, see module docstring
    out_dir     = cfg['data']['output_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)

    # Load the unified window + chi_eff from the closure test (script 14) --
    # same reasoning as scripts 05/13: reused, not recomputed, so every
    # angle is tested against the exact same reference frame.
    closure_path = f"{out_dir}/closure_test.npz"
    if not os.path.exists(closure_path):
        raise FileNotFoundError(
            f"{closure_path} not found -- run scripts/14_closure_test.py first. "
            "This script now reuses its chi_eff and unified window rather than "
            "computing its own, so it depends on that output existing."
        )
    closure = np.load(closure_path)
    chi_eff = float(closure['chi_eff'])
    z_lo, z_hi = float(closure['z_lo']), float(closure['z_hi'])
    d3000_direct_matched = float(np.interp(3000, closure['ell_direct'], closure['Dl_direct']))

    print(f"Angle-of-stitching sweep: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM} "
          f"(fiducial values -- reusing fiducial cache), angles={angles} deg")
    print(f"Using matched window z=[{z_lo:.2f}, {z_hi:.2f}] and chi_eff={chi_eff:.1f} Mpc "
          f"from closure_test.npz (script 14) -- FIXED across every angle.")

    results = {}
    for angle in angles:
        tag = f"angle_{angle:g}"
        print(f"=== angle_deg={angle} ===", flush=True)
        r = run_one_config(BOX_LEN, HII_DIM, z_snapshots, z_min, z_max,
                            cache_dir, tag, angle_deg=float(angle),
                            N_THREADS=sim_cfg['N_THREADS'],
                            random_seed=sim_cfg['random_seed'],
                            z_lo=z_lo, z_hi=z_hi, chi_Mpc=chi_eff)
        results[angle] = r
        print(f"  D_3000={r['D3000']:.4g} uK^2  ksz_map_rms={r['ksz_map_rms']:.4e}", flush=True)

    print(f"\n{'angle_deg':>10} {'D_3000':>12} {'ksz_map_rms':>14}")
    print("-" * 38)
    for angle in angles:
        r = results[angle]
        print(f"{angle:>10} {r['D3000']:>12.4g} {r['ksz_map_rms']:>14.4e}")

    save_dict = {"angles": np.array(angles),
                 "D3000": np.array([results[a]['D3000'] for a in angles]),
                 "ksz_map_rms": np.array([results[a]['ksz_map_rms'] for a in angles]),
                 "chi_eff": chi_eff, "z_lo": z_lo, "z_hi": z_hi,
                 "d3000_direct_matched": d3000_direct_matched}
    for i, angle in enumerate(angles):
        save_dict[f"ell_{i}"] = results[angle]['ell']
        save_dict[f"Dl_{i}"]  = results[angle]['Dl']

    summary_path = f"{out_dir}/convergence_angle_stitched.npz"
    np.savez(summary_path, **save_dict)
    print(f"\nSaved -> {summary_path}")
    print("(NOTE: this OVERWRITES any earlier angle sweep at this path -- if it "
          "included points under the OLD convention, those numbers are superseded, "
          "not preserved alongside these.)")
    print("\nCheck specifically: does the ell~130-180 low-ell spike shrink "
          "or shift as angle_deg increases from 0? A shrinking/shifting "
          "spike would support the periodic-replication hypothesis; an "
          "angle-independent spike would point elsewhere.")
    print(f"\nCompare to coeval-direct's matched-window D_3000 = {d3000_direct_matched:.4g} uK^2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--angles", default="0,15,30,45,60,75,90",
                         help="comma-separated list of angle_deg values")
    args = parser.parse_args()
    angle_list = [float(a) for a in args.angles.split(",")]
    main(args.config, angle_list)
