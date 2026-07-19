#!/usr/bin/env python
"""
Script 13: stitched-lightcone dz (snapshot-interpolation-density)
convergence.

Tests the coherent-addition hypothesis directly: does stitched's D_ell
depend on how many real coeval snapshots get interpolated between to
build the continuous LOS field? If D_3000 shrinks toward coeval-direct's
value (1.7822 uK^2, trusted fiducial) as interpolation is reduced (denser
snapshot sampling, dz_x1 = finest = all 29 snapshots), that supports an
interpolation-artifact explanation for stitched's excess power -- the
~80-pixel-wide interpolated stretches between real snapshots aren't
independent samples, so coherent (constructive) addition in the real-space
sum could be manufacturing power that wouldn't exist with genuinely
independent samples. If D_3000 stays high regardless of dz, that points
toward the excess being real physics coeval-direct's incoherent
(power-summed, not field-summed) Limber approach simply cannot capture.

Uses run_dz_sweep_stitched (convergence/dz_sweep.py) -- written and
patched for N_THREADS/random_seed earlier this session but never wired to
a driver script until now. Reuses the main fiducial cache_dir (not a
sweep-specific subdirectory), so this should be cache-hit and fast: same
BOX_LEN=800/HII_DIM=512 as the already-completed fiducial run.

dz_x1 (finest, all 29 snapshots) uses the exact same config as the
fiducial stitched run (job B) -- its D_3000 should reproduce 4.0111 uK^2
if caching/seeding are both working as expected. A mismatch there would
be a red flag about this script, not about the physics question it's
testing.

Usage
-----
    python scripts/13_convergence_dz_stitched.py --config configs/fiducial.yaml
"""
import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.dz_sweep import run_dz_sweep_stitched
from ksz_pipeline.plotting.convergence_plots import plot_d3000_convergence


def main(config_path, angle_deg=0.0):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    conv_cfg = cfg['convergence']
    sim_cfg  = cfg['21cmfast']
    z_fine   = cfg['coeval_ksz']['z_snapshots']
    dz_multiples = conv_cfg['dz_multiples']

    BOX_LEN = sim_cfg['BOX_LEN']
    HII_DIM = sim_cfg['HII_DIM_coeval']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    cache_dir = cfg['data']['cache_dir']   # main cache -- same as fiducial, deliberate
    out_dir   = cfg['data']['output_dir'].rstrip('/')
    plot_dir  = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Stitched dz convergence: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM}, "
          f"dz_multiples={dz_multiples} (of a {len(z_fine)}-point fine grid), "
          f"angle_deg={angle_deg}")
    print("Testing the coherent-addition/interpolation-artifact hypothesis: "
          "does D_3000 shrink toward coeval-direct's 1.7822 uK^2 as "
          "interpolation between snapshots is reduced?\n")

    results = run_dz_sweep_stitched(z_fine, dz_multiples, BOX_LEN, HII_DIM,
                                     z_min, z_max, cache_dir, angle_deg=angle_deg,
                                     N_THREADS=sim_cfg['N_THREADS'],
                                     random_seed=sim_cfg['random_seed'])

    print(f"\n{'dz label':<10} {'n_z':>6} {'D3000':>12}")
    print("-" * 32)
    D3000_vals, n_z_list = [], []
    save_dict = {}
    for i, m in enumerate(sorted(dz_multiples)):
        label = f"dz_x{m}"
        d3000 = results[label]['D3000']
        n_z = len([z for z in z_fine][::m])
        D3000_vals.append(d3000)
        n_z_list.append(n_z)
        print(f"{label:<10} {n_z:>6} {d3000:>12.4g}")
        save_dict[f"ell_{i}"]    = results[label]['ell']
        save_dict[f"Dl_{i}"]     = results[label]['Dl']
        save_dict[f"Dl_err_{i}"] = results[label]['Dl_err']

    summary_path = f"{out_dir}/convergence_dz_stitched.npz"
    save_dict.update(dz_multiples=sorted(dz_multiples), n_z=n_z_list, D3000=D3000_vals)
    np.savez(summary_path, **save_dict)
    print(f"\nSaved -> {summary_path}")

    plot_path = f"{plot_dir}/D3000_convergence_dz_stitched"
    plot_d3000_convergence(
        n_z_list, {'Stitched lightcone': D3000_vals},
        xlabel='Number of coeval snapshots interpolated from',
        title='Stitched dz (snapshot-interpolation) convergence',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")
    print(f"\nCompare to coeval-direct's D_3000 = 1.7822 uK^2 (trusted, seed-fixed fiducial)")
    print(f"dz_x1 should reproduce the fiducial stitched D_3000 = 4.0111 uK^2 "
          f"(same config as job B) -- a mismatch flags this script, not the physics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--angle-deg", type=float, default=0.0)
    args = parser.parse_args()
    main(args.config, angle_deg=args.angle_deg)
