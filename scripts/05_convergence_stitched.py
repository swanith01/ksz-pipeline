#!/usr/bin/env python
"""
Script 05: stitched-lightcone convergence (box-size or resolution).

Reuses convergence.stitched_sweep, which reuses ksz.stitch_from_coeval,
ksz.optical_depth, ksz.lightcone_integral unchanged -- the same pipeline
03_stitched_lightcone_crosscheck.py runs for one configuration, looped
here over a parameter list. Config needs the same `convergence:` section
as script 04.

Usage
-----
    python scripts/05_convergence_stitched.py --sweep boxsize    --config configs/fiducial.yaml
    python scripts/05_convergence_stitched.py --sweep resolution --config configs/fiducial.yaml
"""

import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.stitched_sweep import run_sweep
from ksz_pipeline.convergence.param_lists import build_param_list
from ksz_pipeline.plotting.convergence_plots import plot_d3000_convergence


def main(config_path, sweep, angle_deg=0.0):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    conv_cfg    = cfg['convergence']
    sim_cfg     = cfg['21cmfast']
    z_snapshots = cfg['coeval_ksz']['z_snapshots']
    z_min, z_max = sim_cfg['z_min'], sim_cfg['z_max']
    cache_dir   = os.path.join(cfg['data']['cache_dir'], f"convergence_{sweep}_stitched")
    out_dir     = cfg['data']['output_dir'].rstrip('/')
    plot_dir    = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    param_list, xlabel, x_values = build_param_list(sweep, conv_cfg)
    print(f"Stitched-lightcone {sweep} convergence sweep: "
          f"{len(param_list)} configurations, angle_deg={angle_deg}")
    for L, N, tag in param_list:
        print(f"  {tag}: BOX_LEN={L} Mpc, HII_DIM={N}, dx={L/N:.3f} Mpc")

    results = run_sweep(param_list, z_snapshots, z_min, z_max, cache_dir,
                         angle_deg=angle_deg, N_THREADS=sim_cfg['N_THREADS'],
                         random_seed=sim_cfg['random_seed'])

    print(f"\n{'tag':<16} {'n_lc_pix':>10} {'D3000':>12}")
    print("-" * 40)
    D3000_vals = []
    for L, N, tag in param_list:
        d3000 = results[tag]['D3000']
        D3000_vals.append(d3000)
        print(f"{tag:<16} {results[tag]['n_lc_pix']:>10} {d3000:>12.4g}")

    summary_path = f"{out_dir}/convergence_stitched_{sweep}.npz"
    np.savez(summary_path, x_values=x_values, D3000=D3000_vals,
             tags=[p[2] for p in param_list])
    print(f"\nSaved -> {summary_path}")

    plot_path = f"{plot_dir}/D3000_convergence_stitched_{sweep}"
    plot_d3000_convergence(
        x_values, {'Stitched lightcone': D3000_vals},
        xlabel=xlabel, title=f'Stitched-lightcone {sweep} convergence',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--sweep", choices=["boxsize", "resolution"], required=True)
    parser.add_argument("--angle-deg", type=float, default=0.0)
    args = parser.parse_args()
    main(args.config, args.sweep, angle_deg=args.angle_deg)
