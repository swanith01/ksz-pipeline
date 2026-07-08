#!/usr/bin/env python
"""
Script 06: coeval dz (redshift-sampling-density) convergence, for both
the direct/Cain-style measurement and the Georgiev Eq.10 reconstruction.

Uses coeval_ksz.z_snapshots as the finest reference grid; dz variants
are "every Nth point" subsamplings of it (convergence.dz_multiples in
the config), reusing the fine grid's cache so only it is ever actually
simulated -- see convergence/dz_sweep.py's module docstring.

Usage
-----
    python scripts/06_convergence_dz.py --config configs/fiducial.yaml
"""

import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.dz_sweep import run_dz_sweep_coeval
from ksz_pipeline.plotting.convergence_plots import plot_d3000_convergence


def main(config_path, force=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    conv_cfg = cfg['convergence']
    sim_cfg  = cfg['21cmfast']
    z_fine   = cfg['coeval_ksz']['z_snapshots']
    dz_multiples = conv_cfg['dz_multiples']

    BOX_LEN = sim_cfg['BOX_LEN']
    HII_DIM = sim_cfg['HII_DIM_coeval']
    cache_dir = os.path.join(cfg['data']['cache_dir'], "convergence_dz_coeval")
    out_dir   = cfg['data']['output_dir'].rstrip('/')
    plot_dir  = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Coeval dz convergence: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM}, "
          f"dz_multiples={dz_multiples} (of a {len(z_fine)}-point fine grid)")

    results = run_dz_sweep_coeval(z_fine, dz_multiples, BOX_LEN, HII_DIM,
                                   cache_dir, tag="dz_fiducial", force=force)

    print(f"\n{'dz label':<10} {'n_z':>6} {'D3000_direct':>14} {'D3000_georgiev':>16}")
    print("-" * 48)
    D3000_direct, D3000_georgiev, n_z_list = [], [], []
    for m in sorted(dz_multiples):
        label = f"dz_x{m}"
        d = results[label]['D3000_direct']
        g = results[label]['D3000_georgiev']
        n_z = len([z for z in z_fine][::m])
        D3000_direct.append(d)
        D3000_georgiev.append(g)
        n_z_list.append(n_z)
        print(f"{label:<10} {n_z:>6} {d:>14.4g} {g:>16.4g}")

    summary_path = f"{out_dir}/convergence_dz_coeval.npz"
    np.savez(summary_path, dz_multiples=sorted(dz_multiples), n_z=n_z_list,
             D3000_direct=D3000_direct, D3000_georgiev=D3000_georgiev)
    print(f"\nSaved -> {summary_path}")

    plot_path = f"{plot_dir}/D3000_convergence_dz_coeval"
    plot_d3000_convergence(
        n_z_list,
        {'Direct (Cain)': D3000_direct, 'Georgiev (Eq.10)': D3000_georgiev},
        xlabel='Number of redshift snapshots',
        title='Coeval dz convergence',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, force=args.force)
