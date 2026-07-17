#!/usr/bin/env python
"""
Script 04: coeval-box convergence (box-size or resolution), for both
the direct/Cain-style P_qperp measurement and the Georgiev Eq.10
reconstruction.

Reuses convergence.coeval_sweep, which reuses coeval.fields,
coeval.momentum, coeval.pee_pvv_pev, coeval.georgiev_convolution,
coeval.limber unchanged. Config needs a `convergence:` section -- see
the block below and the accompanying yaml diff.

Usage
-----
    python scripts/04_convergence_coeval.py --sweep boxsize   --config configs/fiducial.yaml
    python scripts/04_convergence_coeval.py --sweep resolution --config configs/fiducial.yaml
    python scripts/04_convergence_coeval.py --sweep boxsize --config configs/quicktest.yaml --force
"""

import argparse
import os

import numpy as np
import yaml

from ksz_pipeline.convergence.coeval_sweep import run_sweep
from ksz_pipeline.convergence.param_lists import build_param_list
from ksz_pipeline.plotting.convergence_plots import plot_d3000_convergence


def main(config_path, sweep, force=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    conv_cfg   = cfg['convergence']
    z_snapshots = cfg['coeval_ksz']['z_snapshots']
    sim_cfg    = cfg['21cmfast']
    if sweep == "resolution":
        # ref_box_len (800 Mpc) and the largest hii_dims entry (512) match
        # the fiducial run's own BOX_LEN/HII_DIM_coeval exactly -- that
        # point IS the already-completed fiducial run. Point at the SAME
        # cache dir so it's read from disk, not regenerated. py21cmfast's
        # cache is keyed by parameter hash, so the sweep's other HII_DIM
        # configs just add new files here safely -- no collision.
        cache_dir = cfg['data']['cache_dir']
    else:
        cache_dir = os.path.join(cfg['data']['cache_dir'], f"convergence_{sweep}_coeval")
    out_dir    = cfg['data']['output_dir'].rstrip('/')
    plot_dir   = cfg['data']['plot_dir'].rstrip('/')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    param_list, xlabel, x_values = build_param_list(sweep, conv_cfg)
    print(f"Coeval {sweep} convergence sweep: {len(param_list)} configurations")
    for L, N, tag in param_list:
        print(f"  {tag}: BOX_LEN={L} Mpc, HII_DIM={N}, dx={L/N:.3f} Mpc")

    results = run_sweep(param_list, z_snapshots, cache_dir, force=force,
                         N_THREADS=sim_cfg['N_THREADS'])

    # -- summary table --
    print(f"\n{'tag':<16} {'D3000_direct':>14} {'D3000_georgiev':>16}")
    print("-" * 48)
    D3000_direct, D3000_georgiev = [], []
    for L, N, tag in param_list:
        d = results[tag]['D3000_direct']
        g = results[tag]['D3000_georgiev']
        D3000_direct.append(d)
        D3000_georgiev.append(g)
        print(f"{tag:<16} {d:>14.4g} {g:>16.4g}")

    summary_path = f"{out_dir}/convergence_coeval_{sweep}.npz"
    np.savez(summary_path, x_values=x_values,
             D3000_direct=D3000_direct, D3000_georgiev=D3000_georgiev,
             tags=[p[2] for p in param_list])
    print(f"\nSaved -> {summary_path}")

    plot_path = f"{plot_dir}/D3000_convergence_coeval_{sweep}"
    plot_d3000_convergence(
        x_values,
        {'Direct (Cain)': D3000_direct, 'Georgiev (Eq.10)': D3000_georgiev},
        xlabel=xlabel,
        title=f'Coeval {sweep} convergence',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--sweep", choices=["boxsize", "resolution"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.sweep, force=args.force)
