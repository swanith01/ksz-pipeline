#!/usr/bin/env python
"""
Script 05: stitched-lightcone convergence (box-size or resolution).

Reuses convergence.stitched_sweep, which reuses ksz.stitch_from_coeval,
ksz.optical_depth, ksz.lightcone_integral unchanged -- the same pipeline
03_stitched_lightcone_crosscheck.py runs for one configuration, looped
here over a parameter list. Config needs the same `convergence:` section
as script 04.

CHANGE (2026-07-23): now loads chi_eff and the unified window from
closure_test.npz (script 14), same pattern as script 13, and passes them
through to run_sweep for EVERY point in the sweep. Previously this used
each point's own native window and the hardcoded chi_Mpc=7800 default --
same convention mismatch already fixed in script 13. Since run_sweep
recomputes post-processing for the WHOLE param_list each time (cheap --
only the underlying py21cmfast simulation is expensive/cached), adding a
new resolution point (e.g. 1024) automatically regenerates ALL points
consistently under the corrected convention, not just the new one.
NEW DEPENDENCY: requires scripts/14_closure_test.py to have been run
first (closure_test.npz must exist) -- same as script 13.

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

    # Load the unified window + chi_eff from the closure test (script 14) --
    # same reasoning as script 13: reused, not recomputed, so every
    # resolution/boxsize point is tested against the exact same reference
    # frame instead of each computing its own (possibly drifting) window/chi.
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

    param_list, xlabel, x_values = build_param_list(sweep, conv_cfg)
    print(f"Stitched-lightcone {sweep} convergence sweep: "
          f"{len(param_list)} configurations, angle_deg={angle_deg}")
    print(f"Using matched window z=[{z_lo:.2f}, {z_hi:.2f}] and chi_eff={chi_eff:.1f} Mpc "
          f"from closure_test.npz (script 14) -- FIXED across every point in this sweep.")
    for L, N, tag in param_list:
        print(f"  {tag}: BOX_LEN={L} Mpc, HII_DIM={N}, dx={L/N:.3f} Mpc")

    results = run_sweep(param_list, z_snapshots, z_min, z_max, cache_dir,
                         angle_deg=angle_deg, N_THREADS=sim_cfg['N_THREADS'],
                         random_seed=sim_cfg['random_seed'],
                         z_lo=z_lo, z_hi=z_hi, chi_Mpc=chi_eff)

    print(f"\n{'tag':<16} {'n_lc_pix':>10} {'D3000':>12}")
    print("-" * 40)
    D3000_vals = []
    for L, N, tag in param_list:
        d3000 = results[tag]['D3000']
        D3000_vals.append(d3000)
        print(f"{tag:<16} {results[tag]['n_lc_pix']:>10} {d3000:>12.4g}")

    summary_path = f"{out_dir}/convergence_stitched_{sweep}_angle{angle_deg:g}.npz"
    save_dict = dict(x_values=x_values, D3000=D3000_vals,
                      tags=[p[2] for p in param_list],
                      chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi,
                      d3000_direct_matched=d3000_direct_matched)
    # Full D_ell curves were already computed by run_one_config -- save
    # them too, not just the summary D_3000 (added 19Jul2026, so future
    # convergence plots can show full curves, not just a single point).
    for i, (L, N, tag) in enumerate(param_list):
        save_dict[f"ell_{i}"]     = results[tag]['ell']
        save_dict[f"Dl_{i}"]      = results[tag]['Dl']
        save_dict[f"Dl_err_{i}"]  = results[tag]['Dl_err']
    np.savez(summary_path, **save_dict)
    print(f"\nSaved -> {summary_path}")
    print(f"(NOTE: this OVERWRITES any earlier {sweep} sweep at this path -- if it "
          f"included points under the OLD convention, those numbers are superseded, "
          f"not preserved alongside these.)")

    plot_path = f"{plot_dir}/D3000_convergence_stitched_{sweep}_angle{angle_deg:g}"
    plot_d3000_convergence(
        x_values, {'Stitched lightcone': D3000_vals},
        xlabel=xlabel, title=f'Stitched-lightcone {sweep} convergence, matched window',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")
    print(f"\nCompare to coeval-direct's matched-window D_3000 = {d3000_direct_matched:.4g} uK^2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--sweep", choices=["boxsize", "resolution"], required=True)
    parser.add_argument("--angle-deg", type=float, default=0.0)
    args = parser.parse_args()
    main(args.config, args.sweep, angle_deg=args.angle_deg)
