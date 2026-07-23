#!/usr/bin/env python
"""
Script 13: stitched-lightcone dz (snapshot-interpolation-density)
convergence.

Tests the coherent-addition hypothesis directly: does stitched's D_ell
depend on how many real coeval snapshots get interpolated between to
build the continuous LOS field? If D_3000 shrinks toward coeval-direct's
matched-window value as interpolation is reduced (denser snapshot
sampling, dz_x1 = finest = all 29 snapshots), that supports an
interpolation-artifact explanation for stitched's excess power -- the
~80-pixel-wide interpolated stretches between real snapshots aren't
independent samples, so coherent (constructive) addition in the real-space
sum could be manufacturing power that wouldn't exist with genuinely
independent samples. If D_3000 stays high regardless of dz, that points
toward the excess being real physics coeval-direct's incoherent
(power-summed, not field-summed) Limber approach simply cannot capture.

CHANGE (2026-07-22): chi_eff and the unified (matched) window are now
LOADED from closure_test.npz (script 14) rather than each dz variant
computing its own native window and using the 7800 chi_Mpc default.
This is deliberate: letting chi or the window drift per-variant would
let the sweep's D_3000 differences reflect those confounds rather than
purely the interpolation-density variable this script exists to test.
NEW DEPENDENCY: this script now requires scripts/14_closure_test.py to
have been run first (closure_test.npz must exist). chi_eff and the
window were both correct even in the pre-bugfix run (job 1684054) --
they were never affected by the chi_end_reionization labeling bug -- so
this does NOT require waiting for job 1684234's corrected rerun.

Uses run_dz_sweep_stitched (convergence/dz_sweep.py) -- written and
patched for N_THREADS/random_seed earlier this session but never wired to
a driver script until now. Reuses the main fiducial cache_dir (not a
sweep-specific subdirectory), so this should be cache-hit and fast: same
BOX_LEN=800/HII_DIM=512 as the already-completed fiducial run.

dz_x1 (finest, all 29 snapshots) uses the exact same config as script 14's
closure test -- its D_3000 should reproduce closure_test.npz's own
Dl_stitched at ell=3000 if caching/seeding/window/chi are all working as
expected. A mismatch there would be a red flag about this script, not
about the physics question it's testing.

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

    # Load the unified window + chi_eff from the closure test (script 14) --
    # reused here, not recomputed, so this sweep is tested against the
    # EXACT SAME reference frame as the closure test, not a second,
    # independently-derived (and possibly drifting) chi_eff/window.
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
    d3000_stitched_matched_chi_eff = float(np.interp(3000, closure['ell_stitched'], closure['Dl_stitched']))

    print(f"Stitched dz convergence: BOX_LEN={BOX_LEN} Mpc, HII_DIM={HII_DIM}, "
          f"dz_multiples={dz_multiples} (of a {len(z_fine)}-point fine grid), "
          f"angle_deg={angle_deg}")
    print(f"Using matched window z=[{z_lo:.2f}, {z_hi:.2f}] and chi_eff={chi_eff:.1f} Mpc "
          f"from closure_test.npz (script 14) -- FIXED across all dz variants, "
          f"so this sweep isolates snapshot-interpolation density only.")
    print("Testing the coherent-addition/interpolation-artifact hypothesis: "
          f"does D_3000 shrink toward coeval-direct's matched-window value "
          f"({d3000_direct_matched:.4g} uK^2) as interpolation between snapshots is reduced?\n")

    results = run_dz_sweep_stitched(z_fine, dz_multiples, BOX_LEN, HII_DIM,
                                     z_min, z_max, cache_dir, angle_deg=angle_deg,
                                     N_THREADS=sim_cfg['N_THREADS'],
                                     random_seed=sim_cfg['random_seed'],
                                     z_lo=z_lo, z_hi=z_hi, chi_Mpc=chi_eff)

    print(f"\n{'dz label':<10} {'n_z':>6} {'D3000':>12} {'ksz_map_rms':>14}")
    print("-" * 46)
    D3000_vals, n_z_list, rms_vals = [], [], []
    save_dict = {}
    for i, m in enumerate(sorted(dz_multiples)):
        label = f"dz_x{m}"
        d3000 = results[label]['D3000']
        rms = results[label]['ksz_map_rms']
        n_z = len([z for z in z_fine][::m])
        D3000_vals.append(d3000)
        n_z_list.append(n_z)
        rms_vals.append(rms)
        print(f"{label:<10} {n_z:>6} {d3000:>12.4g} {rms:>14.6e}")
        save_dict[f"ell_{i}"]    = results[label]['ell']
        save_dict[f"Dl_{i}"]     = results[label]['Dl']
        save_dict[f"Dl_err_{i}"] = results[label]['Dl_err']

    # ksz_map_rms is computed on the real-space map BEFORE any window/chi
    # post-processing -- an independent, chi-blind check on whether the
    # dz variants' underlying maps genuinely differ. If these come back
    # exactly/suspiciously identical across dz_x1/x2/x4, that's evidence
    # of the caching bug coeval_sweep.py's docstring describes (results
    # instead of results_subset used somewhere), not real convergence.
    if len(set(f"{r:.10e}" for r in rms_vals)) < len(rms_vals):
        print("\nWARNING: two or more dz variants have IDENTICAL ksz_map_rms "
              "to 10 significant figures -- check for a caching bug (e.g. "
              "the full snapshot cache being used instead of the requested "
              "subset) before trusting the D_ell convergence above.")

    summary_path = f"{out_dir}/convergence_dz_stitched.npz"
    save_dict.update(dz_multiples=sorted(dz_multiples), n_z=n_z_list, D3000=D3000_vals,
                      ksz_map_rms=rms_vals,
                      chi_eff=chi_eff, z_lo=z_lo, z_hi=z_hi,
                      d3000_direct_matched=d3000_direct_matched)
    np.savez(summary_path, **save_dict)
    print(f"\nSaved -> {summary_path}")

    plot_path = f"{plot_dir}/D3000_convergence_dz_stitched"
    plot_d3000_convergence(
        n_z_list, {'Stitched lightcone': D3000_vals},
        xlabel='Number of coeval snapshots interpolated from',
        title='Stitched dz (snapshot-interpolation) convergence, matched window',
        out_path=plot_path,
    )
    print(f"Saved -> {plot_path}.png")
    print(f"\nCompare to coeval-direct's matched-window D_3000 = {d3000_direct_matched:.4g} uK^2")
    print(f"dz_x1 should reproduce script 14's matched-window stitched D_3000 = "
          f"{d3000_stitched_matched_chi_eff:.4g} uK^2 (same config, same chi_eff, same window) "
          f"-- a mismatch flags this script, not the physics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fiducial.yaml")
    parser.add_argument("--angle-deg", type=float, default=0.0)
    args = parser.parse_args()
    main(args.config, angle_deg=args.angle_deg)
