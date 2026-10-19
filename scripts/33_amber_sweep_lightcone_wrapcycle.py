"""
scripts/33_amber_sweep_lightcone_wrapcycle.py

Reproduces scripts/40's Chen2023 Fig.10 sweep comparison with a THIRD,
independent method: the stitched-lightcone path (scripts/32,
amber_lightcone_to_ksz_Dl) with wrap-cycle rotation now enabled, run on
the SAME already-validated AMBER run directories scripts/40/30 already
produced. Pure post-processing of existing output/cmb + output/reion
data -- no new AMBER execution, no new runs/ directories.

Why wrap-cycle matters here specifically: the Chen2023-sweep box
(L=256 Mpc/h, h=0.6766 -> ~378 Mpc) is far shorter than the comoving
distance a full czmin..czmax lightcone needs, so the stitched lightcone
wraps the same periodic AMBER box many times over. Without wrap-cycle,
stitch_field reuses the identical 2D slab every time the LOS returns to
a given y_cell (a single fixed angle_deg does not break that -- see
stitch_from_coeval's module docstring and
tests/test_amber_adapter.py::test_wrap_cycle_breaks_periodic_repeat),
biasing the result at the box-length scale. --wrap-cycle-seed (default
123, matching docs/HANDOFF_wrap_cycle.md's reference table) turns this
on for every point here.

Point identification: reads <out-root>/sweep_results.npz (already
written by scripts/40) to get the EXACT set and order of points that
passed validation for each scenario, by label -- not re-derived from
amber_gate.npz's mere presence, which scripts/40 itself does not treat
as sufficient for a point to count as validated on a fresh run (a
failed GATE 1 with a since-written GATE 2 npz is exactly the kind of
stale-looking-complete case that would silently diverge from scripts/40
if presence alone were trusted). Iterates scripts/40's own
CHEN2023_FIG10_SCENARIOS (imported directly, not copied, so this can
never drift out of sync with that file) in the same order, keeping only
points whose label appears in sweep_results.npz for that scenario --
which reproduces scripts/40's own filtered order exactly, since both
scripts process the same canonical point list and only differ in
whether a label survives the filter.

Usage
-----
    python scripts/33_amber_sweep_lightcone_wrapcycle.py --out-root runs/sweep

Writes <out-root>/sweep_results_lightcone.npz, with per-scenario index
i aligned to sweep_results.npz's own index i (same scenario names, same
point order) so scripts/41 can zip the two together directly.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

sweep40 = importlib.import_module('40_run_amber_sweep')

from ksz_pipeline.amber import io
from ksz_pipeline.amber.stitch_from_amber import (
    stitch_lightcone_from_amber, build_los_z_grid, amber_lightcone_to_ksz_Dl)
from ksz_pipeline.utils.constants import ne0_cgs


def _kept_labels(d, name):
    """Labels sweep_results.npz actually kept for scenario `name`, in
    their original saved order (== scripts/40's own filtered order)."""
    n = int(d[f'{name}_n'])
    return [str(d[f'{name}_{i}_label']) for i in range(n)]


def _run_one_point(run_dir, margin, wrap_cycle_seed):
    """amber_lightcone_to_ksz_Dl result for one AMBER run dir."""
    meta_path = os.path.join(run_dir, 'amber_run.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} missing -- this point's directory doesn't look "
            f"like a scripts/40 output (run that script first for this "
            f"--out-root)")

    meta = json.load(open(meta_path))
    N, L_h, h = meta['N'], meta['L'], meta['amber_params']['h']
    cmb_dir = os.path.join(run_dir, 'output', 'cmb')
    reion_dir = os.path.join(run_dir, 'output', 'reion')

    zre = io.read_zre(io.find_zre_file(reion_dir), N)
    field_files = io.list_field_files(cmb_dir)
    snap_z = np.array(sorted(meta['snapshot_z']))

    cell_size = (L_h / h) / N
    m = margin if margin is not None else 0.5 * np.diff(snap_z).min()
    z_min, z_max = snap_z.min() + m, snap_z.max() - m
    z_arr = build_los_z_grid(z_min, z_max, cell_size)

    lc = stitch_lightcone_from_amber(field_files, zre, z_arr, N, L_h, h,
                                     interp=True,
                                     wrap_cycle_seed=wrap_cycle_seed)
    tau_table = np.loadtxt(os.path.join(cmb_dir, 'tau.txt'), skiprows=1)
    ne0 = meta.get('ne0_cgs', ne0_cgs())
    return amber_lightcone_to_ksz_Dl(lc, tau_table, L_h / h, ne0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='runs/sweep',
                    help='SAME --out-root scripts/40 was run with -- '
                         'this reads its run directories and its '
                         'sweep_results.npz, creates no new AMBER runs')
    ap.add_argument('--wrap-cycle-seed', type=int, default=123,
                    help='default 123 matches docs/HANDOFF_wrap_cycle.md')
    ap.add_argument('--margin', type=float, default=None,
                    help='same meaning as scripts/32 --margin')
    ap.add_argument('--scenarios-json', default=None,
                    help='must match whatever scripts/40 was actually '
                         'run with, if not the built-in Chen2023 scenarios')
    a = ap.parse_args()

    results_npz = os.path.join(a.out_root, 'sweep_results.npz')
    if not os.path.exists(results_npz):
        raise SystemExit(
            f"{results_npz} not found -- run scripts/40_run_amber_sweep.py "
            f"--out-root {a.out_root} first (this script post-processes "
            f"its already-validated run directories, it does not run AMBER)")
    d40 = np.load(results_npz)

    scenarios = (json.load(open(a.scenarios_json)) if a.scenarios_json
                else sweep40.CHEN2023_FIG10_SCENARIOS)

    payload = {}
    for name, scen in scenarios.items():
        print(f"\n=== Scenario '{name}' ({scen['axis_label']}) ===", flush=True)
        kept_labels = set(_kept_labels(d40, name))
        kept = []
        for p in scen['points']:
            if p['label'] not in kept_labels:
                print(f"  [{p['label']}] skipped -- not in {results_npz} "
                      f"(scripts/40 excluded it, gate failure or never run)",
                      flush=True)
                continue
            tag = sweep40.tag_for(p)
            run_dir = sweep40.cfg_dir(a.out_root, tag)
            # Printed and flushed BEFORE the (potentially very slow --
            # wrap-cycle lightcone stitching over thousands of LOS pixels,
            # not a cheap box-based sum) call, not just after, so a run
            # that gets killed mid-point (e.g. walltime) still leaves a
            # log saying which point it was stuck on, instead of nothing.
            print(f"  [{tag}] starting...", flush=True)
            t0 = time.time()
            result = _run_one_point(run_dir, a.margin, a.wrap_cycle_seed)
            dt = time.time() - t0
            i3 = int(np.argmin(np.abs(result['ell'] - 3000)))
            print(f"  [{tag}] done in {dt:.0f}s -- D_3000 = "
                  f"{result['Dl'][i3]:.4f} uK^2  chi_eff = "
                  f"{result['chi_eff']:.1f} Mpc", flush=True)
            kept.append((p, result))

        payload[f'{name}_axis'] = scen['axis']
        payload[f'{name}_n'] = len(kept)
        for i, (p, r) in enumerate(kept):
            payload[f'{name}_{i}_ell'] = r['ell']
            payload[f'{name}_{i}_Dlc'] = r['Dl']
            payload[f'{name}_{i}_Dlc_err'] = r['Dl_err']
            payload[f'{name}_{i}_chi_eff'] = r['chi_eff']
            payload[f'{name}_{i}_axisval'] = p[scen['axis']]
            payload[f'{name}_{i}_label'] = p['label']

        if len(kept) != int(d40[f'{name}_n']):
            print(f"  NOTE: kept {len(kept)}/{int(d40[f'{name}_n'])} points "
                  f"that scripts/40 kept -- a run directory for one of "
                  f"them must be missing or incomplete; check the skip/"
                  f"error messages above before trusting this scenario's "
                  f"comparison as apples-to-apples.")

    out = os.path.join(a.out_root, 'sweep_results_lightcone.npz')
    np.savez(out, **payload)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
