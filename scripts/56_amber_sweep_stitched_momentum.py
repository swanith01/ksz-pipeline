"""
scripts/56_amber_sweep_stitched_momentum.py

Corrected replacement for scripts/33's stitched-lightcone curves on an
AMBER sweep (e.g. runs/sweep_chen2023, 256^3, L=256 Mpc/h). Same points,
same patchy-window D_ell for scripts/41's dotted curves, but built with
ksz_pipeline.amber.stream_lightcone instead of stitch_lightcone_from_amber,
fixing the three problems found on 2026-10-26:

  1. v = mom/rho  -> mode='momentum' stitches q = x_HII * mom directly.
     AMBER's interlaced+deconvolved rho goes negative (to -2.8 at z~5 in
     the 1536^3 run); dividing by it inflated D_ell x10-60 at high ell.
  2. bilinear wrap-cycle rotation -> geometry='reorient': per pass through
     the periodic box, a random box axis becomes the LOS (its momentum
     component) plus integer periodic shifts. Exact, no interpolation;
     rotation smoothed high ell to 0.80 / 0.59 at ell = 5000 / 8000.
     Needed here: the z=5-18 LOS is ~7.4 box lengths of this sweep's box.
  3. Planck18 distances -> chi(z) from AMBER's own H(z) for each run's
     cosmology (scripts/55's los_grid).

Points are identified exactly as scripts/33 does (labels kept in
<out-root>/sweep_results.npz, scripts/40's canonical order). Writes
<out-root>/sweep_results_lightcone_<tag>.npz -- same keys scripts/41
reads (<name>_<i>_ell/_Dlc/_label, ...) plus total (z>=zmin) spectra
under _ell_total/_Dl_total -- and leaves scripts/33's original
sweep_results_lightcone.npz untouched. Plot with:

    python scripts/41_plot_amber_sweep.py --run <out-root> \
        --lightcone-npz <out-root>/sweep_results_lightcone_<tag>.npz \
        --suffix _<tag>

Needs raw output/cmb/fields_*.dat for every point (pure post-processing,
no AMBER runs). ~1 GB RAM.
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
los_grid = importlib.import_module('55_amber_stitched_streaming').los_grid

from ksz_pipeline.amber import io
from ksz_pipeline.amber.stream_lightcone import stream_ksz_maps, open_zre
from ksz_pipeline.ksz.lightcone_integral import ksz_map_to_Dl
from ksz_pipeline.utils.constants import ne0_cgs


def _kept_labels(d, name):
    n = int(d[f'{name}_n'])
    return [str(d[f'{name}_{i}_label']) for i in range(n)]


def run_point(run_dir, zmin, seed, mode, geometry):
    meta = json.load(open(os.path.join(run_dir, 'amber_run.json')))
    N, ap = meta['N'], meta['amber_params']
    L = meta['L'] / ap['h']
    cmb = os.path.join(run_dir, 'output', 'cmb')
    files = io.list_field_files(cmb)
    if len(files) < 2:
        raise FileNotFoundError(f"{cmb}: no fields_*.dat (pruned?)")
    snap_z = np.array([io.z_from_name(f) for f in files])
    z_arr, pos = los_grid(max(zmin, snap_z.min()), snap_z.max(), L / N, ap)
    tau = np.loadtxt(os.path.join(cmb, 'tau.txt'), skiprows=1)
    zre = open_zre(io.find_zre_file(os.path.join(run_dir, 'output', 'reion')), N)
    r = stream_ksz_maps(files, zre, N, L, z_arr, pos, tau,
                        meta.get('ne0_cgs', ne0_cgs()), seed,
                        mode=mode, geometry=geometry)
    if r['xe_slices'][0] < 1 - 1e-4:
        print(f"    NOTE: x_e={r['xe_slices'][0]:.5f} at the LOS start "
              f"z={z_arr[0]:.2f} -- reionization not complete there, so the "
              f"patchy window is truncated at the low-z end", flush=True)
    lp, Dp, Ep = ksz_map_to_Dl(r['map_patchy'], L, chi_Mpc=r['chi_eff_patchy'])
    lt, Dt, Et = ksz_map_to_Dl(r['map_total'], L, chi_Mpc=r['chi_eff_total'])
    return dict(ell=lp, Dl=Dp, Dl_err=Ep, chi_eff=r['chi_eff_patchy'],
                ell_total=lt, Dl_total=Dt, Dl_err_total=Et,
                chi_eff_total=r['chi_eff_total'],
                n_box=(pos[-1] - pos[0]) / L, window=r['z_window_patchy'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='runs/sweep_chen2023')
    ap.add_argument('--zmin', type=float, default=5.0,
                    help='LOS start (scripts/33 effectively used 5.0 too)')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--mode', default='momentum', choices=['momentum', 'velocity'])
    ap.add_argument('--geometry', default='reorient', choices=['reorient', 'rotate'])
    ap.add_argument('--scenarios-json', default=None)
    a = ap.parse_args()
    tag = f"{a.mode}_{a.geometry}"

    results_npz = os.path.join(a.out_root, 'sweep_results.npz')
    if not os.path.exists(results_npz):
        raise SystemExit(f"{results_npz} not found -- run scripts/40 first")
    d40 = np.load(results_npz)
    scenarios = (json.load(open(a.scenarios_json)) if a.scenarios_json
                 else sweep40.CHEN2023_FIG10_SCENARIOS)

    payload = {'method': f'stitched lightcone ({a.mode}, {a.geometry})'}
    cache = {}                       # the same run dir appears in >1 scenario
    for name, scen in scenarios.items():
        print(f"\n=== Scenario '{name}' ===", flush=True)
        keep = set(_kept_labels(d40, name))
        kept = []
        for p in scen['points']:
            if p['label'] not in keep:
                print(f"  [{p['label']}] skipped (not in {results_npz})", flush=True)
                continue
            t = sweep40.tag_for(p)
            if t not in cache:
                print(f"  [{t}] starting...", flush=True)
                t0 = time.time()
                cache[t] = run_point(sweep40.cfg_dir(a.out_root, t), a.zmin,
                                     a.seed, a.mode, a.geometry)
                r = cache[t]
                i3 = int(np.argmin(np.abs(r['ell'] - 3000)))
                print(f"  [{t}] done in {time.time() - t0:.0f}s -- LOS "
                      f"{r['n_box']:.1f} box lengths, patchy D_3000 = "
                      f"{r['Dl'][i3]:.4f} uK^2 (window {r['window']})", flush=True)
            else:
                print(f"  [{t}] reused from earlier scenario", flush=True)
            kept.append((p, cache[t]))

        payload[f'{name}_axis'] = scen['axis']
        payload[f'{name}_n'] = len(kept)
        for i, (p, r) in enumerate(kept):
            for k_out, k_in in (('ell', 'ell'), ('Dlc', 'Dl'), ('Dlc_err', 'Dl_err'),
                                ('chi_eff', 'chi_eff'), ('ell_total', 'ell_total'),
                                ('Dl_total', 'Dl_total'),
                                ('Dl_err_total', 'Dl_err_total')):
                payload[f'{name}_{i}_{k_out}'] = r[k_in]
            payload[f'{name}_{i}_axisval'] = p[scen['axis']]
            payload[f'{name}_{i}_label'] = p['label']

    out = os.path.join(a.out_root, f'sweep_results_lightcone_{tag}.npz')
    np.savez(out, **payload)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
