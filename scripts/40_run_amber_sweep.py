"""
scripts/40_run_amber_sweep.py

Runs a set of AMBER configurations at FIXED box size and resolution,
varying only the reionization-history parameters, and validates each one
(scripts/30_amber_validation_gate.py) before trusting it. Two scenarios,
per the original handoff's step 5:

  A: z_mid FIXED, Delta_z varying   -- a clean relabeling of the SAME
     ionization pattern (see reionization.f90: the density/radiation
     field ranking cells is evaluated once, at z_mid; only the ranking
     ->redshift lookup changes with Delta_z). Isolates duration.
  B: Delta_z FIXED, z_mid varying   -- NOT a pure relabeling: the ranking
     field is re-evaluated at the new z_mid each time, so morphology
     changes along with timing. Don't over-read this as a "duration only"
     sweep.

Resumable: a config already at amber_gate.npz is read, not rerun --
lets this reuse an existing, already-validated run (e.g. amber_q01) for
whatever point the two scenarios share, and lets this script be
re-launched after a partial failure without redoing finished work.
Map-making is deliberately OFF for every point here (no --mapmake) --
this compares the direct/P_qperp method only, and map-making's walltime
cost (see docs/amber_integration.md) has nothing to do with this question.

Usage:
  python scripts/40_run_amber_sweep.py --amber-x ~/amber/src/amber.x \\
      --reuse runs/amber_q01 --out-root runs/sweep

Writes runs/sweep/sweep_results.npz for scripts/41_plot_amber_sweep.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def cfg_dir(root, tag):
    return os.path.join(root, tag)


def tag_for(zmid, zdel):
    return f"zmid{zmid:04.1f}_dz{zdel:04.1f}".replace('.', 'p')


REQUIRED_NPZ_KEYS = {'ells', 'D_compute_cell', 'D_amber_window', 'window',
                    'hist_z', 'hist_xH_vol'}


def _npz_is_complete(npz_path):
    """False if the cached amber_gate.npz predates a schema change (e.g.
    an existing run from before hist_z/hist_xH_vol were added) -- such a
    file needs the gate rerun, not a KeyError at plot time."""
    try:
        keys = set(np.load(npz_path).files)
    except Exception:
        return False
    return REQUIRED_NPZ_KEYS.issubset(keys)


def _glob_fields(d):
    import glob
    return glob.glob(os.path.join(d, 'output', 'cmb', 'fields_*.dat'))


def run_one(root, zmid, zdel, zasy, L, N, ncore, amber_x, python_exe,
           reuse_dir=None, timeout=600):
    """Ensure one config exists and is validated. Returns dict of results
    or None if it failed (input generation, AMBER, or the gate)."""
    tag = tag_for(zmid, zdel)
    d = reuse_dir if reuse_dir else cfg_dir(root, tag)
    npz_path = os.path.join(d, 'amber_gate.npz')

    if os.path.exists(npz_path) and _npz_is_complete(npz_path):
        print(f"  [{tag}] already validated at {d} -- reusing")
        return _read_result(d, npz_path, zmid, zdel)

    skip_amber = os.path.isdir(os.path.join(d, 'output', 'cmb')) and \
        len(_glob_fields(d)) > 0
    if skip_amber:
        print(f"  [{tag}] {d} has AMBER output but no (or stale-schema) "
             f"amber_gate.npz -- rerunning the gate only, not amber.x")
    else:
        print(f"  [{tag}] generating input ...")
        os.makedirs(d, exist_ok=True)
        gen = subprocess.run(
            [python_exe, os.path.join(HERE, 'make_amber_input.py'),
             '--out', d, '--L', str(L), '--N', str(N),
             '--zmid', str(zmid), '--zdel', str(zdel), '--zasy', str(zasy),
             '--ncore', str(ncore)],
            capture_output=True, text=True)
        if gen.returncode != 0:
            print(f"  [{tag}] INPUT GENERATION FAILED:\n{gen.stderr[-2000:]}")
            return None

        for sub in ('cosmo', 'reion', 'grf', 'lpt', 'esf', 'mesh', 'cmb'):
            os.makedirs(os.path.join(d, 'output', sub), exist_ok=True)

        print(f"  [{tag}] running amber.x ...")
        t0 = time.time()
        with open(os.path.join(d, 'input', 'input.txt')) as fin, \
             open(os.path.join(d, 'output', 'log.txt'), 'w') as fout:
            try:
                r = subprocess.run([amber_x], stdin=fin, stdout=fout,
                                  stderr=subprocess.STDOUT, timeout=timeout,
                                  cwd=d)
            except subprocess.TimeoutExpired:
                print(f"  [{tag}] TIMED OUT after {timeout}s")
                return None
        dt = time.time() - t0
        log = open(os.path.join(d, 'output', 'log.txt')).read()
        if r.returncode != 0 or 'AMBER completed' not in log:
            print(f"  [{tag}] AMBER FAILED (exit {r.returncode}, {dt:.0f}s) -- "
                 f"see {d}/output/log.txt")
            print('    ' + log.strip().splitlines()[-1] if log.strip() else
                 '    (empty log)')
            return None
        print(f"  [{tag}] amber.x completed in {dt:.0f}s")

    print(f"  [{tag}] running validation gate ...")
    gate = subprocess.run(
        [python_exe, os.path.join(HERE, '30_amber_validation_gate.py'),
         '--run', d],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.path.join(HERE, '..', 'src')})
    print('    ' + '\n    '.join(gate.stdout.strip().splitlines()[-6:]))
    if gate.returncode != 0:
        print(f"  [{tag}] GATE FAILED -- point excluded from results "
             f"(full output in {d}/gate_output.txt)")
        open(os.path.join(d, 'gate_output.txt'), 'w').write(
            gate.stdout + gate.stderr)
        return None

    return _read_result(d, npz_path, zmid, zdel)


def _read_result(d, npz_path, zmid, zdel):
    z = np.load(npz_path)
    return dict(dir=d, zmid=zmid, zdel=zdel,
               ells=z['ells'], D_cc=z['D_compute_cell'],
               D_win=z['D_amber_window'], window=z['window'],
               hist_z=z['hist_z'], hist_xH_vol=z['hist_xH_vol'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amber-x', required=True)
    ap.add_argument('--out-root', default='runs/sweep')
    ap.add_argument('--reuse', default=None,
                    help='existing validated run dir for the shared '
                         '(zmid_fixed, zdel_fixed) point, e.g. runs/amber_q01')
    ap.add_argument('--L', type=float, default=256.0)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--zasy', type=float, default=3.0)
    ap.add_argument('--ncore', type=int, default=16)
    ap.add_argument('--zmid-fixed', type=float, default=8.0)
    ap.add_argument('--zdel-fixed', type=float, default=4.0)
    ap.add_argument('--zdel-values', type=float, nargs='+',
                    default=[2.0, 4.0, 6.0, 8.0],
                    help='scenario A: Delta_z values at zmid-fixed')
    ap.add_argument('--zmid-values', type=float, nargs='+',
                    default=[6.0, 7.0, 8.0, 9.0, 10.0],
                    help='scenario B: z_mid values at zdel-fixed')
    ap.add_argument('--timeout', type=int, default=600,
                    help='seconds per amber.x call (no-map runs are '
                         'fast; generous default, see docs)')
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    amber_x = os.path.expanduser(a.amber_x)
    if not os.path.isfile(amber_x):
        raise SystemExit(f"amber.x not found at {amber_x}")

    def reuse_if_matches(zmid, zdel):
        return (a.reuse if (a.reuse and abs(zmid - a.zmid_fixed) < 1e-9
                            and abs(zdel - a.zdel_fixed) < 1e-9) else None)

    results = {'A': [], 'B': []}
    print(f"=== Scenario A: z_mid={a.zmid_fixed} fixed, "
         f"Delta_z in {a.zdel_values} ===")
    for zdel in a.zdel_values:
        res = run_one(a.out_root, a.zmid_fixed, zdel, a.zasy, a.L, a.N,
                      a.ncore, amber_x, a.python,
                      reuse_dir=reuse_if_matches(a.zmid_fixed, zdel),
                      timeout=a.timeout)
        if res:
            results['A'].append(res)

    print(f"\n=== Scenario B: Delta_z={a.zdel_fixed} fixed, "
         f"z_mid in {a.zmid_values} ===")
    for zmid in a.zmid_values:
        res = run_one(a.out_root, zmid, a.zdel_fixed, a.zasy, a.L, a.N,
                      a.ncore, amber_x, a.python,
                      reuse_dir=reuse_if_matches(zmid, a.zdel_fixed),
                      timeout=a.timeout)
        if res:
            results['B'].append(res)

    out = os.path.join(a.out_root, 'sweep_results.npz')
    payload = {}
    for scen in ('A', 'B'):
        payload[f'{scen}_zmid'] = np.array([r['zmid'] for r in results[scen]])
        payload[f'{scen}_zdel'] = np.array([r['zdel'] for r in results[scen]])
        # ells grids match across a scenario only if L,N are fixed (they
        # are, by construction) and the patchy window is similar; still
        # verify before stacking, since window WILL differ across z_mid
        for i, r in enumerate(results[scen]):
            payload[f'{scen}_{i}_ells'] = r['ells']
            payload[f'{scen}_{i}_Dcc'] = r['D_cc']
            payload[f'{scen}_{i}_Dwin'] = r['D_win']
            payload[f'{scen}_{i}_window'] = r['window']
            payload[f'{scen}_{i}_hist_z'] = r['hist_z']
            payload[f'{scen}_{i}_hist_xH'] = r['hist_xH_vol']
        payload[f'{scen}_n'] = len(results[scen])
    np.savez(out, **payload)
    meta = dict(L=a.L, N=a.N, zasy=a.zasy, zmid_fixed=a.zmid_fixed,
               zdel_fixed=a.zdel_fixed)
    json.dump(meta, open(os.path.join(a.out_root, 'sweep_meta.json'), 'w'))
    print(f"\nwrote {out}  (A: {len(results['A'])}/{len(a.zdel_values)} "
         f"points, B: {len(results['B'])}/{len(a.zmid_values)} points)")
    failed_A = len(a.zdel_values) - len(results['A'])
    failed_B = len(a.zmid_values) - len(results['B'])
    if failed_A or failed_B:
        print(f"WARNING: {failed_A + failed_B} point(s) failed and were "
             f"excluded -- see per-config output/log.txt or gate_output.txt")


if __name__ == '__main__':
    main()
