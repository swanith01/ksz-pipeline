"""
scripts/40_run_amber_sweep.py

Runs a set of AMBER configurations at FIXED box size, resolution and
cosmology, validates each one (scripts/30_amber_validation_gate.py)
before trusting it, and collects D_ell + reionization-history curves for
scripts/41_plot_amber_sweep.py.

A scenario is a NAMED list of points; each point is a full parameter
dict (zmid, zdel, zasy, Mmin, mfp), built by overriding a fiducial. This
(not two hardcoded scenarios) is needed even for a plain z_mid or Delta_z
sweep, because some comparisons need one point to differ in MORE than the
swept axis -- e.g. reproducing Chen, Trac, Mukherjee & Cen 2023's
(arXiv:2203.04337) Figure 10 middle panel, whose Delta_z=12.8 point also
uses z_mid=6.5, A_z=8, mfp=1 Mpc/h (their Sec 5.2), not the fiducial
z_mid/A_z/mfp the rest of that panel shares.

Built-in scenario definitions reproduce that paper's Figure 10 exactly
(see CHEN2023_FIG10_SCENARIOS below); pass --scenarios-json for anything
else. Their fiducial [z_mid, Delta_z, A_z, Mh, mfp] = [8.0, 4.0, 3.0,
1e8, 3.0] is also this repo's own fiducial (amber_q01) -- not a
coincidence worth re-deriving, just confirms that choice was reasonable.

Resumable: a config already at amber_gate.npz is read, not rerun -- lets
this reuse an existing, already-validated run (e.g. amber_q01) for
whatever point matches its exact parameters, and lets this script be
re-launched after a partial failure without redoing finished work.
Map-making is deliberately OFF for every point here (no --mapmake) --
this compares the direct/P_qperp method only, and map-making's walltime
cost (see docs/amber_integration.md) has nothing to do with this question.

Usage:
  python scripts/40_run_amber_sweep.py --amber-x ~/amber/src/amber.x \
      --reuse runs/amber_q01 --out-root runs/sweep --cosmology chen2022

Writes runs/sweep/sweep_results.npz for scripts/41_plot_amber_sweep.py.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FIDUCIAL = dict(zmid=8.0, zdel=4.0, zasy=3.0, Mmin=1e8, mfp=3.0)


def point(label, **overrides):
    p = dict(FIDUCIAL)
    p.update(overrides)
    p['label'] = label
    return p


# Chen, Trac, Mukherjee & Cen 2023 Figure 10: z_mid, Delta_z (+ one
# special-parameter point), and A_z panels, at their stated fiducial.
CHEN2023_FIG10_SCENARIOS = {
    'z_mid': {
        'axis': 'zmid', 'axis_label': r'z$_{mid}$',
        'points': [point(f"zmid={v}", zmid=v)
                  for v in [7.0, 7.5, 8.0, 8.5, 9.0]],
    },
    'delta_z': {
        'axis': 'zdel', 'axis_label': r'$\Delta z$',
        'points': [point(f"dz={v}", zdel=v) for v in [2, 3, 4, 5, 6]]
                 + [point("dz=12.8 (max, Sec 5.2)", zdel=12.8, zmid=6.5,
                          zasy=8.0, mfp=1.0)],
    },
    'asymmetry': {
        'axis': 'zasy', 'axis_label': r'A$_z$',
        'points': [point(f"Az={v}", zasy=v) for v in [1, 2, 3, 5, 8]],
    },
}


def tag_for(p):
    def fmt(x):
        return f"{x:g}".replace('.', 'p').replace('-', 'm')
    return (f"zmid{fmt(p['zmid'])}_dz{fmt(p['zdel'])}_az{fmt(p['zasy'])}"
           f"_mh{fmt(np.log10(p['Mmin']))}_mfp{fmt(p['mfp'])}")


def cfg_dir(root, tag):
    return os.path.join(root, tag)


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
    return glob.glob(os.path.join(d, 'output', 'cmb', 'fields_*.dat'))


def _matches_reuse(p, reuse_dir):
    """True if reuse_dir's own amber_run.json states EXACTLY this point's
    parameters -- not just "the fixed point of some scenario", so a
    Delta_z=12.8-style override with a different z_mid/A_z/mfp is never
    mistaken for the plain fiducial point sharing only zdel's fiducial."""
    if not reuse_dir:
        return False
    meta_path = os.path.join(reuse_dir, 'amber_run.json')
    if not os.path.exists(meta_path):
        return False
    m = json.load(open(meta_path))
    return all(abs(m.get(k, float('nan')) - p[k]) < 1e-9
              for k in ('zmid', 'zdel', 'zasy', 'Mmin', 'mfp'))


def run_one(root, p, L, N, ncore, cosmology, amber_x, python_exe,
           reuse_dir=None, timeout=600):
    """Ensure one config exists and is validated. Returns dict of results
    or None if it failed (input generation, AMBER, or the gate)."""
    tag = tag_for(p)
    d = reuse_dir if _matches_reuse(p, reuse_dir) else cfg_dir(root, tag)
    npz_path = os.path.join(d, 'amber_gate.npz')

    if os.path.exists(npz_path) and _npz_is_complete(npz_path):
        print(f"  [{tag}] already validated at {d} -- reusing")
        return _read_result(d, npz_path, p)

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
             '--zmid', str(p['zmid']), '--zdel', str(p['zdel']),
             '--zasy', str(p['zasy']), '--Mmin', str(p['Mmin']),
             '--mfp', str(p['mfp']), '--ncore', str(ncore),
             '--cosmology', cosmology],
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
            print('    ' + (log.strip().splitlines()[-1] if log.strip() else
                 '(empty log)'))
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

    return _read_result(d, npz_path, p)


def _read_result(d, npz_path, p):
    z = np.load(npz_path)
    return dict(dir=d, params=p,
               ells=z['ells'], D_cc=z['D_compute_cell'],
               D_win=z['D_amber_window'], window=z['window'],
               hist_z=z['hist_z'], hist_xH_vol=z['hist_xH_vol'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amber-x', required=True)
    ap.add_argument('--out-root', default='runs/sweep')
    ap.add_argument('--reuse', default=None,
                    help='existing validated run dir, reused for any '
                         'point whose amber_run.json matches it exactly '
                         '(e.g. amber_q01 for the shared fiducial point)')
    ap.add_argument('--L', type=float, default=256.0)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--ncore', type=int, default=16)
    ap.add_argument('--cosmology', default='planck18',
                    help="passed through to make_amber_input.py -- must "
                         "match --reuse's own cosmology if given")
    ap.add_argument('--scenarios-json', default=None,
                    help='path to a JSON {name: {axis, axis_label, '
                         'points:[{zmid,zdel,zasy,Mmin,mfp,label},...]}} '
                         'dict, overriding the built-in Chen et al. 2023 '
                         'Figure 10 scenarios')
    ap.add_argument('--timeout', type=int, default=600,
                    help='seconds per amber.x call (no-map runs are '
                         'fast; generous default, see docs)')
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    amber_x = os.path.expanduser(a.amber_x)
    if not os.path.isfile(amber_x):
        raise SystemExit(f"amber.x not found at {amber_x}")

    scenarios = (json.load(open(a.scenarios_json)) if a.scenarios_json
                else CHEN2023_FIG10_SCENARIOS)

    results = {}
    for name, scen in scenarios.items():
        print(f"\n=== Scenario '{name}' ({scen['axis_label']}) ===")
        results[name] = []
        for p in scen['points']:
            res = run_one(a.out_root, p, a.L, a.N, a.ncore, a.cosmology,
                          amber_x, a.python, reuse_dir=a.reuse,
                          timeout=a.timeout)
            if res:
                results[name].append(res)

    out = os.path.join(a.out_root, 'sweep_results.npz')
    payload = {}
    for name, scen in scenarios.items():
        pts = results[name]
        payload[f'{name}_axis'] = scen['axis']
        payload[f'{name}_n'] = len(pts)
        for i, r in enumerate(pts):
            payload[f'{name}_{i}_ells'] = r['ells']
            payload[f'{name}_{i}_Dcc'] = r['D_cc']
            payload[f'{name}_{i}_Dwin'] = r['D_win']
            payload[f'{name}_{i}_window'] = r['window']
            payload[f'{name}_{i}_hist_z'] = r['hist_z']
            payload[f'{name}_{i}_hist_xH'] = r['hist_xH_vol']
            payload[f'{name}_{i}_axisval'] = r['params'][scen['axis']]
            payload[f'{name}_{i}_label'] = r['params']['label']
    np.savez(out, **payload)

    meta = dict(L=a.L, N=a.N, cosmology=a.cosmology,
               axis_labels={n: s['axis_label'] for n, s in scenarios.items()})
    json.dump(meta, open(os.path.join(a.out_root, 'sweep_meta.json'), 'w'))

    print(f"\nwrote {out}")
    total_fail = 0
    for name, scen in scenarios.items():
        n_ok, n_req = len(results[name]), len(scen['points'])
        print(f"  {name}: {n_ok}/{n_req} points")
        total_fail += n_req - n_ok
    if total_fail:
        print(f"WARNING: {total_fail} point(s) failed and were excluded "
             f"-- see per-config output/log.txt or gate_output.txt")


if __name__ == '__main__':
    main()
