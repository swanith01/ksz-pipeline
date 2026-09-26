"""
scripts/42_harvest_old_sweep.py

One-off: assembles scripts/41_plot_amber_sweep.py's current npz schema
from a set of ALREADY-COMPLETED AMBER run directories, without touching
or re-running anything. Written for the 256 Mpc/h, 256^3 z_mid/Delta_z
sweep run under the pre-Chen2023-generalization driver (commit before
44f349c) -- its directory names don't match the current tag_for()
naming, so the current driver's own reuse-matching can't find them; this
reads each directory's amber_gate.npz + a hand-specified axis value/label
directly instead.

Edit HARVEST below (paths, relative to wherever you run this from) to
match your actual run directories, then:
  python scripts/42_harvest_old_sweep.py --out-root runs/sweep_zmid_dz_256cubed
"""
import argparse
import os

import numpy as np

# EDIT to match the actual run directories on the cluster.
HARVEST = {
    'z_mid': {
        'axis_label': r'z$_{mid}$',
        'points': [  # (axis value, label, directory)
            (6.0,  'zmid=6.0', 'runs/sweep/zmid06p0_dz04p0'),
            (7.0,  'zmid=7.0', 'runs/sweep/zmid07p0_dz04p0'),
            (8.0,  'zmid=8.0', 'runs/amber_q01'),
            (9.0,  'zmid=9.0', 'runs/sweep/zmid09p0_dz04p0'),
            (10.0, 'zmid=10.0', 'runs/sweep/zmid10p0_dz04p0'),
        ],
    },
    'delta_z': {
        'axis_label': r'$\Delta z$',
        'points': [
            (2.0, 'dz=2.0', 'runs/sweep/zmid08p0_dz02p0'),
            (4.0, 'dz=4.0', 'runs/amber_q01'),
            (6.0, 'dz=6.0', 'runs/sweep/zmid08p0_dz06p0'),
            (8.0, 'dz=8.0', 'runs/sweep/zmid08p0_dz08p0'),
        ],
    },
}

REQUIRED_KEYS = {'ells', 'D_compute_cell', 'D_amber_window', 'window',
                 'hist_z', 'hist_xH_vol'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', required=True,
                    help='where to write sweep_results.npz + sweep_meta.json '
                         '(created if it does not exist; not one of the '
                         'harvested directories)')
    ap.add_argument('--L', type=float, default=256.0)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--cosmology', default='planck18')
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    payload = {}
    axis_labels = {}
    for name, scen in HARVEST.items():
        axis_labels[name] = scen['axis_label']
        n_ok = 0
        for axisval, label, d in scen['points']:
            npz_path = os.path.join(d, 'amber_gate.npz')
            if not os.path.exists(npz_path):
                print(f"[{name}] MISSING: {npz_path} -- skipped")
                continue
            z = np.load(npz_path)
            missing = REQUIRED_KEYS - set(z.files)
            if missing:
                print(f"[{name}] {npz_path} missing {missing} -- run "
                     f"'python scripts/30_amber_validation_gate.py --run "
                     f"{d}' to refresh it, then rerun this harvester")
                continue
            i = n_ok
            payload[f'{name}_{i}_ells'] = z['ells']
            payload[f'{name}_{i}_Dcc'] = z['D_compute_cell']
            payload[f'{name}_{i}_Dwin'] = z['D_amber_window']
            payload[f'{name}_{i}_window'] = z['window']
            payload[f'{name}_{i}_hist_z'] = z['hist_z']
            payload[f'{name}_{i}_hist_xH'] = z['hist_xH_vol']
            payload[f'{name}_{i}_axisval'] = axisval
            payload[f'{name}_{i}_label'] = label
            n_ok += 1
            print(f"[{name}] harvested {label} from {d}")
        payload[f'{name}_axis'] = name  # informational only, not read by 41
        payload[f'{name}_n'] = n_ok

    out = os.path.join(a.out_root, 'sweep_results.npz')
    np.savez(out, **payload)
    import json
    json.dump(dict(L=a.L, N=a.N, cosmology=a.cosmology, axis_labels=axis_labels),
              open(os.path.join(a.out_root, 'sweep_meta.json'), 'w'))
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
