"""
scripts/41_plot_amber_sweep.py

Two-panel comparison figure per scenario, from scripts/40_run_amber_sweep.py's
output (any number of named scenarios, not hardcoded to two):
  LEFT:  patchy-kSZ D_ell, both methods -- solid = our compute_cell,
         dashed = AMBER's own native P_qperp+Limber (same window). Two
         independently-written codes per sweep point, not just ours.
  RIGHT: the reionization history x_HI(z) (volume-weighted) that
         produced each D_ell curve, same color per sweep-parameter value
         as the left panel -- so a rise/shift on the right can be read
         directly against the D_ell change it causes on the left.

Each point's legend label is whatever 40_run_amber_sweep.py's point()
set (e.g. "dz=12.8 (max, Sec 5.2)" for a point that also overrides other
parameters) -- not reconstructed from the swept axis value alone, so an
overridden point's label stays honest about what actually differs.

Usage:
  python scripts/41_plot_amber_sweep.py --run runs/sweep
  # -> data/plots/<run>_<scenario_name>.png, one per scenario in the npz
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CMAPS = ['viridis', 'plasma', 'cividis', 'magma']


def _scenario_names(d):
    return sorted({k[:-len('_axis')] for k in d.files if k.endswith('_axis')})


def _load_scenario(d, name):
    n = int(d[f'{name}_n'])
    pts = []
    for i in range(n):
        pts.append(dict(
            axisval=float(d[f'{name}_{i}_axisval']),
            label=str(d[f'{name}_{i}_label']),
            ells=d[f'{name}_{i}_ells'], D=d[f'{name}_{i}_Dcc'],
            D_amber=d[f'{name}_{i}_Dwin'], window=d[f'{name}_{i}_window'],
            hist_z=d[f'{name}_{i}_hist_z'], hist_xH=d[f'{name}_{i}_hist_xH']))
    return sorted(pts, key=lambda p: p['axisval'])


def _plot_scenario(pts, title, out_path, cmap='viridis'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.9, len(pts)))

    for p, c in zip(pts, colors):
        left.loglog(p['ells'], p['D'], '-', color=c, lw=1.8, label=p['label'])
        left.loglog(p['ells'], p['D_amber'], '--', color=c, lw=1.2, alpha=0.7)
    left.set_xlabel(r'$\ell$')
    left.set_ylabel(r'$D_\ell$  [$\mu$K$^2$]')
    left.set_title('patchy kSZ (solid = ours, dashed = AMBER native)')
    left.legend(frameon=False, fontsize=8.5)

    for p, c in zip(pts, colors):
        right.plot(p['hist_z'], p['hist_xH'], '-', color=c, lw=1.8,
                  label=p['label'])
        z1, z2 = p['window']
        right.axvspan(z1, z2, color=c, alpha=0.06)
    right.set_xlabel(r'$z$')
    right.set_ylabel(r'$x_{HI}$ (volume-weighted)')
    right.set_title('reionization history\n(shaded = patchy window used at left)')
    right.set_ylim(-0.03, 1.03)
    right.invert_xaxis()
    right.legend(frameon=False, fontsize=8.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'wrote {out_path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()

    npz_path = os.path.join(a.run, 'sweep_results.npz')
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found -- run "
                         f"scripts/40_run_amber_sweep.py --out-root {a.run} first")
    d = np.load(npz_path)
    meta_path = os.path.join(a.run, 'sweep_meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    run_name = os.path.basename(os.path.normpath(a.run))
    box_str = (f"L={meta['L']:.0f} Mpc/h, N={meta['N']}, "
              f"cosmology={meta.get('cosmology', '?')}"
              if 'L' in meta else '')
    axis_labels = meta.get('axis_labels', {})

    for i, name in enumerate(_scenario_names(d)):
        pts = _load_scenario(d, name)
        if not pts:
            print(f"{name}: no validated points, skipping")
            continue
        axis_lbl = axis_labels.get(name, name)
        _plot_scenario(
            pts, f"Scenario '{name}': varying {axis_lbl}  ({box_str})",
            os.path.join('data', 'plots', f'{run_name}_{name}.png'),
            cmap=CMAPS[i % len(CMAPS)])


if __name__ == '__main__':
    main()
