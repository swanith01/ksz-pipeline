"""
scripts/41_plot_amber_sweep.py

Two-panel comparison figures from scripts/40_run_amber_sweep.py's output,
per scenario:
  LEFT:  patchy-kSZ D_ell, both methods -- solid = our compute_cell,
         dashed = AMBER's own native P_qperp+Limber (same window). Two
         independently-written codes per sweep point, not just ours.
  RIGHT: the reionization history x_HI(z) (volume-weighted) that
         produced each D_ell curve, same color per sweep-parameter value
         as the left panel -- so a rise/shift on the right can be read
         directly against the D_ell change it causes on the left.

  A: one curve per Delta_z, at fixed z_mid -- a clean duration-only
     comparison (see 40_run_amber_sweep.py's docstring).
  B: one curve per z_mid, at fixed Delta_z -- timing AND morphology both
     changing; don't read this as duration-only.

Usage:
  python scripts/41_plot_amber_sweep.py --run runs/sweep
  # -> data/plots/<run>_scenarioA_delta_z.png
  # -> data/plots/<run>_scenarioB_z_mid.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _load_scenario(d, scen):
    n = int(d[f'{scen}_n'])
    pts = []
    for i in range(n):
        pts.append(dict(
            zmid=float(d[f'{scen}_zmid'][i]), zdel=float(d[f'{scen}_zdel'][i]),
            ells=d[f'{scen}_{i}_ells'], D=d[f'{scen}_{i}_Dcc'],
            D_amber=d[f'{scen}_{i}_Dwin'], window=d[f'{scen}_{i}_window'],
            hist_z=d[f'{scen}_{i}_hist_z'], hist_xH=d[f'{scen}_{i}_hist_xH']))
    return pts


def _plot_scenario(pts, label_fn, sort_key, title, out_path, cmap='viridis'):
    pts = sorted(pts, key=sort_key)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.9, len(pts)))

    for p, c in zip(pts, colors):
        left.loglog(p['ells'], p['D'], '-', color=c, lw=1.8,
                   label=label_fn(p))
        left.loglog(p['ells'], p['D_amber'], '--', color=c, lw=1.2,
                   alpha=0.7)
    left.set_xlabel(r'$\ell$')
    left.set_ylabel(r'$D_\ell$  [$\mu$K$^2$]')
    left.set_title('patchy kSZ (solid = ours, dashed = AMBER native)')
    left.legend(frameon=False, fontsize=8.5)

    for p, c in zip(pts, colors):
        right.plot(p['hist_z'], p['hist_xH'], '-', color=c, lw=1.8,
                  label=label_fn(p))
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
    box_str = (f"L={meta['L']:.0f} Mpc/h, N={meta['N']}"
              if 'L' in meta else '')

    A = _load_scenario(d, 'A')
    if A:
        _plot_scenario(
            A, lambda p: f"$\\Delta z$={p['zdel']:.1f}", lambda p: p['zdel'],
            f"Scenario A: fixed z$_{{mid}}$={A[0]['zmid']:.1f}, "
            f"varying $\\Delta z$  ({box_str})",
            os.path.join('data', 'plots', f'{run_name}_scenarioA_delta_z.png'))
    else:
        print("Scenario A: no validated points, skipping")

    B = _load_scenario(d, 'B')
    if B:
        _plot_scenario(
            B, lambda p: f"z$_{{mid}}$={p['zmid']:.1f}", lambda p: p['zmid'],
            f"Scenario B: fixed $\\Delta z$={B[0]['zdel']:.1f}, "
            f"varying z$_{{mid}}$  ({box_str})  -- NOT duration-only, "
            f"see script docstring",
            os.path.join('data', 'plots', f'{run_name}_scenarioB_z_mid.png'),
            cmap='plasma')
    else:
        print("Scenario B: no validated points, skipping")


if __name__ == '__main__':
    main()

