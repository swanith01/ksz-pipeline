"""
scripts/32_amber_stitched_lightcone_crosscheck.py

Independent cross-check of one AMBER run's kSZ D_ell, built via a
COMPLETELY DIFFERENT method than scripts/30's gate2 / scripts/40's sweep:

  compute_cell (scripts/30 GATE 2b): coeval boxes, box-based Limber sum
    over P_qperp(k) at each snapshot z.

  THIS SCRIPT: AMBER snapshots STITCHED into one lightcone
    (ksz_pipeline.amber.stitch_from_amber.stitch_lightcone_from_amber,
    which reuses stitch_from_coeval's geometry verbatim), then
    LOS-integrated into a real 2D map and FFT'd
    (amber_lightcone_to_ksz_Dl -> compute_ksz_map / ksz_map_to_Dl, the
    SAME functions scripts/01 and scripts/03 use for the 21cmFAST path).
    Same role as script 03's cross-check of script 01, now for AMBER.

All three run-specific pieces this needs (ne0, visibility/tau, chi_Mpc)
are resolved from THIS run's own outputs, not borrowed from the 21cmFAST
path or from another AMBER run -- see
ksz_pipeline.amber.stitch_from_amber.amber_lightcone_to_ksz_Dl's
docstring for the full reasoning (ne0 matches scripts/30 gate2's choice;
visibility comes from AMBER's own tau.txt; chi_Mpc is fresh per run,
never the ksz_map_to_Dl default of 7800 nor the unrelated 21cmFAST
fiducial's chi_eff~8504).

Usage
-----
    python scripts/32_amber_stitched_lightcone_crosscheck.py --run runs/amber_q01
    # with wrap-cycle rotation (see docs/HANDOFF_wrap_cycle.md):
    python scripts/32_amber_stitched_lightcone_crosscheck.py --run runs/amber_q01 \
        --wrap-cycle-seed 123

Produces <run>/ksz_Dl_stitched_amber.npz (or, with --wrap-cycle-seed,
<run>/ksz_Dl_stitched_amber_wrapcycle<seed>.npz, kept separate so re-runs
with and without wrap-cycle never overwrite each other): ell, Dl, Dl_err,
chi_eff, z_window. If <run>/amber_gate.npz exists (scripts/30 already
ran), also prints the D_3000 ratio against compute_cell and AMBER's own
window sum, for a three-way check.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from ksz_pipeline.amber import io
from ksz_pipeline.amber.stitch_from_amber import (
    stitch_lightcone_from_amber, build_los_z_grid, amber_lightcone_to_ksz_Dl)
from ksz_pipeline.utils.constants import ne0_cgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True,
                    help='dir with amber_run.json + output/{cmb,reion}')
    ap.add_argument('--margin', type=float, default=None,
                    help='Mpc-in-z margin kept inside the snapshot '
                         'bracket at each end, so the LOS grid never '
                         'asks stitch_field to extrapolate flatly past '
                         'an edge snapshot. Default: half this run\'s '
                         'smallest snapshot spacing -- not a fixed '
                         'constant, since different sweep points use '
                         'different czdel/cspacing (e.g. the '
                         'dz=12.8/log-spacing Fig. 10 point).')
    ap.add_argument('--wrap-cycle-seed', type=int, default=None,
                    help='Enable wrap-cycle rotation (a fresh '
                         'deterministic rotation angle each time the LOS '
                         'wraps back around the periodic box) with this '
                         'integer seed -- see '
                         'docs/HANDOFF_wrap_cycle.md and '
                         'stitch_from_coeval\'s module docstring. Default '
                         'None preserves the old single-fixed-angle '
                         'behavior exactly (byte-for-byte), which still '
                         'reuses the same box slab every time the LOS '
                         'returns to a given y_cell. Requires interp=True '
                         '(this script always uses interp=True, so no '
                         'extra flag is needed for that).')
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    N, L_h, h = meta['N'], meta['L'], meta['amber_params']['h']
    cmb_dir   = os.path.join(a.run, 'output', 'cmb')
    reion_dir = os.path.join(a.run, 'output', 'reion')

    zre = io.read_zre(io.find_zre_file(reion_dir), N)
    field_files = io.list_field_files(cmb_dir)
    snap_z = np.array(sorted(meta['snapshot_z']))

    cell_size = (L_h / h) / N
    margin = a.margin if a.margin is not None else 0.5 * np.diff(snap_z).min()
    z_min, z_max = snap_z.min() + margin, snap_z.max() - margin
    z_arr = build_los_z_grid(z_min, z_max, cell_size)
    wc_note = (f", wrap-cycle seed={a.wrap_cycle_seed}"
              if a.wrap_cycle_seed is not None else ", no wrap-cycle")
    print(f"run {a.run}: L={L_h} Mpc/h N={N}  stitching {len(field_files)} "
          f"snapshots onto {len(z_arr)} LOS pixels, z={z_min:.2f}-{z_max:.2f} "
          f"(margin={margin:.3f}{wc_note})")

    lc = stitch_lightcone_from_amber(field_files, zre, z_arr, N, L_h, h,
                                     angle_deg=0.0, interp=True,
                                     wrap_cycle_seed=a.wrap_cycle_seed)
    print(f"  [check] mean(1+density)={1.0 + lc['density'].mean():.4f} (want ~1)")
    print(f"  [check] rms(velocity_z)/MPC_CM="
          f"{np.sqrt(np.mean((lc['velocity_z'] / 3.0856775814913673e24)**2)):.4e} "
          f"Mpc/s")

    tau_table = np.loadtxt(os.path.join(cmb_dir, 'tau.txt'), skiprows=1)
    # Same fallback rule as scripts/30 gate2: runs predating the
    # ne0_cgs field always used Planck18, so ne0_cgs() is the correct
    # value for them, not a guess.
    ne0 = meta.get('ne0_cgs', ne0_cgs())

    result = amber_lightcone_to_ksz_Dl(lc, tau_table, L_h / h, ne0)

    if result['z_window'] is None:
        print("  WARNING: no slices fall inside the patchy regime -- "
              "LOS grid may not bracket this run's reionization history")
    else:
        z_lo, z_hi = result['z_window']
        print(f"  Patchy regime: z = {z_hi:.2f} -> {z_lo:.2f}")
    print(f"  [chi_eff] = {result['chi_eff']:.1f} Mpc "
          f"(signal-weighted, this run only)")
    print(f"  ksz_map shape: {result['ksz_map'].shape}")

    # Suffix the filename when wrap-cycle is on, so a wrap-cycle run never
    # silently overwrites the plain (default, no-wrap-cycle) result for
    # the same --run dir -- both can be kept side by side.
    out_name = ('ksz_Dl_stitched_amber.npz' if a.wrap_cycle_seed is None
               else f'ksz_Dl_stitched_amber_wrapcycle{a.wrap_cycle_seed}.npz')
    out_path = os.path.join(a.run, out_name)
    np.savez(out_path, ell=result['ell'], Dl=result['Dl'],
             Dl_err=result['Dl_err'], chi_eff=result['chi_eff'],
             z_window=np.array(result['z_window']
                               if result['z_window'] else [np.nan, np.nan]))
    print(f"Saved to {out_path}")

    ell, Dl = result['ell'], result['Dl']
    i3 = int(np.argmin(np.abs(ell - 3000)))
    print(f"  D_3000 (stitched lightcone) = {Dl[i3]:.4f} uK^2")

    gate_npz = os.path.join(a.run, 'amber_gate.npz')
    if os.path.exists(gate_npz):
        g = np.load(gate_npz)
        j3 = int(np.argmin(np.abs(g['ells'] - 3000)))
        d_cc  = float(g['D_compute_cell'][j3])
        d_win = float(g['D_amber_window'][j3])
        print(f"  D_3000 (compute_cell, GATE 2b)  = {d_cc:.4f} uK^2  "
              f"ratio = {Dl[i3] / d_cc:.3f}")
        print(f"  D_3000 (AMBER native, window)   = {d_win:.4f} uK^2  "
              f"ratio = {Dl[i3] / d_win:.3f}")
    else:
        print(f"  (no {gate_npz} -- run scripts/30_amber_validation_gate.py "
              f"first for a compute_cell/AMBER-native cross-check)")


if __name__ == '__main__':
    main()
