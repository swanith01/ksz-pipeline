"""
scripts/54_overlay_post_eor_vs_chen23.py

Like-for-like comparison against Chen, Trac, Mukherjee & Cen 2023
(arXiv:2203.04337) Fig. 10's fiducial curve.

Why this exists: scripts/53 compares our D_coeval/Limber against AMBER's
native sum over the PATCHY WINDOW only (z ~ 7.25-14.25 for the fiducial
history, x_HI between 0.01% and 99.99%). Chen+23's spectra are not
windowed that way -- Sec 5.2: "our D_ell^pkSZ are integrated from z=5 to
z=30". For the fiducial [z_mid, Delta_z, A_z] = [8, 4, 3] history the
universe is fully ionized below z ~ 7.25, so their curve also contains
the post-reionization (homogeneous) kSZ from z=5 up to the end of
reionization. Their "patchy" label refers to the reionization model, not
to an x_e-based window.

This script therefore computes, from the SAME run:
  - ours, patchy window      (compute_cell default; = scripts/53)
  - ours, z >= zmin          (compute_cell(post_eor_zmin=...)): the
                             patchy window extended down through the
                             fully ionized shells, i.e. P_qperp of the
                             full momentum field there
  - AMBER native, same two ranges (scripts/30's amber_cl_sum, rebuilt
                             from power_z=*.txt)
  - AMBER native, all shells (cl_ksz.txt as written, z=czmin..czmax)
and overlays the digitized Chen+23 fiducial (data/chen23_fiducial.csv).

Upper limit: Chen integrate to z=30; AMBER runs here stop at czmax=18.5.
Shells above the patchy window are >99.99% neutral (AMBER's ionization is
binary: ionized iff z < zre), so their q_perp power is ~0 and the
missing 18.5-30 range is negligible. The script prints the highest
shell's x_HI so this can be checked rather than assumed.

Lower limit: shells are selected by their LOWER EDGE >= --zmin, so with
AMBER's default lin shells (4.5-5.0, 5.0-5.5, ...) --zmin 5 keeps
5.0-5.5 upward and drops 4.5-5.0, matching Chen's z=5 exactly.

Inputs needed in --run (no raw fields_*.dat read -- cheap, login node ok):
  amber_run.json, qperp_checkpoint.pkl (written by scripts/30 gate 2),
  output/cmb/{tau.txt, cl_ksz.txt, power_z=*.txt}

Usage
-----
    PYTHONPATH=src python scripts/54_overlay_post_eor_vs_chen23.py \
        --run runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3
"""
import argparse
import importlib
import json
import os
import pickle
import sys

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

gate30 = importlib.import_module('30_amber_validation_gate')
from make_amber_input import chi_amber

from ksz_pipeline.amber import io, adapter
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.utils.constants import ne0_cgs

XHI_MIN_PATCHY, XHI_MAX_PATCHY = 1.0e-4, 1.0 - 1.0e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3')
    ap.add_argument('--zmin', type=float, default=5.0,
                    help='lower integration limit (Chen+23 Sec 5.2: z=5)')
    ap.add_argument('--chen-csv', default='data/chen23_fiducial.csv')
    ap.add_argument('--out', default='data/plots/amber_post_eor_vs_chen23_fiducial.png')
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    amber_p = meta['amber_params']
    preset = meta.get('cosmology_preset', '(not recorded -> planck18)')
    print(f"run: {a.run}   L={meta['L']} Mpc/h  N={meta['N']}  cosmology={preset}")
    if preset != 'chen2022':
        print("  WARNING: Chen+23's fiducial curve uses their own cosmology "
              "[Om,Ob,s8,ns,h]=[0.3,0.045,0.8,0.96,0.7]; this run doesn't, "
              "so the comparison is not like-for-like.")
    cmb = os.path.join(a.run, 'output', 'cmb')
    ne0 = meta.get('ne0_cgs', ne0_cgs())

    ckpt = os.path.join(a.run, 'qperp_checkpoint.pkl')
    if not os.path.exists(ckpt):
        raise SystemExit(
            f"{ckpt} not found -- run scripts/30_amber_validation_gate.py on "
            f"this run first (its gate 2 writes P_qperp for EVERY shell, "
            f"fully ionized ones included, to this checkpoint)")
    with open(ckpt, 'rb') as fh:
        res = pickle.load(fh)['shells']

    zs = np.array(sorted(res))
    print("\n  shell z   xH_vol      in patchy window?")
    for z in zs:
        xh = res[z]['xH_mean']
        print(f"  {z:6.2f}   {xh:.5f}     "
              f"{'yes' if XHI_MIN_PATCHY <= xh <= XHI_MAX_PATCHY else 'no'}")
    print(f"  highest shell z={zs.max():.2f} has xH_vol={res[zs.max()]['xH_mean']:.6f} "
          f"(~1 => z>{zs.max():.1f} up to Chen's z=30 contributes ~nothing)")

    # Shells kept for the fixed-range integral: lower EDGE >= zmin.
    lo_edges = np.array([gate30.shell_edges(z, meta['czdel'], meta['cspacing'],
                                            meta['czmin'])[0] for z in zs])
    keep = zs[lo_edges >= a.zmin - 1e-9]
    if keep.size == 0:
        raise SystemExit(f"no shells with lower edge >= {a.zmin}")
    z_first = float(keep.min())
    z_win_lo = float(min(z for z in zs
                         if XHI_MIN_PATCHY <= res[z]['xH_mean'] <= XHI_MAX_PATCHY))

    class _AmberCosmo:                     # same wrapper as scripts/30 gate 2b
        def comoving_distance(self, z):
            return chi_amber(z, amber_p) * u.Mpc
    cos = _AmberCosmo()

    ells, D_win, *_r, (zt_win, _), _x = compute_cell(
        res, tau0=adapter.tau_below_amber(cmb, z_win_lo), cosmology=cos, ne0=ne0)
    ells_e, D_ext, *_r2, (zt_ext, _), _x2 = compute_cell(
        res, tau0=adapter.tau_below_amber(cmb, z_first), cosmology=cos, ne0=ne0,
        post_eor_zmin=z_first)
    print(f"\nours, patchy window : shells z={zt_win.min():.2f}-{zt_win.max():.2f}")
    print(f"ours, z >= {a.zmin:<4}    : shells z={zt_ext.min():.2f}-{zt_ext.max():.2f}")

    # AMBER native
    la, Ca = io.read_amber_cl(os.path.join(cmb, 'cl_ksz.txt'))
    D_amb_full = gate30.cl_to_dl(Ca, la)
    try:
        D_amb_win = gate30.cl_to_dl(
            gate30.amber_cl_sum(cmb, meta, la, zt_win.min(), zt_win.max()), la)
        D_amb_ext = gate30.cl_to_dl(
            gate30.amber_cl_sum(cmb, meta, la, zt_ext.min(), 1e3), la)
    except FileNotFoundError as e:
        print(f"  WARNING: {e} -- AMBER-native windowed curves skipped")
        D_amb_win = D_amb_ext = None

    chen = np.loadtxt(a.chen_csv, delimiter=",", skiprows=1)
    l_chen, D_chen = chen[:, 0], chen[:, 1]

    def at(lq, l, D):
        return float(np.interp(lq, l, D))

    print(f"\n   ell   ours(win)  ours(z>={a.zmin:g})  AMBER(z>={a.zmin:g})  "
          f"AMBER(all)  Chen+23   ours(z>={a.zmin:g})/Chen")
    for lq in (1000, 2000, 3000, 5000, 8000):
        amb_e = at(lq, la, D_amb_ext) if D_amb_ext is not None else np.nan
        dc = at(lq, l_chen, D_chen)
        de = at(lq, ells_e, D_ext)
        print(f"  {lq:5d}   {at(lq, ells, D_win):.4f}     {de:.4f}         "
              f"{amb_e:.4f}          {at(lq, la, D_amb_full):.4f}      "
              f"{dc:.4f}    {de / dc:.3f}")
    i3w, i3e = (int(np.argmin(np.abs(x - 3000))) for x in (ells, ells_e))
    print(f"\npost-EoR share of our D_3000 (z>={a.zmin:g}): "
          f"{1 - D_win[i3w] / D_ext[i3e]:.1%}")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(l_chen, D_chen, "o-", color="k", ms=4,
            label="Chen+23 fiducial (digitized Fig. 10, z=5-30)")
    if D_amb_ext is not None:
        ax.plot(la, D_amb_ext, color="C0", lw=2,
                label=f"AMBER native, z={zt_ext.min() - meta['czdel'] / 2:.1f}-{meta['czmax']:.1f}")
        ax.plot(la, D_amb_win, color="C0", lw=1, ls=":",
                label=f"AMBER native, patchy window z={zt_win.min():.2f}-{zt_win.max():.2f}")
    ax.plot(ells_e, D_ext, "s--", color="C1", ms=4, lw=1.5,
            label=f"ours (D_coeval/Limber), z>={a.zmin:g} incl. post-EoR")
    ax.plot(ells, D_win, color="C1", lw=1, ls=":", label="ours, patchy window only")
    ax.plot(la, D_amb_full, color="0.6", lw=1,
            label=f"AMBER cl_ksz.txt, all shells z={meta['czmin']}-{meta['czmax']}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
    ax.set_xlim(0, 8500)
    ax.set_title(f"L={meta['L']:g} Mpc/h, N={meta['N']}, cosmology={preset}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"\nwrote {a.out}")

    npz = os.path.join(a.run, 'post_eor_vs_chen23.npz')
    np.savez(npz, ells_win=ells, D_ours_win=D_win, ells_ext=ells_e, D_ours_ext=D_ext,
             ell_amber=la, D_amber_full=D_amb_full,
             D_amber_win=D_amb_win if D_amb_win is not None else np.array([]),
             D_amber_ext=D_amb_ext if D_amb_ext is not None else np.array([]),
             zmin=a.zmin, window_win=[zt_win.min(), zt_win.max()],
             window_ext=[zt_ext.min(), zt_ext.max()],
             ell_chen=l_chen, D_chen=D_chen)
    print(f"wrote {npz}")


if __name__ == '__main__':
    main()
