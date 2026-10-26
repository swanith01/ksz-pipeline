"""
scripts/57_qparallel_test_amber.py

Is the low-ell excess of the stitched kSZ map over Limber the longitudinal
(q_parallel) part of the momentum field? Our version of Ma et al.'s kSZ
auto-power figure (2-D map power vs Limber q_perp / q_parallel modes),
for one AMBER run, over z >= zmin (post-reionization included):

  Limber   D_qperp  -- compute_cell on P_qperp (as everywhere else)
           D_qpar   -- the SAME Limber kernel applied to P_qparallel. A
                       heuristic, as in Ma et al.: strictly, the Limber
                       projection of the longitudinal mode is zero. Shows
                       where in ell q_parallel power sits, not a prediction.
  Stitched full q   -- momentum-mode stitched map of q = x_HII * mom
           q_perp   -- the SAME stitching of the transverse part of q only
                       (q_parallel removed per snapshot in Fourier space)

The decisive test is the stitched A/B: same snapshots, same LOS grid, same
re-orientations; the only difference is whether q_parallel is present.
If the low-ell hump is q_parallel, it disappears from the q_perp map, and
the q_perp map tracks Limber D_qperp. Box repeats along the LOS (~7.4 for
the 256 Mpc/h sweep) affect both maps alike.

Implementation: per snapshot, read fields, build q (km/s) = ionized *
mom with AMBER's criterion (ionized iff z < zre), measure P_qperp and
P_qpar, and write two derived snapshot sets in AMBER's fields_*.dat
layout (rho := 1, mom := q or q_perp) under --work-dir, with an
all-ionized zre so the streaming code's x_HII factor is 1 (x_HII is
already inside q). Stitched full-q is then bit-for-bit the same
computation as scripts/56's total spectrum for this run.

Disk: 2 x 28 x (1+4N^3) floats (~15 GB at N=256), deleted afterwards
unless --keep-derived. RAM: a few GB at N=256.

Usage
-----
    PYTHONPATH=src python -u scripts/57_qparallel_test_amber.py \
        --run runs/sweep_chen2023/zmid8_dz4_az3_mh8_mfp3
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import time

import numpy as np
import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

gate30 = importlib.import_module('30_amber_validation_gate')
los_grid = importlib.import_module('55_amber_stitched_streaming').los_grid
from make_amber_input import chi_amber

from ksz_pipeline.amber import io, adapter
from ksz_pipeline.amber.stream_lightcone import stream_ksz_maps, open_zre
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.coeval.momentum import qsplit_power_from_q, transverse_part
from ksz_pipeline.ksz.lightcone_integral import ksz_map_to_Dl
from ksz_pipeline.utils.constants import ne0_cgs


def _write(path, z, N, q):
    """AMBER fields_*.dat layout (io.read_fields): z, rho (N,N,N), then mom
    as ONE (3,N,N,N) Fortran-order array -- component index fastest, i.e.
    interleaved, not three consecutive blocks."""
    with open(path, 'wb') as f:
        np.array([z], '<f4').tofile(f)
        np.ones(N**3, '<f4').tofile(f)                      # rho := 1 (unused)
        np.asfortranarray(np.stack(q).astype('<f4')).ravel(order='F').tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/sweep_chen2023/zmid8_dz4_az3_mh8_mfp3')
    ap.add_argument('--zmin', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--geometry', default='reorient', choices=['reorient', 'rotate'])
    ap.add_argument('--work-dir', default=None, help='default <run>/qpar_test')
    ap.add_argument('--keep-derived', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    N, amb = meta['N'], meta['amber_params']
    L = meta['L'] / amb['h']
    cmb = os.path.join(a.run, 'output', 'cmb')
    files = io.list_field_files(cmb)
    if len(files) < 2:
        raise SystemExit(f"{cmb}: no fields_*.dat")
    work = a.work_dir or os.path.join(a.run, 'qpar_test')
    dirs = {k: os.path.join(work, k) for k in ('full', 'perp')}
    for dd in dirs.values():
        os.makedirs(dd, exist_ok=True)
    zre = io.read_zre(io.find_zre_file(os.path.join(a.run, 'output', 'reion')), N)
    run_name = os.path.basename(os.path.normpath(a.run))
    print(f"run {a.run}: N={N}, L={L:.1f} Mpc, {len(files)} snapshots, "
          f"cosmology={meta.get('cosmology_preset')}", flush=True)

    # ---- 1. per snapshot: q, P_qperp / P_qpar, derived fields -------------
    res_perp, res_par = {}, {}
    t0 = time.time()
    for f in files:
        z, rho, mom = io.read_fields(f, N)
        ion = (z < zre)
        q = [mom[c].astype(np.float64) * ion for c in range(3)]      # km/s
        del rho, mom
        k, Pp, Sp, Pl, Sl = qsplit_power_from_q(
            *(c * adapter.KMS_TO_CMS for c in q), L)
        xh = float(1.0 - ion.mean())
        res_perp[z] = dict(k=k, Pqperp=Pp, Pstd=Sp, xH_mean=xh)
        res_par[z] = dict(k=k, Pqperp=Pl, Pstd=Sl, xH_mean=xh)
        tag = os.path.basename(f)
        _write(os.path.join(dirs['full'], tag), z, N, q)
        _write(os.path.join(dirs['perp'], tag), z, N, transverse_part(*q))
        i = int(np.argmin(np.abs(k - 0.05)))
        print(f"  z={z:6.2f}  x_HI={xh:.4f}  P_qpar/P_qperp at k~0.05/Mpc: "
              f"{Pl[i] / Pp[i] if Pp[i] > 0 else np.nan:.3g}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        del q
    ion_zre = os.path.join(work, 'zre_all_ionized.dat')
    np.full(N**3, 99.0, '<f4').tofile(ion_zre)

    # ---- 2. Limber D_qperp, D_qpar over z >= zmin --------------------------
    zs = np.array(sorted(res_perp))
    lo = np.array([gate30.shell_edges(z, meta['czdel'], meta['cspacing'],
                                      meta['czmin'])[0] for z in zs])
    z_first = float(zs[lo >= a.zmin - 1e-9].min())

    class _Cos:
        def comoving_distance(self, z):
            return chi_amber(z, amb) * u.Mpc
    kw = dict(tau0=adapter.tau_below_amber(cmb, z_first), cosmology=_Cos(),
              ne0=meta.get('ne0_cgs', ne0_cgs()), post_eor_zmin=z_first)
    l_lim, D_qperp, *_ = compute_cell(res_perp, **kw)
    l_lim2, D_qpar, *_ = compute_cell(res_par, **kw)
    assert np.array_equal(l_lim, l_lim2)

    # ---- 3. stitched, full q vs q_perp only --------------------------------
    snap_z = np.array([io.z_from_name(f) for f in files])
    z_arr, pos = los_grid(max(a.zmin, snap_z.min()), snap_z.max(), L / N, amb)
    tau = np.loadtxt(os.path.join(cmb, 'tau.txt'), skiprows=1)
    zi = open_zre(ion_zre, N)
    maps = {}
    for key, dd in dirs.items():
        t1 = time.time()
        maps[key] = stream_ksz_maps(io.list_field_files(dd), zi, N, L, z_arr, pos,
                                    tau, meta.get('ne0_cgs', ne0_cgs()), a.seed,
                                    mode='momentum', geometry=a.geometry)
        print(f"stitched {key}: {time.time() - t1:.0f}s, LOS "
              f"{(pos[-1] - pos[0]) / L:.2f} box lengths", flush=True)
    chi = maps['full']['chi_eff_total']          # same ell mapping for both
    l_f, D_f, E_f = ksz_map_to_Dl(maps['full']['map_total'], L, chi_Mpc=chi)
    l_p, D_p, E_p = ksz_map_to_Dl(maps['perp']['map_total'], L, chi_Mpc=chi)

    # ---- 4. report ----------------------------------------------------------
    at = lambda lq, l, D: float(np.interp(lq, l, D))
    print(f"\nLimber shells z>={z_first:.2f}; stitched z={z_arr[0]:.2f}-"
          f"{z_arr[-1]:.2f}, chi_eff {chi:.0f} Mpc, geometry={a.geometry}")
    print("   ell  stitched(q)  stitched(q_perp)  q/q_perp   Limber q_perp  "
          "Limber q_par*  Limber sum*")
    for lq in (100, 150, 200, 300, 500, 1000, 2000, 3000, 5000):
        if lq < max(l_f.min(), l_p.min()):
            continue
        sf, sp = at(lq, l_f, D_f), at(lq, l_p, D_p)
        lp_, lq_ = at(lq, l_lim, D_qperp), at(lq, l_lim, D_qpar)
        print(f"  {lq:5d}  {sf:10.4f}  {sp:12.4f}      {sf / sp:6.2f}     "
              f"{lp_:10.4f}    {lq_:10.4f}   {lp_ + lq_:10.4f}")
    print("  (* heuristic: Limber kernel on P_qparallel, as Ma et al.)")

    npz = os.path.join(a.run, f'qpar_test_{a.geometry}.npz')
    np.savez(npz, ell_stitched_full=l_f, D_stitched_full=D_f, Derr_stitched_full=E_f,
             ell_stitched_perp=l_p, D_stitched_perp=D_p, Derr_stitched_perp=E_p,
             ell_limber=l_lim, D_limber_qperp=D_qperp, D_limber_qpar=D_qpar,
             chi_eff=chi, z_first=z_first,
             z_shells=zs, Pqpar_over_Pqperp_k005=np.array([
                 res_par[z]['Pqperp'][int(np.argmin(np.abs(res_par[z]['k'] - 0.05)))]
                 / max(res_perp[z]['Pqperp'][int(np.argmin(np.abs(res_perp[z]['k'] - 0.05)))], 1e-300)
                 for z in zs]))
    print(f"\nwrote {npz}")

    out = a.out or os.path.join('data', 'plots', f'qpar_test_{run_name}_{a.geometry}.png')
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for axx, log in zip(ax, (False, True)):
        axx.errorbar(l_f, D_f, E_f, color='C3', marker='.', ms=4, lw=1.5, capsize=0,
                     label='stitched map, full q')
        axx.errorbar(l_p, D_p, E_p, color='C2', marker='.', ms=4, lw=1.5, capsize=0,
                     label=r'stitched map, $q_\perp$ only')
        axx.plot(l_lim, D_qperp, 'k-', lw=1.8, label=r'Limber $q_\perp$')
        axx.plot(l_lim, D_qpar, 'k:', lw=1.8, label=r'Limber $q_\parallel$ (heuristic)')
        axx.plot(l_lim, D_qperp + D_qpar, '--', color='0.5', lw=1,
                 label=r'Limber $q_\perp + q_\parallel$')
        axx.set_xscale('log')
        axx.set_xlabel(r'$\ell$')
        axx.set_ylabel(r'$D_\ell\ [\mu K^2]$')
        if log:
            axx.set_yscale('log')
        else:
            axx.set_xlim(80, 1e4)
            top = max(np.nanmax(D_f[(l_f >= 80) & (l_f <= 1e4)]),
                      np.nanmax(D_qperp[l_lim <= 1e4]))
            axx.set_ylim(0, 1.15 * top)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{run_name}: z>={a.zmin:g}, L={meta['L']:g} Mpc/h, N={N}, "
                 f"LOS {(pos[-1] - pos[0]) / L:.1f} box lengths ({a.geometry})",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    if not a.keep_derived:
        shutil.rmtree(work)
        print(f"removed {work}")


if __name__ == '__main__':
    main()
