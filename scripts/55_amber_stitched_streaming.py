"""
scripts/55_amber_stitched_streaming.py

Stitched-lightcone (+ wrap-cycle) kSZ D_ell for an AMBER run too big for
scripts/32 (e.g. runs/sweep_2gpc_1536, 1536^3), via the slab-streaming
ksz_pipeline.amber.stream_lightcone -- ~1 GB RAM instead of >1 TB.

Gives BOTH, from one pass over the LOS:
  patchy : clipped to x_e in [1e-4, 1-1e-4]   (scripts/32/33 convention)
  total  : every slice from --zmin up         (Chen+23's z=5-30 convention,
                                               post-EoR included)
and overlays them on scripts/54's Limber results (post_eor_vs_chen23.npz
in the same run dir, if present) and the digitized Chen+23 fiducial.

Why the 2 Gpc/h run is the clean test: L = 2000/0.7 = 2857 Mpc, while
z=5 -> 18 spans ~2790 Mpc of comoving distance -- the stitched LOS never
re-enters the box (one wrap cycle), so periodic-replication artifacts
and wrap-cycle rotation seams are absent by construction. If stitched
!= Limber here, it is not box periodicity.

Distances: chi(z) from AMBER's own H(z) (make_amber_input.chi_amber with
this run's amber_params), NOT astropy Planck18 -- scripts/32/33 go
through stitch_from_coeval.comoving_distance_mpc, which is Planck18
regardless of the run's cosmology (~2% off at z=5 for chen2022 runs).

Needs in --run: amber_run.json, output/cmb/fields_*.dat (raw dumps!),
output/reion/zre_*.dat, output/cmb/tau.txt. Run BEFORE pruning fields.

Usage
-----
    PYTHONPATH=src python -u scripts/55_amber_stitched_streaming.py \
        --run runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

from make_amber_input import chi_amber
from ksz_pipeline.amber import io
from ksz_pipeline.amber.stream_lightcone import stream_ksz_maps, open_zre
from ksz_pipeline.ksz.lightcone_integral import ksz_map_to_Dl
from ksz_pipeline.utils.constants import ne0_cgs


def los_grid(zmin, zmax, cell, ap, n_fine=4000):
    """Uniform-in-chi LOS grid (step = cell) using AMBER's own chi(z) --
    build_los_z_grid's recipe with the run's cosmology."""
    zf = np.linspace(zmin, zmax, n_fine)
    cf = np.array([chi_amber(z, ap) for z in zf])
    n = int(round((cf[-1] - cf[0]) / cell))
    chi = cf[0] + cell * np.arange(n)
    return np.interp(chi, cf, zf), chi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='runs/sweep_2gpc_1536/zmid8_dz4_az3_mh8_mfp3')
    p.add_argument('--zmin', type=float, default=5.0,
                   help="LOS start (Chen+23: z=5); must be >= lowest snapshot")
    p.add_argument('--zmax', type=float, default=None,
                   help="LOS end; default = highest snapshot z")
    p.add_argument('--wrap-cycle-seed', type=int, default=123)
    p.add_argument('--wrap-mode', default='wrap', choices=['wrap', 'grid-wrap'])
    p.add_argument('--chen-csv', default='data/chen23_fiducial.csv')
    p.add_argument('--out', default='data/plots/amber_stitched_vs_limber_2gpc.png')
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    N, L_h, ap = meta['N'], meta['L'], meta['amber_params']
    h = ap['h']
    L = L_h / h
    cell = L / N
    cmb = os.path.join(a.run, 'output', 'cmb')
    files = io.list_field_files(cmb)
    if len(files) < 2:
        raise SystemExit(f"{cmb}: need fields_*.dat (found {len(files)}) -- "
                         f"raw dumps pruned? regenerate first")
    snap_z = np.array([io.z_from_name(f) for f in files])
    zmax = a.zmax if a.zmax is not None else float(snap_z.max())
    if a.zmin < snap_z.min():
        raise SystemExit(f"--zmin {a.zmin} below lowest snapshot {snap_z.min()}")

    z_arr, pos = los_grid(a.zmin, zmax, cell, ap)
    print(f"run {a.run}: N={N}, L={L:.1f} Mpc (cell {cell:.3f}), "
          f"cosmology={meta.get('cosmology_preset')}", flush=True)
    print(f"LOS z={z_arr.min():.3f}-{z_arr.max():.3f}: {z_arr.size} slices, "
          f"span {pos[-1] - pos[0]:.0f} Mpc = {(pos[-1] - pos[0]) / L:.2f} box "
          f"lengths, {len(files)} snapshots", flush=True)

    tau = np.loadtxt(os.path.join(cmb, 'tau.txt'), skiprows=1)
    zre = open_zre(io.find_zre_file(os.path.join(a.run, 'output', 'reion')), N)
    t0 = time.time()
    r = stream_ksz_maps(files, zre, N, L, z_arr, pos, tau,
                        meta.get('ne0_cgs', ne0_cgs()), a.wrap_cycle_seed,
                        a.wrap_mode, progress_every=50)
    print(f"streamed in {time.time() - t0:.0f}s; chi_eff total "
          f"{r['chi_eff_total']:.1f}, patchy {r['chi_eff_patchy']:.1f} Mpc; "
          f"patchy window {r['z_window_patchy']}", flush=True)

    lt, Dt, Et = ksz_map_to_Dl(r['map_total'], L, chi_Mpc=r['chi_eff_total'])
    lp, Dp, Ep = ksz_map_to_Dl(r['map_patchy'], L, chi_Mpc=r['chi_eff_patchy'])
    k_nyq = np.pi * N / L
    print(f"1D Nyquist: ell ~ {k_nyq * r['chi_eff_total']:.0f} (total), "
          f"{k_nyq * r['chi_eff_patchy']:.0f} (patchy)")

    ref = os.path.join(a.run, 'post_eor_vs_chen23.npz')
    R = np.load(ref) if os.path.exists(ref) else None
    chen = np.loadtxt(a.chen_csv, delimiter=",", skiprows=1)

    def at(lq, l, D):
        return float(np.interp(lq, l, D))

    hdr = "   ell   stitched(total)  stitched(patchy)"
    if R is not None:
        hdr += "  Limber(z>=5)  Limber(win)  total/Limber  patchy/Limber"
    print("\n" + hdr + "   Chen+23")
    for lq in (500, 1000, 2000, 3000, 5000, 8000):
        row = f"  {lq:5d}   {at(lq, lt, Dt):9.4f}        {at(lq, lp, Dp):9.4f}"
        if R is not None:
            le, de = R['ells_ext'], R['D_ours_ext']
            lw, dw = R['ells_win'], R['D_ours_win']
            row += (f"        {at(lq, le, de):.4f}       {at(lq, lw, dw):.4f}"
                    f"       {at(lq, lt, Dt) / at(lq, le, de):.3f}"
                    f"         {at(lq, lp, Dp) / at(lq, lw, dw):.3f}")
        row += f"     {at(lq, chen[:, 0], chen[:, 1]):.4f}" if lq >= chen[0, 0] else ""
        print(row)

    out_npz = os.path.join(a.run, 'stitched_streaming_wc%d.npz' % a.wrap_cycle_seed)
    np.savez(out_npz, ell_total=lt, D_total=Dt, Derr_total=Et,
             ell_patchy=lp, D_patchy=Dp, Derr_patchy=Ep,
             chi_eff_total=r['chi_eff_total'], chi_eff_patchy=r['chi_eff_patchy'],
             xe_slices=r['xe_slices'], z_arr=z_arr, pos_axis=pos,
             map_total=r['map_total'].astype(np.float32),
             map_patchy=r['map_patchy'].astype(np.float32))
    print(f"\nwrote {out_npz}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (axx, xs) in enumerate(zip(ax, ('linear', 'log'))):
        axx.plot(chen[:, 0], chen[:, 1], 'o-', color='k', ms=4,
                 label='Chen+23 fiducial (z=5-30)')
        if R is not None:
            axx.plot(R['ells_ext'], R['D_ours_ext'], color='C1', lw=2,
                     label=f'Limber, z>={a.zmin:g} (post-EoR incl.)')
            axx.plot(R['ells_win'], R['D_ours_win'], color='C1', lw=1, ls=':',
                     label='Limber, patchy window')
        axx.errorbar(lt, Dt, Et, color='C2', lw=1.5, marker='.', capsize=0,
                     label=f'stitched + wrap-cycle, z>={a.zmin:g}')
        axx.errorbar(lp, Dp, Ep, color='C2', lw=1, ls=':', marker='.', ms=3,
                     capsize=0, label='stitched + wrap-cycle, patchy window')
        axx.axvline(k_nyq * r['chi_eff_total'], color='0.6', ls='--', lw=1,
                    label='1D Nyquist')
        axx.set_xlabel(r'$\ell$')
        axx.set_ylabel(r'$D_\ell\ [\mu K^2]$')
        if xs == 'log':
            axx.set_xscale('log'); axx.set_yscale('log')
        else:
            axx.set_xlim(0, 8500)
            axx.set_ylim(0, max(2.5, 1.2 * np.nanmax(chen[:, 1])))
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{os.path.basename(a.run)}: L={L_h:g} Mpc/h, N={N}, "
                 f"LOS = {(pos[-1] - pos[0]) / L:.2f} box lengths", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
