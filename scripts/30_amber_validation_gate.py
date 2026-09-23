"""
scripts/30_amber_validation_gate.py

Run after every NEW AMBER configuration (box, resolution, code version),
before trusting any sweep built on it. Two independent codes, same fields:

GATE 1 (must pass -- failure = bug): per-z P_qperp from this repo's
  qperp_power path vs AMBER's own power_z=*.txt, binned in AMBER's own
  linear bins (no interpolation), converted with amber_pqq_to_repo.
  Pass: max |ratio-1| < 1e-3 (float32 fields; found 4.7e-4 on the
  2026-09-10 sandbox run).

GATE 2 (report + tolerance): compute_cell vs AMBER's Limber C_ell,
  in two steps so a mismatch can be attributed:
  2a. rebuild AMBER's cl_ksz.txt from its own power files with its own
      formula AND its own natural-cubic-spline P_qq interpolation
      (cmbreion.f90 + mkl.f90) -- checks we read AMBER correctly.
      Pass: within 3% for ell >= 2000.
  2b. compute_cell vs the AMBER-formula sum over compute_cell's OWN patchy
      window. Pass: within 5% at ell ~ 3000.
  Also reported: fraction of AMBER's D_3000 coming from OUTSIDE the patchy
  window (post-reionization kSZ, excluded from compute_cell by design --
  24% on the sandbox run; history-dependent, so matters for the sweep).

Usage:
  python scripts/30_amber_validation_gate.py --run runs/amber_q01
Exit code 0 = all gates pass, 1 = a gate failed.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.cosmology import Planck18 as P18
from scipy.interpolate import CubicSpline

from ksz_pipeline.amber import io, adapter
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.utils.constants import SIGMA_T, MPC_CM, C_CGS, T_CMB_K, ne0_cgs


def shell_edges(z, zdel, spacing, zmin):
    """AMBER cmb_ray shell [z1, z2] around midpoint z (lin or log)."""
    if spacing == 'lin':
        return z - zdel / 2, z + zdel / 2
    dlg = np.log10(1 + zdel / (1 + zmin))
    return (1 + z) * 10**(-dlg / 2) - 1, (1 + z) * 10**(dlg / 2) - 1


def gate1(run, N, L_h, h, zre):
    worst, n = 0.0, 0
    Lm = L_h / h
    f = np.fft.fftfreq(N, Lm / N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(f, f, f, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2s = np.where(k2 == 0, np.inf, k2)
    kk = np.rint(np.sqrt(k2) / (2 * np.pi / Lm)).astype(int)
    for fpath in io.list_field_files(os.path.join(run, 'output', 'cmb')):
        z, rho, mom = io.read_fields(fpath, N)
        ion = z < zre
        if ion.mean() < 1e-3 or ion.mean() > 0.999:
            continue
        q = mom.astype(np.float64) * adapter.KMS_TO_CMS * ion
        Q = np.fft.fftn(q, axes=(1, 2, 3)) * (Lm / N)**3
        kd = (Q[0] * kx + Q[1] * ky + Q[2] * kz) / k2s
        p = (np.abs(Q - np.stack([kx, ky, kz]) * kd)**2).sum(0) / Lm**3 / 2
        ours = np.array([p[kk == b].mean() for b in range(1, N // 2 + 1)])
        pf = os.path.join(run, 'output', 'cmb', f'power_z={z:05.2f}.txt')
        a = io.read_amber_power(pf)
        _, Pa = adapter.amber_pqq_to_repo(a['k_h'], a['P_qq'], h)
        r = ours / Pa[:N // 2]
        worst = max(worst, float(np.abs(r - 1).max()))
        n += r.size
    ok = n > 0 and worst < 1e-3
    print(f"GATE 1  per-z P_qperp, same bins, {n} (z,k) points: "
          f"max|ratio-1| = {worst:.2e}   {'PASS' if ok else 'FAIL'}")
    return ok


def gate2(run, meta, N, L_h, h, zre):
    cmb = os.path.join(run, 'output', 'cmb')
    t = np.loadtxt(os.path.join(cmb, 'tau.txt'), skiprows=1)
    la, Ca = io.read_amber_cl(os.path.join(cmb, 'cl_ksz.txt'))
    xe = adapter.electron_fraction_amber(meta['XH'], meta['YHe'])
    ne0_amber = ne0_cgs() / xe                      # nH + 2 nHe

    pw = sorted(glob.glob(os.path.join(cmb, 'power_z=*.txt')), key=io.z_from_name)
    zs = np.array([io.z_from_name(f) for f in pw])
    ps = [io.read_amber_power(f) for f in pw]

    def amber_sum(zlo, zhi):
        C = np.zeros(la.shape)
        for z, p in zip(zs, ps):
            if not (zlo - 1e-9 <= z <= zhi + 1e-9):
                continue
            z1, z2 = shell_edges(z, meta['czdel'], meta['cspacing'], meta['czmin'])
            r = P18.comoving_distance(z).value * h                      # Mpc/h
            dr = (P18.comoving_distance(z2).value
                  - P18.comoving_distance(z1).value) * h
            a = 1 / (1 + z)
            tau = np.interp(z, t[:, 0], t[:, 2])
            # AMBER evaluates P_qq(k=l/r) with a NATURAL cubic spline
            # (mkl.f90 spline_cubic: DF_PP_NATURAL / DF_BC_FREE_END), not
            # linear interpolation. On coarse k tables (small boxes) the
            # two differ by several % at low ell, where k=l/r falls in the
            # first couple of bins -- so match AMBER exactly here.
            Pq = CubicSpline(p['k_h'], p['P_qq'], bc_type='natural')(la / r)
            C += ((SIGMA_T * ne0_amber / C_CGS)**2 * (MPC_CM / h)**2 * 1e10
                  * (Pq / 2) * np.exp(-2 * tau) / (r**2 * a**4) * dr)
        return C

    def D(C, l):
        return l * (l + 1) / (2 * np.pi) * C * (T_CMB_K * 1e6)**2

    Cfull = amber_sum(-1, 1e3)
    sel = la >= 2000
    dev_a = float(np.abs(Cfull[sel] / Ca[sel] - 1).max())
    ok_a = dev_a < 0.03
    print(f"GATE 2a rebuild AMBER cl_ksz.txt from its power files (ell>=2000): "
          f"max|ratio-1| = {dev_a:.2%}   {'PASS' if ok_a else 'FAIL'}")

    res = adapter.results_qperp_from_amber(
        io.list_field_files(cmb), zre, N, L_h, h, verbose=False)
    # tau below the patchy window must come from AMBER's own history, not
    # from the repo's fixed analytic model -- see adapter.tau_below_amber.
    zwin = min(z for z in res
               if 1e-4 <= res[z]['xH_mean'] <= 1 - 1e-4)
    ells, Dl, _, _, _, (zt, _), _ = compute_cell(
        res, tau0=adapter.tau_below_amber(cmb, zwin))
    Cwin = amber_sum(zt.min(), zt.max())
    i3 = int(np.argmin(np.abs(ells - 3000)))
    D_win = np.interp(ells, la, D(Cwin, la))
    r3 = Dl[i3] / D_win[i3]
    ok_b = abs(r3 - 1) < 0.05
    j3 = int(np.argmin(np.abs(la - 3000)))
    outside = 1 - Cwin[j3] / Ca[j3]
    print(f"GATE 2b compute_cell vs AMBER formula, patchy window z="
          f"{zt.min():.2f}-{zt.max():.2f}: D_{ells[i3]} ratio = {r3:.3f}   "
          f"{'PASS' if ok_b else 'FAIL'}")
    print(f"        (info) AMBER D_3000 from OUTSIDE the patchy window: "
          f"{outside:.1%}  [post-reionization kSZ, excluded by design]")
    print("\n   ell   compute_cell  AMBER(window)  AMBER(full)   [uK^2]")
    for l in (1000, 2000, 3000, 5000, 8000):
        i = int(np.argmin(np.abs(ells - l)))
        print(f"  {ells[i]:5d}   {Dl[i]:.4f}        {D_win[i]:.4f}        "
              f"{np.interp(ells[i], la, D(Ca, la)):.4f}")
    np.savez(os.path.join(run, 'amber_gate.npz'), ells=ells, D_compute_cell=Dl,
             D_amber_window=D_win, ell_amber=la, D_amber_full=D(Ca, la),
             window=np.array([zt.min(), zt.max()]), frac_outside=outside,
             hist_z=np.array(sorted(res)),
             hist_xH_vol=np.array([res[z]['xH_mean'] for z in sorted(res)]),
             hist_xH_mass=np.array([res[z]['xH_mass'] for z in sorted(res)]))
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='dir containing amber_run.json')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, 'amber_run.json')))
    N, L_h, h = meta['N'], meta['L'], meta['amber_params']['h']
    if meta.get('cspacing') not in ('lin', 'log'):
        sys.exit("amber_run.json lacks cspacing")
    zre = io.read_zre(io.find_zre_file(os.path.join(a.run, 'output', 'reion')), N)
    print(f"run {a.run}: L={L_h} Mpc/h N={N} z_mid={meta['zmid']} "
          f"dz={meta['zdel']} A_z={meta['zasy']}\n")
    ok = gate1(a.run, N, L_h, h, zre) & gate2(a.run, meta, N, L_h, h, zre)
    print("\nALL GATES PASS" if ok else "\nGATE FAILURE -- do not build on this run")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
