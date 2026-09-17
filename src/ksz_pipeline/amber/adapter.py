"""
ksz_pipeline/amber/adapter.py

Converts AMBER fields into the exact inputs this repo's existing,
validated functions already take -- so AMBER plugs in UNDER
compute_cell / qperp_power / coherence_decomposition rather than
replacing any of them. The kSZ methods stay identical across
simulation codes; only the fields change.

Conventions mapped (each checked against AMBER source, main branch):

  quantity        AMBER                          this repo
  --------------  -----------------------------  -------------------------
  length          Mpc/h                          Mpc      (x 1/h)
  k               h/Mpc                          1/Mpc    (x h)
  velocity        km/s, PROPER peculiar          cm/s, same physical
                  (unit%vel = a*L/t)             quantity (x 1e5)
                                                 -- NO extra (1+z) factor:
                                                 validation_table.md §4's
                                                 Psi*D*f*H/(1+z) is a*dx/dt,
                                                 i.e. this same velocity,
                                                 despite being labelled
                                                 "comoving" there.
  momentum        mom2 = (1+delta)*v             built from delta, v
  ionization      binary: ionized iff z < zre    x_HI field in [0,1]
                  (cmbreion.f90 test)
  electrons       ne0 = nH+2nHe, times           ne0_cgs() = nH+nHe,
                  x_e=(X+Y/4)/(X+Y/2) inside q   applied in prefactor
  P_qperp         P_qq: no 1/2, includes x_e^2   includes 1/2, no x_e
  cosmology       input.txt                      astropy Planck18 (hard-
                                                 coded in limber.py) --
                                                 AMBER MUST be run with
                                                 Planck18 params, see
                                                 scripts/make_amber_input.py

The conversion factor for P_qq is unit-tested against a Python port of
AMBER's own estimator (tests/test_amber_adapter.py).
"""
import numpy as np

from ..coeval.momentum import qperp_power_from_momentum
from . import io

KMS_TO_CMS = 1.0e5


def electron_fraction_amber(XH=0.76, YHe=0.24):
    """AMBER's x_e = n_e/n_e,tot for H ionized + He singly ionized."""
    return (XH + YHe / 4.0) / (XH + YHe / 2.0)


def ionized_mask(zre, z):
    """AMBER's own criterion (cmbreion.f90): cell ionized iff z < zre."""
    return z < zre


def snapshot(zre, rho, mom, z, h):
    """
    One AMBER snapshot in repo units.

    Returns dict with
      'xH'      : (N,N,N) float32 neutral fraction (binary 0/1)
      'delta'   : (N,N,N) rho - 1
      'p_cms'   : (3,N,N,N) (1+delta)*v [cm/s]
      'xH_mean' : volume-weighted mean x_HI (what compute_cell's patchy
                  window and tau use)
      'xH_mass' : mass-weighted mean x_HI (what AMBER's history is
                  specified in -- NOT equal to xH_mean)
      'z'       : float
    """
    ion = ionized_mask(zre, z)
    xH = (~ion).astype(np.float32)
    return {
        'xH': xH,
        'delta': rho - 1.0,
        'p_cms': mom.astype(np.float64) * KMS_TO_CMS,
        'xH_mean': float(xH.mean()),
        'xH_mass': float((xH * rho).sum() / rho.sum()),
        'z': float(z),
    }


def velocity_from_momentum(rho, mom_cms, rho_floor=1e-6):
    """
    v = p/(1+delta) for code paths that insist on (delta, v) separately
    (e.g. stitch_from_coeval). Where rho <= rho_floor -- possible, since
    rho2 is interlaced+deconvolved and can dip to ~0 or below -- v is set
    to 0 (same rule AMBER itself uses for vel1). Returns the masked cell
    fraction so the caller can report it; density_1plus * v then equals
    p everywhere EXCEPT those cells.

    Prefer qperp_power_from_momentum where possible; it needs no division.
    """
    bad = rho <= rho_floor
    safe = np.where(bad, 1.0, rho)
    v = np.where(bad[None], 0.0, mom_cms / safe[None])
    return v, float(bad.mean())


def results_qperp_from_amber(field_files, zre, N, L_box_mpc_h, h,
                             nbins=None, verbose=True):
    """
    Build compute_cell's input dict directly from AMBER snapshots.

    Parameters
    ----------
    field_files : list of fields_*.dat paths (io.list_field_files)
    zre         : (N,N,N) reionization-redshift field (io.read_zre)
    N           : mesh cells per side
    L_box_mpc_h : AMBER box length [Mpc/h]
    h           : Hubble parameter used in the AMBER run

    Returns
    -------
    dict keyed by z (float, from inside each file -- the filename tag
    is rounded to 2 decimals), each entry with 'k', 'Pqperp', 'Pstd',
    'xH_mean' exactly as compute_cell expects, plus 'xH_mass' and
    'rho_mean' for diagnostics.
    """
    L_mpc = L_box_mpc_h / h
    out = {}
    for f in field_files:
        z, rho, mom = io.read_fields(f, N)
        rho_mean = float(rho.mean())
        # rho2 is 1+delta normalized to mean 1 by construction; a large
        # departure means the layout/field assumption is wrong -- fail.
        if abs(rho_mean - 1.0) > 1e-2:
            raise RuntimeError(
                f"{f}: <rho2> = {rho_mean:.4f}, expected ~1. Field is not "
                f"1+delta as assumed -- check the dump patch / layout.")
        s = snapshot(zre, rho, mom, z, h)
        k, P, Ps = qperp_power_from_momentum(
            s['p_cms'][0], s['p_cms'][1], s['p_cms'][2], s['xH'],
            L_mpc, nbins=nbins)
        if z in out:
            raise RuntimeError(f"duplicate snapshot z={z} ({f})")
        out[z] = {'k': k, 'Pqperp': P, 'Pstd': Ps,
                  'xH_mean': s['xH_mean'], 'xH_mass': s['xH_mass'],
                  'rho_mean': rho_mean}
        if verbose:
            print(f"  z={z:7.4f}  xH_vol={s['xH_mean']:.5f}  "
                  f"xH_mass={s['xH_mass']:.5f}  <rho>={rho_mean:.5f}")
    return out


def amber_pqq_to_repo(k_h, P_qq, h, XH=0.76, YHe=0.24):
    """
    AMBER power_<zstr>.txt P_qq -> this repo's P_qperp convention.

      k    [h/Mpc]              -> [1/Mpc]              : x h
      P_qq [(Mpc/h)^3 (km/s)^2] -> [Mpc^3 (cm/s)^2]     : x 1e10 / h^3
      AMBER omits the Limber 1/2 (applies it in C_ell) : x 1/2
      AMBER's q carries x_e; repo's carries it in ne0  : x 1/x_e^2
    """
    xe = electron_fraction_amber(XH, YHe)
    k = np.asarray(k_h) * h
    P = np.asarray(P_qq) * KMS_TO_CMS**2 / h**3 / 2.0 / xe**2
    return k, P


def tau_below_amber(cmb_dir, z):
    """
    tau(0 -> z) from AMBER's own output/cmb/tau.txt, for compute_cell's
    tau0. AMBER integrates tau with the SAME history it simulated, so
    across a (z_mid, Delta_z, A_z) sweep this moves as it should, while
    analytic_tau_below() would stay fixed at one history's value.

    Note tau.txt is mass/volume consistent with AMBER's own x_e(z) and
    includes helium as AMBER treats it; it is the right partner for the
    fields this adapter feeds in.
    """
    import os
    t = np.loadtxt(os.path.join(cmb_dir, 'tau.txt'), skiprows=1)
    if not (t[:, 0].min() <= z <= t[:, 0].max()):
        raise ValueError(f"z={z} outside tau.txt range "
                         f"[{t[0,0]}, {t[-1,0]}]")
    return float(np.interp(z, t[:, 0], t[:, 2]))
