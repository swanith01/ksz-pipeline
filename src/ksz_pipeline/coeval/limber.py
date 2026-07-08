"""
kSZ angular power spectrum via Limber projection.

Extracted from Cell 7 of 28May2026_kSZBoxes.py.

Implements Cain+2024 Eq.(3) / Park+2013 Appendix Eq.(A16):

    C_ell = (sigma_T ne0 / c)^2
            * integral e^{-2*tau} / (s^2 * a^4) * P_{q_perp}(k=l/s) ds

Error propagation (slices treated as independent):
    sigma^2_C = pref^2 * sum_i (w_i * sigma_P_i)^2 * MPC_CM^4

Units (verified dimensionally):
    pref = (sigma_T ne0 / c)^2         [s^2 cm^-4]
    w    = ds / (s^2 a^4)              [Mpc^-1]
    P    = P_{q_perp}(k)              [cm^2 s^-2 Mpc^3]
    MPC_CM^2                           [cm^2 Mpc^-2]
    product: dimensionless             check OK

D_ell = l(l+1)/(2pi) * C_ell * T_CMB^2 [uK^2]
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from ..utils.constants import SIGMA_T, MPC_CM, C_CGS, T_CMB_K, ne0_cgs
from ..ksz.optical_depth import analytic_tau_below


def _interp_loglog(xq, xp, fp):
    """Interpolate in log-log space, clipping to [xp.min, xp.max]."""
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    m  = (xp > 0) & (fp > 0)
    lx = np.log(xp[m])
    lf = np.log(fp[m])
    lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
    return np.exp(np.interp(lq, lx, lf))


def compute_cell(results_qperp, ells_full=None, ne0=None):
    """
    Compute C_ell and D_ell from cached P_{q_perp} results.

    Parameters
    ----------
    results_qperp : dict
        Keyed by redshift z. Each entry must have:
            'k'       : ndarray [Mpc^-1]
            'Pqperp'  : ndarray [cm^2 s^-2 Mpc^3]
            'Pstd'    : ndarray  std per bin
            'xH_mean' : float    volume-averaged x_HI
    ells_full : ndarray, optional
        Multipoles to evaluate. Default: 80 log-spaced points from
        l=100 to l=31623 (10^4.5).
    ne0 : float, optional
        Mean electron density [cm^-3]. Default: ne0_cgs() with helium.

    Returns
    -------
    ells       : ndarray  valid multipoles
    D_ell      : ndarray  [uK^2]
    sigma_D    : ndarray  1-sigma error [uK^2]
    C_ell      : ndarray  [dimensionless]
    sigma_C    : ndarray
    tau_out    : tuple (ZS_asc, tau)  optical depth history
    xe_out     : tuple (ZS_asc, xe)  ionization history
    """
    if ne0 is None:
        ne0 = ne0_cgs()

    if ells_full is None:
        ells_full = np.unique(
            np.round(np.logspace(2.0, 4.5, 80)).astype(int)
        ).astype(int)

    pref    = (SIGMA_T * ne0 / C_CGS)**2    # [s^2 cm^-4]
    c_cms   = C_CGS

    # Patchy regime only: 99.99% neutral to 0.01% neutral (xH_mean bounds),
    # not a hardcoded redshift. FIXED: previously "z >= 5.0", a fixed
    # redshift that isn't guaranteed to track the true start/end of the
    # patchy regime across simulation setups or code versions -- xH_mean
    # is already computed per snapshot, so use it directly instead.
    XHI_MIN_PATCHY = 1.0e-4          # 0.01% neutral -> end of patchy regime
    XHI_MAX_PATCHY = 1.0 - 1.0e-4    # 99.99% neutral -> start of patchy regime
    ZS_asc = np.array(sorted([z for z in results_qperp.keys()
                               if XHI_MIN_PATCHY <= results_qperp[z]['xH_mean']
                                                  <= XHI_MAX_PATCHY]),
                       dtype=float)

    chi_mpc  = np.array([cosmo.comoving_distance(z).value for z in ZS_asc])
    dchi_mpc = np.abs(np.gradient(chi_mpc))
    dchi_cm  = dchi_mpc * MPC_CM

    xe_arr = np.array([1.0 - results_qperp[z]['xH_mean'] for z in ZS_asc])

    # Optical depth tau(z). FIXED: previously started tau=0 at the lowest
    # kept redshift (ZS_asc.min(), >=5 due to the patchy-only filter
    # above), silently dropping the real tau accumulated from z=0 to
    # there -- roughly half the true total for a typical z_min~5-6 (see
    # analytic_tau_below()'s docstring). That made e^{-2tau(z)} too large
    # at every z, inflating D_ell by roughly exp(2*tau_below) ~ 6-8%.
    tau0 = analytic_tau_below(ZS_asc.min())
    tau  = np.full_like(ZS_asc, tau0)
    for i in range(len(ZS_asc) - 1):
        zmid   = 0.5 * (ZS_asc[i]  + ZS_asc[i + 1])
        xe_mid = 0.5 * (xe_arr[i]  + xe_arr[i + 1])
        tau[i + 1] = tau[i] + (SIGMA_T * ne0
                                * xe_mid * (1.0 + zmid)**2
                                * dchi_cm[i])

    # Valid ell range
    k_max_sim     = min(results_qperp[z]['k'].max() for z in ZS_asc)
    k_min_sim     = max(results_qperp[z]['k'].min() for z in ZS_asc)
    ell_max_valid = int(k_max_sim * chi_mpc.min())
    ell_min_valid = int(k_min_sim * chi_mpc.max())
    ells = ells_full[(ells_full >= ell_min_valid) & (ells_full <= ell_max_valid)]

    # C_ell integral with error propagation
    C_ell     = np.zeros(len(ells), dtype=float)
    var_C_ell = np.zeros(len(ells), dtype=float)

    for i, z in enumerate(ZS_asc):
        s_mpc = chi_mpc[i]
        if s_mpc <= 0.0:
            continue
        a_i  = 1.0 / (1.0 + z)
        vis2 = np.exp(-2.0 * tau[i])
        w    = vis2 / (s_mpc**2 * a_i**4) * dchi_mpc[i]   # [Mpc^-1]

        k_ell = ells / s_mpc
        P_now = _interp_loglog(k_ell,
                               results_qperp[z]['k'],
                               results_qperp[z]['Pqperp'])
        S_now = _interp_loglog(k_ell,
                               results_qperp[z]['k'],
                               results_qperp[z]['Pstd'])

        C_ell     += pref * w * P_now * MPC_CM**2
        var_C_ell += (pref * w * S_now * MPC_CM**2)**2

    sigma_C_ell = np.sqrt(var_C_ell)

    prefD   = ells * (ells + 1.0) / (2.0 * np.pi) * T_CMB_K**2 * 1e12
    D_ell   = prefD * C_ell
    sigma_D = prefD * sigma_C_ell

    return (ells, D_ell, sigma_D, C_ell, sigma_C_ell,
            (ZS_asc, tau), (ZS_asc, xe_arr))
