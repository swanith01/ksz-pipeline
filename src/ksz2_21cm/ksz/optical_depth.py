"""
Global optical depth tau(z) and visibility function.

VENDORED from ksz-pipeline (github.com/swanith01/ksz-pipeline), commit
7305482def70cd94ab1f7b553db13070fd619b76 (2026-10-06), file
src/ksz_pipeline/ksz/optical_depth.py, verbatim below this header.
Unchanged -- this is kSZ-side machinery, field-agnostic with respect to
the 21cm addition, so no extension was needed.

Extracted from Cell 4 of 16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.

tau(z) = sigma_T * ne0 * integral_0^z x_e(z') (1+z')^2 dchi
visibility(z) = exp(-tau(z))
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
from scipy.integrate import quad
from ..utils.constants import SIGMA_T, MPC_CM, NE0_HYDROGEN_ONLY


def analytic_tau_below(z_min, Y_He=0.24, z_HeIII=3.5, dz_HeIII=0.5):
    """
    Thomson optical depth from z=0 to z_min, assuming the IGM is fully
    ionized throughout -- i.e. the homogeneous, post-reionization
    universe below whatever redshift the lightcone/coeval pipeline
    actually simulates down to.

    Builds n_H0 from cosmology directly (not from a caller-supplied ne0,
    whose H-only vs H+He convention varies elsewhere -- see constants.py's
    ne0_cgs() docstring), then adds helium explicitly:
      - HeII (singly ionized): assumed complete throughout 0<=z<=z_min.
      - HeIII (doubly ionized): a smooth tanh transition centered at
        z_HeIII (default 3.5, width 0.5) -- CAMB's standard fiducial
        default for helium reionization timing.

    Parameters
    ----------
    z_min    : float   lowest redshift the pipeline actually simulates
    Y_He     : float   helium mass fraction (default 0.24)
    z_HeIII  : float   fiducial HeII->HeIII transition redshift (default 3.5)
    dz_HeIII : float   transition width (default 0.5)

    Returns
    -------
    tau_below : float, the optical depth accumulated over 0 <= z <= z_min
    """
    m_p    = const.m_p.cgs.value
    rho_c0 = cosmo.critical_density0.cgs.value
    X_H    = 1.0 - Y_He
    n_H0   = X_H * (cosmo.Ob0 * rho_c0) / m_p
    y      = Y_He / (4.0 * X_H)          # He/H number ratio

    def integrand(z):
        f_HeIII = 0.5 * (1.0 + np.tanh((z_HeIII - z) / dz_HeIII))
        n_e = n_H0 * (1.0 + y * (1.0 + f_HeIII)) * (1.0 + z)**2
        Hz  = cosmo.H(z).to('1/s').value
        c_cgs = const.c.cgs.value
        return n_e * c_cgs / Hz

    tau_below, _ = quad(integrand, 0.0, z_min)
    return SIGMA_T * tau_below


def compute_tau(x_e_interp, red_axis, pos_axis, ne0=None, tau0=0.0):
    """
    Cumulative Thomson optical depth tau(<z) along the lightcone.

    Midpoint trapezoidal integration over comoving distance slices.
    Pass tau0=analytic_tau_below(red_axis.min()) to correct for the
    optical depth accumulated below the lowest simulated redshift;
    defaults to 0.0 (the old, biased-low behavior) so existing callers
    don't silently change until they opt in.

    Parameters
    ----------
    x_e_interp : ndarray (N,)
        Ionization fraction x_e = 1 - x_HI interpolated onto red_axis.
    red_axis   : ndarray (N,)
        Redshift at each lightcone slice (low-z to high-z).
    pos_axis   : ndarray (N,)
        Comoving distance at each slice [Mpc].
    ne0        : float, optional
        Mean electron density today [cm^-3].
        Defaults to NE0_HYDROGEN_ONLY (2.06e-7) matching lightcone script.
    tau0       : float, optional
        Optical depth already accumulated below red_axis[0], e.g. from
        analytic_tau_below(red_axis.min()). Default 0.0.

    Returns
    -------
    z_mid  : ndarray (N-1,)   midpoint redshifts
    ds     : ndarray (N-1,)   comoving slice widths [Mpc]
    dtau   : ndarray (N-1,)   optical depth per slice
    tau    : ndarray (N-1,)   cumulative tau(<z), including tau0
    """
    if ne0 is None:
        ne0 = NE0_HYDROGEN_ONLY

    s      = np.asarray(pos_axis, dtype=float)
    ds     = np.diff(s)                                    # [Mpc]
    z_mid  = 0.5 * (red_axis[:-1] + red_axis[1:])
    xe_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

    # prefactor: sigma_T * ne0 in Mpc^-1 units
    # sigma_T [cm^2] * ne0 [cm^-3] * ds [Mpc] * MPC_CM [cm/Mpc] -> dimensionless
    prefactor = ne0 * SIGMA_T                              # [cm^-1]
    dtau = prefactor * xe_mid * (1.0 + z_mid)**2 * ds * MPC_CM

    tau = tau0 + np.cumsum(dtau)
    return z_mid, ds, dtau, tau


def compute_patchy_mask(x_e_interp, xe_min=1.0e-4, xe_max=1.0 - 1.0e-4):
    """
    Boolean-as-float mask selecting the patchy regime: 99.99% neutral to
    0.01% neutral (x_e between xe_min and xe_max), NOT a fixed redshift
    cut.

    Use this to clip the kSZ SIGNAL integral (compute_ksz_map's
    patchy_mask_3D argument) to the patchy regime only. Do NOT use it to
    clip tau/visibility -- those need the full, continuous z_min..z_max
    range to be physically correct.

    Parameters
    ----------
    x_e_interp : ndarray (N,)  ionization fraction x_e = 1 - x_HI at
                 each lightcone slice (same array compute_tau takes)
    xe_min, xe_max : float     patchy-regime bounds on x_e (defaults:
                 0.01% neutral to 99.99% neutral)

    Returns
    -------
    mask    : ndarray (N,)      1.0 inside the patchy regime, 0.0 outside
    mask_3D : ndarray (1,1,N)   broadcast-ready for 3D field multiplication
    """
    mask = ((x_e_interp >= xe_min) & (x_e_interp <= xe_max)).astype(float)
    return mask, mask[None, None, :]


def compute_visibility(tau, red_axis, z_mid):
    """
    Interpolate tau onto red_axis and return visibility exp(-tau).

    Parameters
    ----------
    tau      : ndarray (N-1,)  cumulative tau from compute_tau
    red_axis : ndarray (N,)    lightcone redshift grid
    z_mid    : ndarray (N-1,)  midpoint redshifts from compute_tau

    Returns
    -------
    tau_at_lc    : ndarray (N,)   tau interpolated onto red_axis
    visibility   : ndarray (N,)   exp(-tau_at_lc)
    visibility3D : ndarray (1,1,N) broadcast-ready for 3D field multiplication
    """
    tau_extended  = np.concatenate([[0.0], np.asarray(tau, dtype=float)])
    z_extended    = np.concatenate([[red_axis[0]], z_mid])
    tau_at_lc     = np.interp(red_axis, z_extended, tau_extended)
    visibility    = np.exp(-tau_at_lc)
    visibility3D  = visibility[None, None, :]
    return tau_at_lc, visibility, visibility3D
