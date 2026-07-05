"""
Global optical depth tau(z) and visibility function.

Extracted from Cell 4 of 16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.

tau(z) = sigma_T * ne0 * integral_0^z x_e(z') (1+z')^2 dchi
visibility(z) = exp(-tau(z))
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from ..utils.constants import SIGMA_T, MPC_CM, NE0_HYDROGEN_ONLY


def compute_tau(x_e_interp, red_axis, pos_axis, ne0=None):
    """
    Cumulative Thomson optical depth tau(<z) along the lightcone.

    Mirrors Cell 4 exactly: midpoint trapezoidal integration over
    comoving distance slices.

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

    Returns
    -------
    z_mid  : ndarray (N-1,)   midpoint redshifts
    ds     : ndarray (N-1,)   comoving slice widths [Mpc]
    dtau   : ndarray (N-1,)   optical depth per slice
    tau    : ndarray (N-1,)   cumulative tau(<z)
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

    tau = np.cumsum(dtau)
    return z_mid, ds, dtau, tau


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
