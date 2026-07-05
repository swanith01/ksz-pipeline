"""
Patchy optical-depth screening: tau(n_hat, z) cube.

Extracted from Cell 4c of 16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.

In the standard kSZ calculation the visibility is approximated using the
sky-averaged optical depth exp(-tau_bar(z)). Here we construct a fully
patchy visibility field exp(-tau(n_hat, <z)) by integrating the LOCAL
electron density along each line of sight.

Result (Semester-6 notes sec 1.3, Figures 5-7):
    - Mean fractional difference in D_ell: 0.3%
    - Maximum deviation: < 0.6%
    - tau(n_hat) rms fluctuations: 1.4%
    => global-tau approximation is sufficient for current lightcone runs.
"""

import numpy as np
from ..utils.constants import SIGMA_T, MPC_CM, NE0_HYDROGEN_ONLY


def compute_patchy_tau(density_field, xH_field, ind_z, ds, z_mid,
                       ne0=None):
    """
    Build the cumulative patchy optical-depth cube tau(n_hat, <z).

    Parameters
    ----------
    density_field : ndarray (Nx, Ny, Nz_full)  matter overdensity delta
    xH_field      : ndarray (Nx, Ny, Nz_full)  neutral fraction x_HI
    ind_z         : ndarray  indices selecting the redshift range of interest
    ds            : ndarray (Nz-1,)  comoving slice widths [Mpc]
    z_mid         : ndarray (Nz-1,)  midpoint redshifts
    ne0           : float, optional  [cm^-3], defaults to NE0_HYDROGEN_ONLY

    Returns
    -------
    tau_patchy_cube : ndarray (Nx, Ny, Nz)  cumulative tau(n_hat, <z)
    tau_patchy_map  : ndarray (Nx, Ny)      total tau(n_hat) through box
    visibility_patchy_3D : ndarray (Nx, Ny, Nz)  exp(-tau_patchy_cube)
    """
    if ne0 is None:
        ne0 = NE0_HYDROGEN_ONLY

    # Local free electron field: n_e(x,z) / ne0 = (1+delta) * x_HII
    density_1plus = 1.0 + density_field[:, :, ind_z]
    x_HII_field   = 1.0 - xH_field[:, :, ind_z]
    n_e_local     = density_1plus * x_HII_field       # (Nx, Ny, Nz)

    Nx, Ny, Nz = n_e_local.shape

    ds_arr  = np.asarray(ds,    dtype=float)           # (Nz-1,) [Mpc]
    z_mid_a = np.asarray(z_mid, dtype=float)           # (Nz-1,)
    one_plus_z2_mid = (1.0 + z_mid_a)**2              # (Nz-1,)

    # Midpoint electron density
    n_e_mid = 0.5 * (n_e_local[:, :, :-1] + n_e_local[:, :, 1:])

    # Patchy dtau per cell: (Nx, Ny, Nz-1)
    prefactor = ne0 * SIGMA_T                          # [cm^-1]
    dtau_patchy = (prefactor
                   * n_e_mid
                   * one_plus_z2_mid[None, None, :]
                   * ds_arr[None, None, :]
                   * MPC_CM)

    # Cumulative tau from observer outward (low-z → high-z)
    tau_cube = np.cumsum(dtau_patchy, axis=2)          # (Nx, Ny, Nz-1)

    # Prepend zero slice to align with Nz redshift grid
    tau_patchy_cube = np.concatenate(
        [np.zeros((Nx, Ny, 1)), tau_cube], axis=2)    # (Nx, Ny, Nz)

    tau_patchy_map       = tau_patchy_cube[:, :, -1]  # (Nx, Ny)
    visibility_patchy_3D = np.exp(-tau_patchy_cube)   # (Nx, Ny, Nz)

    return tau_patchy_cube, tau_patchy_map, visibility_patchy_3D
