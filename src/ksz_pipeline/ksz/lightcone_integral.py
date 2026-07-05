"""
Line-of-sight kSZ integration: delta_T/T map from the lightcone.

Extracted from Cells 5 and 6 of
16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.

Works on both the full 3D lightcone arrays (Nx, Ny, Nz) and on
the skewer arrays (Nlos, Nz) produced by skewed_los.py.
"""

import numpy as np
from ..utils.constants import SIGMA_T, MPC_CM, C_CGS, T_CMB_K, NE0_HYDROGEN_ONLY


def compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                    red_axis, ds, visibility_3D, ne0=None):
    """
    LOS-integrated kSZ temperature fluctuation map delta_T/T (dimensionless).

    Mirrors Cell 6 exactly.

    Parameters
    ----------
    density_1plus : ndarray (..., Nz)   1 + delta
    x_HII_field   : ndarray (..., Nz)   ionized fraction = 1 - x_HI
    v_los_Mpc_s   : ndarray (..., Nz)   LOS velocity [Mpc/s] = lightcone.velocity / H0
    red_axis      : ndarray (Nz,)       redshift at each slice
    ds            : ndarray (Nz-1,)     comoving slice widths [Mpc]
    visibility_3D : ndarray (..., Nz)   exp(-tau) broadcast to field shape
    ne0           : float, optional     [cm^-3], defaults to NE0_HYDROGEN_ONLY

    Returns
    -------
    ksz_map : ndarray (...)  dimensionless delta_T/T integrated along LOS
    """
    if ne0 is None:
        ne0 = NE0_HYDROGEN_ONLY

    c_Mpc_s   = C_CGS / MPC_CM
    prefactor = ne0 * SIGMA_T * C_CGS             # [1/s]

    a         = 1.0 / (1.0 + red_axis)
    a_squared = a**2

    integrand_base = density_1plus * x_HII_field * v_los_Mpc_s / c_Mpc_s
    integrand_vis  = integrand_base * visibility_3D

    integ_mid = 0.5 * (integrand_vis[..., :-1] + integrand_vis[..., 1:])
    a2_mid    = 0.5 * (a_squared[:-1] + a_squared[1:])
    ds_cm     = np.asarray(ds, dtype=float) * MPC_CM

    full    = (prefactor / a2_mid) * integ_mid * (ds_cm / C_CGS)
    ksz_map = np.sum(full, axis=-1)
    return ksz_map


def ksz_map_to_Dl(ksz_map, box_len_Mpc, chi_Mpc=7800.0, h=0.67, n_kbins=35):
    """
    2D FFT power spectrum of the kSZ map -> D_ell [uK^2].

    Mirrors Cell 8 / angular_power_2d() from the lightcone script.

    Parameters
    ----------
    ksz_map     : ndarray (Npix, Npix)  dimensionless delta_T/T map
    box_len_Mpc : float                 physical side length [Mpc]
    chi_Mpc     : float                 comoving distance proxy for ell<->k
    h           : float                 dimensionless Hubble parameter
    n_kbins     : int                   number of radial k bins

    Returns
    -------
    ell     : ndarray  multipoles (valid bins only)
    Dl      : ndarray  D_ell [uK^2]
    Dl_err  : ndarray  1-sigma error on D_ell
    """
    T_CMB_uK = T_CMB_K * 1e6
    N        = ksz_map.shape[0]
    pix_Mpc  = box_len_Mpc / N

    m     = ksz_map - ksz_map.mean()
    fft_m = np.fft.fftshift(np.fft.fft2(m))
    ps2d  = (pix_Mpc / N)**2 * np.abs(fft_m)**2

    dk = 2.0 * np.pi / (N * pix_Mpc)
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    kg = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kg.max() * 0.9), n_kbins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    k_area    = dk**2

    P1d = np.zeros(len(k_centers))
    Err = np.zeros(len(k_centers))

    for i in range(len(k_centers)):
        mask   = (kg >= k_bins[i]) & (kg < k_bins[i + 1])
        n_meas = mask.sum()
        if n_meas > 0:
            vals     = ps2d[mask]
            P1d[i]   = vals.mean()
            err_samp = vals.std() / np.sqrt(n_meas)
            ring_area = np.pi * (k_bins[i+1]**2 - k_bins[i]**2)
            err_cv   = P1d[i] / np.sqrt(ring_area / k_area)
            Err[i]   = np.sqrt(err_samp**2 + err_cv**2)
        else:
            P1d[i] = np.nan

    ell    = k_centers * chi_Mpc / h
    Cl     = P1d * h**2 * 36 / chi_Mpc**2
    Cl_err = Err * h**2 * 36 / chi_Mpc**2
    fac    = ell * (ell + 1) / (2.0 * np.pi) * T_CMB_uK**2
    Dl     = Cl     * fac
    Dl_err = Cl_err * fac

    valid = ~np.isnan(Dl) & (Dl > 0) & (ell > 10)
    return ell[valid], Dl[valid], Dl_err[valid]
