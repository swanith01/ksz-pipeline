"""
Angular (fixed field-of-view) lightcone kSZ pipeline.

Companion to lightcone_integral.py. compute_ksz_map() there is reused
unchanged -- it just sums whatever field arrays it's given, regardless
of whether the transverse grid is comoving (rectilinear) or angular.
The only new piece needed is the map -> D_ell conversion, because an
angular map sidesteps the single-reference-chi approximation that
ksz_map_to_Dl() has to make for a rectilinear map (see that function's
docstring).

Status: angular_ksz_map_to_Dl() is self-contained numpy and has been
sanity-checked on synthetic data. build_angular_lightcone() is a DRAFT
based on the documented py21cmfast v4 AngularLightconer API -- it has
not been run, and its InputParameters/astro-params handling should be
reconciled with whatever 01_make_ksz_lightcone_maps.py already does
before running it for real.
"""

import numpy as np
from ..utils.constants import T_CMB_K


def angular_ksz_map_to_Dl(ksz_map, pixel_scale_rad, n_kbins=35):
    """
    2D FFT power spectrum of an ANGULAR kSZ map -> D_ell [uK^2].

    For a map built with a fixed field of view (AngularLightconer),
    every pixel already corresponds to a fixed angle. The flat-sky
    multipole is then exactly the 2D Fourier wavenumber conjugate to
    position measured in radians:

        ell = |k|          (k conjugate to angle, radians)
        C_ell = P_2D(ell)   (no chi, no h, no reference-distance choice)

    This is the same FFT/radial-binning machinery as ksz_map_to_Dl(),
    with pixel_scale_rad standing in for pix_Mpc and with no final
    ell = k*chi / Cl = P/chi**2 step -- there's nothing left to
    approximate once the map itself is angular.

    Parameters
    ----------
    ksz_map          : ndarray (Npix, Npix)  dimensionless delta_T/T map
    pixel_scale_rad  : float                 angular size of one pixel [rad]
    n_kbins          : int                   number of radial ell bins

    Returns
    -------
    ell     : ndarray  multipoles (valid bins only)
    Dl      : ndarray  D_ell [uK^2]
    Dl_err  : ndarray  1-sigma error on D_ell
    """
    T_CMB_uK = T_CMB_K * 1e6
    N        = ksz_map.shape[0]

    m     = ksz_map - ksz_map.mean()
    fft_m = np.fft.fftshift(np.fft.fft2(m))
    ps2d  = (pixel_scale_rad / N)**2 * np.abs(fft_m)**2

    dk = 2.0 * np.pi / (N * pixel_scale_rad)
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
            vals      = ps2d[mask]
            P1d[i]    = vals.mean()
            err_samp  = vals.std() / np.sqrt(n_meas)
            ring_area = np.pi * (k_bins[i + 1]**2 - k_bins[i]**2)
            err_cv    = P1d[i] / np.sqrt(ring_area / k_area)
            Err[i]    = np.sqrt(err_samp**2 + err_cv**2)
        else:
            P1d[i] = np.nan

    ell    = k_centers
    Cl     = P1d
    Cl_err = Err
    fac    = ell * (ell + 1) / (2.0 * np.pi) * T_CMB_uK**2
    Dl     = Cl     * fac
    Dl_err = Cl_err * fac

    valid = ~np.isnan(Dl) & (Dl > 0) & (ell > 10)
    return ell[valid], Dl[valid], Dl_err[valid]


def build_angular_lightcone(inputs, match_at_z, max_redshift,
                             quantities=("density", "velocity_z"),
                             cache=None, **run_lightcone_kwargs):
    """
    *** DRAFT -- not yet verified against your installed py21cmfast
    version or the InputParameters/astro-param conventions in your
    actual 01_make_ksz_lightcone_maps.py. Check both before running. ***

    Build an angular (fixed field-of-view) lightcone, resolution-matched
    to a rectilinear lightcone at redshift `match_at_z`, and return the
    angular pixel scale needed by angular_ksz_map_to_Dl().

    Parameters
    ----------
    inputs         : py21cmfast InputParameters
        Same object you already build for the rectilinear run.
    match_at_z     : float
        Redshift at which the angular pixel resolution is matched to
        inputs.simulation_options.cell_size (i.e. your rectilinear
        cell size). Pick something near the middle of your kSZ weight,
        e.g. z ~ 7-8.
    max_redshift   : float
        Upper redshift bound of the lightcone (same role as in your
        rectilinear run).
    quantities     : tuple of str
        Fields to carry through the lightcone. Must include whatever
        compute_ksz_map() needs: density, an ionization field, and the
        LOS velocity (requested separately via get_los_velocity=True
        below, not through `quantities`).
    cache, **run_lightcone_kwargs
        Passed straight through to p21c.run_lightcone -- match whatever
        01_make_ksz_lightcone_maps.py already passes (astro_params,
        flag_options, regenerate, etc.).

    Returns
    -------
    lightcone        : py21cmfast AngularLightcone
    pixel_scale_rad  : float
        Angular size of one pixel [radians], for angular_ksz_map_to_Dl().
    """
    import py21cmfast as p21c
    from astropy.cosmology import Planck18 as cosmo

    lightconer = p21c.AngularLightconer.like_rectilinear(
        match_at_z=match_at_z,
        max_redshift=max_redshift,
        simulation_options=inputs.simulation_options,
        get_los_velocity=True,
        quantities=quantities,
    )

    lightcone = p21c.run_lightcone(
        lightconer=lightconer,
        inputs=inputs,
        cache=cache,
        **run_lightcone_kwargs,
    )

    cell_size_mpc   = inputs.simulation_options.cell_size.to_value("Mpc")
    chi_match       = cosmo.comoving_distance(match_at_z).value
    pixel_scale_rad = cell_size_mpc / chi_match

    return lightcone, pixel_scale_rad
