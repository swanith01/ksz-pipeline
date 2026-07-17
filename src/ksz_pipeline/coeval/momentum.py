"""
Ionized momentum field and transverse power spectrum P_{q_perp}(k).

Extracted from Cells 2 and 5 of 28May2026_kSZBoxes.py.

Momentum field:
    q(x) = (1+delta) * chi * v,   chi = 1 - x_HI,   v in [cm/s]

Transverse projection in Fourier space:
    Q_perp(k) = Q(k) - k_hat [Q(k) . k_hat]

Power estimator (Park+2013 Eq.A16 / Cain+2024 Eq.3):
    P_{q_perp}(k) = <|Q_perp|^2> / V / 2   [cm^2 s^-2 Mpc^3]

The /2 comes from the Limber approximation (Park+2013 Eq.A15).
ne0 is NOT absorbed into q — it enters as (sigma_T ne0/c)^2 in C_ell.
"""

import numpy as np
from ..utils.constants import ne0_cgs


def build_momentum(delta, xH, vx, vy, vz, z):
    """
    Build the ionized electron momentum field q = n_e * v [cm^-2 s^-1].

    n_e(x,z) = ne0 * (1+z)^3 * (1+delta) * chi,   chi = 1 - xH

    Parameters
    ----------
    delta    : ndarray (N,N,N)   matter overdensity delta
    xH       : ndarray (N,N,N)   neutral hydrogen fraction
    vx/vy/vz : ndarray (N,N,N)   physical peculiar velocities [cm/s]
    z        : float             redshift

    Returns
    -------
    qx, qy, qz : ndarrays (N,N,N) [cm^-2 s^-1]
    """
    ne0   = ne0_cgs()
    chi   = 1.0 - xH
    ne_fl = ne0 * (1.0 + z)**3 * (1.0 + delta) * chi   # [cm^-3]
    return ne_fl * vx, ne_fl * vy, ne_fl * vz


def qperp_power(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    """
    Spherically averaged transverse momentum power spectrum P_{q_perp}(k).

    Mirrors Cell 5 (qperp_power function) exactly.

    Parameters
    ----------
    delta    : ndarray (N,N,N)  matter overdensity delta
    xH       : ndarray (N,N,N)  neutral hydrogen fraction
    vx/vy/vz : ndarray (N,N,N)  physical peculiar velocities [cm/s]
    BOX_LEN  : float            comoving box side length [Mpc]
    nbins    : int or None      number of radial k bins (None -> auto)

    Returns
    -------
    k_bins : ndarray [Mpc^-1]
    P_bins : ndarray [cm^2 s^-2 Mpc^3]
    P_std  : ndarray [cm^2 s^-2 Mpc^3]  std per bin
    """
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi

    qx = w * vx
    qy = w * vy
    qz = w * vz

    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    # k grid [Mpc^-1]
    kfreq      = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2         = kx**2 + ky**2 + kz**2
    k_mag      = np.sqrt(k2)
    k2_safe    = np.where(k2 == 0.0, np.inf, k2)

    # FFT: continuous-FT convention Q(k) = sum q(x) exp(-ik.x) dx^3
    Qx = np.fft.fftn(qx) * d**3
    Qy = np.fft.fftn(qy) * d**3
    Qz = np.fft.fftn(qz) * d**3

    # Transverse projection Q_perp = Q - k_hat(Q.k_hat)
    kdotQ_k2 = (Qx * kx + Qy * ky + Qz * kz) / k2_safe
    Qx_perp  = Qx - kdotQ_k2 * kx
    Qy_perp  = Qy - kdotQ_k2 * ky
    Qz_perp  = Qz - kdotQ_k2 * kz

    # P_{q_perp}(k) = <|Q_perp|^2> / V / 2
    Qperp2 = (np.abs(Qx_perp)**2
              + np.abs(Qy_perp)**2
              + np.abs(Qz_perp)**2)
    p_flat = (Qperp2 / V / 2.0).ravel()
    k_flat = k_mag.ravel()

    # Radial log-spaced binning
    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))

    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin  = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax  = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0

    bins  = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)

    k_bins, P_bins, P_std = [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_bins.append(k_flat[mask].mean())
        P_bins.append(p_flat[mask].mean())
        P_std.append(p_flat[mask].std())

    return np.array(k_bins), np.array(P_bins), np.array(P_std)


def qparallel_power(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    """
    Spherically averaged LONGITUDINAL (line-of-sight-parallel) momentum
    power spectrum P_{q_parallel}(k) -- companion to qperp_power()'s
    transverse measurement.

    Standard kSZ treatments (Vishniac 1987; Ma & Fry 2002; this codebase's
    own qperp_power + limber.py pipeline) include ONLY q_perp, since
    q_parallel Fourier modes are argued to phase-cancel on line-of-sight
    projection for a sufficiently long, statistically independent LOS.
    A reference comparison (2-D map power vs. Limber q_perp/q_parallel
    modes, kSZ auto-power figure) shows real, finite-box map power at low
    ell exceeding q_perp-only Limber substantially, with q_parallel's own
    Limber curve narrowing -- but not fully closing -- that gap. Added
    14Jul2026-session-2 to test the analogous question for THIS pipeline:
    does q_perp + q_parallel (Limber) close some of the low-ell gap
    between coeval-direct (q_perp-only Limber) and stitched (direct
    real-space map, no component split, closer to the reference figure's
    "2-D Maps" curve by construction)?

    Mirrors qperp_power's FFT/binning conventions EXACTLY (same N, same
    d, same k grid, same nbins formula, same "/2" normalization) so the
    two are directly addable/bin-comparable without interpolation.

    Parameters
    ----------
    delta    : ndarray (N,N,N)  matter overdensity delta
    xH       : ndarray (N,N,N)  neutral hydrogen fraction
    vx/vy/vz : ndarray (N,N,N)  physical peculiar velocities [cm/s]
    BOX_LEN  : float            comoving box side length [Mpc]
    nbins    : int or None      number of radial k bins (None -> auto,
               same formula as qperp_power so bins align)

    Returns
    -------
    k_bins : ndarray [Mpc^-1]
    P_bins : ndarray [cm^2 s^-2 Mpc^3]  same units/normalization as
             qperp_power's P_qperp, including its "/2"
    P_std  : ndarray [cm^2 s^-2 Mpc^3]  std per bin
    """
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi

    qx = w * vx
    qy = w * vy
    qz = w * vz

    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    kfreq      = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2         = kx**2 + ky**2 + kz**2
    k_mag      = np.sqrt(k2)
    k2_safe    = np.where(k2 == 0.0, np.inf, k2)

    Qx = np.fft.fftn(qx) * d**3
    Qy = np.fft.fftn(qy) * d**3
    Qz = np.fft.fftn(qz) * d**3

    # Longitudinal (parallel) component: the scalar projection Q.k_hat.
    # kdotQ_k2 = (Q.k)/k^2 (same intermediate qperp_power computes to
    # build Q_perp); Q.k_hat = k_mag * kdotQ_k2.
    kdotQ_k2  = (Qx * kx + Qy * ky + Qz * kz) / k2_safe
    Q_par_mag = k_mag * kdotQ_k2

    # P_{q_parallel}(k) = <|Q_parallel|^2> / V / 2 -- same "/2" convention
    # as qperp_power, so P_qperp + P_qparallel is a meaningful sum.
    Qpar2  = np.abs(Q_par_mag)**2
    p_flat = (Qpar2 / V / 2.0).ravel()
    k_flat = k_mag.ravel()

    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))

    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin  = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax  = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0

    bins  = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)

    k_bins, P_bins, P_std = [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_bins.append(k_flat[mask].mean())
        P_bins.append(p_flat[mask].mean())
        P_std.append(p_flat[mask].std())

    return np.array(k_bins), np.array(P_bins), np.array(P_std)
