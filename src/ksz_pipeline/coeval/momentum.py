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

    return _qperp_power_from_q(qx, qy, qz, BOX_LEN, nbins)


def qperp_power_from_momentum(px, py, pz, xH, BOX_LEN, nbins=None):
    """
    P_{q_perp}(k) from a MASS-WEIGHTED momentum field p = (1+delta)*v,
    for codes (e.g. AMBER) that deposit momentum directly rather than
    storing delta and v separately.

    Identical to qperp_power(delta, xH, v) with p = (1+delta)*v, but
    avoids reconstructing v = p/(1+delta), which is ill-defined where
    an interlaced/deconvolved density field is ~0 or slightly negative.

    Parameters
    ----------
    px/py/pz : ndarray (N,N,N)  (1+delta)*v, physical peculiar velocity
               [cm/s] (same velocity convention as qperp_power)
    xH       : ndarray (N,N,N)  neutral hydrogen fraction
    BOX_LEN  : float            comoving box side length [Mpc]
    nbins    : int or None      same meaning/default as qperp_power

    Returns
    -------
    Same as qperp_power.
    """
    chi = 1.0 - xH
    return _qperp_power_from_q(chi * px, chi * py, chi * pz, BOX_LEN, nbins)


def _qperp_power_from_q(qx, qy, qz, BOX_LEN, nbins=None):
    """
    Shared FFT / transverse-projection / binning core of qperp_power.
    Moved here VERBATIM from qperp_power's body (2026-09 refactor) so
    qperp_power and qperp_power_from_momentum cannot drift apart.
    qperp_power's output is bit-identical to the pre-refactor version
    (tests/test_amber_adapter.py::test_qperp_refactor_bit_identical).
    """
    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    # Memory fix: 1D broadcastable k-arrays instead of full (N,N,N) float64
    # kx,ky,kz from meshgrid (was an extra ~87GB resident for N=1536, for
    # nothing -- broadcasting gives numerically IDENTICAL values, same
    # float64 precision throughout, so this does not change
    # test_qperp_refactor_bit_identical's result).
    kfreq = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx1 = kfreq.reshape(N, 1, 1)
    ky1 = kfreq.reshape(1, N, 1)
    kz1 = kfreq.reshape(1, 1, N)
    k2      = kx1**2 + ky1**2 + kz1**2
    k_mag   = np.sqrt(k2)
    k2_safe = np.where(k2 == 0.0, np.inf, k2)
    del k2

    # Memory fix: two passes over the 3 components instead of holding
    # Qx,Qy,Qz and Qx_perp,Qy_perp,Qz_perp all at once (was ~350GB+
    # transient at N=1536). Costs one extra FFT pass; math is identical.
    kdotQ_k2 = np.zeros((N, N, N), dtype=np.complex128)
    for q, k1 in ((qx, kx1), (qy, ky1), (qz, kz1)):
        Q = np.fft.fftn(q) * d**3
        kdotQ_k2 += Q * k1
        del Q
    kdotQ_k2 /= k2_safe

    Qperp2 = np.zeros((N, N, N), dtype=np.float64)
    for q, k1 in ((qx, kx1), (qy, ky1), (qz, kz1)):
        Q = np.fft.fftn(q) * d**3
        Qp = Q - kdotQ_k2 * k1
        Qperp2 += np.abs(Qp)**2
        del Q, Qp
    del kdotQ_k2

    p_flat = (Qperp2 / V / 2.0).ravel()
    k_flat = k_mag.ravel()
    del Qperp2

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


# ---------------------------------------------------------------------------
# Momentum-field (AMBER) q_perp / q_parallel split -- 2026-10-26.
# qparallel_power above takes (delta, v) and so would need v = mom/rho for
# AMBER, which is ill-defined where AMBER's deconvolved rho <= 0. These take
# q directly. Binning/normalisation identical to _qperp_power_from_q.
# ---------------------------------------------------------------------------

def _kgrid(N, L):
    kfreq = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi
    return kfreq, kfreq.reshape(N, 1, 1), kfreq.reshape(1, N, 1), kfreq.reshape(1, 1, N)


def _bin_like_qperp(p_flat, k_flat, kfreq, N, nbins):
    """Exactly _qperp_power_from_q's radial binning."""
    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))
    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0
    bins = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)
    k_b, P_b, S_b = [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_b.append(k_flat[mask].mean())
        P_b.append(p_flat[mask].mean())
        S_b.append(p_flat[mask].std())
    return np.array(k_b), np.array(P_b), np.array(S_b)


def qsplit_power_from_q(qx, qy, qz, BOX_LEN, nbins=None):
    """
    P_{q_perp}(k) and P_{q_parallel}(k) of one momentum field q, same FFT
    normalisation, "/2" convention and k bins as _qperp_power_from_q (so
    P_perp here equals qperp_power_from_momentum's output, and the two are
    directly addable). The k=0 mode (bulk flow) has no direction; it is
    counted in neither.

    Returns k, P_perp, P_perp_std, P_par, P_par_std.
    """
    N = qx.shape[0]
    L = float(BOX_LEN)
    d, V = L / N, L**3
    kfreq, kx1, ky1, kz1 = _kgrid(N, L)
    k2 = kx1**2 + ky1**2 + kz1**2
    k_mag = np.sqrt(k2)
    k2_safe = np.where(k2 == 0.0, np.inf, k2)
    Qs = [np.fft.fftn(q) * d**3 for q in (qx, qy, qz)]
    kdotQ_k2 = (Qs[0] * kx1 + Qs[1] * ky1 + Qs[2] * kz1) / k2_safe
    tot2 = sum(np.abs(Q)**2 for Q in Qs)
    par2 = np.abs(kdotQ_k2)**2 * k2                       # |Q.k_hat|^2
    perp2 = sum(np.abs(Q - kdotQ_k2 * k1)**2 for Q, k1 in zip(Qs, (kx1, ky1, kz1)))
    del Qs, kdotQ_k2, tot2
    k_flat = k_mag.ravel()
    k, Pp, Sp = _bin_like_qperp((perp2 / V / 2.0).ravel(), k_flat, kfreq, N, nbins)
    _, Pl, Sl = _bin_like_qperp((par2 / V / 2.0).ravel(), k_flat, kfreq, N, nbins)
    return k, Pp, Sp, Pl, Sl


def transverse_part(qx, qy, qz):
    """
    Real-space transverse (divergence-free) part of a periodic vector
    field: inverse FFT of Q - k_hat (k_hat.Q). The k=0 mode is kept in
    the transverse part. The longitudinal part is q - transverse_part(q).

    For even N the Nyquist plane of each axis has no -k partner, so the raw
    projection breaks Hermitian symmetry and .real would drop power; the
    Nyquist component of k is set to 0 in the projection (standard
    spectral-derivative convention). The split is then exactly orthogonal:
    |q|^2 = |q_T|^2 + |q_L|^2 summed over all modes.
    """
    N = qx.shape[0]
    kfreq = np.fft.fftfreq(N)                            # units cancel in k_hat
    if N % 2 == 0:
        kfreq[N // 2] = 0.0
    kx1, ky1, kz1 = (kfreq.reshape(N, 1, 1), kfreq.reshape(1, N, 1),
                     kfreq.reshape(1, 1, N))
    k2 = kx1**2 + ky1**2 + kz1**2
    k2_safe = np.where(k2 == 0.0, np.inf, k2)
    Qs = [np.fft.fftn(q) for q in (qx, qy, qz)]
    kdotQ_k2 = (Qs[0] * kx1 + Qs[1] * ky1 + Qs[2] * kz1) / k2_safe
    return tuple(np.fft.ifftn(Q - kdotQ_k2 * k1).real
                 for Q, k1 in zip(Qs, (kx1, ky1, kz1)))
