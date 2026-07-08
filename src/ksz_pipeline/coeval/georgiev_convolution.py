"""
Georgiev+24 Eq. (10): the curl ("q_perp") momentum power spectrum built
from Pee, Pvv, Pev via the Gaussian/Wick mode-coupling convolution.

This module is deliberately independent of how Pee/Pvv/Pev were obtained
-- they're passed in as callables. That means it works identically for:
  - measured spectra from a coeval box (this work's own numbers)
  - Georgiev's analytic model (Eq. 12 Pee, Eq. 11 linear-theory Pvv)
so it's the one piece of the Georgiev-comparison work that could be
written and tested without seeing the coeval box code.

Derivation note (so the integral below is traceable back to the paper):
Eq. (10) defines, via (2pi)^3 delta_D(k-k') ⟨q~_B,e(k) q~*_B,e(k')⟩ ≡
2*pi^2/k^3 * Delta^2_{B,e}(k,z), and since Delta^2 and P are related by
the standard Delta^2(k) = k^3 P(k) / (2 pi^2), the "2pi^2/k^3 Delta^2"
prefactor is just P_{B,e}(k,z) itself -- i.e. P_qperp(k,z), the same
quantity stored as 'Pqperp' in qperp_power.npz. So Eq. (10) reduces to,
after doing the trivial azimuthal (phi) integral of the angle-independent
integrand:

    P_qperp(k) = 1/(2*pi)^2 * int_0^inf dk' k'^2 int_{-1}^{1} dmu (1-mu^2)
                 * [ Pee(s) Pvv(k') - (k'/s) Pev(s) Pev(k') ]

    s = |k - k'| = sqrt(k^2 + k'^2 - 2*k*k'*mu)
    mu = cos(angle between k and k')
"""

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def qperp_from_pee_pvv_pev(k_grid, Pee_of_k, Pvv_of_k, Pev_of_k=None,
                            n_kprime=400, n_mu=64,
                            kprime_min=1e-4, kprime_max=50.0,
                            s_floor=1e-6):
    """
    Evaluate the Eq. (10) convolution at each k in k_grid.

    Parameters
    ----------
    k_grid      : ndarray (Nk,)  external wavenumbers [Mpc^-1]
    Pee_of_k    : callable, k [Mpc^-1] -> Pee(k)  (array-broadcastable)
    Pvv_of_k    : callable, k [Mpc^-1] -> Pvv(k)
    Pev_of_k    : callable or None. If None, the cross term is dropped
                  (Pev=0), useful as a first pass / sanity check.
    n_kprime    : int    number of log-spaced k' integration points
    n_mu        : int    number of Gauss-Legendre points in mu in [-1,1]
    kprime_min, kprime_max : float  integration range for k' [Mpc^-1].
                  CAUTION if Pee_of_k/Pvv_of_k/Pev_of_k are interpolators
                  built from a finite measured k range (as opposed to
                  true analytic functions valid everywhere, like
                  Georgiev's Eq. 12): most interpolators go FLAT beyond
                  their input range, and since the integrand carries a
                  k'^2 measure that keeps growing, a flat tail out to a
                  large kprime_max dominates the integral and gives
                  nonsense -- it does NOT fail loudly, it just returns a
                  large, k-independent, wrong number (confirmed by hand
                  while testing this against real qperp_power() output).
                  For measured spectra, keep kprime_max at or just beyond
                  the actual measured k range -- compare_direct_vs_
                  reconstructed() in pee_pvv_pev.py does this
                  automatically. Only widen freely when Pee/Pvv/Pev are
                  genuine analytic functions.
    s_floor     : float  minimum allowed |k-k'| to avoid the 1/s term
                  blowing up right at s=0; only matters if Pev is given
                  and k happens to sit exactly on the k' grid.

    Returns
    -------
    Pqperp : ndarray (Nk,)  same units as Pee*Pvv (e.g. cm^2 s^-2 Mpc^3
             if Pee is dimensionless-density-power and Pvv is velocity^2,
             matching the 'Pqperp' convention already used in
             qperp_power.npz -- keep Pee/Pvv/Pev in those same units in).
    """
    k_grid = np.atleast_1d(np.asarray(k_grid, dtype=float))

    kp    = np.logspace(np.log10(kprime_min), np.log10(kprime_max), n_kprime)
    mu, w_mu = np.polynomial.legendre.leggauss(n_mu)   # nodes/weights on [-1,1]

    Pvv_kp   = Pvv_of_k(kp)                              # (n_kprime,)
    Pev_kp   = Pev_of_k(kp) if Pev_of_k is not None else None

    Pqperp = np.empty_like(k_grid)

    for i, k in enumerate(k_grid):
        # s(k', mu) on the full (n_kprime, n_mu) grid
        s2 = k**2 + kp[:, None]**2 - 2.0 * k * kp[:, None] * mu[None, :]
        s  = np.sqrt(np.clip(s2, s_floor**2, None))

        Pee_s = Pee_of_k(s)                              # (n_kprime, n_mu)
        term  = Pee_s * Pvv_kp[:, None]                  # broadcast (n_kprime,1)*(n_kprime,n_mu)

        if Pev_kp is not None:
            Pev_s = Pev_of_k(s)
            term  = term - (kp[:, None] / s) * Pev_s * Pev_kp[:, None]

        weight = (1.0 - mu**2)[None, :]                  # (1, n_mu)
        # integrate over mu (Gauss-Legendre) then over k' (trapz in log-k')
        inner_mu = np.sum(term * weight * w_mu[None, :], axis=1)   # (n_kprime,)
        Pqperp[i] = _trapz(kp**2 * inner_mu, kp) / (2.0 * np.pi)**2

    return Pqperp


def qperp_convergence_check(k_grid, Pee_of_k, Pvv_of_k, Pev_of_k=None,
                             resolutions=((200, 32), (400, 64), (800, 128)),
                             kprime_max=50.0):
    """
    Run qperp_from_pee_pvv_pev at increasing (n_kprime, n_mu) and
    kprime_max, and report the fractional change between successive
    resolutions -- a cheap way to confirm the integral has converged
    before trusting it against real Pee/Pvv/Pev. Run this once with your
    actual measured spectra before reading anything into the comparison
    plot.

    Returns
    -------
    dict of {resolution_label: Pqperp array}, plus prints max fractional
    change between consecutive entries.
    """
    results = {}
    prev = None
    for n_kprime, n_mu in resolutions:
        label = f"n_kprime={n_kprime}, n_mu={n_mu}"
        Pq = qperp_from_pee_pvv_pev(k_grid, Pee_of_k, Pvv_of_k, Pev_of_k,
                                     n_kprime=n_kprime, n_mu=n_mu,
                                     kprime_max=kprime_max)
        results[label] = Pq
        if prev is not None:
            frac = np.max(np.abs((Pq - prev) / np.where(prev != 0, prev, 1.0)))
            print(f"  {label}: max fractional change vs previous = {frac:.3e}")
        prev = Pq
    return results
