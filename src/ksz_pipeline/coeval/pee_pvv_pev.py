"""
Electron-weight and velocity power spectra Pee, Pvv, Pev from a coeval box.

Companion to momentum.py's qperp_power(), which computes the DIRECT,
exact curl-projected |Q_perp|^2 power. This module instead measures the
three separate spectra that Georgiev+24 Eq.(10) combines via a
Gaussian/Wick mode-coupling convolution (georgiev_convolution.py) to
*reconstruct* an approximation to the same P_qperp. Comparing the two
-- direct vs reconstructed -- isolates how much of P_qperp comes from
genuine non-Gaussian mode-coupling (reionization's bubble morphology)
that the Wick factorization discards. Lopez, D'Aloisio & Cain (2025,
arXiv:2507.17817) do the same kind of reconstruction for their missing
large-scale modes (their Eq. 2.5) and find it undershoots the true
P_qperp by ~10-20%, growing toward late reionization, due to exactly
this non-Gaussianity -- that's the benchmark to expect here, not a
flat, k-independent offset.

Same FFT grid and radial k-binning as qperp_power(), by construction
(same N, same nbins formula), so the two outputs are directly
bin-comparable without interpolation for most purposes -- but
compare_direct_vs_reconstructed() interpolates anyway, defensively.

Field/normalization conventions
--------------------------------
Pee : power spectrum of w = (1+delta)*(1-xH)   [dimensionless]
      (no ne0, no (1+z)^3 -- same convention as qperp_power()'s q field;
      matches Georgiev+24's electron *overdensity* fluctuation field and
      Lopez/D'Aloisio/Cain's P_{chi(1+delta)}.)

Pvv : power spectrum of the velocity field's component along k_hat,
      v_par(k) = k_hat . v(k). 21cmFAST's Zel'dovich-approximation
      velocity field is curl-free by construction (v(k) is exactly
      parallel to k_hat), so this should equal the full vector power
      |vx|^2+|vy|^2+|vz|^2 to within numerical noise -- the code checks
      this and prints a warning if it doesn't, since a mismatch would
      indicate either a bug upstream or unexpected vorticity.

Pev : cross power spectrum Re[w*(k) . v_par(k)], the density-velocity
      cross term used in Georgiev+24 Eq.(11)/Eq.(10). Can be negative;
      bounded by Cauchy-Schwarz, |Pev(k)| <= sqrt(Pee(k)*Pvv(k)) -- a
      good one-line sanity check on real data.

Normalization vs qperp_power(): DELIBERATELY does not include the
Park+2013 Eq.A15 "/2" that qperp_power() bakes into its P_qperp.
Lopez/D'Aloisio/Cain's Eq. 2.4 (the same Limber formula limber.py
mirrors) carries no such factor, which suggests that "/2" is an
estimator-normalization detail internal to qperp_power(), not a
cross-paper convention mismatch -- but confirm empirically (see
compare_direct_vs_reconstructed) before trusting that assumption.
"""

import numpy as np


def measure_pee_pvv_pev(delta, xH, vx, vy, vz, BOX_LEN, nbins=None):
    """
    Auto- and cross-power spectra of w=(1+delta)*(1-xH) and the
    line-of-sight-of-k-hat velocity component, from a coeval box.

    Parameters
    ----------
    delta    : ndarray (N,N,N)  matter overdensity delta
    xH       : ndarray (N,N,N)  neutral hydrogen fraction
    vx/vy/vz : ndarray (N,N,N)  physical peculiar velocities [cm/s]
    BOX_LEN  : float            comoving box side length [Mpc]
    nbins    : int or None      number of radial k bins (None -> auto,
               same formula as qperp_power() so bins align)

    Returns
    -------
    k_bins  : ndarray [Mpc^-1]
    Pee     : ndarray [dimensionless]
    Pvv     : ndarray [cm^2 s^-2 Mpc^3]
    Pev     : ndarray [cm s^-1 Mpc^3^(1/2)... same units as sqrt(Pee*Pvv)]
    """
    chi = 1.0 - xH
    w   = (1.0 + delta) * chi

    N = w.shape[0]
    L = float(BOX_LEN)
    d = L / N
    V = L**3

    kfreq      = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2         = kx**2 + ky**2 + kz**2
    k_mag      = np.sqrt(k2)
    k_safe     = np.where(k_mag == 0.0, np.inf, k_mag)

    W  = np.fft.fftn(w)  * d**3
    Vx = np.fft.fftn(vx) * d**3
    Vy = np.fft.fftn(vy) * d**3
    Vz = np.fft.fftn(vz) * d**3

    # velocity component along k_hat
    V_par = (Vx * kx + Vy * ky + Vz * kz) / k_safe

    # curl-free check: full vector power vs longitudinal-only power.
    # 21cmFAST's Zel'dovich velocity field should make these equal; a
    # mismatch flags a units/construction problem upstream, not physics.
    nonzero    = k_mag > 0
    full_power = (np.abs(Vx)**2 + np.abs(Vy)**2 + np.abs(Vz)**2)[nonzero]
    par_power  = (np.abs(V_par)**2)[nonzero]
    frac_diff  = np.abs(full_power.sum() - par_power.sum()) / full_power.sum()
    if frac_diff > 0.01:
        print(f"  [warning] velocity field is not purely curl-free: "
              f"full vs longitudinal power differ by {frac_diff:.2%} "
              f"-- check vx/vy/vz units and construction before trusting Pvv")

    Pee_flat = (np.abs(W)**2                       / V).ravel()
    Pvv_flat = (np.abs(V_par)**2                    / V).ravel()
    Pev_flat = (np.real(np.conj(W) * V_par)         / V).ravel()
    k_flat   = k_mag.ravel()

    if nbins is None:
        nbins = max(2, int(np.ceil(np.cbrt(N) * 8)))   # matches qperp_power()

    pos_k = np.abs(kfreq[kfreq > 0.0])
    kmin  = pos_k.min() if pos_k.size > 0 else 1e-6
    kmax  = np.abs(kfreq).max() * np.sqrt(3.0)
    if kmax <= kmin:
        kmax = kmin * 10.0

    bins  = np.geomspace(kmin, kmax, nbins)
    digit = np.digitize(k_flat, bins)

    k_out, Pee_out, Pvv_out, Pev_out = [], [], [], []
    for i in range(1, len(bins)):
        mask = digit == i
        if not np.any(mask):
            continue
        k_out.append(k_flat[mask].mean())
        Pee_out.append(Pee_flat[mask].mean())
        Pvv_out.append(Pvv_flat[mask].mean())
        Pev_out.append(Pev_flat[mask].mean())

    return (np.array(k_out), np.array(Pee_out),
            np.array(Pvv_out), np.array(Pev_out))


def compare_direct_vs_reconstructed(k_direct, P_direct, k_pee, Pee, Pvv, Pev,
                                     verbose=True, **convolution_kwargs):
    """
    Compare qperp_power()'s DIRECT P_qperp(k) against the Georgiev Eq.10
    RECONSTRUCTION built from this module's measured Pee/Pvv/Pev.

    Run this FIRST on real data, before trusting the reconstruction for
    anything. What to look for:
      - a flat ratio (~constant across all k) near a clean factor like 2
        -> the Park+2013 "/2" normalization noted in this module's
        docstring needs reconciling; the physics comparison isn't valid
        until that's resolved.
      - ratio ~1 at low k, rising toward high k / late reionization
        (~1.1-1.2, per Lopez/D'Aloisio/Cain's ~10-20% figure) -> the
        expected, literature-consistent non-Gaussianity signal.
      - anything wildly different from both -> something upstream
        (Pee/Pvv/Pev measurement, or the qperp_power call it's compared
        against) is likely inconsistent; check inputs before reading
        physics into it.

    Parameters
    ----------
    k_direct, P_direct : direct qperp_power() output for one redshift
    k_pee, Pee, Pvv, Pev : measure_pee_pvv_pev() output for the SAME box
    **convolution_kwargs : passed through to
        georgiev_convolution.qperp_from_pee_pvv_pev (e.g. n_kprime, n_mu,
        kprime_max -- widen kprime_max if qperp_convergence_check says so)

    Returns
    -------
    P_reconstructed : ndarray, evaluated at k_direct
    ratio           : ndarray, P_direct / P_reconstructed
    """
    from .georgiev_convolution import qperp_from_pee_pvv_pev

    def _loglog(xq, xp, fp):
        xp, fp = np.asarray(xp), np.asarray(fp)
        m = (xp > 0) & (fp > 0)
        lx, lf = np.log(xp[m]), np.log(fp[m])
        lq = np.log(np.clip(xq, xp[m].min(), xp[m].max()))
        return np.exp(np.interp(lq, lx, lf))

    Pee_f = lambda k: _loglog(k, k_pee, Pee)
    Pvv_f = lambda k: _loglog(k, k_pee, Pvv)
    Pev_f = lambda k: np.interp(k, k_pee, Pev)   # can be negative -> linear, not log

    # Bound k' integration to the measured range (with modest padding),
    # NOT georgiev_convolution's generic defaults -- Pee_f/Pvv_f/Pev_f
    # are interpolators that go flat outside k_pee's range, and the
    # integral's k'^2 measure makes that flat tail dominate if kprime_max
    # is left much larger than where real data exists (see that module's
    # docstring). Explicit kwargs from the caller still win.
    conv_kwargs = dict(kprime_min=float(np.min(k_pee)) * 0.5,
                        kprime_max=float(np.max(k_pee)) * 2.0)
    conv_kwargs.update(convolution_kwargs)

    P_recon = qperp_from_pee_pvv_pev(k_direct, Pee_f, Pvv_f, Pev_f,
                                      **conv_kwargs)
    ratio = P_direct / P_recon

    if verbose:
        print("  k [Mpc^-1]   P_direct       P_reconstructed   ratio")
        for k, pd, pr, r in zip(k_direct, P_direct, P_recon, ratio):
            print(f"  {k:9.4f}   {pd:12.4e}   {pr:12.4e}      {r:8.4f}")

    return P_recon, ratio
