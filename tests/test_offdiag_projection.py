"""
test_offdiag_projection.py
==========================

Self-contained checks on the corrected off-diagonal estimator.  These use a
synthetic Gaussian field, so they test the *implementation*, not the physics.
The physics checks (``check_kpar0_matches_qperp``, ``check_limber_diagonal``
against ``limber.py:compute_cell``) need real snapshots and live in the driver.

Run:  python -m pytest test_offdiag_projection.py -v
  or: python test_offdiag_projection.py

Status when drafted: tests 1-3 pass at machine precision (~3e-16); test 4
passes in the median (0.93) with large per-mode scatter, which is expected --
the Limber reduction is an approximation and the reference is a single k_par=0
mode with ~2 degrees of freedom.
"""

import numpy as np

from ksz_pipeline.coeval import offdiag_projection as op


def _synthetic_pair(N=64, L=200.0, seed=7, kc=0.4):
    """Two correlated 'snapshots': shared phases, different amplitude + noise."""
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(N, N, N))
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    K[0, 0, 0] = 1e-9
    base = np.fft.ifftn(np.fft.fftn(white) * np.exp(-(K / kc) ** 2)).real
    qi = base
    qj = 0.8 * base + 0.05 * rng.normal(size=(N, N, N))
    return qi, qj, L, N


def test_mixed_correlator_matches_configuration_space():
    """xi(Delta) == (1/V) int dz g_i(z) g_j*(z - Delta), g = transverse FFT."""
    qi, qj, L, N = _synthetic_pair()
    xi, _ = op.mixed_correlator(op.fft_field(qi, L), op.fft_field(qj, L), L)
    dz, V = L / N, L ** 3
    gi = np.fft.fft2(qi, axes=(0, 1)) * (L / N) ** 2
    gj = np.fft.fft2(qj, axes=(0, 1)) * (L / N) ** 2
    for m in (0, 1, 5, 17, 31):
        ref = (np.roll(gi, -m, axis=2) * np.conj(gj)).sum(axis=2) * dz / V
        assert np.abs(xi[:, :, m] - ref).max() / np.abs(ref).max() < 1e-12


def test_zero_separation_is_the_kpar_integral():
    """xi(Delta=0) == int dk_par/(2pi) P(k_perp, k_par)."""
    qi, qj, L, N = _synthetic_pair()
    Fi, Fj = op.fft_field(qi, L), op.fft_field(qj, L)
    xi, _ = op.mixed_correlator(Fi, Fj, L)
    P = (Fi * np.conj(Fj)) / L ** 3
    brute = P.sum(axis=2) / L                       # dk_par/(2pi) = 1/L
    assert np.abs(xi[:, :, 0] - brute).max() / np.abs(brute).max() < 1e-12


def test_window_integral_equals_fourier_form():
    """int dDelta K(Delta) xi(Delta) == int dk_par/(2pi) P(k_par) Ktilde*(k_par).

    Exact Parseval identity -- the strongest single check that equation (B) is
    implemented correctly, for both equal and unequal layer centres.
    """
    qi, qj, L, N = _synthetic_pair()
    Fi, Fj = op.fft_field(qi, L), op.fft_field(qj, L)
    xi, dgrid = op.mixed_correlator(Fi, Fj, L)
    P = (Fi * np.conj(Fj)) / L ** 3
    dz = L / N

    li = op.Layer(chi=6000.0, dchi=40.0, a=0.125, tau=0.020)
    for sep in (0.0, 25.0):
        lj = op.Layer(chi=6000.0 - sep, dchi=40.0, a=0.130, tau=0.019)
        K = op.window_overlap(dgrid, li, lj)
        lhs = (xi * K[None, None, :]).sum(axis=2) * dz
        Ktilde = np.fft.fft(K) * dz
        rhs = (P * np.conj(Ktilde)[None, None, :]).sum(axis=2) / L
        assert np.abs(lhs - rhs).max() / np.abs(lhs).max() < 1e-12


def test_window_overlap_normalisation():
    """int K dDelta == dchi_i * dchi_j for top-hats."""
    _, _, L, N = _synthetic_pair()
    dgrid = np.fft.fftfreq(N, d=1.0 / N) * (L / N)
    li = op.Layer(chi=6000.0, dchi=40.0, a=0.125, tau=0.02)
    lj = op.Layer(chi=5975.0, dchi=30.0, a=0.130, tau=0.019)
    K = op.window_overlap(dgrid, li, lj)
    assert abs(K.sum() * (L / N) / (li.dchi * lj.dchi) - 1.0) < 0.02


def test_limber_reduction_in_the_median():
    """i == j with a narrow xi reduces to dchi * P(k_perp, k_par=0).

    Per-mode scatter is large by construction (the reference is one k_par=0
    mode); only the median is meaningful here.
    """
    qi, qj, L, N = _synthetic_pair()
    Fi = op.fft_field(qi, L)
    xi, dgrid = op.mixed_correlator(Fi, Fi, L)
    P = (Fi * np.conj(Fi)) / L ** 3
    li = op.Layer(chi=6000.0, dchi=40.0, a=0.125, tau=0.02)
    K = op.window_overlap(dgrid, li, li)
    lhs = (xi * K[None, None, :]).sum(axis=2).real * (L / N)
    rhs = li.dchi * np.real(P[:, :, 0])
    m = np.abs(rhs) > np.abs(rhs).max() * 1e-3
    assert 0.8 < np.median((lhs / rhs)[m]) < 1.25


def test_separation_suppresses_cross_terms():
    """The missing physics: cross terms fall off with layer separation.

    Illustrative, not a physical measurement -- a periodic 200 Mpc box aliases
    at Delta = L/2, so the residual at large separation is a floor, not signal.
    """
    qi, qj, L, N = _synthetic_pair()
    xi, dgrid = op.mixed_correlator(op.fft_field(qi, L), op.fft_field(qj, L), L)
    li = op.Layer(chi=6000.0, dchi=40.0, a=0.125, tau=0.02)
    dz = L / N

    def amp(sep):
        lj = op.Layer(chi=li.chi - sep, dchi=40.0, a=0.13, tau=0.019)
        K = op.window_overlap(dgrid, li, lj)
        return np.abs((xi * K[None, None, :]).sum(axis=2).real * dz).mean()

    a0 = amp(0.0)
    vals = [amp(s) / a0 for s in (0.0, 10.0, 25.0, 50.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] < 0.4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
