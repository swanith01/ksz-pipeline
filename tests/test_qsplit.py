"""q_perp / q_parallel split for momentum fields."""
import numpy as np

from ksz_pipeline.coeval.momentum import (
    qsplit_power_from_q, transverse_part, _qperp_power_from_q)


def _field(N=16, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((N, N, N)) for _ in range(3)]


def test_perp_matches_existing_qperp_estimator():
    q = _field()
    k, Pp, Sp, Pl, Sl = qsplit_power_from_q(*q, 100.0)
    k0, P0, S0 = _qperp_power_from_q(*q, 100.0)
    np.testing.assert_allclose(k, k0, rtol=1e-12)
    np.testing.assert_allclose(Pp, P0, rtol=1e-10)
    np.testing.assert_allclose(Sp, S0, rtol=1e-8)


def test_transverse_split_is_orthogonal_and_complete():
    q = _field(seed=1)
    qt = transverse_part(*q)
    ql = [a - b for a, b in zip(q, qt)]
    p = lambda f: sum(np.sum(np.abs(np.fft.fftn(c))**2) for c in f)
    np.testing.assert_allclose(p(qt) + p(ql), p(q), rtol=1e-10)
    assert p(ql) > 0.2 * p(q) and p(qt) > 0.5 * p(q)     # ~1/3 vs ~2/3 for noise
    # projecting twice changes nothing
    for a, b in zip(transverse_part(*qt), qt):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_gradient_field_is_all_parallel_curl_field_all_perp():
    N, L = 16, 1.0
    x = np.arange(N) / N
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    phi = np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y) + np.sin(2 * np.pi * Z)
    # spectral gradient of phi -> purely longitudinal
    k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    P = np.fft.fftn(phi)
    grad = [np.fft.ifftn(1j * kk * P).real for kk in (
        k.reshape(N, 1, 1), k.reshape(1, N, 1), k.reshape(1, 1, N))]
    _, Pp, _, Pl, _ = qsplit_power_from_q(*grad, L)
    assert Pp.sum() < 1e-20 * Pl.sum()
    qt = transverse_part(*grad)
    assert max(np.abs(c).max() for c in qt) < 1e-10
    # curl field (0, 0, f(x)) with k along x only -> purely transverse
    curl = [np.zeros_like(phi), np.zeros_like(phi), np.sin(2 * np.pi * X)]
    _, Pp, _, Pl, _ = qsplit_power_from_q(*curl, L)
    assert Pl.sum() < 1e-20 * Pp.sum()
    for a, b in zip(transverse_part(*curl), curl):
        np.testing.assert_allclose(a, b, atol=1e-12)
