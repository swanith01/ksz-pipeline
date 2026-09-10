"""
Tests for the AMBER integration. Each test targets one specific way the
integration could be silently wrong (the category of bug this project
has actually hit: conventions, layouts, factors of h / 2 / x_e).

Run:  pytest tests/test_amber_adapter.py -v
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from ksz_pipeline.coeval.momentum import qperp_power, qperp_power_from_momentum
from ksz_pipeline.amber import io, adapter

from _frozen_qperp_pre_amber import qperp_power_frozen


def _fields(N=24, seed=0):
    rng = np.random.default_rng(seed)
    delta = 0.3 * rng.standard_normal((N, N, N))
    xH = (rng.random((N, N, N)) < 0.4).astype(float)
    v = 3e7 * rng.standard_normal((3, N, N, N))     # cm/s
    return delta, xH, v


# 1. Refactor must not change qperp_power AT ALL ---------------------------
def test_qperp_refactor_bit_identical():
    delta, xH, v = _fields()
    new = qperp_power(delta, xH, *v, 300.0)
    old = qperp_power_frozen(delta, xH, *v, 300.0)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)


# 2. Momentum-input entry point == (delta, v) entry point ------------------
def test_momentum_variant_matches_qperp_power():
    delta, xH, v = _fields()
    p = (1.0 + delta)[None] * v
    k1, P1, S1 = qperp_power(delta, xH, *v, 300.0)
    k2, P2, S2 = qperp_power_from_momentum(*p, xH, 300.0)
    assert np.array_equal(k1, k2)
    np.testing.assert_allclose(P2, P1, rtol=1e-12)
    np.testing.assert_allclose(S2, S1, rtol=1e-12)


# 3. Real Fortran byte layout -> reader, incl. axis order ------------------
FORTRAN_WRITER = """
program w
  implicit none
  integer, parameter :: N = {N}
  real(4) :: rho2(N,N,N), mom2(3,N,N,N), zre(N,N,N)
  integer :: i, j, k, c
  do k=1,N; do j=1,N; do i=1,N
     rho2(i,j,k) = 1.0 + 0.01*i + 0.0001*j + 0.000001*k
     zre(i,j,k)  = 5.0 + i + 0.1*j + 0.01*k
     do c=1,3
        mom2(c,i,j,k) = 1000.0*c + 100.0*i + 10.0*j + k
     enddo
  enddo; enddo; enddo
  open(12,file='fields_z=08.25.dat',access='stream',form='unformatted',status='replace')
  write(12) real(8.25d0,kind=4)
  write(12) rho2
  write(12) mom2
  close(12)
  open(11,file='zre_test.dat',access='stream',form='unformatted',status='replace')
  write(11) zre
  close(11)
end program w
"""


@pytest.mark.skipif(shutil.which('gfortran') is None, reason='no gfortran')
def test_fortran_layout_roundtrip(tmp_path):
    N = 5
    src = tmp_path / 'w.f90'
    src.write_text(FORTRAN_WRITER.format(N=N))
    subprocess.run(['gfortran', str(src), '-o', str(tmp_path / 'w.x')],
                   check=True)
    subprocess.run([str(tmp_path / 'w.x')], cwd=tmp_path, check=True)

    z, rho, mom = io.read_fields(str(tmp_path / 'fields_z=08.25.dat'), N)
    zre = io.read_zre(str(tmp_path / 'zre_test.dat'), N)
    assert z == pytest.approx(8.25)
    assert io.z_from_name('fields_z=08.25.dat') == 8.25
    for (i, j, k) in [(0, 0, 0), (4, 1, 2), (1, 3, 4)]:
        I, J, K = i + 1, j + 1, k + 1            # Fortran 1-based
        assert rho[i, j, k] == pytest.approx(1 + 0.01*I + 1e-4*J + 1e-6*K)
        assert zre[i, j, k] == pytest.approx(5 + I + 0.1*J + 0.01*K)
        for c in range(3):
            assert mom[c, i, j, k] == 1000*(c+1) + 100*I + 10*J + K


def test_reader_rejects_wrong_N(tmp_path):
    f = tmp_path / 'fields_z=07.00.dat'
    np.zeros(1 + 4 * 8**3, dtype='<f4').tofile(f)
    with pytest.raises(ValueError):
        io.read_fields(str(f), 9)


# 4. AMBER P_qq -> repo P_qperp conversion (h, 1e10, 1/2, x_e^2) ----------
def _amber_pqq_port(mom_kms, ion, L_mpc_h, XH, YHe):
    """Python port of cmbreion.f90 cmb_powerspectrum's P_qq branch:
    q = x_e * mom2 * ionized ; forward FFT unscaled ; transverse projection;
    P = |FFT/Nmesh|^2 summed over components ; bin kk = nint(k/Ak) ;
    x Lbox^3 ; bins k = 1..N/2 at k = Ak*kk. No 1/2."""
    N = mom_kms.shape[1]
    xe = adapter.electron_fraction_amber(XH, YHe)
    q = xe * mom_kms * ion[None]
    Q = np.fft.fftn(q, axes=(1, 2, 3)) / N**3
    Ak = 2 * np.pi / L_mpc_h
    f = np.fft.fftfreq(N, d=1.0 / N) * Ak
    kx, ky, kz = np.meshgrid(f, f, f, indexing='ij')
    kr = np.sqrt(kx**2 + ky**2 + kz**2)
    kr_s = np.where(kr == 0, np.inf, kr)
    kdq = (Q[0] * kx + Q[1] * ky + Q[2] * kz) / kr_s
    Qp = Q - np.stack([kx, ky, kz]) / kr_s * kdq
    p = (np.abs(Qp)**2).sum(0)
    kk = np.rint(kr / Ak).astype(int)
    P = np.array([p[(kk == b)].mean() for b in range(1, N // 2 + 1)])
    return Ak * np.arange(1, N // 2 + 1), P * L_mpc_h**3


def test_pqq_conversion_exact_per_mode():
    # Strongest form: identical Fourier modes, AMBER convention vs this
    # repo's convention (momentum.py formulas), before any binning. The
    # conversion must be exact, not merely close.
    N, L_h, h, XH, YHe = 16, 150.0, 0.6766, 0.76, 0.24
    xe = adapter.electron_fraction_amber(XH, YHe)
    rng = np.random.default_rng(5)
    mom = 300.0 * rng.standard_normal((3, N, N, N))
    ion = rng.random((N, N, N)) < 0.5

    def perp2(Q, kx, ky, kz):
        k2 = kx**2 + ky**2 + kz**2
        k2s = np.where(k2 == 0, np.inf, k2)
        kd = (Q[0] * kx + Q[1] * ky + Q[2] * kz) / k2s
        return (np.abs(Q - np.stack([kx, ky, kz]) * kd)**2).sum(0), k2 > 0

    # AMBER (cmbreion.f90): FFT/Nmesh, x Lbox^3 [(Mpc/h)^3], km/s, x_e in q
    fA = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / L_h
    QA = np.fft.fftn(xe * mom * ion, axes=(1, 2, 3)) / N**3
    pA, m = perp2(QA, *np.meshgrid(fA, fA, fA, indexing='ij'))
    pA *= L_h**3
    # repo (momentum.py): FFT*d^3, /V/2 [Mpc^3], cm/s, no x_e
    L = L_h / h
    fR = np.fft.fftfreq(N, L / N) * 2 * np.pi
    QR = np.fft.fftn(mom * 1e5 * ion, axes=(1, 2, 3)) * (L / N)**3
    pR, _ = perp2(QR, *np.meshgrid(fR, fR, fR, indexing='ij'))
    pR /= L**3 * 2.0

    kA = np.sqrt(sum(g**2 for g in np.meshgrid(fA, fA, fA, indexing='ij')))
    k_conv, p_conv = adapter.amber_pqq_to_repo(kA[m], pA[m], h, XH, YHe)
    kR = np.sqrt(sum(g**2 for g in np.meshgrid(fR, fR, fR, indexing='ij')))
    np.testing.assert_allclose(k_conv, kR[m], rtol=1e-12)
    np.testing.assert_allclose(p_conv, pR[m], rtol=1e-10)


def test_pqq_conversion_matches_repo_estimator():
    # Same check through the ACTUAL repo function (qperp_power_from_momentum)
    # and a port of AMBER's actual binning. White-noise momentum -> flat
    # P_qperp, so linear (AMBER) vs log (repo) bins compare without bias.
    # Different bin weighting of the same realized modes leaves ~1% noise:
    # over 40 seeds the ratio is 0.9988 +/- 0.0014 (checked 2026-09-10),
    # hence the 3% tolerance here; exactness is tested per-mode above.
    N, L_h, h, XH, YHe = 32, 200.0, 0.6766, 0.76, 0.24
    rng = np.random.default_rng(3)
    mom_kms = 300.0 * rng.standard_normal((3, N, N, N))
    ion = rng.random((N, N, N)) < 0.6

    k_a, P_a = _amber_pqq_port(mom_kms, ion, L_h, XH, YHe)
    k_conv, P_conv = adapter.amber_pqq_to_repo(k_a, P_a, h, XH, YHe)

    xH = (~ion).astype(float)
    k_r, P_r, _ = qperp_power_from_momentum(*(mom_kms * 1e5), xH, L_h / h)

    # compare over the shared, well-sampled k range
    lo, hi = max(k_conv[2], k_r[2]), min(k_conv[-1], k_r[-1])
    a = P_conv[(k_conv >= lo) & (k_conv <= hi)].mean()
    b = P_r[(k_r >= lo) & (k_r <= hi)].mean()
    assert a / b == pytest.approx(1.0, abs=0.03)
    # and a WRONG conversion (e.g. forgetting h^3) must fail this test
    assert (a * h**3) / b != pytest.approx(1.0, abs=0.03)


# 5. Ionization convention and the two x_HI means --------------------------
def test_snapshot_conventions():
    N = 16
    rng = np.random.default_rng(1)
    zre = 6 + 6 * rng.random((N, N, N)).astype(np.float32)
    rho = (1 + 0.5 * rng.standard_normal((N, N, N))).clip(0.05).astype(np.float32)
    rho /= rho.mean()
    mom = np.zeros((3, N, N, N), np.float32)
    s = adapter.snapshot(zre, rho, mom, 9.0, 0.6766)
    assert np.array_equal(s['xH'] == 0, 9.0 < zre)     # ionized iff z<zre
    assert s['xH_mean'] == pytest.approx((zre <= 9.0).mean())
    assert s['xH_mass'] == pytest.approx(
        (rho * (zre <= 9.0)).sum() / rho.sum(), rel=1e-6)


# 6. End to end: synthetic AMBER dir -> compute_cell ------------------------
def _write_fields(path, z, rho, mom):
    with open(path, 'wb') as f:
        np.array([z], '<f4').tofile(f)
        np.asfortranarray(rho, dtype='<f4').ravel(order='F').tofile(f)
        np.asfortranarray(mom, dtype='<f4').ravel(order='F').tofile(f)


def test_end_to_end_into_compute_cell(tmp_path):
    from ksz_pipeline.coeval.limber import compute_cell
    N, L_h, h = 16, 128.0, 0.6766
    rng = np.random.default_rng(2)
    zre = (5 + 8 * rng.random((N, N, N))).astype(np.float32)
    for z in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        rho = (1 + 0.2 * rng.standard_normal((N, N, N))).astype(np.float32)
        rho /= rho.mean()
        mom = (200 * rng.standard_normal((3, N, N, N))).astype(np.float32)
        _write_fields(tmp_path / f'fields_z={z:05.2f}.dat', z, rho, mom)
    files = io.list_field_files(str(tmp_path))
    assert [io.z_from_name(f) for f in files] == sorted(
        io.z_from_name(f) for f in files)
    res = adapter.results_qperp_from_amber(files, zre, N, L_h, h,
                                           verbose=False)
    ells, D, sD, C, sC, tau, xe = compute_cell(res)
    assert len(ells) > 0 and np.all(np.isfinite(D)) and np.all(D > 0)


def test_rho_guard_trips_on_wrong_field(tmp_path):
    N = 8
    _write_fields(tmp_path / 'fields_z=07.00.dat', 7.0,
                  np.full((N, N, N), 2.0, np.float32),     # <rho> = 2
                  np.zeros((3, N, N, N), np.float32))
    with pytest.raises(RuntimeError, match='expected ~1'):
        adapter.results_qperp_from_amber(
            io.list_field_files(str(tmp_path)),
            np.full((N, N, N), 9.0, np.float32), N, 100.0, 0.7,
            verbose=False)
