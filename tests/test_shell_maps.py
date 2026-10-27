"""Shell-map analysis of AMBER's native lightcone (patch 0002 output)."""
import numpy as np
import pytest

hp = pytest.importorskip('healpy')

from ksz_pipeline.amber.shell_maps import (
    read_kszmap, load_shell_maps, shell_alms, cross_spectra, decompose,
    r_vs_dchi, lattice_vectors, direction_class, lattice_image_correlation)

L = 400.0                     # Mpc/h
NSIDE = 32


def _field(periodic, seed=0, nmin=2, nmax=6):
    """Broadband sum of plane waves. periodic=True: wavevectors on the box
    lattice 2*pi*n/L (exactly periodic); False: scaled off the lattice."""
    rng = np.random.default_rng(seed)
    ks, amps, phs = [], [], []
    for n in np.ndindex(*(2 * nmax + 1,) * 3):
        n = np.array(n) - nmax
        n2 = (n**2).sum()
        if nmin**2 <= n2 <= nmax**2:
            k = 2 * np.pi * n / L
            if not periodic:
                k = k * np.sqrt(2.1) + rng.normal(0, 0.003, 3)
            ks.append(k); amps.append(rng.normal() / np.sqrt(n2))
            phs.append(rng.uniform(0, 2 * np.pi))
    ks, amps, phs = np.array(ks), np.array(amps), np.array(phs)
    return lambda x: (amps * np.cos(x @ ks.T + phs)).sum(axis=-1)


def _write_run(d, f, chi_edges, nside=NSIDE):
    """Running maps in 0002's byte layout, high z (large chi) first."""
    n = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    mid = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    shells = [f(c * n) for c in mid]
    nz = len(mid)
    cum = np.zeros(hp.nside2npix(nside))
    for k in range(nz - 1, -1, -1):              # iz = k+1, highest first
        cum = cum + shells[k]
        with open(d / f'mapksz_z={k:05.2f}.dat', 'wb') as fh:
            np.array([nside, k + 1, nz], '<i8').tofile(fh)
            np.array([k, k + 0.5, k + 1.0], '<f8').tofile(fh)         # z (dummy)
            np.array([chi_edges[k], mid[k], chi_edges[k + 1]], '<f8').tofile(fh)
            cum.astype('<f8').tofile(fh)
    return np.array(shells), mid


def test_read_and_difference_roundtrip(tmp_path):
    edges = np.arange(1000.0, 1801.0, 40.0)
    shells, mid = _write_run(tmp_path, _field(True), edges)
    sm = load_shell_maps(str(tmp_path))
    np.testing.assert_allclose(sm['chi'], mid)
    np.testing.assert_allclose(sm['maps'], shells, atol=1e-10)
    np.testing.assert_allclose(sm['total'], shells.sum(axis=0), atol=1e-10)
    r = read_kszmap(str(tmp_path / 'mapksz_z=00.00.dat'))
    assert r['nside'] == NSIDE and r['iz'] == 1


def test_missing_shell_is_detected(tmp_path):
    edges = np.arange(1000.0, 1401.0, 40.0)
    _write_run(tmp_path, _field(True), edges)
    (tmp_path / 'mapksz_z=03.00.dat').unlink()
    with pytest.raises(ValueError):
        load_shell_maps(str(tmp_path))


def test_cross_spectra_match_healpy_and_sum_to_total(tmp_path):
    edges = np.arange(1000.0, 1401.0, 40.0)
    _write_run(tmp_path, _field(False, seed=3), edges)
    sm = load_shell_maps(str(tmp_path))
    lmax = 3 * NSIDE - 1
    alms = shell_alms(sm['maps'], lmax)
    C = cross_spectra(alms, lmax)
    for i, j in ((0, 0), (1, 4), (7, 2)):
        np.testing.assert_allclose(C[i, j], hp.alm2cl(alms[i], alms[j]),
                                   rtol=1e-10, atol=1e-30)
    dec = decompose(C, sm['chi'], L)
    tot = hp.alm2cl(hp.map2alm(sm['total'], lmax=lmax, iter=1))
    np.testing.assert_allclose(dec['total'], tot, rtol=1e-8, atol=1e-30)
    np.testing.assert_allclose(dec['diag'] + dec['near'] + dec['far'],
                               dec['total'], rtol=1e-12, atol=1e-30)


def test_r_vs_dchi_finds_copied_shells():
    rng = np.random.default_rng(1)
    chi = np.arange(0, 1600, 40.0)
    base = rng.standard_normal((len(chi), hp.nside2npix(NSIDE)))
    maps = base.copy()
    for i, c in enumerate(chi):                  # shell at chi+L copies chi
        j = np.flatnonzero(np.isclose(chi, c + L))
        if j.size:
            maps[j[0]] = base[i]
    lmax = 2 * NSIDE
    C = cross_spectra(shell_alms(maps, lmax), lmax)
    r = r_vs_dchi(C, chi, L, (2, lmax), np.array([0.45, 0.55, 0.95, 1.05]))
    assert r[2] > 0.3 and abs(r[0]) < 0.05


def test_lattice_image_correlation_periodic_vs_aperiodic(tmp_path):
    edges = np.arange(300.0, 1101.0, 10.0)        # reaches sqrt(3) L
    for periodic, d in ((True, tmp_path / 'p'), (False, tmp_path / 'a')):
        d.mkdir()
        _write_run(d, _field(periodic, seed=5, nmin=2, nmax=5), edges, nside=32)
    sp, sa = load_shell_maps(str(tmp_path / 'p')), load_shell_maps(str(tmp_path / 'a'))
    reps = [np.array(v, float) for v in ((1, 0, 0), (0, 1, -1), (1, 1, 1), (-1, 1, 1))]
    for m in reps:
        rho_p, n_p = lattice_image_correlation(sp, L, m)
        rho_c, _ = lattice_image_correlation(sp, L, m, scale=0.5)
        rho_a, _ = lattice_image_correlation(sa, L, m)
        assert n_p > 5000
        assert rho_p > 0.9, (m, rho_p)
        assert abs(rho_c) < 0.2 and abs(rho_a) < 0.2, (m, rho_c, rho_a)


def test_direction_class_counts():
    cls = direction_class(16, 10.0)
    assert set(np.unique(cls)) == {0, 1, 2, 3}
    frac = np.array([(cls == k).mean() for k in (1, 2, 3)])
    # cap solid angle 2pi(1-cos10deg) per direction: 6, 12, 8 directions
    expect = np.array([6, 12, 8]) * 2 * np.pi * (1 - np.cos(np.radians(10))) / (4 * np.pi)
    np.testing.assert_allclose(frac, expect, rtol=0.25)
