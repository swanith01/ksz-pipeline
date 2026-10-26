"""stream_lightcone must reproduce the in-memory stitched pipeline
(stitch_lightcone_from_amber -> compute_ksz_map / amber_lightcone_to_ksz_Dl)
when given the same LOS grid and distances."""
import numpy as np
import pytest

from ksz_pipeline.amber import io
from ksz_pipeline.amber.stitch_from_amber import (
    stitch_lightcone_from_amber, amber_lightcone_to_ksz_Dl)
from ksz_pipeline.amber.stream_lightcone import stream_ksz_maps, open_zre
from ksz_pipeline.ksz.lightcone_integral import compute_ksz_map
from ksz_pipeline.ksz.stitch_from_coeval import (
    build_los_z_grid, comoving_distance_mpc)
from ksz_pipeline.utils.constants import MPC_CM


def _write_fields(path, z, rho, mom):
    with open(path, 'wb') as f:
        np.array([z], '<f4').tofile(f)
        np.asfortranarray(rho, dtype='<f4').ravel(order='F').tofile(f)
        np.asfortranarray(mom, dtype='<f4').ravel(order='F').tofile(f)


@pytest.fixture
def run(tmp_path):
    N, L_h, h = 16, 64.0, 0.6766        # tiny box: LOS wraps many times
    rng = np.random.default_rng(11)
    zre = (6 + 5 * rng.random((N, N, N))).astype(np.float32)
    np.asfortranarray(zre, dtype='<f4').ravel(order='F').tofile(
        tmp_path / 'zre_test.dat')
    for z in np.arange(5.5, 12.6, 1.0):
        rho = (1 + 0.3 * rng.standard_normal((N, N, N))).astype(np.float32)
        rho /= rho.mean()
        mom = (300 * rng.standard_normal((3, N, N, N))).astype(np.float32)
        _write_fields(tmp_path / f'fields_z={z:05.2f}.dat', z, rho, mom)
    zt = np.linspace(0, 30, 301)
    tau = np.c_[zt, np.ones_like(zt), 0.004 * zt]
    return dict(dir=tmp_path, N=N, L_h=L_h, h=h, tau=tau, ne0=2.0e-7,
                files=io.list_field_files(str(tmp_path)))


def _reference(run, z_arr, seed):
    zre = io.read_zre(str(run['dir'] / 'zre_test.dat'), run['N'])
    lc = stitch_lightcone_from_amber(run['files'], zre, z_arr, run['N'],
                                     run['L_h'], run['h'], interp=True,
                                     wrap_cycle_seed=seed)
    L = run['L_h'] / run['h']
    patchy = amber_lightcone_to_ksz_Dl(lc, run['tau'], L, run['ne0'])
    vis = np.exp(-np.interp(z_arr, run['tau'][:, 0], run['tau'][:, 2]))
    total = compute_ksz_map(1.0 + lc['density'], 1.0 - lc['xH_box'],
                            lc['velocity_z'] / MPC_CM, z_arr,
                            np.diff(lc['pos_axis']), vis[None, None, :],
                            ne0=run['ne0'], patchy_mask_3D=None)
    return patchy, total, lc


@pytest.mark.parametrize('seed', [7, 123])
def test_stream_matches_in_memory(run, seed):
    N, L = run['N'], run['L_h'] / run['h']
    z_arr = build_los_z_grid(6.0, 11.5, L / N)
    pos = np.array([comoving_distance_mpc(z) for z in z_arr])
    assert (pos[-1] - pos[0]) / L > 3            # several wrap cycles
    patchy, total, _ = _reference(run, z_arr, seed)
    s = stream_ksz_maps(run['files'], open_zre(str(run['dir'] / 'zre_test.dat'), N),
                        N, L, z_arr, pos, run['tau'], run['ne0'], seed)
    scale = np.abs(total).max()
    np.testing.assert_allclose(s['map_total'], total, rtol=0, atol=2e-5 * scale)
    np.testing.assert_allclose(s['map_patchy'], patchy['ksz_map'], rtol=0,
                               atol=2e-5 * np.abs(patchy['ksz_map']).max())
    assert s['chi_eff_patchy'] == pytest.approx(patchy['chi_eff'], rel=1e-5)
    assert s['z_window_patchy'] == pytest.approx(patchy['z_window'], abs=1e-9)


def test_stream_extrapolates_like_interp1d(run):
    """LOS reaching past the last snapshot: both paths extrapolate."""
    N, L = run['N'], run['L_h'] / run['h']
    z_arr = build_los_z_grid(5.0, 12.9, L / N)    # snapshots span 5.5-12.5
    pos = np.array([comoving_distance_mpc(z) for z in z_arr])
    _, total, _ = _reference(run, z_arr, 3)
    s = stream_ksz_maps(run['files'], open_zre(str(run['dir'] / 'zre_test.dat'), N),
                        N, L, z_arr, pos, run['tau'], run['ne0'], 3)
    np.testing.assert_allclose(s['map_total'], total, rtol=0,
                               atol=2e-5 * np.abs(total).max())


# --- mode='momentum' --------------------------------------------------------
def _uniform_ionized_run(tmp_path, rho_fn):
    N, L_h, h = 16, 64.0, 0.6766
    rng = np.random.default_rng(5)
    zre = np.full((N, N, N), 30.0, np.float32)          # ionized everywhere
    np.asfortranarray(zre, dtype='<f4').ravel(order='F').tofile(
        tmp_path / 'zre_test.dat')
    for z in np.arange(5.5, 12.6, 1.0):
        mom = (300 * rng.standard_normal((3, N, N, N))).astype(np.float32)
        _write_fields(tmp_path / f'fields_z={z:05.2f}.dat', z,
                      rho_fn(rng, N).astype(np.float32), mom)
    L = L_h / h
    z_arr = build_los_z_grid(6.0, 11.5, L / N)
    pos = np.array([comoving_distance_mpc(z) for z in z_arr])
    zt = np.linspace(0, 30, 301)
    tau = np.c_[zt, np.ones_like(zt), 0.004 * zt]

    def go(mode):
        return stream_ksz_maps(io.list_field_files(str(tmp_path)),
                               open_zre(str(tmp_path / 'zre_test.dat'), N),
                               N, L, z_arr, pos, tau, 2e-7, 9, mode=mode)
    return go


def test_momentum_equals_velocity_when_rho_uniform(tmp_path):
    go = _uniform_ionized_run(tmp_path, lambda rng, N: np.ones((N, N, N)))
    v, m = go('velocity'), go('momentum')
    np.testing.assert_allclose(m['map_total'], v['map_total'], rtol=0,
                               atol=1e-6 * np.abs(v['map_total']).max())


def _voidy(rng, N):
    rho = 1 + 0.3 * rng.standard_normal((N, N, N))
    rho[rng.random((N, N, N)) < 0.02] = 1e-4             # near-empty cells
    return np.clip(rho, 1e-4, None)


def test_momentum_mode_ignores_rho_and_velocity_mode_blows_up(tmp_path):
    d1, d2 = tmp_path / 'a', tmp_path / 'b'
    d1.mkdir(); d2.mkdir()
    voidy = _uniform_ionized_run(d1, _voidy)
    flat = _uniform_ionized_run(d2, lambda rng, N: np.ones((N, N, N)))
    # rho-independence of momentum mode is tested separately below
    m_voidy, v_voidy = voidy('momentum'), voidy('velocity')
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    assert rms(v_voidy['map_total']) > 10 * rms(m_voidy['map_total'])
    assert rms(flat('momentum')['map_total']) > 0


def test_momentum_mode_is_independent_of_rho(tmp_path):
    """Same mom, different rho -> identical momentum-mode maps."""
    N = 16
    rng = np.random.default_rng(3)
    moms = {z: (300 * rng.standard_normal((3, N, N, N))).astype(np.float32)
            for z in np.arange(5.5, 12.6, 1.0)}
    maps = []
    for k, rho_fn in enumerate((lambda: np.ones((N, N, N)),
                                lambda: _voidy(np.random.default_rng(1), N))):
        d = tmp_path / f'r{k}'
        d.mkdir()
        np.full((N, N, N), 30.0, '<f4').ravel(order='F').tofile(d / 'zre.dat')
        for z, mom in moms.items():
            _write_fields(d / f'fields_z={z:05.2f}.dat', z,
                          rho_fn().astype(np.float32), mom)
        L = 64.0 / 0.6766
        z_arr = build_los_z_grid(6.0, 11.5, L / N)
        pos = np.array([comoving_distance_mpc(z) for z in z_arr])
        zt = np.linspace(0, 30, 301)
        maps.append(stream_ksz_maps(
            io.list_field_files(str(d)), open_zre(str(d / 'zre.dat'), N), N, L,
            z_arr, pos, np.c_[zt, np.ones_like(zt), 0.004 * zt], 2e-7, 9,
            mode='momentum')['map_total'])
    np.testing.assert_array_equal(maps[0], maps[1])
