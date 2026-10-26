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
