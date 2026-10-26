"""scripts/57 end to end on a synthetic AMBER run with a purely
longitudinal (gradient) velocity field."""
import json
import os
import subprocess
import sys

import numpy as np

from ksz_pipeline.amber import io
from ksz_pipeline.amber.stream_lightcone import stream_ksz_maps, open_zre

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))


def _make(run, N=16, all_ionized=False):
    from make_amber_input import chen2022_for_amber
    p, _ = chen2022_for_amber()
    zs = np.arange(4.75, 18.26, 0.5)
    json.dump(dict(N=N, L=128.0, amber_params=p, ne0_cgs=2e-7,
                   cosmology_preset='chen2022', czmin=4.5, czmax=18.5,
                   czdel=0.5, cspacing='lin'),
              open(run / 'amber_run.json', 'w'))
    (run / 'output' / 'cmb').mkdir(parents=True)
    (run / 'output' / 'reion').mkdir(parents=True)
    rng = np.random.default_rng(2)
    zre = np.full((N, N, N), 99.0) if all_ionized else 6 + 4 * rng.random((N, N, N))
    zre.astype('<f4').ravel(order='F').tofile(
        run / 'output' / 'reion' / 'zre_x.dat')
    k = np.fft.fftfreq(N) * 2 * np.pi
    phi = np.fft.fftn(rng.standard_normal((N, N, N)))
    for ax in range(3):                  # no Nyquist-plane power: a real
        sl = [slice(None)] * 3           # gradient there isn't longitudinal
        sl[ax] = N // 2
        phi[tuple(sl)] = 0.0
    for z in zs:
        mom = np.stack([np.fft.ifftn(1j * kk * phi).real for kk in (
            k.reshape(N, 1, 1), k.reshape(1, N, 1), k.reshape(1, 1, N))]) * 300
        with open(run / 'output' / 'cmb' / f'fields_z={z:05.2f}.dat', 'wb') as f:
            np.array([z], '<f4').tofile(f)
            np.ones(N**3, '<f4').tofile(f)
            np.asfortranarray(mom, '<f4').ravel(order='F').tofile(f)
    zt = np.linspace(0, 30, 301)
    np.savetxt(run / 'output' / 'cmb' / 'tau.txt',
               np.c_[zt, np.ones_like(zt), 0.004 * zt], header='z xe tau', comments='')
    return N


def _run57(run, tmp_path, *extra):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, 'src'))
    r = subprocess.run([sys.executable, 'scripts/57_qparallel_test_amber.py',
                        '--run', str(run), '--out', str(tmp_path / 'p.png'), *extra],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return np.load(run / 'qpar_test_reorient.npz')


def test_pure_gradient_fully_ionized_has_no_qperp(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _make(run, all_ionized=True)
    d = _run57(run, tmp_path)
    # q = grad(phi) exactly -> q_perp = 0 -> its map and Limber D vanish
    assert d['D_stitched_perp'].max() < 1e-12 * d['D_stitched_full'].max()
    assert d['D_limber_qperp'].max() < 1e-12 * d['D_limber_qpar'].max()


def test_script57_full_q_equals_ordinary_stitching(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    N = _make(run)
    d = _run57(run, tmp_path, '--keep-derived')
    assert np.all(d['D_limber_qpar'] > 0) and np.all(d['D_limber_qperp'] > 0)

    # full-q stitched == ordinary momentum-mode stitching of the originals
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import importlib
    los_grid = importlib.import_module('55_amber_stitched_streaming').los_grid
    meta = json.load(open(run / 'amber_run.json'))
    L = meta['L'] / meta['amber_params']['h']
    files = io.list_field_files(str(run / 'output' / 'cmb'))
    z_arr, pos = los_grid(5.0, 18.25, L / N, meta['amber_params'])
    tau = np.loadtxt(run / 'output' / 'cmb' / 'tau.txt', skiprows=1)
    orig = stream_ksz_maps(files, open_zre(io.find_zre_file(str(run / 'output' / 'reion')), N),
                           N, L, z_arr, pos, tau, 2e-7, 123, mode='momentum',
                           geometry='reorient')['map_total']
    work = run / 'qpar_test'
    derived = stream_ksz_maps(io.list_field_files(str(work / 'full')),
                              open_zre(str(work / 'zre_all_ionized.dat'), N),
                              N, L, z_arr, pos, tau, 2e-7, 123, mode='momentum',
                              geometry='reorient')['map_total']
    np.testing.assert_allclose(derived, orig, rtol=0, atol=1e-6 * np.abs(orig).max())
