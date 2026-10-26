"""compute_cell(post_eor_zmin=...) and scripts/54 on a synthetic AMBER run."""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from ksz_pipeline.coeval.limber import compute_cell

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZS = np.round(np.arange(4.75, 18.26, 0.5), 2)        # AMBER lin shells 4.5-18.5


def _xH(z):
    """Fully ionized below 7.25, linear to fully neutral at 14.25."""
    return float(np.clip((z - 7.0) / 7.5, 0.0, 1.0))


def _res():
    k = np.logspace(-2, 0.5, 40)
    out = {}
    for z in ZS:
        xh = _xH(z)
        P = (1 - xh) * 1e14 * k**-1.0 * (1 + z) ** -2      # zero when neutral
        out[float(z)] = {'k': k, 'Pqperp': P, 'Pstd': 0.1 * P,
                         'xH_mean': xh, 'xH_mass': xh}
    return out


def test_default_window_unchanged():
    res = _res()
    a = compute_cell(res)
    b = compute_cell(res, post_eor_zmin=None)
    for x, y in zip(a[:5], b[:5]):
        assert np.array_equal(x, y)
    zt = a[5][0]
    assert all(1e-4 <= res[z]['xH_mean'] <= 1 - 1e-4 for z in zt)


def test_post_eor_extends_down_and_adds_power():
    res = _res()
    win = compute_cell(res, tau0=0.04)
    ext = compute_cell(res, tau0=0.03, post_eor_zmin=5.25)
    zt_w, zt_e = win[5][0], ext[5][0]
    assert zt_e.min() == pytest.approx(5.25)
    assert zt_e.max() == pytest.approx(zt_w.max())     # neutral end untouched
    assert set(zt_w) < set(zt_e)
    assert all(res[z]['xH_mean'] == 0.0 for z in zt_e if z < zt_w.min())
    l3w = int(np.argmin(np.abs(win[0] - 3000)))
    l3e = int(np.argmin(np.abs(ext[0] - 3000)))
    assert ext[1][l3e] > win[1][l3w]


def test_post_eor_zmin_above_everything_raises():
    with pytest.raises(ValueError):
        compute_cell(_res(), post_eor_zmin=50.0)


def _make_run(d):
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    from make_amber_input import planck18_for_amber
    p, _ = planck18_for_amber()
    meta = dict(N=64, L=500.0, czmin=4.5, czmax=18.5, czdel=0.5,
                cspacing='lin', XH=0.76, YHe=0.24, ne0_cgs=2.064357e-7,
                cosmology_preset='planck18', amber_params=p,
                snapshot_z=ZS.tolist())
    os.makedirs(d / 'output' / 'cmb')
    json.dump(meta, open(d / 'amber_run.json', 'w'))
    with open(d / 'qperp_checkpoint.pkl', 'wb') as fh:
        pickle.dump({'shells': _res(), 'files_done': set()}, fh)
    cmb = d / 'output' / 'cmb'
    zt = np.linspace(0, 30, 301)
    xe = np.array([1 - _xH(z) for z in zt])
    tau = np.concatenate([[0], np.cumsum(0.0007 * xe[1:] * (1 + zt[1:]) ** 0.5)])
    np.savetxt(cmb / 'tau.txt', np.c_[zt, xe, tau], header='z xe tau', comments='')
    ell = np.arange(100, 10001, 100, dtype=float)
    np.savetxt(cmb / 'cl_ksz.txt', np.c_[ell, 1e-13 / ell], header='l cl', comments='')
    kh = np.logspace(-2, 0.5, 40)
    for z in ZS:
        Pqq = (1 - _xH(z)) * 1e5 * kh**-1.0 + 1e-30
        np.savetxt(cmb / f'power_z={z:05.2f}.txt',
                   np.c_[kh, kh, kh, Pqq, kh, kh], header='cols', comments='')


def test_script54_runs_on_synthetic_run(tmp_path):
    run = tmp_path / 'run'
    _make_run(run)
    out = tmp_path / 'plot.png'
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, 'src'))
    r = subprocess.run([sys.executable, 'scripts/54_overlay_post_eor_vs_chen23.py',
                        '--run', str(run), '--out', str(out)],
                       cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    d = np.load(run / 'post_eor_vs_chen23.npz')
    assert d['window_ext'][0] == pytest.approx(5.25)     # 4.5-5.0 shell dropped
    i_w = int(np.argmin(np.abs(d['ells_win'] - 3000)))
    i_e = int(np.argmin(np.abs(d['ells_ext'] - 3000)))
    assert d['D_ours_ext'][i_e] > d['D_ours_win'][i_w]
    assert d['D_amber_ext'].size and d['D_amber_win'].size
