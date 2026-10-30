"""scripts/59: toy and real-map modes run and write both figures."""
import importlib.util
import json
import os

import numpy as np
import pytest

pytest.importorskip('healpy')
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'vis59', os.path.join(HERE, '..', 'scripts', '59_amber_native_visuals.py'))
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def test_shell_geometry_matches_run():
    edges, mids = vis.shell_geometry()
    assert len(mids) == 140 and np.all(np.diff(edges) > 0)
    # the run printed mid-shell chi = 5265-7422 and widths 6-42 Mpc/h (radiation ignored here)
    assert abs(mids[0] - 5265) < 10 and abs(mids[-1] - 7422) < 10
    w = np.diff(edges)
    assert 6 < w.min() < 7 and 41 < w.max() < 42


def test_toy_field_is_periodic():
    f = vis.toy_field(300.0)
    x = np.random.default_rng(0).uniform(0, 300, (50, 3))
    np.testing.assert_allclose(f(x), f(x + [300.0, 0, 0]), atol=1e-9)
    np.testing.assert_allclose(f(x), f(x + [0, 300.0, -300.0]), atol=1e-9)


def test_toy_mode_writes_both_figures(tmp_path):
    pre = str(tmp_path / 'toy')
    vis.main(['--prefix', pre, '--nside', '8', '--L', '2000,256'])
    for tail in ('_geometry.png', '_decomposition.png'):
        assert os.path.getsize(pre + tail) > 10_000


def test_real_mode_on_synthetic_run(tmp_path):
    from test_shell_maps import _write_run, _field
    run = tmp_path / 'run'
    cmb = run / 'output' / 'cmb'
    cmb.mkdir(parents=True)
    json.dump({'L': 400.0}, open(run / 'amber_run.json', 'w'))
    edges = np.arange(1000.0, 1401.0, 20.0)
    _write_run(cmb, _field(True, seed=1), edges, nside=16)
    pre = str(tmp_path / 'real')
    vis.main(['--run', str(run), '--prefix', pre])
    for tail in ('_geometry.png', '_decomposition.png'):
        assert os.path.getsize(pre + tail) > 10_000
