"""Thick-slab merging (shell_maps) and scripts/60, 61 smoke tests."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('healpy')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_shell_maps import _write_run, _field                      # noqa: E402
from ksz_pipeline.amber.shell_maps import (                          # noqa: E402
    slab_groups, merge_blocks, cumulative_vs_separation)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, '..', 'scripts', f'{name}.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_slab_groups_cover_and_respect_thickness():
    edges = np.concatenate([[1000.0], 1000.0 + np.cumsum(np.linspace(6, 42, 60))])
    lo, hi = edges[:-1], edges[1:]
    for T in (1.0, 50.0, 200.0, 5000.0):
        g = slab_groups(lo, hi, T)
        flat = np.concatenate(g)
        np.testing.assert_array_equal(flat, np.arange(60))
        for s in g[:-1]:
            assert hi[s[-1]] - lo[s[0]] >= T
    assert len(slab_groups(lo, hi, 1.0)) == 60
    assert len(slab_groups(lo, hi, 5000.0)) == 1


def test_merge_blocks_conserves_total_and_limits():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30, 7))
    C = A + A.transpose(1, 0, 2)                         # symmetric C_ij
    lo = np.arange(30) * 10.0
    hi = lo + 10.0
    tot = C.sum(axis=(0, 1))
    for T in (1.0, 35.0, 95.0, 1e5):
        g = slab_groups(lo, hi, T)
        Cs = merge_blocks(C, g)
        np.testing.assert_allclose(Cs.sum(axis=(0, 1)), tot, rtol=1e-12)
    np.testing.assert_allclose(np.einsum('iil->l', merge_blocks(C, slab_groups(lo, hi, 1.0))),
                               np.einsum('iil->l', C), rtol=1e-12)
    np.testing.assert_allclose(np.einsum('iil->l', merge_blocks(C, slab_groups(lo, hi, 1e5))), tot,
                               rtol=1e-12)


def test_cumulative_reaches_total():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((20, 20, 12))
    C = A + A.transpose(1, 0, 2)
    chi = np.arange(20) * 10.0
    toD = np.ones(12)
    cum = cumulative_vs_separation(C, chi, toD, (2, 10), np.array([0.0, 1e4]))
    diag = np.einsum('iil->il', C)[:, 2:10].sum() / 8
    np.testing.assert_allclose(cum[0], diag, rtol=1e-12)
    np.testing.assert_allclose(cum[1], C[:, :, 2:10].sum() / 8, rtol=1e-12)


def _make_run(path, seed):
    cmb = path / 'output' / 'cmb'
    cmb.mkdir(parents=True)
    json.dump({'L': 400.0, 'amber_params': {'Tcmb': 2.725}}, open(path / 'amber_run.json', 'w'))
    _write_run(cmb, _field(True, seed=seed), np.arange(1000.0, 1241.0, 10.0), nside=16)
    ell = np.arange(10, 60, 5.0)
    np.savetxt(cmb / 'cl_ksz.txt', np.c_[ell, 1e-12 / ell**2], header='ell C')


def test_scripts_60_and_61_smoke(tmp_path, monkeypatch):
    s58, s60, s61 = (_load(n) for n in ('58_amber_native_shell_decomposition',
                                       '60_native_thick_slabs',
                                       '61_overlay_native_vs_stitched'))
    runs = []
    for k in (1, 2):
        r = tmp_path / f'run{k}'
        _make_run(r, seed=k)
        runs.append(str(r))
    monkeypatch.chdir(tmp_path)
    for r in runs:
        monkeypatch.setattr(sys, 'argv', ['58', '--run', r])
        s58.main()
        s60.main(['--run', r, '--thickness', '0,40,100,1000', '--bands', '10,20;20,40'])
        d = np.load(os.path.join(r, 'native_thick_slabs.npz'))
        # thick limit: one slab -> diag == total; thin: diag != total
        np.testing.assert_allclose(d['D_diag'][-1][2:45], d['D_total'][2:45], rtol=1e-8)
        assert d['nslab'][0] == 24
    np.savez(tmp_path / 'st.npz', ell_total=np.array([10., 100, 1000]),
             D_total=np.array([1., 2, 3]), Derr_total=np.array([.1, .1, .1]))
    s61.main(['--native-run', *runs, '--stitched-npz', str(tmp_path / 'st.npz'),
              '--out', str(tmp_path / 'ov.png')])
    assert (tmp_path / 'ov.png').stat().st_size > 10_000
    assert any(p.name.startswith('native_thick_slabs_') for p in (tmp_path / 'data' / 'plots').iterdir())
