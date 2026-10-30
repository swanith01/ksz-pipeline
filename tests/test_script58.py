"""Smoke test of scripts/58 on a synthetic run (signed panel, NaN Limber range)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('healpy')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_shell_maps import _write_run, _field          # noqa: E402

spec = importlib.util.spec_from_file_location(
    's58', os.path.join(HERE, '..', 'scripts', '58_amber_native_shell_decomposition.py'))
s58 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s58)


def test_script58_runs_and_writes_outputs(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    cmb = run / 'output' / 'cmb'
    cmb.mkdir(parents=True)
    json.dump({'L': 400.0, 'amber_params': {'Tcmb': 2.725}}, open(run / 'amber_run.json', 'w'))
    _write_run(cmb, _field(True, seed=2), np.arange(1000.0, 1241.0, 20.0), nside=16)
    ell = np.arange(100, 400, 25.0)                   # Limber file starts at ell = 100
    np.savetxt(cmb / 'cl_ksz.txt', np.c_[ell, 1e-12 / ell**2], header='ell C_ell')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['58', '--run', str(run)])
    s58.main()
    out = capsys.readouterr().out
    assert 'nan' in out.lower()                       # no Limber below ell = 100: NaN, not clamped
    assert (run / 'native_shells.npz').exists()
    assert (tmp_path / 'data' / 'plots' / 'native_shells_run.png').stat().st_size > 10_000
