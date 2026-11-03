"""pixel_window reader and scripts/62 (synthetic runs, synthetic window files)."""
import importlib.util
import os

import numpy as np
import pytest

pytest.importorskip('astropy')
HERE = os.path.dirname(os.path.abspath(__file__))
from ksz_pipeline.amber.shell_maps import pixel_window                # noqa: E402


def _w(ell, nside):
    return np.exp(-0.5 * (ell / (2.0 * nside))**2)                    # smooth, <1


def _write_pixwin(d, nside):
    from astropy.io import fits
    ell = np.arange(4 * nside + 1)
    w = _w(ell, nside)
    cols = [fits.Column(name='TEMPERATURE', format='D', array=w),
            fits.Column(name='POLARIZATION', format='D', array=0.9 * w)]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        os.path.join(d, f'pixel_window_n{nside:04d}.fits'))


def test_pixel_window_reads_temperature_column(tmp_path):
    _write_pixwin(str(tmp_path), 16)
    w = pixel_window(16, 40, str(tmp_path))
    np.testing.assert_allclose(w, _w(np.arange(41), 16))
    with pytest.raises(ValueError):
        pixel_window(16, 200, str(tmp_path))


def test_script62_corrected_runs_agree(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        's62', os.path.join(HERE, '..', 'scripts', '62_native_pixwin_check.py'))
    s62 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s62)
    wdir = tmp_path / 'win'
    wdir.mkdir()
    runs = []
    for nside in (16, 32):
        _write_pixwin(str(wdir), nside)
        ell = np.arange(3 * nside)
        Dtrue = 1.0 + 100.0 / (1 + ell)
        r = tmp_path / f'run{nside}'
        r.mkdir()
        np.savez(r / 'native_shells.npz', ell=ell, D_total=Dtrue * _w(ell, nside)**2,
                 nside=nside, ell_limber=np.arange(10, 90.0), D_limber=np.full(80, 1.0), L=400.0)
        runs.append(str(r))
    np.savez(tmp_path / 'st.npz', ell_total=np.array([10., 30, 60, 90]),
             D_total=np.array([8., 4, 2, 1.5]), Derr_total=np.full(4, .1))
    out = tmp_path / 'c.png'
    s62.main(['--native-run', runs[1], runs[0], '--stitched-npz', str(tmp_path / 'st.npz'),
              '--pixwin-dir', str(wdir), '--bands', '10,20;20,40;40,60;60,90',
              '--out', str(out)])
    txt = capsys.readouterr().out
    assert out.stat().st_size > 10_000
    line = [l for l in txt.splitlines() if 'max |corrected ratio - 1|' in l][0]
    assert float(line.split(':')[1]) < 1e-9                    # window removed exactly
    # raw ratio is NOT 1 (windows differ), so the check is meaningful
    raw = [l for l in txt.splitlines() if l.strip().startswith('raw')][0].split()[1:]
    assert max(abs(float(v) - 1) for v in raw) > 0.01
