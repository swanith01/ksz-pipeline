"""
ksz_pipeline/amber/io.py

Readers for AMBER (github.com/hytrac/amber) outputs. Pure I/O: no unit
conversion here (see adapter.py), so every array comes back in AMBER's
own units, with AMBER's own axis order.

Files and layouts (checked against AMBER source, main branch, 2026-09):

  output/reion/zre_<fstr>.dat   written by reion_write (reion make='write')
      form='binary' (Intel raw stream, no record markers)
      real(4) zre(N,N,N)

  output/cmb/fields_<zstr>.dat  written by cmb_dumpfields -- NOT upstream,
      added by external/amber_patch/0001-dump-fields-for-ksz-pipeline.patch
      access='stream' (no record markers)
      real(4) z
      real(4) rho2(N,N,N)        1+delta, interlaced + deconvolved
      real(4) mom2(3,N,N,N)      (1+delta)*v [km/s], v = proper peculiar vel.

  output/cmb/power_<zstr>.txt   cmb_powerspectrum, one header line, columns
      k [h/Mpc], P_mm, P_ee, P_qq [(Mpc/h)^3 (km/s)^2], b_em, r_em
      NOTE P_qq has NO 1/2 and DOES include x_e^2 (He singly ionized).

  output/cmb/cl_ksz.txt         AMBER's own Limber C_ell (header + l, C)

Axis convention: Fortran arrays are column-major; reshaping with order='F'
gives numpy arr[i, j, k] == Fortran arr(i, j, k), i.e. axis 0 = x. For
mom2, arr[c, i, j, k] with c = 0,1,2 -> x,y,z components. This matches
qperp_power's meshgrid(indexing='ij') (axis 0 <-> kx).
"""
import glob
import os
import re

import numpy as np

_DT = np.dtype('<f4')   # AMBER runs on x86 (little-endian) nodes
_ZTAG = re.compile(r'z=\s*([0-9]+(?:\.[0-9]*)?)')


def read_zre(path, N):
    """Reionization-redshift field zre(N,N,N), float32."""
    a = np.fromfile(path, dtype=_DT)
    if a.size != N**3:
        raise ValueError(f"{path}: {a.size} floats, expected N^3={N**3} "
                         f"(wrong N, or not a raw-stream zre file)")
    return a.reshape((N, N, N), order='F')


def read_fields(path, N):
    """
    One cmb_dumpfields snapshot.

    Returns
    -------
    z   : float
    rho : ndarray (N,N,N)   1+delta
    mom : ndarray (3,N,N,N) (1+delta)*v [km/s]
    """
    a = np.fromfile(path, dtype=_DT)
    n3 = N**3
    if a.size != 1 + 4 * n3:
        raise ValueError(f"{path}: {a.size} floats, expected 1+4N^3="
                         f"{1 + 4 * n3} (wrong N, or file from an unpatched "
                         f"AMBER / different layout)")
    z = float(a[0])
    rho = a[1:1 + n3].reshape((N, N, N), order='F')
    mom = a[1 + n3:].reshape((3, N, N, N), order='F')
    return z, rho, mom


def z_from_name(path):
    """Redshift parsed from AMBER's '..._z=xx.xx...' filename tag."""
    m = _ZTAG.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"no z= tag in {path}")
    return float(m.group(1).rstrip('.'))


def list_field_files(cmb_dir):
    """fields_*.dat in cmb_dir, sorted by increasing z."""
    files = glob.glob(os.path.join(cmb_dir, 'fields_*.dat'))
    return sorted(files, key=z_from_name)


def find_zre_file(reion_dir):
    files = glob.glob(os.path.join(reion_dir, 'zre_*.dat'))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one zre_*.dat in {reion_dir}, found "
            f"{len(files)} -- one reionization history per output dir")
    return files[0]


def read_amber_power(path):
    """AMBER power_<zstr>.txt -> dict of columns (AMBER units)."""
    d = np.loadtxt(path, skiprows=1)
    return {'k_h': d[:, 0], 'P_mm': d[:, 1], 'P_ee': d[:, 2],
            'P_qq': d[:, 3], 'b_em': d[:, 4], 'r_em': d[:, 5]}


def read_amber_cl(path):
    """AMBER cl_ksz.txt / cl_tau.txt -> (ell, C_ell)."""
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]
