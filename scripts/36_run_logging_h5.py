#!/usr/bin/env python
"""
Run any python script unchanged, and record every HDF5 file it opens.

Why: py21cmfast cache files are named by parameter hash
(IonizedBox_<md5>_r<seed>.h5), so from the filenames alone you cannot tell
which boxes a given run needs. Running the real job through this wrapper
gives the exact list, which scripts/37_cache_prune.py then uses as its
keep-list.

Usage (same arguments you would give the script, after the script path):
    H5_LOG=data/h5_touched_28.txt python scripts/36_run_logging_h5.py \
        scripts/28_stitched_dz_convergence.py --config configs/fiducial.yaml \
        --wrap-cycle-seed 123

Writes one absolute path per line to $H5_LOG at exit (also on error), and
also every 50 new files, so a walltime kill still leaves a usable partial log.
Files opened for reading OR writing are both recorded (both are "needed").
"""
import atexit
import os
import runpy
import sys

import h5py

LOG = os.environ.get("H5_LOG")
if not LOG:
    sys.exit("set H5_LOG=<output file> first")

_seen = set()
_orig_init = h5py.File.__init__


def _dump():
    tmp = LOG + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(sorted(_seen)) + ("\n" if _seen else ""))
    os.replace(tmp, LOG)


def _patched(self, name, *args, **kwargs):
    try:
        p = os.path.realpath(os.fspath(name))
        if p not in _seen:
            _seen.add(p)
            if len(_seen) % 50 == 0:
                _dump()
    except Exception:
        pass
    return _orig_init(self, name, *args, **kwargs)


h5py.File.__init__ = _patched
atexit.register(_dump)

if len(sys.argv) < 2:
    sys.exit(__doc__)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
