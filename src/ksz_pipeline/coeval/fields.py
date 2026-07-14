"""
Shared coeval-box field loader.

Extracted from scripts/02_make_ksz_coeval_boxes.py's run_coeval_fields()
so there is exactly one implementation -- reused by both that script and
ksz/stitch_from_coeval.py. Velocity conversion is velocity_conversion_factor()
from coeval/velocity.py, independently confirmed twice: the quicktest v/c
sanity check (8Jul2026) and the dz-convergence plot landing on Reichardt's
D_pkSZ at full 800Mpc/128^3 resolution. Do not reimplement this elsewhere --
that's exactly how the pipeline ended up with silently-different velocity
conventions in different places before.

FIXED (14Jul2026): user_params never set N_THREADS, silently defaulting to
1 regardless of OMP_NUM_THREADS or however many cores were requested from
PBS. Confirmed via a live fiducial run stuck at 21+ hours with a single
python process pinned at 100% CPU (not ~3200%, which 32 real threads would
show) and cput tracking walltime 1:1. Script 01's lightcone run, which
DOES pass N_THREADS explicitly, finished the same 800Mpc/128^3 scale
problem in under 4 minutes -- direct confirmation this was the cause, not
a coincidence of problem size. N_THREADS now defaults to OMP_NUM_THREADS
if set (matching every existing PBS script's `export OMP_NUM_THREADS=32`
convention automatically), falling back to 1 only if that's genuinely
unset (e.g. interactive/quicktest use on a small allocation).
"""

import os

import py21cmfast as p21c

from .velocity import velocity_conversion_factor


def run_coeval_fields(z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=None):
    """
    Run (or load from py21cmfast's own cache) a coeval box at redshift z
    and return its density, neutral fraction, and velocity fields.

    Parameters
    ----------
    z         : float   redshift
    HII_DIM   : int      grid resolution
    BOX_LEN   : float    comoving box side length [Mpc]
    cache_dir : str       py21cmfast cache directory
    N_THREADS : int, optional. Defaults to int(os.environ['OMP_NUM_THREADS'])
                if that's set, else 1. Pass explicitly to override.

    Returns
    -------
    delta : ndarray (HII_DIM,)^3   raw density contrast (mean ~0, NOT 1+delta)
    xH    : ndarray (HII_DIM,)^3   neutral fraction
    vx, vy, vz : ndarray (HII_DIM,)^3   physical peculiar velocities [cm/s]
    """
    if N_THREADS is None:
        N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

    coeval = p21c.run_coeval(
        redshift    = float(z),
        user_params = {"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN),
                        "N_THREADS": int(N_THREADS)},
        write       = False,
        direc       = cache_dir,
    )
    fac   = velocity_conversion_factor(z)
    delta = coeval.density
    xH    = coeval.xH_box
    vx    = coeval.lowres_vx * fac * 1e5
    vy    = coeval.lowres_vy * fac * 1e5
    vz    = coeval.lowres_vz * fac * 1e5
    return delta, xH, vx, vy, vz
