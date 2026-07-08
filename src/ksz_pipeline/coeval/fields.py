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
"""

import py21cmfast as p21c

from .velocity import velocity_conversion_factor


def run_coeval_fields(z, HII_DIM, BOX_LEN, cache_dir):
    """
    Run (or load from py21cmfast's own cache) a coeval box at redshift z
    and return its density, neutral fraction, and velocity fields.

    Parameters
    ----------
    z         : float   redshift
    HII_DIM   : int      grid resolution
    BOX_LEN   : float    comoving box side length [Mpc]
    cache_dir : str       py21cmfast cache directory

    Returns
    -------
    delta : ndarray (HII_DIM,)^3   raw density contrast (mean ~0, NOT 1+delta)
    xH    : ndarray (HII_DIM,)^3   neutral fraction
    vx, vy, vz : ndarray (HII_DIM,)^3   physical peculiar velocities [cm/s]
    """
    coeval = p21c.run_coeval(
        redshift    = float(z),
        user_params = {"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN)},
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
