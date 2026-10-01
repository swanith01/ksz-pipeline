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

ADDED (1Oct2026): astro_params/flag_options passthrough, for the
Nikolic/Mesinger/Gorce (2023) replication, which uses 21cmFAST's flexible
mass-dependent SFR/escape-fraction astrophysics model (F_STAR10,
ALPHA_STAR, F_ESC10, ALPHA_ESC, M_TURN, t_STAR) rather than this
pipeline's existing constant-efficiency HII_EFF_FACTOR model. Both new
kwargs default to None and are omitted entirely from the run_coeval() call
in that case -- the default path is byte-identical to before this change.
IMPORTANT: astro_params alone does nothing in 21cmFAST -- F_STAR10 and
friends are silently ignored (falling back to HII_EFF_FACTOR) unless
flag_options also sets USE_MASS_DEPENDENT_ZETA=True. Pass both together.
Verify which model actually ran (don't assume): the returned coeval
object's `.astro_params` and `.flag_options` reflect what 21cmFAST
actually used, not just what was requested.
"""

import os

import py21cmfast as p21c

from .velocity import velocity_conversion_factor


def run_coeval_fields(z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=None, random_seed=None,
                       astro_params=None, flag_options=None):
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
    astro_params : dict, optional
        Passed straight through to py21cmfast.run_coeval(astro_params=...).
        Default None: omitted from the call entirely, identical to the
        pre-existing behavior (21cmFAST's own default AstroParams, i.e.
        HII_EFF_FACTOR=30). Requires flag_options with
        USE_MASS_DEPENDENT_ZETA=True to actually take effect -- see module
        docstring.
    flag_options : dict, optional
        Passed straight through to py21cmfast.run_coeval(flag_options=...).
        Default None: omitted from the call entirely, identical to the
        pre-existing behavior.

    Returns
    -------
    delta : ndarray (HII_DIM,)^3   raw density contrast (mean ~0, NOT 1+delta)
    xH    : ndarray (HII_DIM,)^3   neutral fraction
    vx, vy, vz : ndarray (HII_DIM,)^3   physical peculiar velocities [cm/s]
    """
    if N_THREADS is None:
        N_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

    run_coeval_kwargs = dict(
        redshift    = float(z),
        user_params = {"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN),
                        "N_THREADS": int(N_THREADS)},
        random_seed = random_seed,
        write       = True,   # FIXED 18Jul2026: was False, silently disabling
                                # all persistent caching -- every call, across
                                # every script, regenerated boxes from scratch
                                # regardless of matching params/seed. Confirmed
                                # against py21cmfast's own docs. Does not affect
                                # correctness of any existing result, only speed.
        direc       = cache_dir,
    )
    if astro_params is not None:
        run_coeval_kwargs["astro_params"] = astro_params
    if flag_options is not None:
        run_coeval_kwargs["flag_options"] = flag_options

    coeval = p21c.run_coeval(**run_coeval_kwargs)
    fac   = velocity_conversion_factor(z)
    delta = coeval.density
    xH    = coeval.xH_box
    vx    = coeval.lowres_vx * fac * 1e5
    vy    = coeval.lowres_vy * fac * 1e5
    vz    = coeval.lowres_vz * fac * 1e5
    return delta, xH, vx, vy, vz
