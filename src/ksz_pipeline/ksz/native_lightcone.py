"""
ksz_pipeline/ksz/native_lightcone.py

Builds a lightcone from py21cmfast's OWN run_lightcone() -- an alternative
to stitch_from_coeval.stitch_lightcone_from_coeval(), NOT a replacement.
Both are kept; this module exists purely to add a second data source that
matches the first one's output contract exactly, so everything downstream
(optical_depth.py, coherence_decomposition.py, and any driver script) works
unchanged regardless of which loader built the lightcone.

Contract matched exactly (see stitch_from_coeval.py's own docstring for the
authoritative statement of it):
    density    : ndarray (Nx,Ny,Nz)  RAW delta (mean ~0), NOT (1+delta)
    xH_box     : ndarray             neutral fraction
    velocity_z : ndarray             physical LOS peculiar velocity [cm/s]
    z_arr      : ndarray             redshift per slice
    pos_axis   : ndarray             comoving distance [Mpc] per slice

Two conversions are needed to reach that contract from what run_lightcone()
natively returns, and they carry very different levels of confidence:

1. DENSITY -- confirmed, not a guess. stitch_from_coeval.py's own module
   docstring states outright: "the native lightcone script, where
   lightcone.density (post-rotation) is already (1+delta)" (citing
   lightcone_integral.py's diagnostic print and git history, which this
   session did not have access to for independent re-verification -- if
   that file is available, worth a quick confirming look, but the claim is
   stated as settled fact in already-reviewed code, not speculation). So
   this module returns `lightcone.density - 1.0`, restoring the same raw-
   delta convention every downstream "density_1plus = 1.0 + stitched['density']"
   call already assumes.

2. VELOCITY -- NOT independently validated, by explicit agreement (17-22
   Sep 2026): for a pure periodicity EXISTENCE check (does P_diag vs
   n_groups show the same plateau/non-plateau structure the coeval path
   shows), an overall wrong scale factor on velocity only rescales D_ell by
   that factor squared -- it cannot manufacture or hide the grouping-
   dependence SHAPE the test is actually asking about. So the conversion
   below is the one the reference exploratory script
   (16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py) actually used
   in its real physics cells (lines 1721/2204/538: v_los[Mpc/s] =
   lightcone.velocity / H0) -- NOT the OTHER, inconsistent factor-of-1e17
   version that appears elsewhere in that same file (line 925) and is
   never actually used downstream there. H0 is taken from this pipeline's
   own Planck18 cosmology (astropy), not the reference script's literal
   "67.4", to stay internally consistent with the rest of this codebase
   rather than importing a second, slightly different assumed cosmology.
   DO NOT trust the resulting D_ell's ABSOLUTE amplitude for anything
   beyond the grouping-convergence shape test until this is independently
   checked -- e.g. against Reichardt+2021, the way fields.py's coeval
   velocity_conversion_factor was checked, before this module's numbers.

velocity_z here is IN cm/s, matching stitch_from_coeval's own contract, NOT
the Mpc/s the reference script printed -- converted via MPC_CM so callers
(which do `v_los_Mpc_s = stitched['velocity_z'] / MPC_CM`, script 20/21's own
convention) work identically regardless of loader.

Rotation/skewed-LOS (the reference script's periodicity-SUPPRESSION
technique, distinct from this module's periodicity-EXISTENCE check) is
deliberately NOT implemented here -- out of scope until the existence
check itself has run. angle_deg is accepted and validated but only 0.0 is
currently supported; passing anything else raises rather than silently
ignoring it.
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo

from ..utils.constants import MPC_CM

# py21cmfast is imported lazily inside stitch_lightcone_native, matching
# stitch_from_coeval.py's own pattern -- no reason to require it just to
# import this module.


def stitch_lightcone_native(z_min, z_max, HII_DIM, BOX_LEN, cache_dir,
                             angle_deg=0.0, N_THREADS=None, random_seed=None,
                             user_params_extra=None):
    """
    Build a (density, xH_box, velocity_z) lightcone via py21cmfast's own
    run_lightcone(), in the SAME dict shape stitch_lightcone_from_coeval
    returns -- see module docstring for the two unit conversions applied
    to get there, and their very different confidence levels.

    Parameters
    ----------
    z_min, z_max : float
        Passed through to run_lightcone as redshift=z_min (final/lowest)
        and max_redshift=z_max (starting/highest) -- same meaning as the
        reference script's own z_min/z_max.
    HII_DIM, BOX_LEN : int, float
        Same meaning as everywhere else in this pipeline.
    cache_dir : str
        py21cmfast cache directory.
    angle_deg : float
        Must be 0.0 -- see module docstring. Present in the signature (not
        just omitted) so a future skewed-LOS extension has an obvious slot,
        and so a caller who copies the coeval call's angle_deg=0.0 doesn't
        need to remember this loader has a different default.
    N_THREADS : int, optional
        Passed to UserParams. Defaults to py21cmfast's own default if None
        -- unlike coeval/fields.py's run_coeval_fields, this loader does NOT
        independently default to OMP_NUM_THREADS; pass it explicitly from
        the same config value the coeval path uses, for consistency.
    random_seed : int, optional
    user_params_extra : dict, optional
        Extra UserParams overrides merged in (e.g. USE_INTERPOLATION_TABLES).

    Returns
    -------
    dict, same keys/units as stitch_lightcone_from_coeval's return value.
    """
    if angle_deg != 0.0:
        raise NotImplementedError(
            "stitch_lightcone_native only supports angle_deg=0.0 for now -- "
            "the reference script's skewed-LOS rotation was deliberately "
            "left out of this first pass (periodicity EXISTENCE check, not "
            "yet the suppression technique). See module docstring.")

    import py21cmfast as p21c

    up_kwargs = dict(HII_DIM=int(HII_DIM), BOX_LEN=float(BOX_LEN))
    if N_THREADS is not None:
        up_kwargs["N_THREADS"] = int(N_THREADS)
    if user_params_extra:
        up_kwargs.update(user_params_extra)
    user_params = p21c.UserParams(**up_kwargs)

    lightcone = p21c.run_lightcone(
        redshift=float(z_min),
        max_redshift=float(z_max),
        # brightness_temp is NOT used by anything downstream of this
        # function, but py21cmfast v3.3.1's LightCone.shape/.n_slices/
        # .lightcone_distances/.lightcone_redshifts are all internally
        # wired through self.brightness_temp.shape (outputs.py, confirmed
        # by traceback 23 Sep 2026: AttributeError: 'LightCone' object has
        # no attribute 'brightness_temp', raised from inside
        # lightcone_redshifts itself) -- so it must be requested even
        # though we discard it, or reading the axes alone crashes before
        # density/xH_box/velocity are ever touched.
        lightcone_quantities=("brightness_temp", "density", "xH_box", "velocity"),
        user_params=user_params,
        random_seed=random_seed,
        direc=cache_dir,
    )

    red_axis_full = np.asarray(lightcone.lightcone_redshifts, dtype=float)
    pos_axis_full = np.asarray(lightcone.lightcone_distances, dtype=float)

    # Defensive trim, mirroring the reference script: run_lightcone can run
    # slightly past max_redshift.
    keep = red_axis_full <= float(z_max)
    z_arr = red_axis_full[keep]
    pos_axis = pos_axis_full[keep]

    # -- density: see module docstring point 1 --
    density = np.asarray(lightcone.density, dtype=np.float32)[:, :, keep] - 1.0

    # -- xH_box: no conversion needed --
    xH_box = np.asarray(lightcone.xH_box, dtype=np.float32)[:, :, keep]

    # -- velocity: see module docstring point 2 --
    H0_km_s_Mpc = float(cosmo.H0.value)
    v_los_Mpc_s = np.asarray(lightcone.velocity, dtype=np.float32)[:, :, keep] / H0_km_s_Mpc
    velocity_z = v_los_Mpc_s * MPC_CM   # -> cm/s, matching the shared contract

    return dict(density=density, xH_box=xH_box, velocity_z=velocity_z,
                z_arr=z_arr, pos_axis=pos_axis)
