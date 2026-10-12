"""
Build a lightcone by stitching coeval boxes along the line of sight,
completely independent of the native p21c.run_lightcone()-based path.

VENDORED + EXTENDED from ksz-pipeline (github.com/swanith01/ksz-pipeline),
commit 7305482def70cd94ab1f7b553db13070fd619b76 (2026-10-06), file
src/ksz_pipeline/ksz/stitch_from_coeval.py.

THE ONE CHANGE from upstream is in stitch_lightcone_from_coeval() at the
bottom of this file: it now also stitches brightness_temp (21cmFAST's own
delta_T box, via the extended simulate/coeval_fields.py::run_coeval_fields)
as a fourth field, through the exact same stitch_field() call -- same
angle_deg / wrap_cycle_seed / wrap_mode -- as density, xH, and velocity_z.
ALL GEOMETRY HELPER FUNCTIONS BELOW (build_los_z_grid through
get_slab_bilinear, and stitch_field itself) are untouched, byte-for-byte:
they are field-agnostic, they just interpolate whatever ndarray dict
they're handed, so no change was needed to extend them to a 4th field.

THIS IS THE LOAD-BEARING CORRECTNESS POINT for the whole kSZ-21cm
cross-correlation: stitch_field's rotation (whether the legacy single
angle_deg, or a wrap_cycle_seed draw) is selected ONCE per LOS position
and must be IDENTICAL across every field stitched at that position, or
kSZ and 21cm end up sampled from different spatial locations after a
periodic wrap -- which would spuriously suppress (or corrupt) exactly the
cross-correlation this repo exists to measure, and would do so silently
(the stitching would still run and produce plausible-looking maps). See
tests/test_shared_rotation.py for a synthetic check of this property
before trusting any real run.

Purpose: an independent cross-check of the native lightcone's ~100-1000x
D_ell excess, using ONLY already-validated pieces -- coeval box generation
and velocity handling from coeval/fields.py, which has now been confirmed
twice (the quicktest v/c sanity check, and the dz-convergence plot landing
on Reichardt's D_pkSZ at full 800Mpc/128^3). If this stitched lightcone's
D_ell also lands near the coeval number, the bug is isolated to the native
run_lightcone() path specifically.

Adapted from ksz_lae_xcorr's lightcone/stitch.py (a DIFFERENT project's
kSZ-LAE cross-correlation pipeline -- not this repo's own prior code).
Kept: the geometry (comoving-pixel indexing along the LOS, rotation,
per-redshift interpolation across snapshots). Dropped: all halo/LAE/LBG
tracer code (out of scope here) and, deliberately, its own velocity
conversion, which used an unaudited constant -- replaced with
coeval.fields.run_coeval_fields().

IMPORTANT density convention, read before using: the density field
returned here is RAW delta (mean ~0), matching run_coeval_fields()'s
convention -- NOT (1+delta). This is deliberately different from a native
run_lightcone()-based script, where lightcone.density (post-rotation) may
already be (1+delta). Callers must add "+1" themselves, exactly once,
before passing this to compute_ksz_map. brightness_temp needs NO such
treatment -- it's 21cmFAST's own delta_T box in mK, already the physical
fluctuation field, not a density-contrast convention.

Rotation: angle_deg=0 (no rotation) is the recommended default for a
FIRST cross-check run against the old (native-lightcone) ksz2-21cm result.

UPDATE: a single, fixed angle_deg (whether 0 or nonzero) does not break
a snapshot's own periodic repeat along the LOS -- verified directly: the
same (ir,jr) rotation applied at a y_cell that later wraps back to the
same value pulls a bit-identical slab both times, since nothing about
the sampling differs between the two visits. For that, use the
wrap_cycle_seed parameter (stitch_field / stitch_lightcone_from_coeval)
instead, which varies the rotation per periodic wrap-cycle (one BOX_LEN
of comoving distance) rather than using one fixed angle throughout, and
uses bilinear rather than nearest-neighbor sampling to avoid the 10-31%
pixel aliasing nearest-neighbor rotation has at generic angles. This is
additive: angle_deg alone (wrap_cycle_seed=None, the default) is
untouched.
"""

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

# run_coeval_fields is imported lazily, inside stitch_lightcone_from_coeval,
# not here -- it pulls in py21cmfast, and the geometry/interpolation
# functions below have no reason to require that just to be imported/tested.


def build_los_z_grid(z_min, z_max, cell_size, z_oversample=4000):
    """
    Build a line-of-sight redshift grid with UNIFORM COMOVING DISTANCE
    spacing (step = cell_size), not uniform in z.

    dchi/dz is not constant over z=4-20. Implementation: build chi(z) on a
    fine auxiliary z grid, then invert (interpolate z as a function of
    chi) onto a uniform chi grid with exactly cell_size spacing.

    Parameters
    ----------
    z_min, z_max : float
    cell_size    : float, comoving Mpc -- the LOS step size to target,
                   normally BOX_LEN/HII_DIM so ds matches the box's own
                   resolution
    z_oversample : int, resolution of the auxiliary z grid used only to
                   invert chi(z)

    Returns
    -------
    z_arr : ndarray, ascending, with (by construction) uniform comoving
            spacing == cell_size between consecutive entries
    """
    z_fine   = np.linspace(z_min, z_max, z_oversample)
    chi_fine = np.array([comoving_distance_mpc(z) for z in z_fine])

    chi_min, chi_max = chi_fine[0], chi_fine[-1]
    n_pix = int(round((chi_max - chi_min) / cell_size))
    chi_target = chi_min + cell_size * np.arange(n_pix)

    return np.interp(chi_target, chi_fine, z_fine)


def comoving_distance_mpc(z):
    return cosmo.comoving_distance(z).to(u.Mpc).value


def comoving_pixel(z, z0, cell_size, ngrid):
    """LOS pixel index for redshift z, relative to reference redshift z0."""
    d = comoving_distance_mpc(z) - comoving_distance_mpc(z0)
    return int(round(d / cell_size)) % ngrid


def rotated_indices(ngrid, angle_deg):
    """
    Vectorized transverse index grid, rotated by angle_deg and
    periodically wrapped on both axes.

    Returns
    -------
    ir, jr : ndarray (ngrid, ngrid) int   rotated, wrapped index grids
    """
    a = np.deg2rad(angle_deg)
    i, j = np.meshgrid(np.arange(ngrid), np.arange(ngrid), indexing='ij')
    ir = np.round(np.cos(a) * i - np.sin(a) * j).astype(int) % ngrid
    jr = np.round(np.sin(a) * i + np.cos(a) * j).astype(int) % ngrid
    return ir, jr


def get_slab(box, y_cell, ir, jr):
    """Rotated (ngrid, ngrid) transverse slab of `box` at LOS index y_cell."""
    return box[ir, jr, y_cell]


# ---------------------------------------------------------------------
# Per-wrap-cycle rotation (opt-in via stitch_field's wrap_cycle_seed).
#
# A SINGLE, FIXED rotation (angle_deg above, applied identically at every
# LOS position) does NOT break the periodic repeat of any one snapshot's
# own box: box[ir,jr,y_cell] is bit-identical the next time y_cell wraps
# back to the same value, for ANY fixed (ir,jr). The rotation angle must
# instead depend on WHICH periodic repeat ("wrap cycle") of the box a
# given LOS position falls into -- one full BOX_LEN of comoving distance
# per cycle.
#
# The angle must be the SAME for every snapshot AND every field queried
# at a given LOS position: stitch_field already queries every snapshot at
# the same y_cell for a given target z, so keying the angle on the LOS
# position's cycle index (not on which snapshot, and not on which field)
# automatically shares one rotation across everything blended there.
#
# Nearest-neighbor rotation (rotated_indices/get_slab above) aliases
# 10-31% of pixels for generic (non-multiple-of-90) angles. Since per-
# wrap-cycle rotation needs many distinct, generic angles, this path uses
# BILINEAR interpolation instead (scipy.ndimage.map_coordinates, order=1).
# ---------------------------------------------------------------------

def wrap_cycle_index(z, z0, cell_size, ngrid):
    """
    Which periodic wrap-cycle (integer, 0-indexed) this redshift's LOS
    position falls into, relative to z0. One cycle = one full BOX_LEN
    (= cell_size * ngrid) of comoving distance.
    """
    d = comoving_distance_mpc(z) - comoving_distance_mpc(z0)
    box_len = cell_size * ngrid
    return int(d // box_len)


def cycle_angle(cycle_idx, seed):
    """
    Deterministic rotation angle (degrees) for a given wrap-cycle index.
    np.random.default_rng accepts a sequence of ints as a seed, so
    (seed, cycle_idx) fully determines the angle with no shared mutable
    state and no caching needed: calling this twice with the same
    (seed, cycle_idx) always returns the same angle -- exactly what's
    needed for different snapshots AND different fields (density/xH/
    velocity/brightness_temp) queried at the same LOS position to agree.
    """
    rng = np.random.default_rng((int(seed), int(cycle_idx)))
    return float(rng.uniform(0.0, 360.0))


def rotated_coords(ngrid, angle_deg):
    """
    Continuous (non-rounded) rotated transverse coordinate grids, for
    bilinear sampling via get_slab_bilinear.

    Returns
    -------
    ic, jc : ndarray (ngrid, ngrid) float   continuous rotated coordinates
    """
    a = np.deg2rad(angle_deg)
    i, j = np.meshgrid(np.arange(ngrid), np.arange(ngrid), indexing='ij')
    ic = np.cos(a) * i - np.sin(a) * j
    jc = np.sin(a) * i + np.cos(a) * j
    return ic, jc


def get_slab_bilinear(box, y_cell, ic, jc, wrap_mode='wrap'):
    """
    Bilinearly-interpolated, periodically-wrapped rotated (ngrid, ngrid)
    transverse slab of `box` at LOS index y_cell, at the continuous
    rotated coordinates (ic, jc) from rotated_coords().

    wrap_mode : 'wrap' (default) or 'grid-wrap'.
        'wrap' is scipy's mode='wrap', which treats index 0 and index n-1
        as the SAME point, i.e. the box is wrapped with period n-1, not n
        (verified: on arange(8), x=8.0 returns 1.0, not 0.0). Every
        upstream result before 6Oct2026 used this mode; it is the default
        here so those results stay reproducible.
        'grid-wrap' has the correct period n for a periodic grid.
    """
    if wrap_mode not in ('wrap', 'grid-wrap'):
        raise ValueError(f"wrap_mode must be 'wrap' or 'grid-wrap', got {wrap_mode!r}")
    return map_coordinates(box[:, :, y_cell], [ic, jc], order=1,
                            mode=wrap_mode)


def stitch_field(snapshot_boxes, snap_z, z_arr, z0, cell_size, ngrid,
                  angle_deg=0.0, wrap_cycle_seed=None, wrap_mode='wrap'):
    """
    Interpolate one field, already loaded per snapshot redshift, onto a
    continuous LOS redshift grid z_arr.

    For each target redshift z_arr[n]: compute the LOS pixel y_cell
    (relative to z0), extract the transverse slab at that SAME y_cell
    from every available snapshot box, then interpolate ACROSS SNAPSHOTS
    (in redshift) to the value at z_arr[n].

    Parameters
    ----------
    snapshot_boxes : dict {z: ndarray(ngrid,ngrid,ngrid)}
    snap_z         : ndarray, sorted snapshot redshifts (keys above)
    z_arr          : ndarray (N_LC_PIX,), target LOS redshifts
    z0             : float, reference redshift for the LOS pixel index
                     (use z_arr.min(), matching where compute_ksz_map's
                     red_axis starts)
    cell_size      : float, comoving Mpc per cell
    ngrid          : int
    angle_deg      : float, transverse rotation (default 0). Ignored
                     entirely if wrap_cycle_seed is not None.
    wrap_cycle_seed : int or None, default None.
                     None: single angle_deg, nearest-neighbor rotation,
                     applied identically at every LOS position.
                     int: a DIFFERENT, deterministic rotation angle per
                     periodic wrap-cycle (see wrap_cycle_index/
                     cycle_angle above), with bilinear sampling.
    wrap_mode      : 'wrap' (default) or 'grid-wrap'. Only used when
                     wrap_cycle_seed is not None.

    Returns
    -------
    lc : ndarray (ngrid, ngrid, len(z_arr)), float32
    """
    if wrap_cycle_seed is None:
        # Original code path, byte-for-byte unchanged.
        ir, jr = rotated_indices(ngrid, angle_deg)
        lc = np.empty((ngrid, ngrid, len(z_arr)), dtype=np.float32)

        for n, z in enumerate(z_arr):
            y_cell = comoving_pixel(z, z0, cell_size, ngrid)
            slabs = np.stack([get_slab(snapshot_boxes[sz], y_cell, ir, jr)
                               for sz in snap_z], axis=-1)
            interp = interp1d(snap_z, slabs, axis=-1, bounds_error=False,
                               fill_value="extrapolate")
            lc[:, :, n] = interp(z)
        return lc

    # Per-wrap-cycle rotation, bilinear sampling (see helpers above).
    lc = np.empty((ngrid, ngrid, len(z_arr)), dtype=np.float32)
    coords_by_cycle = {}
    for n, z in enumerate(z_arr):
        y_cell = comoving_pixel(z, z0, cell_size, ngrid)
        cycle = wrap_cycle_index(z, z0, cell_size, ngrid)
        if cycle not in coords_by_cycle:
            angle = cycle_angle(cycle, wrap_cycle_seed)
            coords_by_cycle[cycle] = rotated_coords(ngrid, angle)
        ic, jc = coords_by_cycle[cycle]
        slabs = np.stack([get_slab_bilinear(snapshot_boxes[sz], y_cell, ic, jc,
                                               wrap_mode)
                           for sz in snap_z], axis=-1)
        interp = interp1d(snap_z, slabs, axis=-1, bounds_error=False,
                           fill_value="extrapolate")
        lc[:, :, n] = interp(z)
    return lc


def stitch_lightcone_from_coeval(z_snapshots, z_arr, HII_DIM, BOX_LEN,
                                  cache_dir, angle_deg=0.0, N_THREADS=None,
                                  random_seed=None, wrap_cycle_seed=None,
                                  astro_params=None, flag_options=None,
                                  wrap_mode='wrap'):
    """
    Build a full (density, xH, velocity_z, brightness_temp) lightcone by
    running/loading coeval boxes at z_snapshots (via the shared, validated
    run_coeval_fields) and stitching them onto the continuous LOS grid
    z_arr.

    EXTENDED from upstream ksz-pipeline: brightness_temp is now a fourth
    stitched field, sharing the exact same angle_deg/wrap_cycle_seed/
    wrap_mode as density/xH/velocity_z -- see module docstring for why
    that sharing is load-bearing for the kSZ-21cm cross-correlation.

    Parameters
    ----------
    z_snapshots : sequence of float, coeval snapshot redshifts to compute
                  (should bracket z_arr's range with some margin, since
                  stitch_field extrapolates flatly outside the snapshot
                  range and that's not something to rely on)
    z_arr       : ndarray, target LOS redshifts (ascending)
    HII_DIM     : int
    BOX_LEN     : float, comoving Mpc
    cache_dir   : str, py21cmfast cache directory (reused across snapshots)
    angle_deg   : float, see module docstring -- default 0 recommended
                  for a first cross-check. Ignored if wrap_cycle_seed
                  is not None (see stitch_field).
    N_THREADS   : int, optional -- passed through to run_coeval_fields.
    wrap_cycle_seed : int or None, default None -- see stitch_field.
                  Passed identically to ALL FOUR field calls below so
                  density/xH/velocity/brightness_temp share the SAME
                  rotation at every LOS position; using different seeds
                  per field would decorrelate kSZ from 21cm at exactly
                  the periodic-wrap locations the fix is meant to handle.
    wrap_mode   : 'wrap' (default) or 'grid-wrap' -- see stitch_field /
                  get_slab_bilinear. Only matters when wrap_cycle_seed is
                  set.
    astro_params : dict, optional -- straight passthrough to
                  simulate/coeval_fields.py's run_coeval_fields() at every
                  snapshot. Also controls brightness_temp's T_S physics
                  via flag_options.USE_TS_FLUCT -- see coeval_fields.py's
                  docstring.
    flag_options : dict, optional -- straight passthrough to
                  run_coeval_fields() at every snapshot.

    Returns
    -------
    dict with:
      density         : ndarray (HII_DIM,HII_DIM,len(z_arr))  RAW delta,
                         caller must add 1.0 before compute_ksz_map -- see
                         module docstring
      xH_box          : ndarray, neutral fraction
      velocity_z      : ndarray, physical LOS peculiar velocity [cm/s]
      brightness_temp : ndarray, 21cmFAST's own delta_T [mK] -- no "+1"
                         or other transform needed, used as-is for the
                         21cm side of the cross-correlation
      z_arr           : ndarray, the input z_arr (echoed back for convenience)
      pos_axis        : ndarray, comoving distance [Mpc] at each z_arr slice
    """
    cell_size = float(BOX_LEN) / int(HII_DIM)
    z0 = float(np.min(z_arr))

    from .coeval_fields import run_coeval_fields
    snap_z = np.array(sorted(z_snapshots), dtype=float)
    delta_boxes, xH_boxes, vz_boxes, dTb_boxes = {}, {}, {}, {}
    for z in snap_z:
        delta, xH, _vx, _vy, vz, dTb = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=N_THREADS,
            random_seed=random_seed, astro_params=astro_params,
            flag_options=flag_options)
        delta_boxes[z] = delta
        xH_boxes[z]    = xH
        vz_boxes[z]    = vz
        dTb_boxes[z]   = dTb

    density    = stitch_field(delta_boxes, snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed, wrap_mode)
    xH_box     = stitch_field(xH_boxes,    snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed, wrap_mode)
    velocity_z = stitch_field(vz_boxes,    snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed, wrap_mode)
    brightness_temp = stitch_field(dTb_boxes, snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed, wrap_mode)

    pos_axis = np.array([comoving_distance_mpc(z) for z in z_arr])

    return dict(density=density, xH_box=xH_box, velocity_z=velocity_z,
                brightness_temp=brightness_temp,
                z_arr=np.asarray(z_arr), pos_axis=pos_axis)
