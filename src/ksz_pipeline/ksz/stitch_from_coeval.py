"""
Build a lightcone by stitching coeval boxes along the line of sight,
completely independent of the native p21c.run_lightcone()-based path
(01_make_ksz_lightcone_maps.py / skewed_los.py).

Purpose: an independent cross-check of the native lightcone's ~100-1000x
D_ell excess, using ONLY already-validated pieces -- coeval box generation
and velocity handling from coeval/fields.py, which has now been confirmed
twice (the quicktest v/c sanity check, and the dz-convergence plot landing
on Reichardt's D_pkSZ at full 800Mpc/128^3). If this stitched lightcone's
D_ell also lands near the coeval number, the bug is isolated to
skewed_los.py / the native run_lightcone() path specifically. If it
reproduces the same excess, the bug is somewhere shared (e.g.
compute_ksz_map or ksz_map_to_Dl), which would be an important, different
conclusion.

Adapted from ksz_lae_xcorr's lightcone/stitch.py (a DIFFERENT project's
kSZ-LAE cross-correlation pipeline -- not this repo's own prior code).
Kept: the geometry (comoving-pixel indexing along the LOS, rotation,
per-redshift interpolation across snapshots). Dropped: all halo/LAE/LBG
tracer code (out of scope for bare kSZ) and, deliberately, its own
velocity conversion, which used an unaudited constant (3.086e19, not
MPC_CM or anything else recognizable in this pipeline) -- replaced with
coeval.fields.run_coeval_fields(). Porting that formula unexamined would
have made this "independent" cross-check depend on a third, unverified
velocity convention, defeating the point of building it.

IMPORTANT density convention, read before using: the density field
returned here is RAW delta (mean ~0), matching run_coeval_fields()'s
convention -- NOT (1+delta). This is deliberately different from the
native lightcone script, where lightcone.density (post-rotation) is
already (1+delta) -- see lightcone_integral.py's diagnostic print and
its git history. Callers must add "+1" themselves, exactly once, before
passing this to compute_ksz_map. Getting this wrong (0 or 2 times instead
of once) is precisely the bug class this pipeline already hit and fixed
once this session -- don't reintroduce it here.

Rotation: angle_deg=0 (no rotation) is the recommended default for the
FIRST cross-check run. The native lightcone's OWN rotation path
(skewed_los.py) already turned out to have an unrelated bug (z-axis
clamped instead of wrapped) that didn't explain its D_ell excess --
starting this independent check unrotated avoids conflating a second,
separate rotation implementation with whatever the real problem is.

UPDATE: a single, fixed angle_deg (whether 0 or nonzero) does not break
a snapshot's own periodic repeat along the LOS -- verified directly: the
same (ir,jr) rotation applied at a y_cell that later wraps back to the
same value pulls a bit-identical slab both times, since nothing about
the sampling differs between the two visits. For that, use the new
wrap_cycle_seed parameter (stitch_field / stitch_lightcone_from_coeval)
instead, which varies the rotation per periodic wrap-cycle (one BOX_LEN
of comoving distance) rather than using one fixed angle throughout, and
uses bilinear rather than nearest-neighbor sampling to avoid the 10-31%
pixel aliasing nearest-neighbor rotation has at generic angles. This is
additive: angle_deg alone (wrap_cycle_seed=None, the default) is
untouched and remains exactly what every prior result in this
investigation used.
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

    dchi/dz is not constant over z=4-20 -- checked directly: for a
    100 Mpc/32^3 box (cell_size=3.125 Mpc), naive np.linspace(z_min,
    z_max, n) gives ds ranging from 1.14 to 9.71 Mpc, an 8.5x spread.
    At low z this UNDER-resolves relative to the box's own cell_size
    (genuinely missing structure between samples); at high z it
    OVER-samples (consecutive z_arr points can round to the same
    integer LOS pixel in comoving_pixel(), since less than one cell's
    worth of comoving distance separates them -- wasted compute, not
    new information). compute_ksz_map's LOS integral is ds-weighted,
    so non-constant, resolution-mismatched ds directly biases the
    result, not just wastes samples.

    Implementation: build chi(z) on a fine auxiliary z grid, then invert
    (interpolate z as a function of chi) onto a uniform chi grid with
    exactly cell_size spacing.

    Parameters
    ----------
    z_min, z_max : float
    cell_size    : float, comoving Mpc -- the LOS step size to target,
                   normally BOX_LEN/HII_DIM so ds matches the box's own
                   resolution
    z_oversample : int, resolution of the auxiliary z grid used only to
                   invert chi(z) -- has nothing to do with the returned
                   grid's size, just needs to resolve chi(z)'s curvature
                   finely enough; 4000 is comfortably oversampled for
                   z=4-20 by a wide margin.

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
    periodically wrapped on BOTH axes (unlike skewed_los.py's
    rotated_skewer, which wraps x but clamps z -- not replicated here;
    see module docstring).

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
# own box: verified directly -- box[ir,jr,y_cell] is bit-identical the
# next time y_cell wraps back to the same value, for ANY fixed (ir,jr),
# since nothing about the sampling changed between the two visits. The
# rotation angle must instead depend on WHICH periodic repeat ("wrap
# cycle") of the box a given LOS position falls into -- one full BOX_LEN
# of comoving distance per cycle, since that's the box's own literal
# repeat scale, not a free parameter to tune.
#
# The angle must be the SAME for every snapshot queried at a given LOS
# position (not per-snapshot): stitch_field already queries every
# snapshot at the same y_cell for a given target z, so keying the angle
# on the LOS position's cycle index (not on which snapshot) automatically
# shares one rotation across all snapshots blended there -- preserving
# their real physical time-correlation while still varying between
# different visits to the same underlying box.
#
# Nearest-neighbor rotation (rotated_indices/get_slab above) was measured
# to alias 10-31% of pixels for generic (non-multiple-of-90) angles --
# collisions where multiple output pixels round to the same source pixel
# while others are never sampled at all -- symmetric in angle-mod-90,
# worst at 45 deg past any multiple of 90, zero exactly at multiples.
# Since per-wrap-cycle rotation needs many distinct, generic angles (not
# just 0/90/180/270), this path uses BILINEAR interpolation instead
# (scipy.ndimage.map_coordinates, order=1, mode='wrap' for periodic
# boundaries) -- the standard fix for this class of artifact, and it
# removes the angle-dependent aliasing entirely rather than picking
# angles that merely minimize it.
# ---------------------------------------------------------------------

def wrap_cycle_index(z, z0, cell_size, ngrid):
    """
    Which periodic wrap-cycle (integer, 0-indexed) this redshift's LOS
    position falls into, relative to z0. One cycle = one full BOX_LEN
    (= cell_size * ngrid) of comoving distance -- the box's own literal
    periodic repeat scale, not a tunable chunk size.
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
    (seed, cycle_idx) always returns the same angle, which is exactly
    what's needed for different snapshots (and different fields --
    density/xH/velocity) queried at the same LOS position to agree.
    """
    rng = np.random.default_rng((int(seed), int(cycle_idx)))
    return float(rng.uniform(0.0, 360.0))


def rotated_coords(ngrid, angle_deg):
    """
    Continuous (non-rounded) rotated transverse coordinate grids, for
    bilinear sampling via get_slab_bilinear -- the fractional-coordinate
    analogue of rotated_indices' integer (ir, jr), deliberately NOT
    rounded or wrapped here (map_coordinates' mode='wrap' handles the
    periodic wrapping during interpolation itself).

    Returns
    -------
    ic, jc : ndarray (ngrid, ngrid) float   continuous rotated coordinates
    """
    a = np.deg2rad(angle_deg)
    i, j = np.meshgrid(np.arange(ngrid), np.arange(ngrid), indexing='ij')
    ic = np.cos(a) * i - np.sin(a) * j
    jc = np.sin(a) * i + np.cos(a) * j
    return ic, jc


def get_slab_bilinear(box, y_cell, ic, jc):
    """
    Bilinearly-interpolated, periodically-wrapped rotated (ngrid, ngrid)
    transverse slab of `box` at LOS index y_cell, at the continuous
    rotated coordinates (ic, jc) from rotated_coords(). order=1 is
    bilinear; mode='wrap' makes the interpolation itself periodic, so a
    coordinate just past the last pixel blends smoothly with the first
    (the continuous analogue of get_slab's integer '% ngrid' wrap).
    """
    return map_coordinates(box[:, :, y_cell], [ic, jc], order=1,
                            mode='wrap')


def stitch_field(snapshot_boxes, snap_z, z_arr, z0, cell_size, ngrid,
                  angle_deg=0.0, wrap_cycle_seed=None):
    """
    Interpolate one field, already loaded per snapshot redshift, onto a
    continuous LOS redshift grid z_arr.

    For each target redshift z_arr[n]: compute the LOS pixel y_cell
    (relative to z0), extract the transverse slab at that SAME y_cell
    from every available snapshot box, then interpolate ACROSS
    SNAPSHOTS (in redshift) to the value at z_arr[n]. This samples a
    different periodic replica of each snapshot at each z_arr[n]
    (reducing periodic-replication artifacts when angle_deg != 0) while
    still capturing the physical time-evolution between snapshots.

    CORRECTION: the parenthetical above does not hold as stated -- a
    single fixed angle_deg does NOT reduce periodic-replication
    artifacts; verified directly (see module docstring's UPDATE note).
    Use wrap_cycle_seed for that instead.

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
    angle_deg      : float, transverse rotation (default 0, see module note).
                     Ignored entirely if wrap_cycle_seed is not None.
    wrap_cycle_seed : int or None, default None.
                     None (default): EXACT original behavior below --
                     single angle_deg, nearest-neighbor rotation, applied
                     identically at every LOS position. This is the path
                     every result in this investigation has used so far
                     (always with angle_deg=0.0); untouched by this
                     parameter's addition.
                     int: use a DIFFERENT, deterministic rotation angle
                     per periodic wrap-cycle instead (see wrap_cycle_index/
                     cycle_angle above), with bilinear (not nearest-
                     neighbor) sampling to avoid the aliasing nearest-
                     neighbor rotation has at generic angles.

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
        slabs = np.stack([get_slab_bilinear(snapshot_boxes[sz], y_cell, ic, jc)
                           for sz in snap_z], axis=-1)
        interp = interp1d(snap_z, slabs, axis=-1, bounds_error=False,
                           fill_value="extrapolate")
        lc[:, :, n] = interp(z)
    return lc


def stitch_lightcone_from_coeval(z_snapshots, z_arr, HII_DIM, BOX_LEN,
                                  cache_dir, angle_deg=0.0, N_THREADS=None,
                                  random_seed=None, wrap_cycle_seed=None):
    """
    Build a full (density, xH, velocity_z) lightcone by running/loading
    coeval boxes at z_snapshots (via the shared, validated
    run_coeval_fields) and stitching them onto the continuous LOS grid
    z_arr.

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
                  for the first cross-check. Ignored if wrap_cycle_seed
                  is not None (see stitch_field).
    N_THREADS   : int, optional -- passed through to run_coeval_fields.
                  Defaults to OMP_NUM_THREADS if unset (see
                  coeval/fields.py). Pass explicitly (e.g. from config's
                  21cmfast.N_THREADS, matching script 01's convention)
                  for anything beyond quick interactive testing.
    wrap_cycle_seed : int or None, default None -- see stitch_field.
                  Passed identically to all three field calls below so
                  density/xH/velocity share the SAME rotation at every
                  LOS position; using different seeds per field would
                  break their physical correspondence (e.g. velocity
                  rotated one way, density another, at the same point).

    Returns
    -------
    dict with:
      density    : ndarray (HII_DIM,HII_DIM,len(z_arr))  RAW delta,
                   caller must add 1.0 before compute_ksz_map -- see
                   module docstring
      xH_box     : ndarray, neutral fraction
      velocity_z : ndarray, physical LOS peculiar velocity [cm/s]
      z_arr      : ndarray, the input z_arr (echoed back for convenience)
      pos_axis   : ndarray, comoving distance [Mpc] at each z_arr slice
    """
    cell_size = float(BOX_LEN) / int(HII_DIM)
    z0 = float(np.min(z_arr))

    from ..coeval.fields import run_coeval_fields
    snap_z = np.array(sorted(z_snapshots), dtype=float)
    delta_boxes, xH_boxes, vz_boxes = {}, {}, {}
    for z in snap_z:
        delta, xH, _vx, _vy, vz = run_coeval_fields(
            z, HII_DIM, BOX_LEN, cache_dir, N_THREADS=N_THREADS,
            random_seed=random_seed)
        delta_boxes[z] = delta
        xH_boxes[z]    = xH
        vz_boxes[z]    = vz

    density    = stitch_field(delta_boxes, snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed)
    xH_box     = stitch_field(xH_boxes,    snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed)
    velocity_z = stitch_field(vz_boxes,    snap_z, z_arr, z0, cell_size,
                               HII_DIM, angle_deg, wrap_cycle_seed)

    pos_axis = np.array([comoving_distance_mpc(z) for z in z_arr])

    return dict(density=density, xH_box=xH_box, velocity_z=velocity_z,
                z_arr=np.asarray(z_arr), pos_axis=pos_axis)
