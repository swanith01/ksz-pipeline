"""
ksz_pipeline/amber/stitch_from_amber.py

AMBER-sourced sibling of ksz.stitch_from_coeval.stitch_lightcone_from_coeval.
Reuses stitch_from_coeval's geometry functions VERBATIM (build_los_z_grid,
comoving_pixel, rotated_indices, stitch_field) -- these have no 21cmFAST
dependency at all, only stitch_lightcone_from_coeval() does (via
run_coeval_fields). This module supplies AMBER snapshots in their place,
so the two lightcones differ ONLY in which simulation made the fields,
not in how they're stitched -- which is the point of the comparison.

Density convention: matches stitch_from_coeval's documented contract --
returns RAW delta (mean ~0), NOT (1+delta). Callers must add +1 exactly
once before compute_ksz_map, same as the 21cmFAST path.

Velocity convention: v_z = mom2_z / (1+delta), in cm/s (adapter.KMS_TO_CMS
already applied), i.e. the physical peculiar velocity's z-component --
the same quantity stitch_from_coeval's velocity_z holds. AMBER's box axes
are the LOS axes here: no separate line-of-sight choice is made, the
stitching is along the mesh's own z axis, exactly as the 21cmFAST path
stitches along its z axis (see stitch_from_coeval's comoving_pixel/
get_slab: y_cell always indexes the LAST array axis).

Ionization: two variants, opt-in, never silently swapped for one another:

  interp=True  (default): x_HI stitched the SAME way density/velocity
      are -- per-snapshot binary field, interpolated across snapshots in
      z by stitch_field. Directly comparable to the 21cmFAST stitched
      path, same interpolation error included.

  interp=False: EXACT per-pixel ionization from z < zre(cell), no
      snapshot interpolation at all -- possible only because AMBER
      supplies one static zre(x) field rather than a handful of x_HI
      snapshots. This is a genuinely different method, not a more
      accurate version of the same one: it removes an approximation the
      21cmFAST path is stuck with, which was flagged as a plausible
      P_off contributor. Report both, don't average them.
"""
import numpy as np

from ..ksz.stitch_from_coeval import (
    build_los_z_grid, comoving_distance_mpc, comoving_pixel,
    rotated_indices, get_slab, stitch_field,
)
from . import io
from .adapter import KMS_TO_CMS


def _load_amber_snapshots(field_files, zre):
    """{z: (delta, xH, v_z)} for every fields_*.dat file, in repo units."""
    snaps = {}
    for f in field_files:
        z, rho, mom = io.read_fields(f, zre.shape[0])
        rho_mean = float(rho.mean())
        if abs(rho_mean - 1.0) > 1e-2:
            raise RuntimeError(f"{f}: <rho2>={rho_mean:.4f}, expected ~1")
        delta = rho - 1.0
        xH = (zre <= z).astype(np.float32)          # AMBER's own criterion
        vz = np.where(rho > 1e-6, mom[2] / np.where(rho > 1e-6, rho, 1.0),
                     0.0).astype(np.float32) * KMS_TO_CMS
        snaps[float(z)] = (delta, xH, vz)
    return snaps


def stitch_lightcone_from_amber(field_files, zre, z_arr, N, L_box_mpc_h, h,
                                angle_deg=0.0, interp=True):
    """
    Build (density, xH, velocity_z) on the LOS grid z_arr from AMBER
    snapshots, using stitch_from_coeval's exact stitching geometry.

    Parameters
    ----------
    field_files  : list of fields_*.dat paths (io.list_field_files),
                   sorted ascending in z -- SHOULD BRACKET z_arr's range
                   with margin; stitch_field extrapolates flatly outside
                   it (see stitch_from_coeval docstring), same caveat as
                   the 21cmFAST path.
    zre          : (N,N,N) reionization-redshift field (io.read_zre)
    z_arr        : ndarray, target LOS redshifts (ascending) -- e.g. from
                   build_los_z_grid(z_min, z_max, L_box_mpc_h/h/N)
    N            : mesh cells per side
    L_box_mpc_h  : AMBER box length [Mpc/h]
    h            : Hubble parameter used in the AMBER run
    angle_deg    : transverse rotation, passed through to stitch_field
                   (default 0, matching stitch_from_coeval's own default
                   and its stated reason: isolate one variable at a time)
    interp       : True (default) = x_HI interpolated across snapshots,
                   directly comparable to the 21cmFAST stitched path.
                   False = exact z<zre per LOS pixel, no interpolation.
                   See module docstring -- report both, don't blend.

    Returns
    -------
    dict with density (RAW delta, caller adds +1), xH_box, velocity_z,
    z_arr, pos_axis -- SAME keys/conventions as
    stitch_from_coeval.stitch_lightcone_from_coeval, so compute_ksz_map
    takes either lightcone unchanged.
    """
    L_mpc = L_box_mpc_h / h
    cell_size = L_mpc / N
    z0 = float(np.min(z_arr))

    snaps = _load_amber_snapshots(field_files, zre)
    snap_z = np.array(sorted(snaps), dtype=float)
    delta_boxes = {z: snaps[z][0] for z in snap_z}
    xH_boxes    = {z: snaps[z][1] for z in snap_z}
    vz_boxes    = {z: snaps[z][2] for z in snap_z}

    density    = stitch_field(delta_boxes, snap_z, z_arr, z0, cell_size,
                               N, angle_deg)
    velocity_z = stitch_field(vz_boxes,    snap_z, z_arr, z0, cell_size,
                               N, angle_deg)

    if interp:
        xH_box = stitch_field(xH_boxes, snap_z, z_arr, z0, cell_size,
                              N, angle_deg)
    else:
        # Exact per-pixel ionization: z_arr[n] < zre at that pixel's
        # rotated transverse position -- no snapshot interpolation.
        ir, jr = rotated_indices(N, angle_deg)
        xH_box = np.empty((N, N, len(z_arr)), dtype=np.float32)
        for n, z in enumerate(z_arr):
            y_cell = comoving_pixel(z, z0, cell_size, N)
            # NEUTRAL iff NOT ionized iff NOT(z < zre) iff z >= zre. Do not
            # write "z < zre" here -- that is the IONIZED condition, and
            # putting it straight into xH (neutral fraction) inverts the
            # whole history. (Caught by test_exact_matches_ionized_mask
            # after this line was first written the wrong way round.)
            xH_box[:, :, n] = (z >= get_slab(zre, y_cell, ir, jr)).astype(
                np.float32)

    pos_axis = np.array([comoving_distance_mpc(z) for z in z_arr])
    return dict(density=density, xH_box=xH_box, velocity_z=velocity_z,
                z_arr=np.asarray(z_arr), pos_axis=pos_axis)
