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
from ..ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl
from ..ksz.optical_depth import compute_patchy_mask
from ..utils.constants import MPC_CM
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


def amber_lightcone_to_ksz_Dl(lc, tau_table, box_len_mpc, ne0,
                              xe_min=1.0e-4, xe_max=1.0 - 1.0e-4,
                              n_kbins=35):
    """
    Wire a stitch_lightcone_from_amber() output through
    ksz.lightcone_integral.compute_ksz_map / ksz_map_to_Dl -- the
    AMBER-sourced sibling of ksz.stitch_from_coeval's 21cmFAST path
    (scripts/03_stitched_lightcone_crosscheck.py), with the three
    run-specific pieces resolved AMBER's own way rather than borrowed
    from that path or from a previous run:

      ne0       : NOT defaulted here -- pass this run's own
                  meta['ne0_cgs'] (amber/scripts/make_amber_input.py),
                  the SAME value scripts/30's gate2 already passes to
                  compute_cell for this run, built from its actual
                  XH/YHe/Ob. A cosmology override paired with the
                  wrong ne0 is exactly the silent-mismatch bug class
                  gate2 already had to catch once; ne0_cgs() (Planck18)
                  is the correct fallback ONLY for runs predating that
                  field, which always used Planck18.

      visibility_3D : exp(-tau(z)) built from tau_table, AMBER's own
                  output/cmb/tau.txt (z, x_e, tau(0->z) columns) --
                  interpolated onto the FULL red_axis, unclipped. AMBER
                  integrates tau against the history it actually
                  simulated, so unlike the 21cmFAST path (whose
                  compute_tau()/analytic_tau_below() patch a gap
                  21cmFAST itself doesn't track) there is nothing to
                  patch here.

      chi_Mpc   : ALWAYS recomputed from this lightcone's own signal
                  (RMS-weighted mean comoving distance over the patchy
                  window -- same recipe script 03 uses for the 21cmFAST
                  path), never passed in and never defaulted. Neither
                  ksz_map_to_Dl's stale chi_Mpc=7800 default nor the
                  21cmFAST fiducial run's own chi_eff~8504 Mpc (left
                  untouched precisely because it is that ONE run's
                  number) apply to an AMBER run with a different window
                  -- and the AMBER sweep moves z_mid 7->9, so a fixed
                  chi_Mpc would silently stop matching its own map as
                  the window shifts.

    Parameters
    ----------
    lc          : dict from stitch_lightcone_from_amber -- density (RAW
                  delta), xH_box, velocity_z (cm/s), z_arr, pos_axis
    tau_table   : ndarray (M, >=3), AMBER's output/cmb/tau.txt loaded
                  with np.loadtxt(path, skiprows=1); column 0 = z,
                  column 2 = tau(0->z). Must bracket lc['z_arr'].
    box_len_mpc : float, AMBER box length in Mpc (L_box_mpc_h / h) --
                  NOT Mpc/h, same convention ksz_map_to_Dl expects.
    ne0         : float [cm^-3], see above -- required, no default.
    xe_min, xe_max : patchy-regime bounds passed to
                  optical_depth.compute_patchy_mask (defaults match
                  that function's own: 99.99%-to-0.01%-neutral).
    n_kbins     : passed through to ksz_map_to_Dl.

    Returns
    -------
    dict: ell, Dl, Dl_err, chi_eff, ksz_map, z_window (float (z_lo, z_hi)
    tuple, or None if no slice falls inside the patchy regime -- this
    is a sign the LOS grid doesn't bracket this run's reionization
    history, not a value to silently fall back from).
    """
    density_1plus = 1.0 + lc['density']
    x_HII_field   = 1.0 - lc['xH_box']
    v_los_Mpc_s   = lc['velocity_z'] / MPC_CM
    z_arr, pos_axis = lc['z_arr'], lc['pos_axis']

    tau_table = np.asarray(tau_table, dtype=float)
    if not (tau_table[:, 0].min() <= z_arr.min() and
            z_arr.max() <= tau_table[:, 0].max()):
        raise ValueError(
            f"LOS grid z=[{z_arr.min():.2f},{z_arr.max():.2f}] falls "
            f"outside tau_table's range "
            f"[{tau_table[:,0].min():.2f},{tau_table[:,0].max():.2f}]")
    tau_at_lc = np.interp(z_arr, tau_table[:, 0], tau_table[:, 2])
    visibility_3D = np.exp(-tau_at_lc)[None, None, :]

    x_e_interp = x_HII_field.mean(axis=(0, 1))
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e_interp, xe_min, xe_max)
    z_patchy = z_arr[patchy_mask.astype(bool)]
    z_window = (float(z_patchy.min()), float(z_patchy.max())) \
        if z_patchy.size > 0 else None

    ksz_map = compute_ksz_map(density_1plus, x_HII_field, v_los_Mpc_s,
                              z_arr, np.diff(pos_axis), visibility_3D,
                              ne0=ne0, patchy_mask_3D=patchy_mask_3D)

    # chi_eff: same weighting compute_ksz_map itself integrates (signal
    # x visibility x patchy mask), collapsed per-slice to RMS (not mean
    # -- kSZ is a fluctuation field around zero) and used to weight
    # pos_axis. Computed fresh every call -- see docstring above.
    integrand = density_1plus * x_HII_field * v_los_Mpc_s * visibility_3D
    integrand = integrand * patchy_mask_3D
    slice_rms = np.sqrt(np.mean(integrand**2, axis=(0, 1)))
    chi_eff = float(np.sum(slice_rms * pos_axis) / np.sum(slice_rms))

    ell, Dl, Dl_err = ksz_map_to_Dl(ksz_map, box_len_mpc, chi_Mpc=chi_eff,
                                    n_kbins=n_kbins)
    return dict(ell=ell, Dl=Dl, Dl_err=Dl_err, chi_eff=chi_eff,
                ksz_map=ksz_map, z_window=z_window)
