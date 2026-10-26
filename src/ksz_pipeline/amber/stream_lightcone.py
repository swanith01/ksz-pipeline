"""
ksz_pipeline/amber/stream_lightcone.py

Memory-light equivalent of

    stitch_lightcone_from_amber(..., interp=True, wrap_cycle_seed=seed)
    -> amber_lightcone_to_ksz_Dl(...)'s compute_ksz_map step

for AMBER runs too large to hold in RAM (e.g. 1536^3: one snapshot's
(delta, xH, v_z) is ~43 GB, 28 snapshots ~1.2 TB, before the lightcone
itself). Instead of loading every snapshot and materialising the 3D
lightcone, this walks the LOS one slice at a time:

  for each LOS pixel n (redshift z_n, comoving position pos_n):
    y_cell  = round((pos_n - pos_0)/cell) % N           (comoving_pixel)
    cycle   = (pos_n - pos_0) // (N*cell)               (wrap_cycle_index)
    read ONLY slab [:, :, y_cell] of the two snapshots bracketing z_n
      (np.memmap -- in AMBER's Fortran-order dumps that slab of rho/zre
       is contiguous on disk, mom_z is a 3N^2 contiguous block)
    linear-in-z blend of (delta, v_z, x_HI) between the two snapshots
    rotate by cycle_angle(cycle, seed), bilinear (map_coordinates)
    add this slice's trapezoid term to the kSZ map

Exactly the same operations as the in-memory path, in a different
order: stitch_field rotates each snapshot's slab and then interpolates
in z (scipy interp1d, linear, extrapolating outside the snapshot range);
here the z-blend comes first and the rotation second. Both are linear
operators on the slab, so they commute -- results agree to float
rounding (float32 vs float64 storage). Verified against the in-memory
pipeline in tests/test_stream_lightcone.py.

Two maps are accumulated in the same pass:
  'patchy' : signal clipped to slices with x_e in [xe_min, xe_max]
             (= amber_lightcone_to_ksz_Dl, compute_patchy_mask)
  'total'  : no clipping -- every slice in z_arr, so with z_arr starting
             at z=5 this includes the post-reionization (homogeneous)
             kSZ, i.e. Chen+23's z=5-30 convention.
Visibility uses the full, unclipped tau(z) in both, as before.
"""
import time

import numpy as np
from scipy.ndimage import map_coordinates

from ..ksz.stitch_from_coeval import cycle_angle, rotated_coords
from ..utils.constants import SIGMA_T, MPC_CM, C_CGS
from .adapter import KMS_TO_CMS

_DT = np.dtype('<f4')


_Z_VIEW = (2, (0, 0, 0))     # LOS = box z axis, no shift (original geometry)


def reorient_params(cycle_idx, seed, N):
    """(los_axis, (sx, sy, sz)) for one wrap cycle: which box axis is the
    line of sight and an integer periodic shift on every axis. Deterministic
    in (seed, cycle_idx), like cycle_angle, but a separate random stream."""
    rng = np.random.default_rng((int(seed), int(cycle_idx), 1))
    return int(rng.integers(3)), tuple(int(v) for v in rng.integers(0, N, size=3))


def view_slab(arr, y, view=_Z_VIEW):
    """2D slab of a 3D box at LOS index y under `view` = (axis, shift):
    the slab perpendicular to `axis` at (y + shift[axis]) % N, with the two
    remaining axes (in increasing order) periodically shifted. The default
    view is exactly arr[:, :, y]. Exact -- integer indexing, no interpolation."""
    axis, shift = view
    s = np.take(arr, (y + shift[axis]) % arr.shape[axis], axis=axis)
    o = [k for k in range(3) if k != axis]
    if shift[o[0]] or shift[o[1]]:
        s = np.roll(s, (-shift[o[0]], -shift[o[1]]), axis=(0, 1))
    return s


class _Snapshot:
    """Lazy (memmapped) view of one AMBER fields_*.dat dump."""

    def __init__(self, path, N):
        n3 = N**3
        mm = np.memmap(path, dtype=_DT, mode='r', shape=(1 + 4 * n3,))
        self.z = float(mm[0])
        self.rho = mm[1:1 + n3].reshape((N, N, N), order='F')
        self.mom = mm[1 + n3:].reshape((3, N, N, N), order='F')

    def slabs(self, y, zre_slab, view=_Z_VIEW):
        """(delta, v_los [cm/s], x_HI) float64 slabs at LOS index y, with
        the SAME conventions as stitch_from_amber._load_amber_snapshots.
        v_los is the momentum component along view's LOS axis."""
        rho = np.asarray(view_slab(self.rho, y, view), dtype=np.float32)
        mz = np.asarray(view_slab(self.mom[view[0]], y, view), dtype=np.float32)
        vz = (np.where(rho > 1e-6, mz / np.where(rho > 1e-6, rho, 1.0), 0.0)
              .astype(np.float32) * KMS_TO_CMS)
        xH = (zre_slab <= self.z).astype(np.float32)
        return (rho.astype(np.float64) - 1.0, vz.astype(np.float64),
                xH.astype(np.float64))

    def q_slab(self, y, zre_slab, view=_Z_VIEW):
        """(q_los = x_HII * mom_los [cm/s], x_HI) float64 slabs -- the SAME
        cell-wise q compute_cell's P_qperp is built from
        (qperp_power_from_momentum), no division by rho."""
        mz = np.asarray(view_slab(self.mom[view[0]], y, view),
                        dtype=np.float64) * KMS_TO_CMS
        xH = (zre_slab <= self.z).astype(np.float64)
        return (1.0 - xH) * mz, xH

    def rho_slab(self, y, view=_Z_VIEW):
        return np.asarray(view_slab(self.rho, y, view), dtype=np.float64)


def open_zre(path, N):
    return np.memmap(path, dtype=_DT, mode='r', shape=(N, N, N), order='F')


def stream_ksz_maps(field_files, zre, N, box_len_mpc, z_arr, pos_axis,
                    tau_table, ne0, wrap_cycle_seed, wrap_mode='wrap',
                    xe_min=1.0e-4, xe_max=1.0 - 1.0e-4, progress_every=0,
                    mode='velocity', geometry='rotate'):
    """
    mode : 'velocity' (default) -- exactly the in-memory stitched path:
           delta, v_z = mom_z/rho and x_HI each blended in z and rotated
           separately, integrand = (1+delta) x_HII v_z.
           'momentum' -- q_z = x_HII * mom_z built cell-wise per
           snapshot (as compute_cell's P_qperp is), then blended and
           rotated; integrand = q_z. No division by rho, so near-empty
           cells (AMBER's rho2 is interlaced+deconvolved and can be ~0)
           can't produce huge v that interpolation then pairs with a
           neighbour's normal rho. In this mode every progress line also
           reports, for that slice, rms of the velocity-path integrand
           over the momentum-path one and the slab's rho statistics.

    geometry : how successive passes through the periodic box (wrap
           cycles) are decorrelated, when wrap_cycle_seed is not None:
           'rotate' (default) -- random in-plane rotation per cycle with
           bilinear sampling (= stitch_field). Interpolation smooths:
           measured 0.80 / 0.59 of the unrotated power at ell = 5000 / 8000
           on the 2 Gpc/h, 1536^3 run.
           'reorient' -- per cycle, a random box axis becomes the LOS
           (momentum component along that axis) plus a random integer
           periodic shift on every axis (reorient_params). Exact, no
           interpolation; breaks the repeat because each pass samples the
           box along a different direction and offset.
           Ignored when wrap_cycle_seed is None (no transformation).

    Parameters
    ----------
    field_files : fields_*.dat paths (any order; sorted by z here)
    zre         : (N,N,N) array or memmap (open_zre)
    N           : mesh cells per side
    box_len_mpc : box side [Mpc] (L_h / h)
    z_arr       : (n,) ascending LOS redshifts, uniform in comoving
                  distance with step box_len_mpc/N
    pos_axis    : (n,) comoving distance [Mpc] at z_arr -- from the SAME
                  cosmology that produced z_arr. Used for LOS pixel
                  indexing, wrap cycles, ds and chi_eff, exactly where the
                  in-memory path uses comoving_distance_mpc().
    tau_table   : AMBER tau.txt array (col 0 z, col 2 tau(0->z))
    ne0         : [cm^-3], meta['ne0_cgs']
    wrap_cycle_seed : int (per-wrap-cycle rotation, bilinear, as
                  stitch_field) or None: NO rotation, slabs used exactly
                  as stored (= stitch_field's wrap_cycle_seed=None path
                  with angle_deg=0) -- no interpolation, so no smoothing,
                  but periodic repeats are not broken. Appropriate when
                  the LOS spans <= 1 box length.
    wrap_mode   : 'wrap' (default, as stitch_field) or 'grid-wrap'
    progress_every : print a progress line every this many slices (0=off)

    Returns
    -------
    dict with map_total, map_patchy (N,N) dimensionless dT/T;
    chi_eff_total, chi_eff_patchy [Mpc]; xe_slices (n,); z_arr; pos_axis;
    z_window_patchy (zlo, zhi) or None
    """
    if mode not in ('velocity', 'momentum'):
        raise ValueError(f"mode must be 'velocity' or 'momentum', got {mode!r}")
    if geometry not in ('rotate', 'reorient'):
        raise ValueError(f"geometry must be 'rotate' or 'reorient', got {geometry!r}")
    reorient = geometry == 'reorient' and wrap_cycle_seed is not None
    rotate = geometry == 'rotate' and wrap_cycle_seed is not None
    z_arr = np.asarray(z_arr, dtype=float)
    pos_axis = np.asarray(pos_axis, dtype=float)
    cell = box_len_mpc / N
    snaps = sorted((_Snapshot(f, N) for f in field_files), key=lambda s: s.z)
    sz = np.array([s.z for s in snaps])
    if sz.size < 2:
        raise ValueError("need at least two snapshots")

    tau_table = np.asarray(tau_table, dtype=float)
    if not (tau_table[:, 0].min() <= z_arr.min()
            and z_arr.max() <= tau_table[:, 0].max()):
        raise ValueError("LOS grid falls outside tau_table's z range")
    vis = np.exp(-np.interp(z_arr, tau_table[:, 0], tau_table[:, 2]))

    c_Mpc_s = C_CGS / MPC_CM
    prefactor = ne0 * SIGMA_T * C_CGS
    a2 = (1.0 / (1.0 + z_arr))**2

    map_tot = np.zeros((N, N))
    map_pat = np.zeros((N, N))
    xe_slices = np.empty(z_arr.size)
    rms_tot = np.empty(z_arr.size)
    rms_pat = np.empty(z_arr.size)
    coords = {}
    prev = None                         # (integrand, mask) of slice n-1
    t0 = time.time()

    for n, z in enumerate(z_arr):
        d = pos_axis[n] - pos_axis[0]
        y = int(round(d / cell)) % N
        cyc = int(d // (cell * N))
        if cyc not in coords:
            if rotate:
                coords[cyc] = rotated_coords(N, cycle_angle(cyc, wrap_cycle_seed))
            elif reorient:
                coords[cyc] = reorient_params(cyc, wrap_cycle_seed, N)
            else:
                coords[cyc] = None
        ic, jc = coords[cyc] if rotate else (None, None)
        view = coords[cyc] if reorient else _Z_VIEW

        i0 = int(np.clip(np.searchsorted(sz, z) - 1, 0, sz.size - 2))
        w1 = (z - sz[i0]) / (sz[i0 + 1] - sz[i0])      # extrapolates outside
        w0 = 1.0 - w1
        zre_slab = np.asarray(view_slab(zre, y, view), dtype=np.float32)
        report = progress_every and (n + 1) % progress_every == 0
        diag = ''

        def _rot(f):
            if not rotate:
                return f
            return map_coordinates(f, [ic, jc], order=1, mode=wrap_mode)

        def _vel_integrand():
            a0 = snaps[i0].slabs(y, zre_slab, view)
            a1 = snaps[i0 + 1].slabs(y, zre_slab, view)
            delta, vz, xH = (_rot(w0 * f0 + w1 * f1) for f0, f1 in zip(a0, a1))
            xHII = 1.0 - xH
            return (1.0 + delta) * xHII * (vz / MPC_CM) / c_Mpc_s * vis[n], xHII

        if mode == 'velocity':
            integ, xHII = _vel_integrand()
            xe = float(xHII.mean())
        else:
            q0, h0 = snaps[i0].q_slab(y, zre_slab, view)
            q1, h1 = snaps[i0 + 1].q_slab(y, zre_slab, view)
            integ = _rot(w0 * q0 + w1 * q1) / MPC_CM / c_Mpc_s * vis[n]
            xe = float(1.0 - (w0 * h0 + w1 * h1).mean())
            if report:
                iv, _ = _vel_integrand()
                rq = float(np.sqrt(np.mean(integ**2)))
                rv = float(np.sqrt(np.mean(iv**2)))
                rho = snaps[i0].rho_slab(y, view)
                diag = (f"  | rms vel-path/mom-path {rv / rq if rq else np.nan:.3f}"
                        f"  rho min {rho.min():.3g}, frac rho<0.05 "
                        f"{(rho < 0.05).mean():.2e}")
        m = 1.0 if xe_min <= xe <= xe_max else 0.0
        xe_slices[n] = xe
        r = float(np.sqrt(np.mean(integ**2)))
        rms_tot[n], rms_pat[n] = r, r * m

        if prev is not None:
            pint, pm = prev
            ds_cm = (pos_axis[n] - pos_axis[n - 1]) * MPC_CM
            w = prefactor / (0.5 * (a2[n - 1] + a2[n])) * ds_cm / C_CGS
            map_tot += w * 0.5 * (pint + integ)
            if pm or m:
                map_pat += w * 0.5 * (pint * pm + integ * m)
        prev = (integ, m)

        if report:
            el = time.time() - t0
            print(f"  slice {n + 1}/{z_arr.size}  z={z:.3f}  x_e={xe:.4f}  "
                  f"{el:.0f}s elapsed, ~{el / (n + 1) * (z_arr.size - n - 1):.0f}s left"
                  + diag, flush=True)

    def _chi(rms):
        return float(np.sum(rms * pos_axis) / np.sum(rms)) if rms.sum() > 0 else np.nan

    win = z_arr[(xe_slices >= xe_min) & (xe_slices <= xe_max)]
    return dict(map_total=map_tot, map_patchy=map_pat,
                chi_eff_total=_chi(rms_tot), chi_eff_patchy=_chi(rms_pat),
                xe_slices=xe_slices, z_arr=z_arr, pos_axis=pos_axis,
                z_window_patchy=(float(win.min()), float(win.max()))
                if win.size else None)
