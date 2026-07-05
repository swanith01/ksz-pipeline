"""
Skewed line-of-sight ray tracing for the lightcone kSZ calculation.

Extracted from Cell 2.5 of 16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.

To reduce artefacts from periodic replication of the simulation volume,
the line of sight is tilted by angle_deg and periodically wrapped across
box boundaries. This ensures successive lightcone slices sample different
regions of the simulation.

Key references: Semester-6 notes sec 1.2, Figure 3.
"""

import numpy as np


def _periodic(n, ngrid):
    return int(round(float(n))) % int(ngrid)


def rotated_skewer(field, x_start, y_idx, s_axis, delta_z, cell_size,
                   Ndim, Nbins, cos_a, sin_a):
    """
    Extract one rotated LOS skewer with linear z-interpolation.

    Parameters
    ----------
    field     : ndarray (Ndim, Ndim, Nbins_box)  3D field to sample
    x_start   : int    starting x-cell index
    y_idx     : int    fixed y-cell index
    s_axis    : ndarray (Nbins,)  relative comoving distance starting at 0
    delta_z   : float  comoving slice thickness [Mpc]
    cell_size : float  cell size [Mpc]
    Ndim      : int    grid dimension
    Nbins     : int    number of output LOS bins
    cos_a     : float  cos(angle_rad)
    sin_a     : float  sin(angle_rad)

    Returns
    -------
    skewer : ndarray (Nbins,) float32
    """
    Nbins_box = field.shape[2]
    skewer    = np.empty(Nbins, dtype=np.float32)

    for i in range(Nbins):
        s      = s_axis[i]
        z_cont = s * cos_a / delta_z
        x_cont = float(x_start) + s * sin_a / cell_size

        z0 = int(np.floor(z_cont))
        z1 = z0 + 1
        fz = z_cont - z0

        z0 = min(max(z0, 0), Nbins_box - 1)
        z1 = min(max(z1, 0), Nbins_box - 1)
        x  = _periodic(x_cont, Ndim)

        skewer[i] = (field[x, y_idx, z0] * (1 - fz) +
                     field[x, y_idx, z1] * fz)
    return skewer


def extract_skewers(Delta_3d, xHI_3d, vel_3d, LOS_ind, s_axis, delta_z,
                    cell_size, Ndim, Nbins, cos_a, sin_a):
    """
    Extract unrotated and rotated skewer arrays for all LOS positions.

    Parameters
    ----------
    Delta_3d : ndarray (Ndim, Ndim, Nbins)  density field 1+delta
    xHI_3d   : ndarray (Ndim, Ndim, Nbins)  neutral fraction
    vel_3d   : ndarray (Ndim, Ndim, Nbins)  velocity / H0 [Mpc]
    LOS_ind  : ndarray (Nlos, 2)            (x, y) grid positions
    s_axis   : ndarray (Nbins,)             relative comoving distances
    delta_z  : float                         slice thickness [Mpc]
    cell_size: float                         cell size [Mpc]
    Ndim     : int                           grid dimension
    Nbins    : int                           number of LOS bins
    cos_a    : float                         cos(angle_rad)
    sin_a    : float                         sin(angle_rad)

    Returns
    -------
    lightcone_unrot : dict  keys: density, xH_box, velocity (Nlos, Nbins)
    lightcone_rot   : dict  keys: density, xH_box, velocity (Nlos, Nbins)
    """
    Nlos = len(LOS_ind)

    density_unrot  = np.zeros((Nlos, Nbins), dtype=np.float32)
    xH_box_unrot   = np.zeros((Nlos, Nbins), dtype=np.float32)
    velocity_unrot = np.zeros((Nlos, Nbins), dtype=np.float32)
    density_rot    = np.zeros((Nlos, Nbins), dtype=np.float32)
    xH_box_rot     = np.zeros((Nlos, Nbins), dtype=np.float32)
    velocity_rot   = np.zeros((Nlos, Nbins), dtype=np.float32)

    for k, (x0, y0) in enumerate(LOS_ind):
        ix, iy = int(x0) % Ndim, int(y0) % Ndim

        # Unrotated — direct z-axis extraction
        density_unrot[k]  = Delta_3d[ix, iy, :]
        xH_box_unrot[k]   = xHI_3d[ix,  iy, :]
        velocity_unrot[k] = vel_3d[ix,   iy, :]

        # Rotated — diagonal interpolated extraction
        density_rot[k]  = rotated_skewer(Delta_3d, int(x0), int(y0),
                                          s_axis, delta_z, cell_size,
                                          Ndim, Nbins, cos_a, sin_a)
        xH_box_rot[k]   = rotated_skewer(xHI_3d,   int(x0), int(y0),
                                          s_axis, delta_z, cell_size,
                                          Ndim, Nbins, cos_a, sin_a)
        velocity_rot[k] = rotated_skewer(vel_3d,    int(x0), int(y0),
                                          s_axis, delta_z, cell_size,
                                          Ndim, Nbins, cos_a, sin_a)

    lightcone_unrot = dict(density=density_unrot, xH_box=xH_box_unrot,
                           velocity=velocity_unrot)
    lightcone_rot   = dict(density=density_rot,   xH_box=xH_box_rot,
                           velocity=velocity_rot)
    return lightcone_unrot, lightcone_rot


def make_los_grid(Nlos, Ndim):
    """
    Build (x, y) grid positions for Nlos skewers over the box face.

    Mirrors Cell 2.5 exactly: full grid if Nlos == Ndim^2,
    otherwise ceil(sqrt) spacing.

    Returns
    -------
    LOS_ind : ndarray (Nlos, 2)
    """
    Nlos_max = Ndim * Ndim
    if Nlos >= Nlos_max:
        LOS_ind = np.array([[i, j]
                             for i in range(Ndim)
                             for j in range(Ndim)])
    else:
        Nlos_perrow = int(np.ceil(np.sqrt(Nlos)))
        ind_step    = int(np.ceil(Ndim / Nlos_perrow))
        LOS_ind = np.array([
            [int((i + 0.5) * ind_step) % Ndim,
             int((j + 0.5) * ind_step) % Ndim]
            for i in range(Nlos_perrow)
            for j in range(Nlos_perrow)
        ])[:Nlos]
    return LOS_ind


class RotatedLightcone:
    """
    Thin wrapper that patches a py21cmfast LightCone object with
    rotated field arrays, so downstream cells see rotated fields
    without any code changes.

    Mirrors the _RotatedLightcone class in Cell 2.5.
    """
    def __init__(self, original, density_r, xH_r, vel_r):
        self._orig     = original
        self._density  = density_r[np.newaxis, :, :]
        self._xH_box   = xH_r[np.newaxis, :, :]
        self._velocity = vel_r[np.newaxis, :, :]
        self.lightcone_redshifts  = original.lightcone_redshifts
        self.lightcone_distances  = original.lightcone_distances
        self.lightcone_dimensions = original.lightcone_dimensions

    def __getattr__(self, name):
        return getattr(self._orig, name)

    @property
    def density(self):   return self._density
    @property
    def xH_box(self):    return self._xH_box
    @property
    def velocity(self):  return self._velocity
