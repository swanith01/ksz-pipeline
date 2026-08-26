"""
ksz_pipeline/utils/angular.py

Mpc <-> arcmin conversion for kSZ maps, using the small-angle
approximation theta ~ physical_distance / chi. Valid whenever the map's
angular extent is well under ~1 radian -- true for every box size used
in this pipeline (e.g. 800 Mpc at chi_eff=8504 Mpc subtends ~0.094 rad,
5.4 deg -- comfortably small-angle). If this is ever used for a genuinely
wide-field map, re-derive with the exact angular-diameter relation
instead of assuming this holds.

Also provides ell_min_for_box(): the largest angular scale (smallest
multipole) a given box size can even measure, set purely by its
fundamental Fourier mode (k_min = 2*pi/L_box). Below this ell, a finite
box has essentially zero independent modes to average over -- any
"excess" or noisy behavior there is expected from sample variance alone,
not necessarily new physics or a numerical bug. Two boxes of different
size have DIFFERENT ell_min, scaling as 1/L_box -- this is the basis of
the low-ell mode-counting check (see
notebooks/exploratory/low_ell_mode_counting_check.py).
"""
import numpy as np


def mpc_to_arcmin(distance_mpc, chi_mpc):
    """
    Convert a transverse comoving distance to an angular size on the
    sky, via the small-angle approximation theta = distance/chi.

    Parameters
    ----------
    distance_mpc : float or ndarray, transverse comoving distance [Mpc]
    chi_mpc      : float, reference comoving distance to the epoch in
                   question (e.g. chi_eff for a patchy-kSZ map)

    Returns
    -------
    float or ndarray, angular size [arcmin]
    """
    theta_rad = np.asarray(distance_mpc) / chi_mpc
    return theta_rad * (180.0 * 60.0 / np.pi)


def arcmin_to_mpc(theta_arcmin, chi_mpc):
    """Inverse of mpc_to_arcmin -- angular size [arcmin] -> transverse
    comoving distance [Mpc] at reference distance chi_mpc."""
    theta_rad = np.asarray(theta_arcmin) * (np.pi / (180.0 * 60.0))
    return theta_rad * chi_mpc


def arcmin_per_pixel(box_len_mpc, hii_dim, chi_mpc):
    """Angular size of one map pixel [arcmin/pixel], for a box of
    box_len_mpc split into hii_dim pixels per side, at reference
    distance chi_mpc."""
    pix_mpc = box_len_mpc / hii_dim
    return mpc_to_arcmin(pix_mpc, chi_mpc)


def ell_min_for_box(chi_mpc, box_len_mpc):
    """
    Smallest reliably-measurable multipole for a box of this size --
    set by the box's fundamental Fourier mode, dk = 2*pi/box_len_mpc,
    converted to ell via ell = k * chi_mpc (same convention as
    lightcone_integral.ksz_map_to_Dl and coherence_decomposition.py).

    Below this ell, the box contains essentially zero independent
    Fourier modes -- any measured power there is dominated by sample
    variance from a handful of modes, not a reliable estimate. Two
    different box sizes have DIFFERENT ell_min (larger box -> smaller
    ell_min -> reliable down to bigger scales).
    """
    dk = 2.0 * np.pi / box_len_mpc
    return dk * chi_mpc


def plot_map_arcmin(ax, ksz_map, box_len_mpc, chi_mpc, cmap='RdBu_r', **imshow_kwargs):
    """
    Convenience plotting helper: imshow a 2D kSZ map with arcmin axes
    instead of pixel/Mpc axes, using this module's own conversion so
    every caller stays consistent rather than each re-deriving the
    extent calculation.

    Parameters
    ----------
    ax          : matplotlib Axes to draw on
    ksz_map     : ndarray (N,N), the map
    box_len_mpc : float, the map's physical side length [Mpc]
    chi_mpc     : float, reference comoving distance (e.g. chi_eff)
    cmap, **imshow_kwargs : passed through to ax.imshow

    Returns
    -------
    the imshow AxesImage, for adding a colorbar etc.
    """
    extent_arcmin = mpc_to_arcmin(box_len_mpc, chi_mpc)
    im = ax.imshow(ksz_map, origin='lower', cmap=cmap,
                    extent=[0, extent_arcmin, 0, extent_arcmin],
                    **imshow_kwargs)
    ax.set_xlabel("x [arcmin]")
    ax.set_ylabel("y [arcmin]")
    return im
