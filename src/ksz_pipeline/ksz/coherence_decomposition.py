"""
ksz_pipeline/ksz/coherence_decomposition.py

Direct test of the coherent-addition hypothesis, per the advisor's
2026-07-23 suggestion: decompose the stitched map's total power into

    P_total(k) = |sum_i Theta_i(k)|^2   -- current coherent map power
    P_diag(k)  = sum_i |Theta_i(k)|^2   -- sum of individual slice
                                            AUTO-powers, no cross terms
                                            -- structurally IDENTICAL to
                                            what coeval-direct's
                                            incoherent Limber sum computes
    P_off(k)   = P_total(k) - P_diag(k) -- genuine cross-slice
                                            correlation power that Limber
                                            cannot capture by construction

Design choices, and why:
- compute_ksz_map_per_slice() mirrors lightcone_integral.compute_ksz_map's
  EXACT formula, just without the final np.sum(axis=-1) -- guarantees
  P_total reconstructed from the per-slice decomposition below is
  IDENTICAL to what the existing, trusted compute_ksz_map/ksz_map_to_Dl
  pipeline already computes (not a re-derivation that could silently
  drift from it). Verify this with the sanity checks in script 16 before
  trusting anything else here.
- Each slice's own mean is subtracted BEFORE FFTing (theta_i - mean(theta_i)),
  not the total map's mean afterward. Since mean is linear,
  sum_i(theta_i - mean(theta_i)) == sum_i(theta_i) - mean(sum_i theta_i)
  exactly -- so this still reproduces ksz_map_to_Dl's own mean-subtraction
  convention (m = ksz_map - ksz_map.mean()) exactly, while making each
  slice independently mean-zero, which is what the P_diag/P_off
  decomposition needs to be well-defined.
- chi per slice uses the SAME midpoint-z convention as compute_ksz_map's
  own a2_mid (0.5*(z_i+z_{i+1})), via stitch_from_coeval.comoving_distance_mpc,
  for consistency with chi_eff and every other chi computed this session.

MEMORY WARNING: this keeps ALL per-slice 2D arrays in memory
simultaneously (shape Nx, Ny, Nz-1). At full fiducial resolution
(512x512x~2320), that is ~9.9 TB in complex128 -- completely infeasible.
This module is intended for QUICK, LOW-RESOLUTION tests only (small
HII_DIM, e.g. 16-32) until a snapshot-grouped or online/incremental
version is written for a full-resolution cluster run. See script 16.

cross_power_by_dchi() is similarly O(Nz^2) pairs -- fine for a quick
test (Nz ~ 100s), NOT fine at full resolution (Nz ~ 2320, ~2.7M pairs)
without a coarser grouping first.
"""
import numpy as np

from .stitch_from_coeval import comoving_distance_mpc
from ..utils.constants import SIGMA_T, MPC_CM, C_CGS, T_CMB_K, NE0_HYDROGEN_ONLY


def compute_ksz_map_per_slice(density_1plus, x_HII_field, v_los_Mpc_s,
                               red_axis, ds, visibility_3D, ne0=None,
                               patchy_mask_3D=None):
    """
    Same math as lightcone_integral.compute_ksz_map, EXCEPT returns each
    trapezoidal LOS segment's own 2D contribution Theta_i(x,y) separately
    instead of summing over them. sum(theta_slices, axis=-1) reproduces
    compute_ksz_map's output EXACTLY (same formula, just not summed yet)
    -- verify this with an assert in the driver script before trusting
    anything downstream.

    Returns
    -------
    theta_slices : ndarray (Nx, Ny, Nz-1) -- per-segment Theta_i(x,y)
    chi_mid_mpc  : ndarray (Nz-1,) -- comoving distance to each segment's
                   midpoint redshift, same convention as compute_ksz_map's
                   own a2_mid
    """
    if ne0 is None:
        ne0 = NE0_HYDROGEN_ONLY

    c_Mpc_s   = C_CGS / MPC_CM
    prefactor = ne0 * SIGMA_T * C_CGS

    a         = 1.0 / (1.0 + red_axis)
    a_squared = a**2

    integrand_base = density_1plus * x_HII_field * v_los_Mpc_s / c_Mpc_s
    if patchy_mask_3D is not None:
        integrand_base = integrand_base * patchy_mask_3D
    integrand_vis  = integrand_base * visibility_3D

    integ_mid = 0.5 * (integrand_vis[..., :-1] + integrand_vis[..., 1:])
    a2_mid    = 0.5 * (a_squared[:-1] + a_squared[1:])
    ds_cm     = np.asarray(ds, dtype=float) * MPC_CM

    theta_slices = (prefactor / a2_mid) * integ_mid * (ds_cm / C_CGS)

    z_mid = 0.5 * (red_axis[:-1] + red_axis[1:])
    chi_mid_mpc = np.array([comoving_distance_mpc(z) for z in z_mid])

    return theta_slices, chi_mid_mpc


def decompose_p_total_diag_off(theta_slices, box_len_mpc, chi_Mpc, n_kbins=35):
    """
    FFT each slice independently (after subtracting ITS OWN mean -- see
    module docstring), then build P_total/P_diag/P_off. Radial binning
    and the D_ell conversion (ell=k*chi_Mpc, Cl=P/chi_Mpc^2,
    fac=ell(ell+1)/2pi * T_CMB_uK^2) exactly mirror
    lightcone_integral.ksz_map_to_Dl's own conventions, so ell_out/Dl_total
    below should reproduce ksz_map_to_Dl(sum(theta_slices,axis=-1), ...)
    to numerical precision -- treat any mismatch as a bug in THIS
    function, not a physics result. Check this in the driver script.

    Returns
    -------
    ell        : ndarray, valid multipoles
    Dl_total   : ndarray -- should match the existing trusted pipeline
    Dl_diag    : ndarray -- compare this to coeval-direct's D_ell
    Dl_off     : ndarray -- CAN BE NEGATIVE (destructive interference is
                 physically expected here, not an error)
    """
    T_CMB_uK = T_CMB_K * 1e6
    Nx, Ny, Nz = theta_slices.shape
    N = Nx
    pix_Mpc = box_len_mpc / N

    theta_slices_zeroed = theta_slices - theta_slices.mean(axis=(0, 1), keepdims=True)

    theta_hat = np.fft.fftshift(
        np.fft.fft2(theta_slices_zeroed, axes=(0, 1)), axes=(0, 1)
    )  # (Nx, Ny, Nz), complex

    norm = (pix_Mpc / N) ** 2
    total_hat  = theta_hat.sum(axis=-1)
    P_total_2d = norm * np.abs(total_hat) ** 2
    P_diag_2d  = norm * np.sum(np.abs(theta_hat) ** 2, axis=-1)
    P_off_2d   = P_total_2d - P_diag_2d

    dk = 2.0 * np.pi / (N * pix_Mpc)
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    kg = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kg.max() * 0.9), n_kbins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    digit     = np.digitize(kg.ravel(), k_bins)

    def _bin(arr2d):
        out = np.full(len(k_centers), np.nan)
        flat = arr2d.ravel()
        for i in range(len(k_centers)):
            mask = digit == i + 1
            if np.any(mask):
                out[i] = flat[mask].mean()
        return out

    P1d_total = _bin(P_total_2d)
    P1d_diag  = _bin(P_diag_2d)
    P1d_off   = _bin(P_off_2d)

    ell = k_centers * chi_Mpc
    fac = ell * (ell + 1.0) / (2.0 * np.pi) * T_CMB_uK ** 2

    valid = ~np.isnan(P1d_total) & (ell > 10)
    ell_out  = ell[valid]
    Dl_total = (P1d_total[valid] / chi_Mpc ** 2) * fac[valid]
    Dl_diag  = (P1d_diag[valid]  / chi_Mpc ** 2) * fac[valid]
    Dl_off   = (P1d_off[valid]   / chi_Mpc ** 2) * fac[valid]

    return ell_out, Dl_total, Dl_diag, Dl_off


def cross_power_by_dchi(theta_slices, chi_mid_mpc, box_len_mpc, n_dchi_bins=20):
    """
    Pairwise real-space cross-correlation between every pair of slices
    i != j, binned by radial separation |chi_i - chi_j|. By Parseval's
    theorem this equals each pair's k-integrated cross-power contribution
    to P_off -- computed directly in real space here, which is exactly
    equivalent and avoids redoing per-pair FFT bookkeeping.

        cross_ij = 2 * sum_{x,y}(theta_i(x,y) * theta_j(x,y)) * pix_area

    (factor of 2 for the (i,j)+(j,i) terms, equal for a real-valued map)

    COST WARNING: O(Nz^2) pairs -- see module docstring.

    Returns
    -------
    dchi_bin_centers : ndarray
    cross_power_mean  : ndarray -- mean cross_ij in each Delta-chi bin
    cross_power_std   : ndarray
    n_pairs_per_bin   : ndarray (int)
    """
    Nx, Ny, Nz = theta_slices.shape
    pix_area = (box_len_mpc / Nx) * (box_len_mpc / Ny)
    theta_zeroed = theta_slices - theta_slices.mean(axis=(0, 1), keepdims=True)

    pairs_dchi, pairs_cross = [], []
    for i in range(Nz):
        for j in range(i + 1, Nz):
            dchi = abs(chi_mid_mpc[i] - chi_mid_mpc[j])
            cross = 2.0 * np.sum(theta_zeroed[:, :, i] * theta_zeroed[:, :, j]) * pix_area
            pairs_dchi.append(dchi)
            pairs_cross.append(cross)

    pairs_dchi  = np.array(pairs_dchi)
    pairs_cross = np.array(pairs_cross)

    bins    = np.linspace(0, pairs_dchi.max(), n_dchi_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_out = np.full(n_dchi_bins, np.nan)
    std_out  = np.full(n_dchi_bins, np.nan)
    n_out    = np.zeros(n_dchi_bins, dtype=int)

    digit = np.digitize(pairs_dchi, bins)
    for i in range(n_dchi_bins):
        mask = digit == i + 1
        n_out[i] = mask.sum()
        if n_out[i] > 0:
            mean_out[i] = pairs_cross[mask].mean()
            std_out[i]  = pairs_cross[mask].std()

    return centers, mean_out, std_out, n_out
