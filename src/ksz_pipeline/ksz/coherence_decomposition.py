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

MEMORY (CORRECTED 2026-08-14, was wrongly stated as ~9.9 TB in an earlier
version -- that was a units error): at full fiducial resolution
(512x512x~2320), theta_hat (complex128) is ~9.7 GB. Including theta_slices
and its zeroed copy (real float64, ~4.9 GB each) held simultaneously in
decompose_p_total_diag_off, peak memory there is closer to ~20 GB.
compute_ksz_map_per_slice's own inputs (density_1plus, x_HII_field,
v_los_Mpc_s, visibility_3D, patchy_mask_3D, all real float64 at full
res, ~4.9 GB each) push the pipeline's overall peak higher, roughly
30-50 GB across the full run -- still comfortably within the 125-515 GB
node memory seen on this cluster. NOT a hard blocker for a full-resolution
run, contrary to the earlier (wrong) TB estimate.

That said, nothing here is written as an incremental/streaming
accumulator (compute one slice's FFT, add its |.|^2 to a running P_diag
total, discard, move to the next slice) the way the advisor specifically
suggested for P_diag -- everything currently materializes the full
(Nx,Ny,Nz) array at once. Given the corrected memory math above, this is
a deliberate scope choice (simpler code, fits comfortably in memory on
the nodes available), not a necessity -- worth revisiting if this is
ever run on a smaller-memory node, or just to follow the advisor's
suggestion literally.

cross_power_by_dchi() is a SEPARATE cost concern from memory -- it's
O(Nz^2) PAIRS in a pure-Python loop. Fine once fed snapshot-grouped
slices (Nz~15-29, trivial). NOT fine fed raw thin LOS pixels at full
resolution (Nz~2320, ~2.7M pairs -- would be prohibitively slow, not
just memory-heavy). Always call it on group_slices_by_snapshot's output.
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


def group_slices_by_snapshot(theta_slices, chi_mid_mpc, z_snapshots):
    """
    Sum thin LOS-pixel slices into thicker groups, one per z_snapshot,
    so 'diagonal' matches what coeval-direct's own per-snapshot P_qperp
    treats as diagonal (each snapshot's full box depth, not each thin
    LOS pixel). Bucket boundaries = comoving-distance midpoints between
    adjacent sorted z_snapshots.

    FIXED: outer edges now guaranteed to bound chi_mid_mpc's ACTUAL
    range, not just z_snapshots' own range -- z_arr (and therefore
    chi_mid_mpc) intentionally extends beyond z_snapshots with margin
    (see stitch_from_coeval's own docstring), so without this fix, LOS
    pixels in that margin would silently fall outside [lo,hi] and get
    DROPPED from every group, with no warning. Now hard-fails instead
    of silently losing data if this is ever violated.
    """
    z_sorted = sorted(z_snapshots)
    chi_snap = np.array([comoving_distance_mpc(z) for z in z_sorted])
    if len(chi_snap) > 1:
        lo = chi_snap[0] - (chi_snap[1] - chi_snap[0]) / 2
        hi = chi_snap[-1] + (chi_snap[-1] - chi_snap[-2]) / 2
    else:
        lo, hi = chi_snap[0] - 1, chi_snap[0] + 1
    # Guarantee coverage of the ACTUAL LOS pixel range -- see FIXED note.
    lo = min(lo, float(chi_mid_mpc.min()) - 1e-6)
    hi = max(hi, float(chi_mid_mpc.max()) + 1e-6)
    edges = np.sort(np.concatenate(([lo], 0.5 * (chi_snap[:-1] + chi_snap[1:]), [hi])))
    digit = np.digitize(chi_mid_mpc, edges)

    grouped, chi_grouped = [], []
    n_used = 0
    for g in range(1, len(edges)):
        mask = digit == g
        if np.any(mask):
            grouped.append(theta_slices[:, :, mask].sum(axis=-1))
            chi_grouped.append(chi_mid_mpc[mask].mean())
            n_used += int(mask.sum())
    if n_used != theta_slices.shape[-1]:
        raise RuntimeError(
            f"group_slices_by_snapshot: {theta_slices.shape[-1] - n_used} of "
            f"{theta_slices.shape[-1]} LOS pixels were not assigned to any group "
            f"-- bucket edges do not cover the full chi_mid_mpc range. This is a "
            f"bug to fix, not a warning to ignore.")
    return np.stack(grouped, axis=-1), np.array(chi_grouped)


def group_slices_uniform(theta_slices, chi_mid_mpc, n_groups):
    """
    Sum thin LOS-pixel slices into n_groups EQUAL-WIDTH comoving-distance
    bins spanning the full chi_mid_mpc range -- unlike
    group_slices_by_snapshot, NOT tied to actual simulation snapshot
    boundaries. Purpose: test whether P_diag's value depends on the
    specific choice of grouping into 26 snapshot-matched groups (per
    Girish's 2026-08-18 request), by sweeping n_groups independently of
    where real snapshots happen to sit.

    IMPORTANT, worth understanding before reading a sweep's results:
    P_diag is NOT expected to converge to a single value as n_groups
    grows without bound. P_total = P_diag + P_off always holds
    (grouping-invariant -- same map, just summed in a different order),
    so finer grouping structurally moves real correlation OUT of P_diag
    and INTO P_off by construction (more, thinner groups -> more
    distinct pairs -> more terms counted as cross- rather than
    self-power). P_diag should DECREASE monotonically as n_groups grows,
    toward some floor (already seen: the fully-ungrouped, per-native-
    pixel case gives a P_diag well below direct, 31-56% low depending
    on scale/resolution tested). The meaningful question is whether
    there's a PLATEAU in a reasonable range around n_groups=26 -- a
    stable region where P_diag doesn't change much -- which would mean
    the 8.5% match at n_groups=26 is robust, not a coincidence tied to
    that exact number.

    Returns
    -------
    grouped, chi_grouped : same shape convention as group_slices_by_snapshot
    """
    lo = float(chi_mid_mpc.min())
    hi = float(chi_mid_mpc.max()) + 1e-6  # ensure the max value falls inside the last bin
    edges = np.linspace(lo, hi, n_groups + 1)
    digit = np.digitize(chi_mid_mpc, edges)

    grouped, chi_grouped = [], []
    n_used = 0
    for g in range(1, n_groups + 1):
        mask = digit == g
        if np.any(mask):
            grouped.append(theta_slices[:, :, mask].sum(axis=-1))
            chi_grouped.append(chi_mid_mpc[mask].mean())
            n_used += int(mask.sum())
    if n_used != theta_slices.shape[-1]:
        raise RuntimeError(
            f"group_slices_uniform: {theta_slices.shape[-1] - n_used} of "
            f"{theta_slices.shape[-1]} LOS pixels were not assigned to any group "
            f"-- bug in bin-edge construction, not a warning to ignore.")
    return np.stack(grouped, axis=-1), np.array(chi_grouped)


def random_shift_slices(theta_slices, seed=None):
    """
    Independent random cyclic (toroidal) shift per slice in (x,y).
    Preserves each slice's OWN power spectrum exactly (translation is a
    pure Fourier phase rotation) -- destroys any FIXED cross-slice
    spatial alignment, e.g. periodicity-induced correlation from
    stitching identical/correlated box copies at zero relative offset.
    Advisor's requested control: breaks artificial periodic coherence,
    preserves per-slice power.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = theta_slices.shape
    shifted = np.empty_like(theta_slices)
    for i in range(Nz):
        dx, dy = int(rng.integers(0, Nx)), int(rng.integers(0, Ny))
        shifted[:, :, i] = np.roll(theta_slices[:, :, i], shift=(dx, dy), axis=(0, 1))
    return shifted


def _interp_ell_signed(ell_q, ell_p, y_p):
    """Interpolate possibly-NEGATIVE y-values vs ell, using log(ell) as
    the x-axis (monotonic, fine even though y isn't all positive --
    plain loglog_interp can't be used since P_off/D_ell CAN be
    negative). Returns NaN outside [ell_p.min(), ell_p.max()] --
    deliberately NOT extrapolated, since a given pair's own k-range
    only maps to a limited ell range through its own chi_ij; treating
    it as contributing 0 outside that range (see caller) is the
    physically correct choice, not an artifact to paper over.
    """
    ell_p = np.asarray(ell_p); y_p = np.asarray(y_p)
    order = np.argsort(ell_p)
    return np.interp(np.log(ell_q), np.log(ell_p[order]), y_p[order],
                      left=np.nan, right=np.nan)


def decompose_off_pairwise_chi(theta_slices, chi_mid_mpc, box_len_mpc,
                                n_kbins=35, n_ell_out=40):
    """
    P_off, but converting EACH (i,j) pair's own k-space cross-term to
    ell using that PAIR's own effective distance chi_ij = sqrt(chi_i *
    chi_j) -- instead of decompose_p_total_diag_off's single shared
    chi_eff for the whole map.

    RECTILINEAR CAVEAT, discussed and deliberately accepted (not a
    Limber-style non-Limber projection): this does NOT implement
    Alvarez et al. (2016, arXiv:1511.02846) equations (4)/(5)/(8), the
    true spherical/Bessel-function non-Limber treatment, where a single
    3D k-mode maps to a RANGE of ell via j_ell(k*chi) (peaked near
    ell~k*chi but with real tails), not a single ell = k*chi. That
    requires genuinely angular geometry -- a full lightcone rebuild,
    out of scope here (see lightcone_integral.py's own docstring
    pointer to an angular_lightcone.py / angular_ksz_map_to_Dl that may
    exist but has not been located/verified this session). This
    function stays entirely inside the existing rectilinear/flat-sky
    pixel-grid framework: correlations only exist between MATCHING k in
    a flat, translation-invariant plane (theta_i_hat(k) only correlates
    with theta_j_hat(k), never a different k' -- that part is already
    correct upstream). What this fixes is narrower: every pair used to
    be converted to ell through ONE shared chi_eff for the whole map,
    which is wrong for pairs far from that reference distance. Using
    chi_ij per pair is a real, honest improvement within the flat-sky
    approximation -- not a substitute for the full spherical treatment.

    Each pair's P(k)->D_ell conversion (chi_ij used for BOTH the
    ell=k*chi_ij mapping AND the Cl=P/chi_ij^2 amplitude normalization,
    for internal self-consistency) is done fully in that pair's own
    frame BEFORE interpolating onto the shared output ell grid -- not
    the other way around (converting in a shared frame first would
    reintroduce the same single-chi assumption this function exists to
    remove).

    Returns
    -------
    ell_out    : ndarray, shared output multipole grid
    Dl_off_pairwise : ndarray, D_ell (uK^2) -- CAN BE NEGATIVE, summed
                 across all pairs' own contributions at each ell (pairs
                 whose own ell range doesn't reach a given output bin
                 contribute 0 there, not extrapolated)
    n_pairs_used : int, sanity check -- should equal Nz*(Nz-1)/2 for a
                 well-behaved run; fewer means many pairs had degenerate
                 chi_i~chi_j and got dropped, worth knowing
    """
    T_CMB_uK = T_CMB_K * 1e6
    Nx, Ny, Nz = theta_slices.shape
    N = Nx
    pix_Mpc = box_len_mpc / N

    theta_zeroed = theta_slices - theta_slices.mean(axis=(0, 1), keepdims=True)
    theta_hat = np.fft.fftshift(np.fft.fft2(theta_zeroed, axes=(0, 1)), axes=(0, 1))
    norm = (pix_Mpc / N) ** 2

    dk = 2.0 * np.pi / (N * pix_Mpc)
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    kg = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    k_bins    = np.logspace(np.log10(dk), np.log10(kg.max() * 0.9), n_kbins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    digit_k   = np.digitize(kg.ravel(), k_bins)

    def _bin_k(arr2d):
        out = np.full(len(k_centers), np.nan)
        flat = arr2d.ravel()
        for i in range(len(k_centers)):
            mask = digit_k == i + 1
            if np.any(mask):
                out[i] = flat[mask].mean()
        return out

    chi_lo, chi_hi = float(chi_mid_mpc.min()), float(chi_mid_mpc.max())
    chi_axis_ref = np.sqrt(chi_lo * chi_hi)  # axis bounds/labeling ONLY
    ell_out = np.logspace(np.log10(k_centers.min() * chi_axis_ref * 0.5),
                           np.log10(k_centers.max() * chi_axis_ref * 1.5), n_ell_out)

    Dl_off_accum = np.zeros(n_ell_out)
    n_pairs_used = 0
    for i in range(Nz):
        for j in range(i + 1, Nz):
            cross_2d = norm * 2.0 * np.real(theta_hat[:, :, i] * np.conj(theta_hat[:, :, j]))
            P1d_k = _bin_k(cross_2d)
            chi_ij = np.sqrt(chi_mid_mpc[i] * chi_mid_mpc[j])

            ell_this_pair = k_centers * chi_ij
            Cl_this_pair  = P1d_k / chi_ij ** 2
            fac_this_pair = ell_this_pair * (ell_this_pair + 1.0) / (2.0 * np.pi) * T_CMB_uK ** 2
            Dl_this_pair  = Cl_this_pair * fac_this_pair

            valid = ~np.isnan(Dl_this_pair) & (ell_this_pair > 1)
            if valid.sum() < 2:
                continue
            Dl_on_common = _interp_ell_signed(ell_out, ell_this_pair[valid], Dl_this_pair[valid])
            Dl_off_accum += np.nan_to_num(Dl_on_common, nan=0.0)
            n_pairs_used += 1

    return ell_out, Dl_off_accum, n_pairs_used


def cross_power_by_dchi(theta_slices, chi_mid_mpc, box_len_mpc, n_dchi_bins=20):
    """
    Pairwise real-space cross-correlation between every pair of slices
    i != j, binned by radial separation |chi_i - chi_j|. By Parseval's
    theorem this equals each pair's k-integrated cross-power contribution
    to P_off -- computed directly in real space here, which is exactly
    equivalent and avoids redoing per-pair FFT bookkeeping.

        cross_ij = 2 * sum_{x,y}(theta_i(x,y) * theta_j(x,y)) * pix_area

    (factor of 2 for the (i,j)+(j,i) terms, equal for a real-valued map)

    COST WARNING: O(Nz^2) pairs -- fine once fed SNAPSHOT-GROUPED slices
    (Nz~15-29, a few hundred pairs). NOT fine fed raw thin LOS pixels at
    full resolution (Nz~2320, ~2.7M pairs) -- that is a pure-Python
    double loop and would be prohibitively SLOW (not just memory-heavy),
    independent of the earlier memory correction. ALWAYS call this on
    group_slices_by_snapshot's output at full resolution, never on the
    raw per-pixel theta_slices.

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
