"""
ksz_pipeline/coeval/qperp_cross_z.py

Cross-redshift correlation of q_perp between INDEPENDENT coeval boxes at
different z (z_i != z_j) -- the periodicity-free counterpart to
coherence_decomposition.py's stitched-map P_off.

MOTIVATION: stitched's P_off mixes together (a) genuine cross-redshift
physics Limber discards and (b) box-periodicity artifacts from
LOS-stitching a repeating simulation volume (confirmed real via the
Delta-chi/L_box test). Coeval boxes at DIFFERENT z are INDEPENDENT calls
to run_coeval_fields -- never stitched, tiled, or LOS-interpolated -- so
a nonzero cross-correlation here CANNOT be a periodicity artifact by
construction. It also uses q_perp specifically, the SAME physical
channel compute_cell/direct already trusts for its own diagonal
(z_i=z_j) terms -- unlike stitched, built entirely from q_parallel/v_los.

IMPORTANT CAVEAT: different-z coeval boxes from THIS pipeline are NOT
independent random realizations -- they share the SAME initial density
field (same random_seed), just evolved to different redshifts via
py21cmfast's own Zel'dovich/2LPT machinery. A nonzero cross-correlation
is therefore not automatically "new physics" purely because it's
nonzero -- part of it may reflect the deterministic, correlated
time-evolution of the SAME underlying field, which Limber's "treat each
z as independent" framing also implicitly assumes away. Worth keeping
this distinct from a broader claim about generic cross-redshift physics
-- see the driver script's printed interpretation notes.

RECTILINEAR CAVEAT (same as coherence_decomposition.py): stays flat-sky.
Extracts the k_z=0 plane of each box's full 3D Q_perp(k) field, then
cross-correlates that 2D slice between z_i,z_j pairs using the same
chi_ij = sqrt(chi_i*chi_j) pairwise projection as
coherence_decomposition.decompose_off_pairwise_chi. NOT the full
spherical/Bessel-function non-Limber treatment.

NORMALIZATION NOTE (CORRECTED 2026-08-27, was a real bug, not just a
caveat): the first version of this module omitted the (sigma_T*ne0/c)^2
physical prefactor entirely -- compute_cell applies this to convert raw
momentum power [cm^2 s^-2 Mpc^3] into a dimensionless (Delta_T/T)^2
quantity; the "/2" alone (mirroring qperp_power's own convention) is
NOT that conversion. Omitting it produced D_3000 ~1e26-1e28 uK^2 instead
of the physically sensible O(1) scale -- caught because the self-check
in script 22 was rebuilt to compare against compute_cell's OWN output
directly (not a hand-derived formula), which is far harder to fool with
a missing global constant than the previous manual-formula comparison
was. STILL NOT REPLICATED: compute_cell's per-z weight also includes
visibility^2 * a^-4 * dchi_mpc (see limber.py) -- none of that is
applied here. Treat this module's absolute amplitude as order-of-
magnitude / qualitative, not precision-matched to compute_cell's full
treatment.
"""
import numpy as np


def diagonal_reference_dl(k, P, chi_Mpc, box_len_mpc=None, ne0=None):
    """
    D_ell for a SINGLE z's raw P(k) (e.g. qperp_power's own output),
    using the EXACT SAME formula cross_power_qperp_pairwise_chi applies
    to each pair's diagonal (z_i=z_j) case -- pref*P(k)*MPC_CM^2/chi^2,
    NO visibility/a^-4/dchi weighting (deliberately, matching the
    pairwise function's own scope -- see module docstring).

    Exists so the self-check in script 22 calls this SAME function
    rather than re-deriving the formula a third time -- a second
    independent formula is exactly what let the missing-prefactor bug
    slip past undetected the first time (see NORMALIZATION NOTE above).
    """
    from ..utils.constants import T_CMB_K, SIGMA_T, C_CGS, MPC_CM, ne0_cgs
    if ne0 is None:
        ne0 = ne0_cgs()
    pref = (SIGMA_T * ne0 / C_CGS) ** 2
    T_CMB_uK = T_CMB_K * 1e6

    ell = np.asarray(k) * chi_Mpc
    Cl = pref * np.asarray(P) * MPC_CM ** 2 / chi_Mpc ** 2
    fac = ell * (ell + 1.0) / (2.0 * np.pi) * T_CMB_uK ** 2
    return ell, Cl * fac


def compute_qperp_transverse_components(delta, xH, vx, vy, vz, BOX_LEN):
    """
    Mirrors momentum.qperp_power's FFT/projection logic EXACTLY, up to
    but NOT including the radial binning to 1D P(k) -- returns the
    kz=0 plane of the full 3D transverse-projected momentum field's
    THREE COMPONENTS SEPARATELY (needed for a correct vector dot
    product when cross-correlating two different redshifts -- summing
    components before cross-correlating would incorrectly mix phase
    information between components).

    Returns
    -------
    Qx_perp_2d, Qy_perp_2d, Qz_perp_2d : ndarray (N,N) complex, the
        kz=0 plane of each transverse-projected momentum component
    kx, ky : ndarray (N,) -- the transverse wavenumber grid [Mpc^-1]
    """
    chi = 1.0 - xH
    w = (1.0 + delta) * chi
    qx, qy, qz = w * vx, w * vy, w * vz

    N = qx.shape[0]
    L = float(BOX_LEN)
    d = L / N

    kfreq = np.fft.fftfreq(N, d=d) * 2.0 * np.pi
    kx3, ky3, kz3 = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k2 = kx3 ** 2 + ky3 ** 2 + kz3 ** 2
    k2_safe = np.where(k2 == 0.0, np.inf, k2)

    Qx = np.fft.fftn(qx) * d ** 3
    Qy = np.fft.fftn(qy) * d ** 3
    Qz = np.fft.fftn(qz) * d ** 3

    kdotQ_k2 = (Qx * kx3 + Qy * ky3 + Qz * kz3) / k2_safe
    Qx_perp = Qx - kdotQ_k2 * kx3
    Qy_perp = Qy - kdotQ_k2 * ky3
    Qz_perp = Qz - kdotQ_k2 * kz3

    # kz=0 plane -- direct analog of Limber's implicit k_perp-only
    # restriction on the diagonal terms (see module docstring). fftfreq's
    # zero-frequency bin sits at index 0 before shifting.
    Qx_perp_2d = np.fft.fftshift(Qx_perp[:, :, 0])
    Qy_perp_2d = np.fft.fftshift(Qy_perp[:, :, 0])
    Qz_perp_2d = np.fft.fftshift(Qz_perp[:, :, 0])
    kx_1d = np.fft.fftshift(kfreq)

    return Qx_perp_2d, Qy_perp_2d, Qz_perp_2d, kx_1d, kx_1d  # ky grid == kx grid


def cross_power_qperp_pairwise_chi(qperp_components, chi_dict, xH_mean_dict, box_len_mpc,
                                    ne0=None, n_kbins=35, n_ell_out=40):
    """
    Cross-correlate every PAIR of z's q_perp transverse (kz=0) components
    (full vector dot product, not summed scalars), converting each pair
    to ell via its own chi_ij = sqrt(chi_i*chi_j) -- same projection
    convention as coherence_decomposition.decompose_off_pairwise_chi,
    applied here to q_perp/coeval data instead of stitched theta data.

    WEIGHTING (added 2026-08-28, after the first version's D_3000 came
    back ~1e5x too small): each pair now carries a weight
        w_ij = sqrt(vis2_i*vis2_j) / (a_i^2*a_j^2) * sqrt(dchi_i*dchi_j)
    -- the GEOMETRIC MEAN of compute_cell's own diagonal per-z weight
    (vis2/a^4 * dchi), chosen SPECIFICALLY because it reduces EXACTLY to
    compute_cell's diagonal weight at i=j (sqrt(x*x)=x for x=vis2_i/a_i^4*dchi_i,
    a mathematical identity, not an empirical coincidence -- true for any
    non-negative vis2, dchi, which they always are). This is the same
    generalization logic as chi_ij=sqrt(chi_i*chi_j) already uses for the
    ell-conversion, extended to the remaining weighting factors.
    dchi_i is computed via np.gradient over chi_dict's own sorted values;
    vis2_i via the SAME tau-accumulation logic compute_cell uses internally
    (mirrored here, not imported, to avoid compute_cell's own multi-z
    np.gradient requirement -- see script 22's earlier crash).

    Parameters
    ----------
    qperp_components : dict {z: (Qx_perp_2d, Qy_perp_2d, Qz_perp_2d, kx, ky)}
        from compute_qperp_transverse_components, one entry per redshift
    chi_dict         : dict {z: chi_Mpc}
    xH_mean_dict     : dict {z: xH_mean} -- needed for the tau/visibility
        weighting (x_e = 1 - xH_mean)
    box_len_mpc       : float

    Returns
    -------
    ell_out        : ndarray
    Dl_qperp_cross : ndarray -- CAN BE NEGATIVE, same D_ell convention
                     as decompose_off_pairwise_chi
    n_pairs_used   : int
    """
    from ..utils.constants import T_CMB_K, SIGMA_T, C_CGS, MPC_CM, ne0_cgs
    from ..ksz.optical_depth import analytic_tau_below
    from ..ksz.coherence_decomposition import _interp_ell_signed  # reuse, don't duplicate

    if ne0 is None:
        ne0 = ne0_cgs()
    pref = (SIGMA_T * ne0 / C_CGS) ** 2  # [s^2 cm^-4] -- SAME prefactor compute_cell
    # applies to convert raw momentum power into a dimensionless (Delta_T/T)^2
    # quantity. THIS WAS MISSING in the first version of this function --
    # its absence is exactly what produced ~1e26-1e28 uK^2 "results" instead
    # of the physically sensible O(1) scale. See module docstring.

    T_CMB_uK = T_CMB_K * 1e6
    zs = sorted(qperp_components.keys())
    Qx0, Qy0, Qz0, kx0, ky0 = qperp_components[zs[0]]
    N = Qx0.shape[0]
    pix_Mpc = box_len_mpc / N
    V = box_len_mpc ** 3
    dk = 2.0 * np.pi / (N * pix_Mpc)

    kg = np.sqrt(kx0[:, None] ** 2 + ky0[None, :] ** 2)
    k_bins = np.logspace(np.log10(dk), np.log10(kg.max() * 0.9), n_kbins)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    digit_k = np.digitize(kg.ravel(), k_bins)

    def _bin_k(arr2d):
        out = np.full(len(k_centers), np.nan)
        flat = arr2d.ravel()
        for i in range(len(k_centers)):
            mask = digit_k == i + 1
            if np.any(mask):
                out[i] = flat[mask].mean()
        return out

    # ---- per-z weighting ingredients, mirroring compute_cell's own
    # tau-accumulation loop exactly (limber.py), inlined here rather than
    # calling compute_cell directly -- that function needs >=2 z's for
    # its OWN internal np.gradient, which crashed the earlier self-check.
    chi_arr = np.array([chi_dict[z] for z in zs])
    dchi_arr = np.abs(np.gradient(chi_arr))
    xe_arr = np.array([1.0 - xH_mean_dict[z] for z in zs])
    a_arr = 1.0 / (1.0 + np.array(zs))

    tau0 = analytic_tau_below(zs[0])
    tau_arr = np.full(len(zs), tau0)
    for i in range(len(zs) - 1):
        zmid = 0.5 * (zs[i] + zs[i + 1])
        xe_mid = 0.5 * (xe_arr[i] + xe_arr[i + 1])
        tau_arr[i + 1] = tau_arr[i] + SIGMA_T * ne0 * xe_mid * (1.0 + zmid) ** 2 * (dchi_arr[i] * MPC_CM)
    vis2_arr = np.exp(-2.0 * tau_arr)

    chi_lo, chi_hi = min(chi_dict.values()), max(chi_dict.values())
    chi_axis_ref = np.sqrt(chi_lo * chi_hi)
    ell_out = np.logspace(np.log10(k_centers.min() * chi_axis_ref * 0.5),
                           np.log10(k_centers.max() * chi_axis_ref * 1.5), n_ell_out)

    Dl_accum = np.zeros(n_ell_out)
    n_pairs_used = 0
    for idx_a in range(len(zs)):
        for idx_b in range(idx_a + 1, len(zs)):
            zi, zj = zs[idx_a], zs[idx_b]
            Qxi, Qyi, Qzi, _, _ = qperp_components[zi]
            Qxj, Qyj, Qzj, _, _ = qperp_components[zj]

            # full vector dot product -- see module docstring on why
            # components must stay separate until this step
            dot = (Qxi * np.conj(Qxj) + Qyi * np.conj(Qyj) + Qzi * np.conj(Qzj))
            cross_2d = np.real(dot) / V / 2.0  # "/2" mirrors qperp_power -- see NORMALIZATION NOTE
            P1d_k = _bin_k(cross_2d)

            chi_ij = np.sqrt(chi_dict[zi] * chi_dict[zj])
            ell_this_pair = k_centers * chi_ij

            # geometric-mean pair weight -- see docstring for the exact-at-i=j
            # derivation. Factor of 2 accounts for the (i,j)+(j,i) symmetric
            # pair (same convention as coherence_decomposition's off-diagonal
            # sum) -- MISSING in the first version of this function.
            w_pair = np.sqrt(vis2_arr[idx_a] * vis2_arr[idx_b]) / (a_arr[idx_a] ** 2 * a_arr[idx_b] ** 2) \
                     * np.sqrt(dchi_arr[idx_a] * dchi_arr[idx_b])
            Cl_this_pair = 2.0 * pref * P1d_k * MPC_CM ** 2 / chi_ij ** 2 * w_pair
            # (previously: Cl_this_pair = pref * P1d_k * MPC_CM**2 / chi_ij**2 -- missing
            # BOTH the w_pair weighting and the factor-of-2 symmetric-pair count. This
            # is what produced D_3000 ~1e-5 uK^2 instead of a physically comparable scale.
            # precision-matched normalization against compute_cell's full
            fac_this_pair = ell_this_pair * (ell_this_pair + 1.0) / (2.0 * np.pi) * T_CMB_uK ** 2
            Dl_this_pair = Cl_this_pair * fac_this_pair

            valid = ~np.isnan(Dl_this_pair) & (ell_this_pair > 1)
            if valid.sum() < 2:
                continue
            Dl_on_common = _interp_ell_signed(ell_out, ell_this_pair[valid], Dl_this_pair[valid])
            Dl_accum += np.nan_to_num(Dl_on_common, nan=0.0)
            n_pairs_used += 1

    return ell_out, Dl_accum, n_pairs_used
