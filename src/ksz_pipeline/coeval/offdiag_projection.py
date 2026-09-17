"""
offdiag_projection.py
=====================

Corrected non-Limber off-diagonal (layer-pair) kSZ projection from coeval
snapshots.  Replacement for the k_par = 0 estimator in
``coeval/qperp_cross_z.py`` (commit 4020d19).

Background
----------
The quantity we want is the flat-sky layer-pair angular cross-power

    C_l^{ij} = (1/chi_0^2) int dchi int dchi' W_i(chi) W_j(chi')
               int dk_par/(2pi) P_f(l/chi_0, k_par; t_i, t_j)
               exp[i k_par (chi - chi')]                              (A)

with f = n_e v_los.  The previous implementation evaluated P_f on the single
plane k_par = 0 and dropped the radial integral entirely.  That is neither the
exact projection nor its Limber limit: Limber evaluates the *smooth* power at
k_par ~ 0 but still integrates the exponential, and it is that integral which
produces the delta_D(Delta chi) localisation that suppresses cross-layer terms.
Deleting it removes the suppression, which is why the old green curve
overshoots the stitched P_off instead of being a fraction of it.

Two facts make the fix cheap:

1.  For a statistically isotropic vector field,

        < q_i q_j^* > = khat_i khat_j P_L(k) + (delta_ij - khat_i khat_j) P_T(k)/2

    so  P_{q_z}(k_perp, k_par) = mu^2 P_L + (1 - mu^2) P_T / 2,  mu = k_par/k.
    At mu = 0 this equals P_T/2 = (1/2)<|Q_perp|^2>, i.e. exactly the old
    estimator.  The old normalisation (including its factor 1/2) was therefore
    *correct on the plane it was evaluated on*.  The bug is only the restriction
    to that plane.  Consequence: we do not need a new field or a new projection
    operator.  We take q_z = (1+delta) x_HII v_z in real space, FFT it, and its
    cross-spectrum *is* P_f(k_perp, k_par) for every k_par.  The Q_perp
    projection was only ever a device for reaching mu = 0 and is dropped here.

2.  Equation (A) is most cheaply evaluated in mixed space.  Define

        xi_ij(k_perp, Delta) = int dk_par/(2pi) P_ij(k_perp, k_par) e^{i k_par Delta}
        K_ij(Delta)          = int dchi W_i(chi) W_j(chi - Delta)

    then

        C_l^{ij} = (1/chi_ij^2) int dDelta K_ij(Delta) xi_ij(l/chi_ij, Delta).   (B)

    xi_ij is obtained by inverse-FFT of the cross-spectrum along the line-of-sight
    axis only, so the whole Delta dependence comes out in one pass.

    For i = j and a xi that decays fast compared with the layer width, (B)
    reduces to h_i P_ii(l/chi, k_par = 0) / chi^2, i.e. the existing validated
    Limber weight -- with no interpolation choice.  The geometric-mean weight
    sqrt(h_i h_j) in the old code is not needed; it is recovered where it is
    correct and replaced where it is not.  ``check_limber_diagonal`` below is
    the regression test for this.

Box periodicity (IMPORTANT)
---------------------------
xi_ij measured from a periodic box of side L is periodic in Delta with period L.
Any layer pair whose true separation exceeds L/2 is aliased back onto a small
separation -- this is the same box periodicity the stitched map suffers from,
re-entering through the radial integral.  The corrected calculation is therefore
NOT automatically "periodicity-free".

The prescription used here: evaluate xi at the TRUE Delta chi_ij and hard-zero
it for |Delta| > max_sep (default L/2).  Physically this asserts that radial
correlations have decayed by half a box, which ``radial_coherence_diagnostic``
must be used to verify.  If it has not decayed, the box is too small and no
conclusion about P_off can be drawn from it.

This gives the decomposition you actually want:

    corrected direct   ->  uses true Delta chi          -> Limber failure only
    stitched P_off     ->  uses Delta chi mod L         -> Limber failure + periodicity
    difference         ->  periodicity contribution, isolated

Transverse band limit
---------------------
k_perp = l / chi, and the box fundamental is 2 pi / L, so
l_min ~ 2 pi chi / L  (e.g. ~140 for chi = 6800 Mpc, L = 300 Mpc).
Below l_min the box contains no modes.  Routines here return NaN there rather
than extrapolating.

Conventions
-----------
Lengths in comoving Mpc (NOT Mpc/h -- convert on input if your boxes are h-units).
Velocities in units of c (helpers provided for km/s).
The line-of-sight axis is the last axis (axis=2) of every field array, matching
the 21cmFAST coeval convention used elsewhere in the pipeline.

The scale factor exponent ``a_power`` in ``LayerWeights`` is deliberately exposed
rather than hard-coded: it encodes whether your q field carries comoving or
physical density and comoving or physical velocity.  Do not guess it -- fix it
by running ``check_limber_diagonal`` against ``coeval/limber.py:compute_cell``
and adopting whichever value reproduces it.

Author: drafted as a correction to qperp_cross_z.py per the referee note of
2026-09; not independently validated against the repo -- run the checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants (cgs unless stated)
# ----------------------------------------------------------------------------
SIGMA_T_CGS = 6.6524587158e-25      # cm^2
C_LIGHT_KMS = 2.99792458e5          # km/s
MPC_CM = 3.0856775814913673e24      # cm per Mpc
T_CMB_UK = 2.7255e6                 # micro-K


# ----------------------------------------------------------------------------
# Geometry / weight containers
# ----------------------------------------------------------------------------
@dataclass
class Layer:
    """One radial layer, backed by one coeval snapshot.

    Attributes
    ----------
    chi : float
        Comoving distance to the layer centre [Mpc].
    dchi : float
        Radial width of the layer [Mpc].  In the old code this was
        ``h_i = |gradient(chi)_i|``; keep that definition so the diagonal
        matches the existing Limber sum.
    a : float
        Scale factor at the snapshot redshift.
    tau : float
        Optical depth from the observer to this layer.
    label : str
        Anything identifying the snapshot (redshift string, filename, ...).
    """
    chi: float
    dchi: float
    a: float
    tau: float
    label: str = ""


@dataclass
class LayerWeights:
    """Amplitude convention for the radial weight W(chi).

        W_i(chi) = B * a(chi)**a_power * exp(-tau(chi))

    with B = sigma_T * nbar_e0 * MPC_CM  [per Mpc], and the q field carrying
    v/c so that C_l comes out dimensionless.

    ``a_power`` MUST be set to whatever reproduces ``limber.py:compute_cell``
    on the diagonal.  The old eq. (18) implies a_power = 2 for its own q
    convention; verify, do not assume.
    """
    nbar_e0_cgs: float          # mean electron number density today [cm^-3]
    a_power: float = 2.0

    @property
    def B_per_Mpc(self) -> float:
        return SIGMA_T_CGS * self.nbar_e0_cgs * MPC_CM

    def w(self, layer: Layer) -> float:
        return self.B_per_Mpc * (layer.a ** self.a_power) * np.exp(-layer.tau)


# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
def q_los(delta: np.ndarray,
          x_HII: np.ndarray,
          v_los: np.ndarray,
          v_units: str = "c") -> np.ndarray:
    """Line-of-sight ionised-momentum field q_z = (1 + delta) x_HII v_z / c.

    This is the field whose 3D power spectrum is P_f(k_perp, k_par) directly.
    No Q_perp projection: that operator only exists to reach the k_par = 0
    plane, which is exactly what we are no longer restricted to.

    Parameters
    ----------
    delta, x_HII, v_los : ndarray, shape (N, N, N)
        Density contrast, ionised fraction, and the line-of-sight velocity
        component (component along axis 2).
    v_units : {"c", "km/s"}
    """
    if delta.shape != x_HII.shape or delta.shape != v_los.shape:
        raise ValueError("delta, x_HII, v_los must have identical shapes")
    v = np.asarray(v_los, dtype=np.float64)
    if v_units == "km/s":
        v = v / C_LIGHT_KMS
    elif v_units != "c":
        raise ValueError("v_units must be 'c' or 'km/s'")
    return (1.0 + delta) * x_HII * v


def fft_field(q: np.ndarray, L: float) -> np.ndarray:
    """Continuum-normalised FFT: qtilde(k) = int d^3x q(x) e^{-ik.x}."""
    N = q.shape[0]
    dV = (L / N) ** 3
    return np.fft.fftn(q) * dV


# ----------------------------------------------------------------------------
# Mixed-space correlator xi_ij(k_perp, Delta)
# ----------------------------------------------------------------------------
def mixed_correlator(qk_i: np.ndarray,
                     qk_j: np.ndarray,
                     L: float) -> Tuple[np.ndarray, np.ndarray]:
    """xi_ij(k_perp, Delta) = int dk_par/(2pi) P_ij(k_perp,k_par) e^{i k_par Delta}.

    P_ij(k) = (1/V) qtilde_i(k) qtilde_j^*(k).  The radial integral is a single
    inverse FFT along the line-of-sight axis.

    Returns
    -------
    xi : complex ndarray, shape (N, N, N)
        Indexed [kx, ky, m], where Delta = delta_grid[m].
    delta_grid : ndarray, shape (N,)
        Radial separations [Mpc], in FFT order (0, dz, ..., -2dz, -dz).

    Notes
    -----
    xi is complex for i != j (unequal-time spectra are not real).  Only the
    real part enters D_l^off via the 2 Re C^{ij} pair sum.
    """
    if qk_i.shape != qk_j.shape:
        raise ValueError("snapshot grids differ in shape")
    N = qk_i.shape[0]
    V = L ** 3
    dz = L / N

    P = (qk_i * np.conj(qk_j)) / V                 # [Mpc^3]
    # int dk_par/(2pi) ... e^{+i k_par Delta}  ->  (1/dz) * ifft along axis 2
    xi = np.fft.ifft(P, axis=2) / dz               # [Mpc^2]

    delta_grid = np.fft.fftfreq(N, d=1.0 / N) * dz
    return xi, delta_grid


def kperp_grid(N: int, L: float) -> np.ndarray:
    """|k_perp| on the (kx, ky) grid [1/Mpc]."""
    k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    return np.sqrt(kx ** 2 + ky ** 2)


def ring_average(xi: np.ndarray,
                 L: float,
                 n_bins: int = 24,
                 kmin: Optional[float] = None,
                 kmax: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ring-average xi over transverse annuli, for every Delta.

    Returns
    -------
    k_centres : (n_bins,) ndarray      geometric bin centres [1/Mpc]
    xi_ring   : (n_bins, N) complex    ring-averaged xi
    counts    : (n_bins,) ndarray      modes per ring (0 => bin unusable)
    """
    N = xi.shape[0]
    kperp = kperp_grid(N, L)
    k_f = 2.0 * np.pi / L                       # fundamental
    k_nyq = np.pi * N / L
    kmin = k_f if kmin is None else kmin
    kmax = k_nyq if kmax is None else kmax

    edges = np.geomspace(kmin * 0.999, kmax * 1.001, n_bins + 1)
    flat_k = kperp.ravel()
    # drop the k_perp = 0 column: it carries no transverse information
    good = flat_k > 0.5 * k_f
    idx = np.digitize(flat_k[good], edges) - 1
    inside = (idx >= 0) & (idx < n_bins)

    flat_xi = xi.reshape(-1, N)[good][inside]
    idx = idx[inside]

    counts = np.bincount(idx, minlength=n_bins)
    xi_ring = np.zeros((n_bins, N), dtype=complex)
    for m in range(N):
        col = flat_xi[:, m]
        re = np.bincount(idx, weights=col.real, minlength=n_bins)
        im = np.bincount(idx, weights=col.imag, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            xi_ring[:, m] = (re + 1j * im) / np.maximum(counts, 1)
    xi_ring[counts == 0, :] = np.nan

    k_centres = np.sqrt(edges[:-1] * edges[1:])
    return k_centres, xi_ring, counts


# ----------------------------------------------------------------------------
# Window overlap K_ij(Delta)
# ----------------------------------------------------------------------------
def window_overlap(delta_grid: np.ndarray,
                   li: Layer,
                   lj: Layer,
                   window: str = "tophat") -> np.ndarray:
    """K_ij(Delta) = int dchi W_i(chi) W_j(chi - Delta), unit-height windows.

    For top-hats of widths dchi_i, dchi_j centred at chi_i, chi_j this is a
    trapezoid centred at Delta chi_ij = chi_i - chi_j, with
    int K dDelta = dchi_i * dchi_j.  Amplitude factors (B, a^p, e^-tau) are
    applied separately in ``cell_pair``.

    A 'gaussian' option is provided for layers whose radial support is better
    described by a smooth snapshot weight; it is normalised identically.
    """
    d0 = li.chi - lj.chi
    u = delta_grid - d0
    wi, wj = li.dchi, lj.dchi

    if window == "tophat":
        half_sum = 0.5 * (wi + wj)
        half_dif = 0.5 * abs(wi - wj)
        K = np.clip((half_sum - np.abs(u)) / (half_sum - half_dif + 1e-30),
                    0.0, 1.0)
        K *= min(wi, wj)
    elif window == "gaussian":
        # sigma chosen so the FWHM matches the top-hat width
        s2 = (wi ** 2 + wj ** 2) / (8.0 * np.log(2.0))
        K = np.exp(-0.5 * u ** 2 / s2)
        norm = np.sqrt(2.0 * np.pi * s2)
        K *= (wi * wj) / norm
    else:
        raise ValueError("window must be 'tophat' or 'gaussian'")
    return K


# ----------------------------------------------------------------------------
# One layer pair -> C_l^{ij}
# ----------------------------------------------------------------------------
def cell_pair(ell: np.ndarray,
              k_centres: np.ndarray,
              xi_ring: np.ndarray,
              delta_grid: np.ndarray,
              li: Layer,
              lj: Layer,
              weights: LayerWeights,
              L: float,
              max_sep: Optional[float] = None,
              window: str = "tophat",
              ) -> Tuple[np.ndarray, dict]:
    """Evaluate equation (B) for one layer pair.

        C_l^{ij} = (w_i w_j / chi_ij^2) * int dDelta K_ij(Delta) xi(l/chi_ij, Delta)

    Returns
    -------
    cl : ndarray, shape (n_ell,)
        NaN wherever l/chi_ij falls outside the box band limit.
    info : dict
        Diagnostics: band-limit ell range, fraction of window weight discarded
        by the |Delta| > max_sep cut, and the aliasing flag.
    """
    ell = np.atleast_1d(np.asarray(ell, dtype=float))
    chi_ij = np.sqrt(li.chi * lj.chi)
    max_sep = 0.5 * L if max_sep is None else max_sep

    # --- radial integral -----------------------------------------------------
    dz = float(np.abs(delta_grid[1] - delta_grid[0]))
    K = window_overlap(delta_grid, li, lj, window=window)

    keep = np.abs(delta_grid) <= max_sep
    discarded = float(K[~keep].sum() / max(K.sum(), 1e-300))
    Kc = np.where(keep, K, 0.0)

    # int dDelta K(Delta) xi(k, Delta)  for every ring
    integral = (xi_ring * Kc[None, :]).sum(axis=1).real * dz   # (n_bins,)

    # --- transverse band limit ----------------------------------------------
    k_f = 2.0 * np.pi / L
    k_nyq = np.pi * (xi_ring.shape[1]) / L
    ell_min = k_f * chi_ij
    ell_max = k_nyq * chi_ij

    k_want = ell / chi_ij
    cl = np.full_like(ell, np.nan, dtype=float)
    ok = (k_want >= k_centres[0]) & (k_want <= k_centres[-1])

    finite = np.isfinite(integral)
    if finite.sum() >= 2:
        cl[ok] = np.interp(np.log(k_want[ok]),
                           np.log(k_centres[finite]),
                           integral[finite])

    pref = weights.w(li) * weights.w(lj) / chi_ij ** 2
    cl *= pref

    info = {
        "chi_ij": chi_ij,
        "delta_chi": li.chi - lj.chi,
        "ell_min_box": ell_min,
        "ell_max_box": ell_max,
        "window_weight_discarded": discarded,
        "aliasing_risk": abs(li.chi - lj.chi) > max_sep,
    }
    return cl, info


# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------
def dl_off(ell: np.ndarray,
           layers: Sequence[Layer],
           load_q: Callable[[Layer], np.ndarray],
           L: float,
           weights: LayerWeights,
           n_bins: int = 24,
           max_sep: Optional[float] = None,
           window: str = "tophat",
           include_diagonal: bool = False,
           progress: bool = True,
           ) -> Tuple[np.ndarray, dict]:
    """D_l^off = T_CMB^2 l(l+1)/(2pi) * 2 sum_{i<j} Re C_l^{ij}.

    Parameters
    ----------
    load_q : callable
        ``load_q(layer) -> q_z array (N,N,N)``.  Build it with ``q_los``.
        Called once per layer; the FFT is cached, so peak memory is
        ~n_layers * N^3 * 16 bytes.  For many layers / large N, wrap this in a
        chunked driver instead (see the driver script).
    include_diagonal : bool
        If True, also return the i == j sum -- useful only for the Limber
        regression check, not part of D_l^off.

    Returns
    -------
    dl : ndarray, shape (n_ell,)
    report : dict
        Per-pair diagnostics plus aggregate warnings.
    """
    ell = np.asarray(ell, dtype=float)
    n = len(layers)

    log.info("FFTing %d snapshots", n)
    qk = []
    for li in layers:
        q = load_q(li)
        if q.ndim != 3 or q.shape[0] != q.shape[1] or q.shape[0] != q.shape[2]:
            raise ValueError("expected a cubic (N,N,N) field")
        qk.append(fft_field(q, L))
        del q

    cl_sum = np.zeros_like(ell)
    cl_diag = np.zeros_like(ell)
    pair_info = []
    aliased_pairs = 0

    pairs: Iterable[Tuple[int, int]] = (
        [(i, j) for i in range(n) for j in range(i, n)] if include_diagonal
        else [(i, j) for i in range(n) for j in range(i + 1, n)]
    )

    for count, (i, j) in enumerate(pairs):
        xi, dgrid = mixed_correlator(qk[i], qk[j], L)
        kc, xir, _ = ring_average(xi, L, n_bins=n_bins)
        del xi
        cl, info = cell_pair(ell, kc, xir, dgrid, layers[i], layers[j],
                             weights, L, max_sep=max_sep, window=window)
        pair_info.append(((i, j), info))
        if info["aliasing_risk"]:
            aliased_pairs += 1
        contrib = np.nan_to_num(cl, nan=0.0)
        if i == j:
            cl_diag += contrib
        else:
            cl_sum += 2.0 * contrib      # both orderings
        if progress and count % 25 == 0:
            log.info("pair %d/%d  (i=%d j=%d)", count, len(list(pairs)) if
                     isinstance(pairs, list) else -1, i, j)

    conv = T_CMB_UK ** 2 * ell * (ell + 1.0) / (2.0 * np.pi)
    dl = conv * cl_sum

    report = {
        "pair_info": pair_info,
        "n_pairs": len(pair_info),
        "n_pairs_beyond_half_box": aliased_pairs,
        "ell_min_box": 2.0 * np.pi * min(l.chi for l in layers) / L,
        "dl_diagonal": conv * cl_diag if include_diagonal else None,
        "band_limit_note": (
            "C_l set to NaN (and excluded from the sum) where l/chi falls "
            "outside [2 pi chi / L, pi N chi / L]."
        ),
    }
    if aliased_pairs:
        log.warning(
            "%d/%d pairs have |Delta chi| > max_sep; their cross terms were "
            "zeroed. Run radial_coherence_diagnostic to confirm xi has decayed "
            "by that separation, otherwise the box is too small.",
            aliased_pairs, len(pair_info))
    return dl, report


# ----------------------------------------------------------------------------
# Validation / diagnostics -- run these before trusting any number
# ----------------------------------------------------------------------------
def check_kpar0_matches_qperp(q_z: np.ndarray,
                              Q_perp_power_kpar0: np.ndarray,
                              L: float,
                              n_bins: int = 24,
                              rtol: float = 0.05) -> dict:
    """Confirm P_{q_z}(k_perp, k_par=0) == (1/2) <|Q_perp|^2> on the k_par=0 plane.

    This reproduces the OLD estimator from the NEW field.  If it passes, the old
    code was correctly normalised on its plane and the only error was the
    missing radial integral -- which is the claim this whole correction rests on.

    ``Q_perp_power_kpar0`` should be the ring-averaged output of the existing
    ``momentum.qperp_power`` on the same snapshot, on the same k_perp bins.
    """
    N = q_z.shape[0]
    qk = fft_field(q_z, L)
    P = (np.abs(qk[:, :, 0]) ** 2) / L ** 3      # k_par = 0 plane
    kperp = kperp_grid(N, L)
    k_f = 2.0 * np.pi / L
    edges = np.geomspace(k_f * 0.999, np.pi * N / L * 1.001, n_bins + 1)
    idx = np.digitize(kperp.ravel(), edges) - 1
    good = (kperp.ravel() > 0.5 * k_f) & (idx >= 0) & (idx < n_bins)
    counts = np.bincount(idx[good], minlength=n_bins)
    prof = np.bincount(idx[good], weights=P.ravel()[good], minlength=n_bins)
    prof = prof / np.maximum(counts, 1)

    ref = np.asarray(Q_perp_power_kpar0, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = prof / ref
    ok = np.nanmedian(np.abs(ratio - 1.0)) < rtol
    return {"passed": bool(ok), "ratio": ratio,
            "k_centres": np.sqrt(edges[:-1] * edges[1:]),
            "note": "ratio should be ~1 if the old 1/2 convention was right"}


def check_limber_diagonal(ell: np.ndarray,
                          layers: Sequence[Layer],
                          load_q: Callable[[Layer], np.ndarray],
                          L: float,
                          weights: LayerWeights,
                          compute_cell_reference: np.ndarray,
                          n_bins: int = 24,
                          window: str = "tophat") -> dict:
    """Regression test: the i == j part of equation (B) must reproduce limber.py.

    This is what replaces the old sqrt(h_i h_j) choice.  If the ratio is not
    ~1, the ``a_power`` / unit convention in ``LayerWeights`` is wrong -- fix it
    here, before running any pair sum.
    """
    dl, rep = dl_off(ell, layers, load_q, L, weights, n_bins=n_bins,
                     window=window, include_diagonal=True, progress=False)
    diag = rep["dl_diagonal"]
    ref = np.asarray(compute_cell_reference, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = diag / ref
    band = np.isfinite(ratio) & (ratio != 0)
    return {"dl_diagonal": diag, "ratio": ratio,
            "median_ratio": float(np.nanmedian(ratio[band])) if band.any() else np.nan,
            "passed": bool(band.any() and abs(np.nanmedian(ratio[band]) - 1) < 0.05)}


def radial_coherence_diagnostic(qk_i: np.ndarray,
                                qk_j: np.ndarray,
                                L: float,
                                n_bins: int = 12) -> dict:
    """|xi(k_perp, Delta)| / |xi(k_perp, 0)| versus Delta.

    The hard-zeroing of cross terms beyond L/2 is only defensible if this has
    fallen into the noise well before Delta = L/2.  If it has not, the box is
    too small to say anything about P_off and the result must not be quoted.
    Plot this for the widest-separated pair you intend to keep.
    """
    xi, dgrid = mixed_correlator(qk_i, qk_j, L)
    kc, xir, _ = ring_average(xi, L, n_bins=n_bins)
    order = np.argsort(dgrid)
    dsort = dgrid[order]
    prof = np.abs(xir[:, order])
    norm = np.abs(xir[:, 0])[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = prof / norm

    half = dsort >= 0
    # coherence length per ring: first Delta where |xi| drops below 10%
    lcoh = np.full(len(kc), np.nan)
    for b in range(len(kc)):
        r = rel[b, half]
        below = np.where(r < 0.1)[0]
        if below.size:
            lcoh[b] = dsort[half][below[0]]
    return {"delta": dsort, "k_centres": kc, "rel": rel,
            "coherence_length": lcoh, "half_box": 0.5 * L,
            "verdict": ("OK" if np.nanmax(lcoh) < 0.25 * L else
                        "BOX TOO SMALL -- xi has not decayed by L/4")}


def periodicity_split(dl_direct_true_sep: np.ndarray,
                      dl_stitched_off: np.ndarray) -> np.ndarray:
    """Isolate the periodicity contribution to the stitched P_off.

    The corrected direct calculation evaluates xi at the TRUE Delta chi; the
    stitched map effectively evaluates it at Delta chi mod L.  Their difference
    is the box-periodicity contribution, with the Limber-failure part cancelled.
    """
    return np.asarray(dl_stitched_off) - np.asarray(dl_direct_true_sep)
