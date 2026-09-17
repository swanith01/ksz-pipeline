#!/usr/bin/env python
"""
Script 23: corrected non-Limber off-diagonal projection.

Replaces scripts/22_qperp_cross_z.py, which evaluated the layer-pair power on
the k_par = 0 plane and dropped the radial integral
int dk_par P(k_perp,k_par) exp(i k_par dchi).  That integral is what suppresses
widely separated layers, so deleting it guaranteed an overshoot -- the green
curve in the referee note.  See src/ksz_pipeline/coeval/offdiag_projection.py
for the derivation and the corrected estimator.

Stages, run in order -- the first two are gates:

    --stage convention   scan a_power, require the i==j sum to reproduce
                         limber.compute_cell.  Expected answer is -2 (see
                         LayerWeights docstring); anything else means the q
                         convention differs from momentum.py's.
    --stage coherence    plot |xi(k_perp,Delta)|/|xi(k_perp,0)|.  The
                         hard-zeroing of pairs beyond L/2 is only defensible if
                         this has decayed well before then.
    --stage run          the pair sum.
    --stage kpar0        optional postmortem, see stage_kpar0.

Expectations, stated in advance so the result can falsify itself:
  * D_l^off should be a FRACTION of the stitched P_off, not a multiple.
  * It should be free to go negative at low l (cross terms cancel).
  * Everything below l_min = 2 pi chi / L (~58 at fiducial) is NaN.
  * If it comes back large and positive again, the estimator is still wrong.

Usage
-----
    python scripts/23_offdiag_projection.py --config configs/fiducial.yaml \
        --stage convention
"""
import argparse
import logging
import os
import pickle

import numpy as np
import yaml
from astropy.cosmology import Planck18 as cosmo

from ksz_pipeline.coeval.fields import run_coeval_fields
from ksz_pipeline.coeval.limber import compute_cell
from ksz_pipeline.coeval.momentum import qperp_power
from ksz_pipeline.coeval.offdiag_projection import (
    Layer, LayerWeights, q_los, fft_field, dl_off,
    radial_coherence_diagnostic, check_limber_diagonal,
    check_kpar0_matches_qperp, periodicity_split,
)
from ksz_pipeline.ksz.optical_depth import analytic_tau_below
from ksz_pipeline.utils.constants import ne0_cgs, SIGMA_T, MPC_CM

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

log = logging.getLogger("offdiag")

# Same patchy-window thresholds as script 14 / compute_cell
XHI_MIN_PATCHY, XHI_MAX_PATCHY = 1.0e-4, 1.0 - 1.0e-4


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def build_layers(ZS_win, tau_of_z):
    """Layer table over the patchy window.

    dchi keeps the old h_i = |gradient(chi)_i| definition so the diagonal
    matches the existing Limber sum term by term.
    """
    chi = np.array([cosmo.comoving_distance(z).value for z in ZS_win])
    dchi = np.abs(np.gradient(chi))
    return [Layer(chi=c, dchi=d, a=1.0 / (1.0 + z), tau=float(tau_of_z(z)),
                  label=f"z={z:.2f}")
            for z, c, d in zip(ZS_win, chi, dchi)]


def limber_tau_history(ZS_asc, results_qperp_all, ne0=None):
    """tau(z) replicated EXACTLY from limber.compute_cell.

    Reproduced rather than imported because compute_cell computes it inline
    and returns it only bundled with everything else.  Any drift between this
    and compute_cell's loop shows up directly as a constant offset in the
    diagonal ratio, which is what --stage convention checks.

    Note the tau0 = analytic_tau_below(z_min) floor: dropping it (starting
    tau=0 at the lowest kept z) inflates D_ell by ~6-8%, per limber.py's own
    fix note.
    """
    if ne0 is None:
        ne0 = ne0_cgs()
    ZS_asc = np.asarray(sorted(ZS_asc), dtype=float)
    chi_mpc = np.array([cosmo.comoving_distance(z).value for z in ZS_asc])
    dchi_cm = np.abs(np.gradient(chi_mpc)) * MPC_CM
    xe = np.array([1.0 - results_qperp_all[z]["xH_mean"] for z in ZS_asc])

    tau = np.full_like(ZS_asc, analytic_tau_below(ZS_asc.min()))
    for i in range(len(ZS_asc) - 1):
        zmid = 0.5 * (ZS_asc[i] + ZS_asc[i + 1])
        xe_mid = 0.5 * (xe[i] + xe[i + 1])
        tau[i + 1] = tau[i] + SIGMA_T * ne0 * xe_mid * (1.0 + zmid) ** 2 * dchi_cm[i]
    return tau


def patchy_window(ZS, results_qperp_all):
    """Coeval xH_mean window -- the same filter compute_cell applies."""
    xH = np.array([results_qperp_all[z]["xH_mean"] for z in ZS])
    xe = 1.0 - xH
    keep = (xe >= XHI_MIN_PATCHY) & (xe <= XHI_MAX_PATCHY)
    ZS_win = [float(z) for z in np.array(ZS)[keep]]
    log.info("patchy window: z=[%.2f, %.2f], %d of %d snapshots",
             min(ZS_win), max(ZS_win), len(ZS_win), len(ZS))
    return ZS_win


# ---------------------------------------------------------------------------
# Field loading
# ---------------------------------------------------------------------------
def _fields_for(cfg, z):
    """Raw coeval fields for one redshift: (delta, xH, vx, vy, vz).

    run_coeval_fields returns a TUPLE, with delta the raw contrast (mean ~0,
    not 1+delta) and velocities already in cm/s via velocity_conversion_factor.
    py21cmfast caches on (params, seed), so repeated calls for the same z are
    cheap after the first.
    """
    sim = cfg["21cmfast"]
    return run_coeval_fields(z=z, HII_DIM=sim["HII_DIM_coeval"],
                             BOX_LEN=sim["BOX_LEN"],
                             cache_dir=cfg["data"]["cache_dir"],
                             N_THREADS=sim["N_THREADS"],
                             random_seed=sim["random_seed"])


def build_reference_results(cfg, ZS_win):
    """Fresh qperp_power reference at the config's ACTUAL box_len/hii_dim.

    stage_convention used to compare against results_all from
    qperp_power.pkl, which was cached at the fiducial resolution (512^3).
    With --hii-dim/--box-len overrides for fast testing, that compared a
    small-box direct calculation against a large-box reference -- confirmed
    (17 Sep run: ratio 4.59 instead of ~1.0 at a_power=-2, traced to exactly
    this mismatch) to produce a spurious factor with nothing to do with the
    estimator. Recomputing here makes both sides use identical fields.

    Cheap in practice: py21cmfast caches coeval boxes to disk on
    (params, seed), so this is a fast re-read, not a recomputation -- the
    same "Existing ... found and read in" lines seen in the 17 Sep log.
    """
    out = {}
    for z in ZS_win:
        delta, xH, vx, vy, vz = _fields_for(cfg, z)
        k, P, Pstd = qperp_power(delta, xH, vx, vy, vz, cfg["21cmfast"]["BOX_LEN"])
        out[z] = {"k": k, "Pqperp": P, "Pstd": Pstd, "xH_mean": float(xH.mean())}
    return out


def make_loader(cfg, z_by_label):
    """layer -> q_z = (1+delta) * xHII * v_z / c.

    Line-of-sight momentum in REAL space -- no Q_perp projection.  That
    operator only ever existed to reach k_par = 0, which is exactly the
    restriction being removed.  Velocities are cm/s, matching
    momentum.build_momentum; ne0 and (1+z)^3 stay out of q and live in
    LayerWeights, also matching momentum.py.
    """
    def load_q(layer: Layer) -> np.ndarray:
        z = z_by_label[layer.label]
        delta, xH, _, _, v_z = _fields_for(cfg, z)
        return q_los(delta, 1.0 - xH, v_z, v_units="cm/s")
    return load_q


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_convention(ell, layers, load_q, L, nbar, results_win):
    """Gate 1: which power of a reproduces limber.compute_cell?"""
    ell_d, Dl_d, *_ = compute_cell(results_win)
    Dl_ref = np.interp(ell, ell_d, Dl_d)

    log.info("%-10s %-16s", "a_power", "median ratio")
    best = None
    for a_power in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0):
        w = LayerWeights(nbar_e0_cgs=nbar, a_power=a_power)
        res = check_limber_diagonal(ell, layers, load_q, L, w, Dl_ref)
        r = res["median_ratio"]
        log.info("%-10.1f %-16.4f %s", a_power, r,
                 "<-- PASS" if res["passed"] else "")
        if np.isfinite(r) and (best is None or
                               abs(np.log(abs(r) + 1e-300)) <
                               abs(np.log(abs(best[1]) + 1e-300))):
            best = (a_power, r)
    if best:
        log.info("closest: a_power=%.1f (ratio %.4f). Expected -2 from "
                 "momentum.py's convention.", *best)
    log.info("a_power=-2 is confirmed analytically against limber.compute_cell: "
             "its w = e^-2tau/(chi^2 a^4) dchi times pref=(sigma_T ne0/c)^2 "
             "equals B^2 a^2p e^-2tau dchi/chi^2 with B=sigma_T ne0 MPC_CM "
             "exactly when p=-2, the 1/c^2 coming from q_los dividing v by c. "
             "This scan is therefore a regression test, not a search: anything "
             "but -2 winning means something drifted.")


def stage_coherence(layers, load_q, L, outdir):
    """Gate 2: has xi decayed within half a box?"""
    i, j = 0, min(1, len(layers) - 1)
    Fi = fft_field(load_q(layers[i]), L)
    Fj = fft_field(load_q(layers[j]), L)
    d = radial_coherence_diagnostic(Fi, Fj, L)
    log.info("coherence length per k_perp bin [Mpc]: %s",
             np.round(d["coherence_length"], 1))
    log.info("half box = %.0f Mpc.  VERDICT: %s", d["half_box"], d["verdict"])

    if plt is not None:
        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        keep = d["delta"] >= 0
        for b in range(0, len(d["k_centres"]), 2):
            ax.plot(d["delta"][keep], d["rel"][b][keep],
                    label=rf"$k_\perp$={d['k_centres'][b]:.2f}")
        ax.axvline(d["half_box"], ls="--", c="k")
        ax.text(d["half_box"], 0.5, "  L/2: aliasing", fontsize=8)
        ax.set_xlabel(r"$\Delta\chi$ [Mpc]")
        ax.set_ylabel(r"$|\xi(k_\perp,\Delta)|/|\xi(k_\perp,0)|$")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{outdir}/radial_coherence.png", dpi=140)
        log.info("wrote %s/radial_coherence.png", outdir)
    return d


def load_stitched_off(npz_path, ell, ell_key=None, off_key=None):
    """Load the blue curve -- Dl_off from decompose_p_total_diag_off.

    That function RETURNS (ell, Dl_total, Dl_diag, Dl_off) rather than saving,
    so the key names depend on whichever driver (17/22) wrote the npz.  Rather
    than guess, try the likely names and print what is actually in the file.

    Dl_total is NOT the blue curve -- it is total power.  Comparing against it
    would silently make the corrected direct curve look far too small.
    """
    z = np.load(npz_path, allow_pickle=True)
    ell_cands = [ell_key] if ell_key else ["ell_off", "ell_decomp", "ell", "ell_stitched"]
    off_cands = [off_key] if off_key else ["Dl_off", "dl_off", "P_off", "Dl_offdiag"]
    ek = next((k for k in ell_cands if k and k in z.files), None)
    ok = next((k for k in off_cands if k and k in z.files), None)
    if ek is None or ok is None:
        raise KeyError(
            f"could not find the off-diagonal curve in {npz_path}.\n"
            f"  keys present: {sorted(z.files)}\n"
            f"  pass --stitched-ell-key / --stitched-off-key explicitly.\n"
            f"  NOTE: Dl_total is the TOTAL, not P_off -- do not substitute it.")
    log.info("stitched P_off from %s [%s, %s]", npz_path, ek, ok)
    return np.interp(ell, np.asarray(z[ek], float), np.asarray(z[ok], float))


def warn_ne0_mismatch():
    """The two sides of this comparison can disagree about helium.

    limber.compute_cell defaults to ne0_cgs() (helium included).
    coherence_decomposition.compute_ksz_map_per_slice defaults to
    NE0_HYDROGEN_ONLY.  C_ell goes as ne0^2, so if the driver that produced
    the blue curve did not pass ne0 explicitly, the blue and green curves
    differ by (ne0_He/ne0_H)^2 -- order tens of percent -- for reasons that
    have nothing to do with Limber or periodicity.

    Script 14 does pass ne0=ne0_cgs() explicitly to compute_ksz_map. Confirm
    whichever script produced the P_off npz does the same.
    """
    try:
        from ksz_pipeline.utils.constants import NE0_HYDROGEN_ONLY
        r = (ne0_cgs() / NE0_HYDROGEN_ONLY) ** 2
        log.warning("ne0 convention: limber uses ne0_cgs()=%.4g, "
                    "coherence_decomposition defaults to NE0_HYDROGEN_ONLY=%.4g. "
                    "If the P_off npz was made without an explicit ne0, the blue "
                    "curve is low by a factor %.3f in D_ell for that reason alone.",
                    ne0_cgs(), NE0_HYDROGEN_ONLY, r)
    except ImportError:
        pass


def stage_run(ell, layers, load_q, L, weights, outdir, stitched_npz,
              ell_key=None, off_key=None):
    dl, rep = dl_off(ell, layers, load_q, L, weights)
    os.makedirs(outdir, exist_ok=True)

    log.info("pairs kept %d, skipped %d, window-truncated %d",
             rep["n_pairs"], rep["n_pairs_skipped"],
             rep["n_pairs_window_truncated"])
    log.info("band limit: l > %.0f", rep["ell_min_box"])
    warn_ne0_mismatch()
    log.info("chi convention: this curve uses chi_ij per pair; "
             "decompose_p_total_diag_off uses a single chi_Mpc (chi_eff) for "
             "both ell=k*chi and Cl=P/chi^2. Near pairs agree closely; the "
             "spread across the window is the size of that approximation.")

    np.savez(f"{outdir}/dl_off_corrected.npz", ell=ell, dl=dl,
             ell_min_box=rep["ell_min_box"], n_pairs=rep["n_pairs"],
             n_pairs_skipped=rep["n_pairs_skipped"], max_sep=rep["max_sep"])

    dl_stitched = None
    if stitched_npz and os.path.exists(stitched_npz):
        dl_stitched = load_stitched_off(stitched_npz, ell, ell_key, off_key)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        band = ell > rep["ell_min_box"]
        ax.plot(ell[band], dl[band], "g.-",
                label="direct, corrected (Limber failure)")
        if dl_stitched is not None:
            ax.plot(ell, dl_stitched, "s-", c="tab:blue",
                    label=r"stitched $P_{\rm off}$")
            frac = periodicity_split(dl, dl_stitched)
            ax.plot(ell[band], frac[band], "r--",
                    label="difference = periodicity")
            np.savez(f"{outdir}/periodicity_split.npz", ell=ell, dl_direct=dl,
                     dl_stitched=dl_stitched, dl_periodicity=frac)
        ax.axhline(0, ls=":", c="k")
        ax.axvline(rep["ell_min_box"], ls="--", c="grey")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{outdir}/dl_off_corrected.png", dpi=140)
        log.info("wrote %s/dl_off_corrected.png", outdir)
    return dl, rep


def stage_kpar0(layers, load_q, cfg, z_by_label):
    """Optional postmortem: reproduce the OLD estimator from the NEW field.

    If P_{q_z}(k_perp, k_par=0) == (1/2)<|Q_perp|^2>, then the old code was
    correctly normalised on the plane it used, and the ONLY error was the
    missing radial integral.  Worth running once: it makes the diagnosis
    airtight rather than "we changed something and the number moved".
    """
    layer = layers[len(layers) // 2]
    z = z_by_label[layer.label]
    q = load_q(layer)
    _, P_ref, _ = qperp_power(*_fields_for(cfg, z), cfg["21cmfast"]["BOX_LEN"])
    res = check_kpar0_matches_qperp(q, P_ref, cfg["21cmfast"]["BOX_LEN"])
    log.info("k_par=0 vs qperp_power at %s: passed=%s, median ratio %.4f",
             layer.label, res["passed"], np.nanmedian(res["ratio"]))
    log.info("ratio ~1 => old normalisation was right, only the radial "
             "integral was missing.")
    return res


# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fiducial.yaml")
    p.add_argument("--stage", required=True,
                   choices=["convention", "coherence", "run", "kpar0"])
    p.add_argument("--a-power", type=float, default=-2.0,
                   help="fixed by --stage convention; -2 expected")
    p.add_argument("--box-len", type=float, default=None,
                   help="override BOX_LEN for fast interactive testing")
    p.add_argument("--hii-dim", type=int, default=None,
                   help="override HII_DIM_coeval for fast interactive testing")
    p.add_argument("--n-z-subset", type=int, default=None,
                   help="use only the first N snapshots in the window")
    p.add_argument("--stitched", default="data/products/coherence_decomposition.npz",
                   help="npz holding Dl_off from decompose_p_total_diag_off")
    p.add_argument("--stitched-ell-key", default=None)
    p.add_argument("--stitched-off-key", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.box_len is not None:
        cfg["21cmfast"]["BOX_LEN"] = args.box_len
    if args.hii_dim is not None:
        cfg["21cmfast"]["HII_DIM_coeval"] = args.hii_dim

    L = cfg["21cmfast"]["BOX_LEN"]
    ps = cfg["power_spectrum"]
    ell = np.geomspace(ps["ell_min"], ps["ell_max"], ps["n_ell_bins"])
    outdir = os.path.join(cfg["data"]["plot_dir"].rstrip("/"), "23_offdiag")

    ZS = sorted(cfg["coeval_ksz"]["z_snapshots"])
    with open(os.path.join(cfg["data"]["cache_dir"], "qperp_power.pkl"), "rb") as f:
        results_all = pickle.load(f)

    ZS_win = patchy_window(ZS, results_all)
    if args.n_z_subset:
        ZS_win = ZS_win[:args.n_z_subset]
        log.info("subset: using %d snapshots", len(ZS_win))
    results_win = {z: results_all[z] for z in ZS_win}

    tau_arr = limber_tau_history(ZS_win, results_all)
    tau_of_z = dict(zip(ZS_win, tau_arr)).__getitem__

    layers = build_layers(ZS_win, tau_of_z)
    z_by_label = {l.label: z for l, z in zip(layers, ZS_win)}
    load_q = make_loader(cfg, z_by_label)
    weights = LayerWeights(nbar_e0_cgs=ne0_cgs(), a_power=args.a_power)

    log.info("L=%.0f Mpc, HII_DIM=%d, dz=%.3f Mpc, l_min=%.0f, layers=%d",
             L, cfg["21cmfast"]["HII_DIM_coeval"],
             L / cfg["21cmfast"]["HII_DIM_coeval"],
             2 * np.pi * min(l.chi for l in layers) / L, len(layers))

    if args.stage == "convention":
        ref = build_reference_results(cfg, ZS_win)
        stage_convention(ell, layers, load_q, L, ne0_cgs(), ref)
    elif args.stage == "coherence":
        stage_coherence(layers, load_q, L, outdir)
    elif args.stage == "kpar0":
        stage_kpar0(layers, load_q, cfg, z_by_label)
    else:
        stage_run(ell, layers, load_q, L, weights, outdir, args.stitched,
                  args.stitched_ell_key, args.stitched_off_key)


if __name__ == "__main__":
    main()
