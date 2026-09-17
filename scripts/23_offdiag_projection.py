#!/usr/bin/env python
"""
scripts/23_offdiag_projection.py
================================

Corrected replacement for ``scripts/22_qperp_cross_z.py``.

Recomputes the "direct" off-diagonal curve with the full radial projection
instead of the k_par = 0 plane, and compares it against the stitched P_off.

Run order -- do NOT skip the gates:

    python scripts/23_offdiag_projection.py --stage convention
        Fixes LayerWeights.a_power by requiring the i==j sum to reproduce
        limber.py:compute_cell.  Nothing downstream is meaningful until this
        passes.

    python scripts/23_offdiag_projection.py --stage coherence
        Plots |xi(k_perp, Delta)| / |xi(k_perp, 0)|.  If radial correlations
        have not decayed by ~L/4, the box is too small and the run stops.

    python scripts/23_offdiag_projection.py --stage run
        The pair sum.

Sanity expectations for the output, stated in advance so they can falsify it:
  * D_l^off should be SMALL -- a fraction of the stitched P_off, not a multiple.
  * It should be free to go negative at low l (cross terms cancel).
  * Everything below l_min ~ 2 pi chi / L is NaN and must not be plotted.
  * If it comes back large and positive again, the estimator is still wrong;
    do not reach for a physical interpretation.

TODO markers are the places that need the real repo -- I/O and the two
reference curves.  Everything else runs as-is.
"""

import argparse
import logging
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ksz_pipeline.coeval.offdiag_projection import (           # TODO: -> ksz_pipeline.coeval.offdiag_projection
    Layer, LayerWeights, q_los, fft_field, dl_off,
    mixed_correlator, radial_coherence_diagnostic,
    check_limber_diagonal, check_kpar0_matches_qperp, periodicity_split,
)

log = logging.getLogger("offdiag")

# --------------------------------------------------------------------------
# Configuration -- fill in from the repo's existing config
# --------------------------------------------------------------------------
BOX_L_MPC = 300.0            # TODO: comoving Mpc (NOT Mpc/h)
NBAR_E0_CGS = 2.1e-7         # TODO: use the pipeline's own normalisation
OUTDIR = "outputs/23_offdiag"


def build_layers():
    """TODO: replace with the pipeline's snapshot table.

    ``dchi`` must keep the old definition h_i = |gradient(chi)_i| so the
    diagonal matches the existing Limber sum.
    """
    # z, chi, a, tau per snapshot, from the same source 14_closure_test.py uses
    z = np.array([...])            # TODO
    chi = np.array([...])          # TODO  cosmology.comoving_distance(z) [Mpc]
    tau = np.array([...])          # TODO  optical_depth.tau_to(z)
    dchi = np.abs(np.gradient(chi))
    return [Layer(chi=c, dchi=d, a=1.0 / (1.0 + zz), tau=t, label=f"z={zz:.2f}")
            for zz, c, d, t in zip(z, chi, dchi, tau)]


def make_loader(v_units="km/s"):
    """TODO: return a callable layer -> q_z field.

    Note this is the *line-of-sight* momentum in real space -- no Q_perp
    projection.  Reuse whatever loader momentum.py uses for delta / xHII / v,
    but keep v_z rather than projecting.
    """
    def load_q(layer: Layer) -> np.ndarray:
        delta = ...      # TODO load density contrast for layer.label
        xHII = ...       # TODO load ionised fraction
        v_z = ...        # TODO load velocity component along axis 2
        return q_los(delta, xHII, v_z, v_units=v_units)
    return load_q


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_convention(ell, layers, load_q, weights):
    """Gate 1: fix a_power by matching the Limber diagonal."""
    cl_ref = ...   # TODO: limber.compute_cell(ell) -> D_l, same units/binning
    for a_power in (-2.0, -1.0, 0.0, 1.0, 2.0):
        w = LayerWeights(nbar_e0_cgs=weights.nbar_e0_cgs, a_power=a_power)
        res = check_limber_diagonal(ell, layers, load_q, BOX_L_MPC, w, cl_ref)
        log.info("a_power=%+.1f  median ratio to compute_cell = %.4f  %s",
                 a_power, res["median_ratio"], "PASS" if res["passed"] else "")
    log.info("Adopt the a_power whose ratio is 1.000 and hard-code it. "
             "If none is, the q-field unit convention differs from limber.py "
             "by more than a power of a -- resolve that before continuing.")


def stage_coherence(layers, load_q):
    """Gate 2: has xi decayed within the box?"""
    i, j = 0, min(1, len(layers) - 1)
    Fi = fft_field(load_q(layers[i]), BOX_L_MPC)
    Fj = fft_field(load_q(layers[j]), BOX_L_MPC)
    d = radial_coherence_diagnostic(Fi, Fj, BOX_L_MPC)
    log.info("coherence lengths per k_perp bin [Mpc]: %s",
             np.round(d["coherence_length"], 1))
    log.info("verdict: %s", d["verdict"])
    if plt is not None:
        os.makedirs(OUTDIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        keep = d["delta"] >= 0
        for b in range(0, len(d["k_centres"]), 2):
            ax.plot(d["delta"][keep], d["rel"][b][keep],
                    label=rf"$k_\perp$={d['k_centres'][b]:.2f}")
        ax.axvline(d["half_box"], ls="--", c="k")
        ax.text(d["half_box"], 0.9, " L/2: aliasing", fontsize=8)
        ax.set_xlabel(r"$\Delta\chi$ [Mpc]")
        ax.set_ylabel(r"$|\xi(k_\perp,\Delta)|/|\xi(k_\perp,0)|$")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/radial_coherence.png", dpi=140)
        log.info("wrote %s/radial_coherence.png", OUTDIR)
    return d


def stage_run(ell, layers, load_q, weights):
    dl, rep = dl_off(ell, layers, load_q, BOX_L_MPC, weights)
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(f"{OUTDIR}/dl_off_corrected.npz", ell=ell, dl=dl,
             ell_min_box=rep["ell_min_box"],
             n_pairs=rep["n_pairs"],
             n_pairs_beyond_half_box=rep["n_pairs_beyond_half_box"])
    log.info("pairs: %d, of which %d beyond half a box (zeroed)",
             rep["n_pairs"], rep["n_pairs_beyond_half_box"])
    log.info("box band limit: l > %.0f", rep["ell_min_box"])

    dl_stitched = ...   # TODO load the blue curve from the stitched run
    frac = periodicity_split(dl, dl_stitched)
    log.info("periodicity contribution = stitched - corrected direct; "
             "this is the quantity the whole exercise was after")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        band = ell > rep["ell_min_box"]
        ax.plot(ell[band], dl[band], "g.-", label="direct, corrected (Limber failure)")
        ax.plot(ell, dl_stitched, "s-", c="tab:blue", label="stitched $P_{off}$")
        ax.plot(ell[band], frac[band], "r--", label="difference = periodicity")
        ax.axhline(0, ls=":", c="k")
        ax.axvline(rep["ell_min_box"], ls="--", c="grey")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell\ [\mu K^2]$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUTDIR}/dl_off_corrected.png", dpi=140)
    return dl, rep


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["convention", "coherence", "run"],
                   required=True)
    p.add_argument("--a-power", type=float, default=2.0,
                   help="fixed by --stage convention")
    args = p.parse_args()

    ell = np.geomspace(40, 3e4, 32)
    layers = build_layers()
    load_q = make_loader()
    weights = LayerWeights(nbar_e0_cgs=NBAR_E0_CGS, a_power=args.a_power)

    if args.stage == "convention":
        stage_convention(ell, layers, load_q, weights)
    elif args.stage == "coherence":
        stage_coherence(layers, load_q)
    else:
        stage_run(ell, layers, load_q, weights)


if __name__ == "__main__":
    main()
