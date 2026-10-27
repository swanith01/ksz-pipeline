#!/usr/bin/env python
"""
Script 45: how much of a lightcone's kSZ D_ell is Limber, how much is genuine
short-range cross-shell (non-Limber) power, and how much is box periodicity?

Input: one or more theta_groups*.npz files written by
    scripts/17_coherence_decomposition_fiducial.py --save-theta-width <Mpc>
(per-slice kSZ maps summed into uniform comoving-distance groups i, centred at
chi_i). No py21cmfast needed -- runs anywhere numpy/matplotlib are available.

For every pair of groups (i, j) the l-resolved cross-power D_ij(l) is computed
with EXACTLY the k-binning and single-chi_eff l conversion of
coherence_decomposition.decompose_p_total_diag_off (so sum over all pairs and
autos reproduces script 17's D_total). Then

    D_diag  = sum_i D_ii                                  (autos, Limber-like)
    D_near  = sum_{i<j, |chi_i - chi_j| <  L/2} D_ij      (physical cross-shell)
    D_far   = sum_{i<j, |chi_i - chi_j| >= L/2} D_ij      (contains box replicas)
    D_total = D_diag + D_near + D_far
    D_trunc = D_diag + D_near      <- "periodicity removed" estimator

Within half a box the lightcone is one continuous realisation, so D_near is
physical; at |dchi| >= L/2 a periodic lightcone correlates a slice with a copy
of itself, so D_far mixes physics with artefact. D_trunc is grouping-invariant
up to edge effects (it is the power of all map pairs within L/2), and it is the
quantity to compare with coeval-direct Limber:
    D_trunc ~ D_direct   ->  no non-Limber residual resolvable below L/2
    D_trunc - D_direct   ->  genuine short-range non-Limber term, which should
                             then be checked for convergence with box size.
Modes longer than the box, and correlations beyond L/2, are out of reach of any
finite-box lightcone -- that is the stated scope of this test.

The second figure resolves the cross-power by separation: for two l-bands,
sum of D_ij over pairs in each |dchi|/L bin, as a fraction of D_direct in that
band. A periodic lightcone shows spikes at |dchi| = L, 2L, ...; wrap-cycle
rotation should remove them.

Usage:
    python scripts/45_limber_scope_dchi.py data/products/theta_groups*.npz \
        --out data/plots/45_limber_scope
"""
import argparse
import os

import numpy as np

T_CMB_UK = 2.7255e6

COLORS = {  # fixed categorical order, assigned by lightcone kind (never by rank)
    "stitched, no rotation": "#1f77b4",
    "stitched, wrap-cycle": "#ff7f0e",
    "native v3 rectilinear": "#2ca02c",
    "native v4 rectilinear": "#d62728",
    "native v4 angular": "#9467bd",
}


def kind_of(d):
    src = str(d["source"])
    if src == "coeval":
        return "stitched, wrap-cycle" if int(d["wrap_cycle_seed"]) >= 0 else "stitched, no rotation"
    return {"native": "native v3 rectilinear", "native_v4": "native v4 rectilinear",
            "native_v4_ang": "native v4 angular"}.get(src, src)


def taper2d(N, alpha=0.25):
    """Separable Tukey (cosine-edge) window, used for non-periodic angular patches."""
    x = np.arange(N) / (N - 1)
    w = np.ones(N)
    e = alpha / 2
    lo, hi = x < e, x > 1 - e
    w[lo] = 0.5 * (1 - np.cos(np.pi * x[lo] / e))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (1 - x[hi]) / e))
    return np.outer(w, w)


def pair_spectra(theta, L, chi_eff, n_kbins=35, taper=False):
    """
    Same binning/normalisation as decompose_p_total_diag_off.
    L is the transverse map length used for ell = k*chi_eff: the box face for
    rectilinear maps, N*pix_angle*chi_eff for angular ones (script 17 saves it
    as map_len). taper=True apodises the (non-periodic) angular patch with a
    Tukey window and divides the power by mean(w^2).
    Returns ell (valid), Dauto[G, nl], Dpair[G, G, nl] (upper triangle i<j filled,
    already including the factor 2 for (i,j)+(j,i)).
    """
    N = theta.shape[0]
    G = theta.shape[-1]
    pix = L / N
    th = theta.astype(np.float64)
    th = th - th.mean(axis=(0, 1), keepdims=True)
    norm = (pix / N) ** 2
    if taper:
        w = taper2d(N)
        th = th * w[:, :, None]
        norm = norm / float(np.mean(w ** 2))
    that = np.fft.fft2(th, axes=(0, 1))
    dk = 2.0 * np.pi / (N * pix)
    k1 = np.fft.fftfreq(N) * N * dk
    kg = np.sqrt(k1[:, None] ** 2 + k1[None, :] ** 2)
    kmax_shifted = np.sqrt(2) * (N // 2) * dk          # same kg.max() as the fftshifted grid
    k_bins = np.logspace(np.log10(dk), np.log10(kmax_shifted * 0.9), n_kbins)
    kc = 0.5 * (k_bins[:-1] + k_bins[1:])
    digit = np.digitize(kg.ravel(), k_bins)            # bin b <-> digit == b+1
    nb = len(kc)
    counts = np.bincount(digit, minlength=nb + 2)[1:nb + 1].astype(float)

    def bin2d(a2d):
        s = np.bincount(digit, weights=a2d.ravel(), minlength=nb + 2)[1:nb + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, s / counts, np.nan)

    ell = kc * chi_eff
    fac = ell * (ell + 1.0) / (2.0 * np.pi) * T_CMB_UK ** 2 / chi_eff ** 2
    valid = (counts > 0) & (ell > 10)
    flat = that.reshape(N * N, G)
    Dauto = np.empty((G, valid.sum()))
    Dpair = np.zeros((G, G, valid.sum()))
    for i in range(G):
        Dauto[i] = (bin2d(norm * np.abs(that[:, :, i]) ** 2) * fac)[valid]
        for j in range(i + 1, G):
            c = norm * 2.0 * np.real(flat[:, i] * np.conj(flat[:, j])).reshape(N, N)
            Dpair[i, j] = (bin2d(c) * fac)[valid]
    return ell[valid], Dauto, Dpair


def analyse(path):
    d = dict(np.load(path, allow_pickle=True))
    L = float(d["box_len"]); chi_eff = float(d["chi_eff"])
    pix_angle = float(d["pix_angle"]) if "pix_angle" in d else 0.0
    Lmap = float(d["map_len"]) if "map_len" in d else L
    ang = pix_angle > 0
    th, chi = d["theta_uniform"], d["chi_uniform"]
    ell, Dauto, Dpair = pair_spectra(th, Lmap, chi_eff, taper=ang)
    G = len(chi)
    iu, ju = np.triu_indices(G, k=1)
    dchi = np.abs(chi[iu] - chi[ju])
    P = Dpair[iu, ju]                                   # (npairs, nl)
    near = dchi < 0.5 * L
    out = dict(
        path=path, kind=kind_of(d), L=L, N=int(d["hii_dim"]), chi_eff=chi_eff, ell=ell,
        D_diag=Dauto.sum(0), D_near=P[near].sum(0), D_far=P[~near].sum(0),
        dchi=dchi, Dpairs=P, width=float(d["group_width_mpc"]), G=G,
        ell_direct=np.asarray(d["ell_direct"]), Dl_direct=np.asarray(d["Dl_direct"]),
        direct_is_fresh=bool(d.get("direct_is_fresh", False)),
    )
    out["D_total"] = out["D_diag"] + out["D_near"] + out["D_far"]
    out["D_trunc"] = out["D_diag"] + out["D_near"]
    # internal check: total power is grouping-invariant -> compare with snapshot groups
    if "theta_snapshot" in d:
        e2, a2, p2 = pair_spectra(d["theta_snapshot"], Lmap, chi_eff, taper=ang)
        tot2 = a2.sum(0) + p2.sum((0, 1))
        rel = np.nanmax(np.abs(tot2 - out["D_total"]) / np.abs(out["D_total"]))
        print(f"[check] {os.path.basename(path)}: D_total uniform vs snapshot grouping, "
              f"max rel diff {rel:.2e} (should be <~1e-3; maps are stored as float32)")
    return out


def direct_for_box(results, L):
    """Prefer a coeval-source file's direct (fresh at this box, or fiducial closure)."""
    cands = [r for r in results if r["L"] == L and r["kind"].startswith("stitched")
             and len(r["ell_direct"])]
    if not cands:
        cands = [r for r in results if r["L"] == L and len(r["ell_direct"])]
    if not cands:
        return None, None
    r = sorted(cands, key=lambda r: not r["direct_is_fresh"])[0]
    return r["ell_direct"], r["Dl_direct"]


def band(ell, D, lo, hi):
    """Mean of D over ell in [lo, hi] along the last axis (float for 1-D input)."""
    D = np.asarray(D)
    m = (ell >= lo) & (ell <= hi)
    if not m.any():
        return np.nan if D.ndim == 1 else np.full(D.shape[:-1], np.nan)
    out = np.mean(D[..., m], axis=-1)
    return float(out) if D.ndim == 1 else out


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": False, "font.size": 9})
    os.makedirs(a.out, exist_ok=True)

    results = [analyse(p) for p in a.files]
    boxes = sorted({r["L"] for r in results})
    bands = [(300.0, 1000.0), (2000.0, 4000.0)]

    print(f"\n{'lightcone':26s} {'L':>5s} {'band':>11s} {'direct':>8s} {'total':>8s} "
          f"{'diag':>8s} {'trunc':>8s} {'far':>8s} {'trunc/dir':>9s} {'total/dir':>9s}")
    for r in results:
        ed, Dd = direct_for_box(results, r["L"])
        for lo, hi in bands:
            dd = band(ed, Dd, lo, hi) if ed is not None else np.nan
            vals = [band(r["ell"], r[k], lo, hi) for k in ("D_total", "D_diag", "D_trunc", "D_far")]
            print(f"{r['kind']:26s} {r['L']:5.0f} {int(lo):5d}-{int(hi):<5d} {dd:8.3f} "
                  + " ".join(f"{v:8.3f}" for v in vals)
                  + f" {vals[2] / dd:9.2f} {vals[0] / dd:9.2f}")

    # -------- figure 1: D_ell, total vs truncated vs direct --------
    fig, axs = plt.subplots(2, len(boxes), figsize=(5.2 * len(boxes), 7), sharex=True,
                            squeeze=False, gridspec_kw=dict(height_ratios=[2.2, 1]),
                            constrained_layout=True)
    for c, L in enumerate(boxes):
        ax, axr = axs[0, c], axs[1, c]
        ed, Dd = direct_for_box(results, L)
        if ed is not None:
            ax.plot(ed, Dd, color="k", lw=2.5, label="coeval-direct (Limber)")
        for r in [r for r in results if r["L"] == L]:
            col = COLORS.get(r["kind"], "0.4")
            ax.plot(r["ell"], r["D_trunc"], color=col, lw=2,
                    label=f"{r['kind']}: |Δχ| < L/2")
            ax.plot(r["ell"], r["D_total"], color=col, lw=1.2, ls="--",
                    label=f"{r['kind']}: all pairs")
            if ed is not None:
                ref = np.exp(np.interp(np.log(r["ell"]), np.log(ed), np.log(Dd),
                                       left=np.nan, right=np.nan))
                axr.plot(r["ell"], r["D_trunc"] / ref, color=col, lw=2)
                axr.plot(r["ell"], r["D_total"] / ref, color=col, lw=1.2, ls="--")
        ax.axvline(3000, color="0.6", ls=":", lw=1)
        axr.axvline(3000, color="0.6", ls=":", lw=1)
        axr.axhline(1, color="k", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
        ax.set_title(f"L = {L:.0f} Mpc")
        axr.set_xscale("log"); axr.set_ylabel("ratio to direct")
        axr.set_xlabel(r"$\ell$")
        axr.set_ylim(0, 3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Lightcone kSZ with box-replica pairs removed (|Δχ| < L/2, solid) "
                 "vs all pairs (dashed) vs Limber")
    fig.savefig(os.path.join(a.out, "limber_scope_Dl.png"), dpi=150)
    plt.close(fig)

    # -------- figure 2: cross-power vs separation --------
    fig, axs = plt.subplots(len(bands), len(boxes), figsize=(5.2 * len(boxes), 3.4 * len(bands)),
                            squeeze=False, constrained_layout=True)
    for c, L in enumerate(boxes):
        ed, Dd = direct_for_box(results, L)
        for b, (lo, hi) in enumerate(bands):
            ax = axs[b, c]
            dd = band(ed, Dd, lo, hi) if ed is not None else 1.0
            for r in [r for r in results if r["L"] == L]:
                x = r["dchi"] / L
                bw = max(r["width"] / L, 0.05)
                edges = np.arange(0, x.max() + bw, bw)
                pb = band(r["ell"], r["Dpairs"], lo, hi)        # per pair, band-mean
                s = np.array([pb[(x >= edges[k]) & (x < edges[k + 1])].sum()
                              for k in range(len(edges) - 1)]) / dd
                ax.step(0.5 * (edges[:-1] + edges[1:]), s, where="mid",
                        color=COLORS.get(r["kind"], "0.4"), lw=1.5, label=r["kind"])
            ax.axvspan(0, 0.5, color="0.92", zorder=0)
            for k in range(1, int(np.ceil(max(r["dchi"].max() for r in results) / L)) + 1):
                ax.axvline(k, color="0.5", ls=":", lw=1)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"L = {L:.0f} Mpc, ℓ = {int(lo)}–{int(hi)}")
            ax.set_xlabel(r"$|\Delta\chi| / L_{\rm box}$  (shaded: kept in $D_{\rm trunc}$)")
            ax.set_ylabel("cross-power / D_direct")
            ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.out, "limber_scope_dchi.png"), dpi=150)
    plt.close(fig)

    # -------- figure 3: cross-power vs separation, normalised to each lightcone's OWN diag
    # (needed for lightcones from a different realisation/astro model, e.g. v4) --------
    fig, axs = plt.subplots(len(bands), len(boxes), figsize=(5.2 * len(boxes), 3.4 * len(bands)),
                            squeeze=False, constrained_layout=True)
    for c, L in enumerate(boxes):
        for b, (lo, hi) in enumerate(bands):
            ax = axs[b, c]
            for r in [r for r in results if r["L"] == L]:
                x = r["dchi"] / L
                bw = max(r["width"] / L, 0.05)
                edges = np.arange(0, x.max() + bw, bw)
                pb = band(r["ell"], r["Dpairs"], lo, hi)
                own = band(r["ell"], r["D_diag"], lo, hi)
                s = np.array([pb[(x >= edges[k]) & (x < edges[k + 1])].sum()
                              for k in range(len(edges) - 1)]) / own
                ax.step(0.5 * (edges[:-1] + edges[1:]), s, where="mid",
                        color=COLORS.get(r["kind"], "0.4"), lw=1.5, label=r["kind"])
            ax.axvspan(0, 0.5, color="0.92", zorder=0)
            for k in range(1, int(np.ceil(max(r["dchi"].max() for r in results) / L)) + 1):
                ax.axvline(k, color="0.5", ls=":", lw=1)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"L = {L:.0f} Mpc, ℓ = {int(lo)}–{int(hi)}")
            ax.set_xlabel(r"$|\Delta\chi| / L_{\rm box}$")
            ax.set_ylabel("cross-power / own D_diag")
            ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.out, "limber_scope_dchi_selfnorm.png"), dpi=150)
    plt.close(fig)

    print(f"\n{'lightcone':26s} {'L':>5s} {'band':>11s} {'far/total':>9s} {'near/diag':>9s}"
          "   (internal ratios: independent of velocity units and of the Limber reference)")
    for r in results:
        for lo, hi in bands:
            tot, far = band(r["ell"], r["D_total"], lo, hi), band(r["ell"], r["D_far"], lo, hi)
            near, diag = band(r["ell"], r["D_near"], lo, hi), band(r["ell"], r["D_diag"], lo, hi)
            print(f"{r['kind']:26s} {r['L']:5.0f} {int(lo):5d}-{int(hi):<5d} "
                  f"{far / tot:9.2f} {near / diag:9.2f}")

    np.savez(os.path.join(a.out, "limber_scope.npz"),
             **{f"{i}__{k}": np.asarray(r[k]) for i, r in enumerate(results)
                for k in ("ell", "D_total", "D_diag", "D_near", "D_far", "D_trunc", "dchi")},
             files=np.array([r["path"] for r in results]),
             kinds=np.array([r["kind"] for r in results]),
             boxes=np.array([r["L"] for r in results]))
    print("wrote", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="theta_groups*.npz from script 17 --save-theta-width")
    p.add_argument("--out", default="data/plots/45_limber_scope")
    main(p.parse_args())
