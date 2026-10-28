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


def ref_for(r, results, v4ref):
    """
    Limber reference for one lightcone: native v4 lightcones are compared with
    the v4 coeval-direct Limber (script 46, same py21cmfast version, inputs and
    realisation) when one is given for that box size; everything else with the
    v3 coeval-direct from a --source coeval run (direct_for_box).
    """
    if v4ref is not None and r["kind"].startswith("native v4") and float(v4ref["box_len"]) == r["L"]:
        return np.asarray(v4ref["ell_direct"]), np.asarray(v4ref["Dl_direct"])
    return direct_for_box(results, r["L"])


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
    v4ref = dict(np.load(a.v4_direct)) if a.v4_direct else None
    if v4ref is not None:
        print(f"[ref] native v4 lightcones are compared with the v4 coeval-direct in {a.v4_direct} "
              f"(L = {float(v4ref['box_len']):.0f} Mpc)")
    boxes = sorted({r["L"] for r in results})
    bands = [(300.0, 1000.0), (2000.0, 4000.0)]

    print(f"\n{'lightcone':26s} {'L':>5s} {'band':>11s} {'direct':>8s} {'total':>8s} "
          f"{'diag':>8s} {'trunc':>8s} {'far':>8s} {'trunc/dir':>9s} {'total/dir':>9s}")
    for r in results:
        ed, Dd = ref_for(r, results, v4ref)
        for lo, hi in bands:
            dd = band(ed, Dd, lo, hi) if ed is not None else np.nan
            vals = [band(r["ell"], r[k], lo, hi) for k in ("D_total", "D_diag", "D_trunc", "D_far")]
            print(f"{r['kind']:26s} {r['L']:5.0f} {int(lo):5d}-{int(hi):<5d} {dd:8.3f} "
                  + " ".join(f"{v:8.3f}" for v in vals)
                  + f" {vals[2] / dd:9.2f} {vals[0] / dd:9.2f}")

    # ---------------- styling shared by all figures ----------------
    # Native lightcones are the "heroes" when present; the stitched ones become
    # faint context. A column whose native lightcones are all v4 (and has a v4
    # reference) shows only the v4 Limber curve -- the v3 one is a different
    # universe there. Legends go in a separate right-hand panel.
    has_native = any(r["kind"].startswith("native") for r in results)

    def style(r):
        faint = has_native and r["kind"].startswith("stitched")
        return dict(color=COLORS.get(r["kind"], "0.4"), alpha=0.35 if faint else 1.0,
                    lw_solid=1.2 if faint else 2.2, lw_dash=0.8 if faint else 1.3,
                    lw_step=1.0 if faint else 1.8, zorder=1 if faint else 3)

    def v4_only_column(L):
        nat = [r for r in results if r["L"] == L and r["kind"].startswith("native")]
        return (v4ref is not None and float(v4ref["box_len"]) == L and nat
                and all(r["kind"].startswith("native v4") for r in nat))

    def legend_panel(fig, ax_leg, axes, note=None):
        seen, hs, ls = set(), [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); hs.append(h); ls.append(l)
        ax_leg.axis("off")
        ax_leg.legend(hs, ls, loc="upper left", frameon=False, fontsize=8)
        if note:
            ax_leg.text(0.0, 0.0, note, transform=ax_leg.transAxes, fontsize=7.5,
                        va="bottom", ha="left", color="0.25", wrap=True)

    def stitched_note():
        if has_native and any(r["kind"].startswith("stitched") for r in results):
            return "Faint: stitched lightcones (v3), for context;\ntheir ratios are to the v3 Limber curve."
        return ""

    # -------- figure 1: D_ell, total vs truncated vs direct --------
    nb = len(boxes)
    fig = plt.figure(figsize=(5.0 * nb + 3.6, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, nb + 1, height_ratios=[2.2, 1], width_ratios=[1] * nb + [0.75])
    top_axes = []
    for c, L in enumerate(boxes):
        ax = fig.add_subplot(gs[0, c])
        axr = fig.add_subplot(gs[1, c], sharex=ax)
        top_axes.append(ax)
        v4col = v4_only_column(L)
        ed, Dd = direct_for_box(results, L)
        if ed is not None and not v4col:
            ax.plot(ed, Dd, color="k", lw=2.6, zorder=4,
                    label="Limber (coeval-direct), v3" if v4ref is not None else "Limber (coeval-direct)")
        if (v4ref is not None and float(v4ref["box_len"]) == L
                and any(r["kind"].startswith("native v4") for r in results if r["L"] == L)):
            ax.plot(v4ref["ell_direct"], v4ref["Dl_direct"], color="k", zorder=4,
                    lw=2.6 if v4col else 2.0, ls="-" if v4col else "-.",
                    label="Limber (coeval-direct), v4")
        for r in [r for r in results if r["L"] == L]:
            st = style(r)
            ed, Dd = ref_for(r, results, v4ref)
            ax.plot(r["ell"], r["D_trunc"], color=st["color"], alpha=st["alpha"], lw=st["lw_solid"],
                    zorder=st["zorder"], label=f"{r['kind']}: |Δχ| < L/2")
            ax.plot(r["ell"], r["D_total"], color=st["color"], alpha=st["alpha"], lw=st["lw_dash"],
                    ls="--", zorder=st["zorder"], label=f"{r['kind']}: all pairs")
            if ed is not None:
                ref = np.exp(np.interp(np.log(r["ell"]), np.log(ed), np.log(Dd),
                                       left=np.nan, right=np.nan))
                axr.plot(r["ell"], r["D_trunc"] / ref, color=st["color"], alpha=st["alpha"],
                         lw=st["lw_solid"], zorder=st["zorder"])
                axr.plot(r["ell"], r["D_total"] / ref, color=st["color"], alpha=st["alpha"],
                         lw=st["lw_dash"], ls="--", zorder=st["zorder"])
        for x in (ax, axr):
            x.axvline(3000, color="0.6", ls=":", lw=1)
            x.set_xscale("log")
        axr.axhline(1, color="k", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
        ax.set_title(f"L = {L:.0f} Mpc")
        ax.tick_params(labelbottom=False)
        axr.set_ylabel("ratio to own Limber")
        axr.set_xlabel(r"$\ell$")
        axr.set_ylim(0, 3)
    legend_panel(fig, fig.add_subplot(gs[:, nb]), top_axes,
                 "Solid, |Δχ| < L/2: only shell pairs closer\nthan half a box (box replicas removed).\n"
                 "Dashed: all shell pairs.\n\n" + stitched_note())
    fig.suptitle("Lightcone kSZ vs Limber, with and without box-replica pairs")
    fig.savefig(os.path.join(a.out, "limber_scope_Dl.png"), dpi=150)
    plt.close(fig)

    # -------- figures 2 and 3: cross-power vs separation --------
    def dchi_figure(fname, norm_kind, ylabel, xlabel_extra):
        fig = plt.figure(figsize=(5.0 * nb + 3.6, 3.4 * len(bands)), constrained_layout=True)
        gs = fig.add_gridspec(len(bands), nb + 1, width_ratios=[1] * nb + [0.75])
        axes = []
        for c, L in enumerate(boxes):
            for b, (lo, hi) in enumerate(bands):
                ax = fig.add_subplot(gs[b, c]); axes.append(ax)
                for r in [r for r in results if r["L"] == L]:
                    st = style(r)
                    if norm_kind == "direct":
                        er, Dr = ref_for(r, results, v4ref)
                        dd = band(er, Dr, lo, hi) if er is not None else 1.0
                    else:
                        dd = band(r["ell"], r["D_diag"], lo, hi)
                    x = r["dchi"] / L
                    bw = max(r["width"] / L, 0.05)
                    edges = np.arange(0, x.max() + bw, bw)
                    pb = band(r["ell"], r["Dpairs"], lo, hi)        # per pair, band-mean
                    sv = np.array([pb[(x >= edges[k]) & (x < edges[k + 1])].sum()
                                   for k in range(len(edges) - 1)]) / dd
                    ax.step(0.5 * (edges[:-1] + edges[1:]), sv, where="mid", color=st["color"],
                            alpha=st["alpha"], lw=st["lw_step"], zorder=st["zorder"], label=r["kind"])
                ax.axvspan(0, 0.5, color="0.92", zorder=0)
                for k in range(1, int(np.ceil(max(r["dchi"].max() for r in results) / L)) + 1):
                    ax.axvline(k, color="0.5", ls=":", lw=1)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_title(f"L = {L:.0f} Mpc, ℓ = {int(lo)}–{int(hi)}")
                ax.set_xlabel(r"$|\Delta\chi| / L_{\rm box}$" + xlabel_extra)
                ax.set_ylabel(ylabel)
        legend_panel(fig, fig.add_subplot(gs[:, nb]), axes,
                     "Shaded: |Δχ| < L/2, the shell pairs kept\nin the solid curves of the D_ell figure.\n"
                     "Dotted: whole box lengths.\n\n" + stitched_note())
        fig.savefig(os.path.join(a.out, fname), dpi=150)
        plt.close(fig)

    dchi_figure("limber_scope_dchi.png", "direct", "cross-power / own Limber", "")
    dchi_figure("limber_scope_dchi_selfnorm.png", "self", "cross-power / own D_diag", "")

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
    p.add_argument("--v4-direct", default=None,
                   help="npz from scripts/46_v4_coeval_limber.py (ell_direct, Dl_direct, box_len): "
                        "Limber reference used for native v4 lightcones at that box size")
    main(p.parse_args())
