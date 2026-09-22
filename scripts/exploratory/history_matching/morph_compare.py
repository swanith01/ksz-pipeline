#!/usr/bin/env python
"""
morph_compare.py -- do history-matched 21cmFAST parameter sets differ in morphology?

Takes the (zeta, log10 Tvir, R_BUBBLE_MAX) solutions from solve_history*.py, runs
ionisation boxes for each from the SAME initial conditions (same seed, box, threads),
and compares, at fixed redshift:

  1. P_ee(k)   power spectrum of the free-electron overdensity  delta_e = e/<e> - 1,
               e = x_HII (1+delta)  (the quantity in Gorce+ Eq. 3), plus ratio to the
               default-R_BUBBLE_MAX solution.
  2. MFP       distance from random ionised points along random rays to the first
               neutral cell (Mesinger & Furlanetto 2007).  Robust at any x_HII.
  3. CCL       connected-component labelling with CORRECT periodic merging
               (replaces the wrap-around step in 29May2026_Bubble_parameter_play.py,
               which pairs differently-masked face arrays).  Also reports the fraction
               of ionised volume in the largest component (percolation).

Usage (p21c_v3 env, 16 cores):
  python morph_compare.py --json solutions_400_64.json --z 9 8 7
  python morph_compare.py --plot-only            # replot from morph_out/results.pkl

Match --box-len/--hii-dim/--seed/--threads to the solver run, or the realisation
(and the history match) will not be the one that was fitted.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np


# =============================================================================
# Statistics (no 21cmFAST dependency)
# =============================================================================
def power_spectrum(field, box_len):
    """
    Spherically averaged P(k) [Mpc^3] of a periodic cubic field, in linear shells of
    width k_f = 2 pi / L.  Returns k_mean, P, N_independent_modes.
    """
    N = field.shape[0]
    dx = box_len / N
    fk = np.fft.fftn(field) * dx ** 3
    p3 = np.abs(fk) ** 2 / box_len ** 3
    kax = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    kk = np.sqrt(kax[:, None, None] ** 2 + kax[None, :, None] ** 2
                 + kax[None, None, :] ** 2)
    kf = 2 * np.pi / box_len
    edges = (np.arange(0, N // 2 + 1) + 0.5) * kf          # shells centred on n*kf
    idx = np.digitize(kk.ravel(), edges) - 1               # -1 => k < 0.5 kf (incl. k=0)
    nb = len(edges) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.bincount(idx[ok], weights=p3.ravel()[ok], minlength=nb) / cnt
        kmean = np.bincount(idx[ok], weights=kk.ravel()[ok], minlength=nb) / cnt
    return kmean, P, cnt / 2.0          # /2: +k and -k are the same mode


def label_periodic(mask):
    """
    6-connected labelling of a boolean cube with periodic wrap-around.
    Labels of components touching opposite faces are merged with union-find using
    ALIGNED face pairs (both cells must be ionised).  Returns (labels, n_components)
    with labels 0 = neutral, 1..n = components.
    """
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ax in range(3):
        a = np.take(lab, -1, axis=ax)
        b = np.take(lab, 0, axis=ax)
        m = (a > 0) & (b > 0)                       # same mask for both faces
        if not m.any():
            continue
        for u, v in np.unique(np.stack([a[m], b[m]], axis=1), axis=0):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(i) for i in range(n + 1)])
    merged = roots[lab]
    uniq = np.unique(merged[merged > 0])
    remap = np.zeros(roots.max() + 1, dtype=int)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[merged], len(uniq)


def mfp_distribution(xH, box_len, thresh=0.5, n_samples=100000, seed=0):
    """
    Distance [Mpc] from random points in ionised cells, along random isotropic
    directions, to the first neutral cell (step = half a cell, capped at L/2).
    Returns (distances_of_rays_that_hit, fraction_that_never_hit_within_L/2).
    """
    rng = np.random.default_rng(seed)
    N = xH.shape[0]
    dx = box_len / N
    ion = xH < thresh
    cells = np.argwhere(ion)
    if len(cells) == 0:
        return np.array([]), 1.0
    start = cells[rng.integers(0, len(cells), n_samples)] + rng.random((n_samples, 3))
    d = rng.normal(size=(n_samples, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    step = 0.5
    max_steps = int(N / 2 / step)
    dist = np.full(n_samples, np.nan)
    alive = np.arange(n_samples)
    for s in range(1, max_steps + 1):
        pos = start[alive] + d[alive] * (s * step)
        ci = np.floor(pos).astype(int) % N
        hit = ~ion[ci[:, 0], ci[:, 1], ci[:, 2]]
        # crossing lies somewhere in ((s-1)*step, s*step]: place it uniformly there
        # (removes the half-cell quantisation that otherwise combs the histogram)
        dist[alive[hit]] = (s - rng.random(int(hit.sum()))) * step * dx
        alive = alive[~hit]
        if len(alive) == 0:
            break
    never = np.isnan(dist)
    return dist[~never], float(never.mean())


def summarize_box(xH, delta, box_len, n_mfp):
    """All morphology statistics for one (parameter set, redshift) box."""
    x = 1.0 - xH
    e = x * (1.0 + delta)
    out = dict(xbar=float(x.mean()))
    dee = e / e.mean() - 1.0
    dxx = x / x.mean() - 1.0
    k, Pee, Nm = power_spectrum(dee, box_len)
    _, Pxx, _ = power_spectrum(dxx, box_len)
    out.update(k=k, Pee=Pee, Pxx=Pxx, Nmodes=Nm)

    lab, n = label_periodic(xH < 0.5)
    N = xH.shape[0]
    cell_size = box_len / N
    if n > 0:
        vols_cells = np.bincount(lab.ravel())[1:]
        vols_mpc3 = vols_cells * cell_size ** 3
        radii_mpc = (3.0 * vols_mpc3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        out.update(n_comp=int(n), giant_frac=float(vols_cells.max() / vols_cells.sum()),
                  bubble_radii=radii_mpc, bubble_vols=vols_mpc3)
    else:
        out.update(n_comp=0, giant_frac=0.0,
                  bubble_radii=np.array([]), bubble_vols=np.array([]))

    dist, never = mfp_distribution(xH, box_len, n_samples=n_mfp)
    out.update(mfp=dist, mfp_never=never,
               mfp_mean=float(dist.mean()) if len(dist) else np.nan,
               mfp_med=float(np.median(dist)) if len(dist) else np.nan)
    return out


# =============================================================================
# 21cmFAST driver (lazy import so the statistics above can be tested standalone)
# =============================================================================
def load_sets(json_paths, max_resid):
    """One entry per R_BUBBLE_MAX (lowest residual wins); None -> the 15 Mpc default."""
    best = {}
    for p in json_paths:
        for row in json.load(open(p)):
            if row["max_abs_dx"] > max_resid:
                continue
            r = row["rmax"]
            if r not in best or row["max_abs_dx"] < best[r]["max_abs_dx"]:
                best[r] = row
    sets = {}
    for r, row in sorted(best.items(), key=lambda kv: 15.0 if kv[0] is None else kv[0]):
        label = "Rmax=15 (default)" if r is None else f"Rmax={r:g}"
        sets[label] = dict(zeta=row["zeta"], tvir=row["log10_Tvir"], rmax=r,
                           resid=row["max_abs_dx"])
    return sets


def compute(a):
    import py21cmfast as p21c

    sets = load_sets(a.json, a.max_resid)
    if len(sets) < 2:
        raise SystemExit(f"Need >=2 solutions with max_abs_dx <= {a.max_resid}; got "
                         f"{list(sets)}")
    print("Parameter sets:")
    for lab, s in sets.items():
        print(f"  {lab:20s} zeta={s['zeta']:.3f}  log10Tvir={s['tvir']:.4f}  "
              f"solver resid={s['resid']:.4f}")

    os.makedirs(a.direc, exist_ok=True)
    up = p21c.UserParams(HII_DIM=a.hii_dim, BOX_LEN=a.box_len, N_THREADS=a.threads)
    ic = p21c.initial_conditions(user_params=up, random_seed=a.seed, direc=a.direc)
    fl = p21c.FlagOptions()
    pf = {z: p21c.perturb_field(redshift=z, init_boxes=ic, direc=a.direc) for z in a.z}

    res = {lab: {} for lab in sets}
    for lab, s in sets.items():
        kw = dict(HII_EFF_FACTOR=s["zeta"], ION_Tvir_MIN=s["tvir"])
        if s["rmax"] is not None:
            kw["R_BUBBLE_MAX"] = s["rmax"]
        astro = p21c.AstroParams(**kw)
        for z in a.z:
            t0 = time.time()
            ib = p21c.ionize_box(redshift=z, init_boxes=ic, perturbed_field=pf[z],
                                 astro_params=astro, flag_options=fl, write=False)
            xH = np.asarray(ib.xH_box, dtype=float)
            delta = np.asarray(pf[z].density, dtype=float)
            res[lab][z] = summarize_box(xH, delta, a.box_len, a.n_mfp)
            print(f"  {lab:20s} z={z:4.1f}  done in {time.time() - t0:5.1f}s", flush=True)

    meta = dict(box_len=a.box_len, hii_dim=a.hii_dim, seed=a.seed, z=list(a.z),
                sets=sets)
    os.makedirs(a.outdir, exist_ok=True)
    pickle.dump(dict(meta=meta, res=res), open(os.path.join(a.outdir, "results.pkl"), "wb"))
    return meta, res


# =============================================================================
# Report + plots
# =============================================================================
def k_ratio(r, ref, k0):
    i = int(np.nanargmin(np.abs(r["k"] - k0)))
    return r["Pee"][i] / ref["Pee"][i], r["k"][i]


def report(meta, res):
    labs = list(res)
    ref = next((l for l in labs if "default" in l), labs[0])
    dx = meta["box_len"] / meta["hii_dim"]
    print(f"\nBox {meta['box_len']:g} Mpc / {meta['hii_dim']}^3  (dx = {dx:.2f} Mpc), "
          f"reference = {ref}")
    print(f"{'set':20s} {'z':>4s} {'xHII':>6s} {'giant':>6s} {'Ncomp':>6s} "
          f"{'MFPmean':>8s} {'MFPmed':>7s} {'never':>6s} {'Pee/ref@0.05':>13s} "
          f"{'Pee/ref@0.2':>12s}")
    for lab in labs:
        for z in meta["z"]:
            r, rr = res[lab][z], res[ref][z]
            q1, _ = k_ratio(r, rr, 0.05)
            q2, _ = k_ratio(r, rr, 0.2)
            print(f"{lab:20s} {z:4.1f} {r['xbar']:6.3f} {r['giant_frac']:6.2f} "
                  f"{r['n_comp']:6d} {r['mfp_mean']:8.1f} {r['mfp_med']:7.1f} "
                  f"{r['mfp_never']:6.2f} {q1:13.3f} {q2:12.3f}")
    print("\nMFP in comoving Mpc; 'never' = fraction of rays not stopped within L/2; "
          "'giant' = ionised volume in the largest connected component.")
    print("Same initial conditions for every set, so ratios are paired comparisons.")


def make_plots(meta, res, outdir):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style = {
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 13, "axes.labelsize": 14,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.fontsize": 10, "savefig.dpi": 300, "savefig.bbox": "tight",
    }
    labs = list(res)
    ref = next((l for l in labs if "default" in l), labs[0])
    zs = meta["z"]
    cmap = plt.get_cmap("viridis")
    others = [l for l in labs if l != ref]
    col = {l: cmap(i / max(1, len(others) - 1)) for i, l in enumerate(others)}
    col[ref] = "k"

    with mpl.rc_context(style):
        fig, axes = plt.subplots(4, len(zs), figsize=(4.6 * len(zs), 14.5),
                                 sharex="row", squeeze=False,
                                 gridspec_kw=dict(hspace=0.1, wspace=0.05))
        GIANT_THRESH = 0.5   # above this, one component dominates -- CCL "bubble
                             # size" stops being a meaningful question (see MFP
                             # row instead); percolation transition in this data
                             # happens between the z=10 and z=9 columns
        for j, z in enumerate(zs):
            a0, a1, a2, a3 = axes[0, j], axes[1, j], axes[2, j], axes[3, j]
            rr = res[ref][z]
            for lab in labs:
                r = res[lab][z]
                lw = 2.2 if lab == ref else 1.5
                a0.loglog(r["k"], r["Pee"], color=col[lab], lw=lw, label=lab)
                a1.semilogx(r["k"], r["Pee"] / rr["Pee"], color=col[lab], lw=lw)
                if len(r["mfp"]):
                    bins = np.logspace(np.log10(meta["box_len"] / meta["hii_dim"]),
                                       np.log10(meta["box_len"] / 2), 16)
                    h, e = np.histogram(r["mfp"], bins=bins)
                    dlnr = np.diff(np.log(e))
                    pdf = h / max(h.sum(), 1) / dlnr
                    a2.semilogx(np.sqrt(e[:-1] * e[1:]), pdf, color=col[lab], lw=lw)
                if r["giant_frac"] <= GIANT_THRESH and len(r["bubble_radii"]):
                    rb = r["bubble_radii"]
                    vb = r["bubble_vols"]
                    bbins = np.logspace(np.log10(max(rb.min(), meta["box_len"] / meta["hii_dim"])),
                                        np.log10(meta["box_len"] / 2), 16)
                    h, e = np.histogram(rb, bins=bbins, weights=vb)
                    dlnr = np.diff(np.log(e))
                    pdf = h / max(h.sum(), 1) / dlnr
                    a3.semilogx(np.sqrt(e[:-1] * e[1:]), pdf, color=col[lab], lw=lw)
            band = np.sqrt(2.0 / np.maximum(rr["Nmodes"], 1))
            a1.fill_between(rr["k"], 1 - band, 1 + band, color="gray", alpha=0.2,
                            lw=0, label="unpaired sample-variance scale")
            a1.axhline(1, color="gray", lw=0.8)
            a0.set_title(rf"$z={z:g}$  ($\bar x_{{\rm HII}}\approx"
                         rf"{rr['xbar']:.2f}$)")
            a1.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
            a2.set_xlabel(r"distance to neutral cell [cMpc]")
            a3.set_xlabel(r"bubble radius $R$ [cMpc]")
            if all(res[lab][z]["giant_frac"] > GIANT_THRESH for lab in labs):
                a3.text(0.5, 0.5, "percolated\n(giant $>$ %d%%)\nsee MFP row instead"
                        % int(100 * GIANT_THRESH),
                        transform=a3.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")
                a3.set_xticks([]); a3.set_yticks([])
            if j == 0:
                a0.set_ylabel(r"$P_{ee}(k)\ [{\rm Mpc}^3]$")
                a1.set_ylabel(r"$P_{ee}/P_{ee}^{\rm ref}$")
                a2.set_ylabel(r"MFP PDF  $dP/d\ln R$")
                a3.set_ylabel(r"bubble-size PDF  $dP/d\ln R$")
            else:
                for a in (a0, a1, a2, a3):
                    a.tick_params(labelleft=False)
            a1.set_ylim(0.5, 1.5)
        axes[0, 0].legend(loc="lower left")
        axes[1, 0].legend(loc="lower right", fontsize=8)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"morphology_compare.{ext}"))
        plt.close(fig)
    print(f"Saved {outdir}/morphology_compare.{{pdf,png}}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", default=["solutions_400_64.json"])
    ap.add_argument("--max-resid", type=float, default=0.02,
                    help="keep only solutions whose solver max|dx_e| is below this")
    ap.add_argument("--z", type=float, nargs="+", default=[9.0, 8.0, 7.0])
    ap.add_argument("--box-len", type=float, default=400.0)
    ap.add_argument("--hii-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
    ap.add_argument("--outdir", default="morph_out")
    ap.add_argument("--n-mfp", type=int, default=100000)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()

    if a.plot_only:
        d = pickle.load(open(os.path.join(a.outdir, "results.pkl"), "rb"))
        meta, res = d["meta"], d["res"]
    else:
        meta, res = compute(a)
    report(meta, res)
    make_plots(meta, res, a.outdir)


if __name__ == "__main__":
    main()
