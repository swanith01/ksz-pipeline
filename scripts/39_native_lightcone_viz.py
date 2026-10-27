#!/usr/bin/env python
"""
Script 39: look at NATIVE 21cmFAST lightcones and measure their field units.
No kSZ.

Three native lightcones, from two code versions:
    v4 rectilinear   py21cmfast 4.1  RectilinearLightconer        (env p21c_v41)
    v4 angular       py21cmfast 4.1  AngularLightconer.like_rectilinear
    v3 rectilinear   py21cmfast 3.3.1 run_lightcone()              (env p21c_v3)
v3 has no angular lightconer.

Why measure units instead of assuming them: the velocity field is stored in a
different convention in each place we have looked --
  * v3 coeval  lowres_vz      = IC displacement Psi; this pipeline converts with
                                v_pec[km/s] = Psi * D f H/(1+z)   (coeval/velocity.py)
  * v4 coeval  velocity_z     = comoving-coordinate rate dx/dt in Mpc/s, so
                                v_pec = velocity_z/(1+z)  (ksz-lae-xcorr note, 6 Oct 2026)
  * v3 native lightcone       = native_lightcone.py uses raw/H0 as Mpc/s, explicitly
                                NOT validated
  * v4 native lightcone       = script 07 took raw as Mpc/s; its rms (2.1e-17 Mpc/s
                                = ~650 km/s) looks ~(1+z) too big
-- and a native lightcone need not match the coeval box of the same version.

THE VELOCITY TEST (rectilinear lightcones). A chunk of a rectilinear lightcone
exactly L_box long along the LOS is a periodic N^3 box (the transverse axes are
periodic, and the LOS is tiled). From the chunk's own density we predict the
linear LOS peculiar velocity in km/s,
        v_k = i a H f delta_k k_z / k^2       (continuity; f = Om(z)^0.55)
smooth both fields (Gaussian, R = 15 Mpc, where linear theory is good at z>5),
and regress   raw = A(z) * v_pred .   A(z) is the unit factor in "raw units per
km/s of peculiar velocity", measured at every chunk redshift. Each hypothesis
about the raw field maps raw to v_pec = raw * g_h(z); the right one gives
A*g_h = 1 at every z:
    H1 v_pec in km/s                                    g = 1
    H2 v_pec in Mpc/s                                   g = 3.0857e19
    H3 comoving rate dx/dt in Mpc/s (v4-coeval-like)    g = 3.0857e19/(1+z)
    H4 IC displacement Psi (v3-coeval-like)             g = D f H/(1+z)
    H5 native_lightcone.py v3 convention, raw/H0 Mpc/s  g = 3.0857e19/H0
The corr(raw, v_pred) is printed too: it should be close to +1 (or -1, which
means only a sign-convention flip). A low correlation means the comparison
itself failed, and A should not be trusted.
Caveat (from the LAE note): this uses the lightcone's own density, so check the
density separately first -- mean (0 for delta, 1 for 1+delta), min >= -1,
std growing toward low z. Those are printed and plotted.

ANGULAR lightcone: not a periodic grid, so no regression. Instead its velocity
rms is compared slice by slice with the v4 rectilinear one (same inputs): ratio
~1 means the same units. The angular pixel ordering is checked by correlating the
angular face at z_min with the rectilinear face under all 8 flips/transposes.

Stages (the plot stage needs only numpy/scipy/matplotlib/astropy):
  --code v4 --stage run   -> <out_dir>/native_lightcones_v4.npz  (env p21c_v41)
  --code v3 --stage run   -> <out_dir>/native_lightcones_v3.npz  (env p21c_v3)
  --stage plot            reads whichever npz files exist and makes:
      native_lightcones_los_<lab>.png   LOS slices: delta, x_HI, raw v_los,
                                        dotted lines every L_box from chi_min
      native_lightcones_faces.png       density faces at z = 6, 8, 10
      native_lightcones_units.png       velocity unit test + density checks
"""
import argparse
import glob
import os

import numpy as np

MPC_KM = 3.0857e19          # km per Mpc
LABELS = {"v4_rect": "v4 rectilinear", "v4_ang": "v4 angular", "v3_rect": "v3 rectilinear"}


# ============================================================================
# stage: run
# ============================================================================
def run_v4(a, npz_path):
    from ksz_pipeline.ksz.lightcone_v4 import (build_inputs, build_node_redshifts,
                                                run_rectilinear, run_angular)
    import py21cmfast as p21c
    print("py21cmfast", p21c.__version__)
    node_z = build_node_redshifts(a.z_min, a.z_max, n_nodes=a.n_nodes)
    inputs = build_inputs(a.random_seed, a.HII_DIM, a.BOX_LEN, node_z,
                          HII_EFF_FACTOR=a.HII_EFF_FACTOR, N_THREADS=a.n_threads)
    q = ("density", "velocity_z", "neutral_fraction")
    save = meta(a, "v4")
    for lab in ("v4_rect", "v4_ang"):
        if lab == "v4_rect":
            lc, f = run_rectilinear(inputs, a.z_min, a.z_max, a.cache_dir, quantities=q)
        else:
            lc, f = run_angular(inputs, a.z_min, a.z_min, a.z_max, a.cache_dir, quantities=q)
        put(save, lab, lc.lightcone_redshifts, lc.lightcone_distances,
            f["density"], f["neutral_fraction"], f["velocity_z"])
    np.savez_compressed(npz_path, **save)
    print("saved", npz_path)


def run_v3(a, npz_path):
    import py21cmfast as p21c
    print("py21cmfast", p21c.__version__)
    up = dict(HII_DIM=int(a.HII_DIM), BOX_LEN=float(a.BOX_LEN), N_THREADS=int(a.n_threads))
    os.makedirs(a.cache_dir, exist_ok=True)
    lc = p21c.run_lightcone(
        redshift=float(a.z_min), max_redshift=float(a.z_max),
        # brightness_temp must be requested in v3.3.1 or the LightCone axes crash
        # (see native_lightcone.py); it is not used here.
        lightcone_quantities=("brightness_temp", "density", "xH_box", "velocity"),
        user_params=p21c.UserParams(**up),
        astro_params=p21c.AstroParams(HII_EFF_FACTOR=float(a.HII_EFF_FACTOR)),
        random_seed=int(a.random_seed),
        direc=a.cache_dir,
    )
    z = np.asarray(lc.lightcone_redshifts, dtype=float)
    keep = z <= a.z_max
    save = meta(a, "v3")
    put(save, "v3_rect", z[keep], np.asarray(lc.lightcone_distances)[keep],
        np.asarray(lc.density)[..., keep], np.asarray(lc.xH_box)[..., keep],
        np.asarray(lc.velocity)[..., keep])
    np.savez_compressed(npz_path, **save)
    print("saved", npz_path)


def meta(a, code):
    return dict(code=code, BOX_LEN=a.BOX_LEN, HII_DIM=a.HII_DIM, z_min=a.z_min,
                z_max=a.z_max, random_seed=a.random_seed)


def put(save, lab, z, chi, dens, xhi, v):
    save[f"{lab}__z"] = np.asarray(z, dtype=float)
    save[f"{lab}__chi"] = np.asarray(chi, dtype=float)
    save[f"{lab}__density"] = np.asarray(dens, dtype=np.float32)
    save[f"{lab}__xHI"] = np.asarray(xhi, dtype=np.float32)
    save[f"{lab}__v_raw"] = np.asarray(v, dtype=np.float32)
    print(f"[{lab}] shape {np.shape(dens)}, z {np.min(z):.2f}-{np.max(z):.2f}")


# ============================================================================
# cosmology helpers (standalone, same formulas as coeval/velocity.py)
# ============================================================================
def _cosmo():
    from astropy.cosmology import Planck18
    return Planck18


def growth_factor(z):
    from scipy.integrate import quad
    c = _cosmo()
    f = lambda zp: (1.0 + zp) / c.H(zp).value ** 3
    return c.H(z).value * quad(f, z, np.inf)[0] / (c.H(0).value * quad(f, 0, np.inf)[0])


def growth_rate(z):
    return _cosmo().Om(z) ** 0.55


def g_hypotheses(z):
    """v_pec[km/s] = raw * g_h(z) under each hypothesis."""
    c = _cosmo()
    H0 = c.H0.value
    return {
        "H1 v_pec km/s": 1.0,
        "H2 v_pec Mpc/s": MPC_KM,
        "H3 dx/dt Mpc/s, /(1+z)": MPC_KM / (1 + z),
        "H4 IC Psi, *DfH/(1+z)": growth_factor(z) * growth_rate(z) * c.H(z).value / (1 + z),
        "H5 raw/H0 as Mpc/s": MPC_KM / H0,
    }


# ============================================================================
# diagnostics
# ============================================================================
def sorted_lc(d, lab):
    chi = d[f"{lab}__chi"]; o = np.argsort(chi)
    return (d[f"{lab}__z"][o], chi[o], d[f"{lab}__density"][..., o],
            d[f"{lab}__xHI"][..., o], d[f"{lab}__v_raw"][..., o])


def density_is_1plus(dens):
    return float(np.mean(dens)) > 0.5


def density_report(lab, z, dens):
    one_plus = density_is_1plus(dens)
    dl = dens - 1.0 if one_plus else dens
    flat = dl.reshape(-1, dl.shape[-1])
    mean, std, mn = flat.mean(0), flat.std(0), flat.min(0)
    print(f"[density:{lab}] global mean {np.mean(dens):+.3f} -> stored as "
          f"{'1+delta' if one_plus else 'delta'}; std z_min->z_max "
          f"{std[0]:.3f}->{std[-1]:.3f} (should fall with z); min delta {mn.min():+.3f} "
          f"({'OK' if mn.min() >= -1.0001 else 'BELOW -1: not an evolved delta'})")
    return dict(mean=mean, std=std, min=mn, one_plus=one_plus)


def velocity_regression(lab, z, chi, dens, v, L, N, R=15.0):
    cell = L / N
    dchi = float(np.median(np.diff(chi)))
    nL = int(round(L / dchi))
    if abs(nL * dchi - L) > 0.5 * dchi or abs(dchi - cell) > 0.05 * cell:
        print(f"[velocity:{lab}] slice spacing {dchi:.3f} Mpc vs cell {cell:.3f}: "
              f"chunks are not periodic boxes, skipping the regression")
        return None
    one_plus = density_is_1plus(dens)
    c = _cosmo()
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=cell)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    W = np.exp(-0.5 * K2 * R ** 2)
    K2s = K2.copy(); K2s[0, 0, 0] = 1.0
    rows = []
    for s in range(0, len(chi) - nL + 1, max(nL // 2, 1)):
        D = dens[..., s:s + nL].astype(np.float64)
        D = D / D.mean() - 1.0 if one_plus else D - D.mean()
        V = v[..., s:s + nL].astype(np.float64); V = V - V.mean()
        zc = float(np.interp(chi[s] + 0.5 * L, chi, z))
        aHf = c.H(zc).value / (1 + zc) * growth_rate(zc)                # km/s/Mpc
        vk = 1j * aHf * np.fft.fftn(D) * KZ / K2s; vk[0, 0, 0] = 0
        vp = np.fft.ifftn(vk * W).real                                   # km/s
        vr = np.fft.ifftn(np.fft.fftn(V) * W).real                       # raw units
        r = float(np.sum(vr * vp) / np.sqrt(np.sum(vr ** 2) * np.sum(vp ** 2)))
        A = float(np.sum(vr * vp) / np.sum(vp ** 2))                     # raw per km/s
        rows.append((zc, r, A, float(np.std(vp))))
    zc, r, A, vp_rms = map(np.array, zip(*rows))
    print(f"[velocity:{lab}] linear-theory regression, {len(zc)} chunks of L={L:.0f} Mpc, "
          f"R={R:.0f} Mpc smoothing. corr(raw, v_pred): {r.min():+.2f} .. {r.max():+.2f}"
          f"{'  (negative: sign convention flipped)' if np.median(r) < 0 else ''}")
    print(f"[velocity:{lab}]   smoothed linear v_pec rms {vp_rms.min():.0f}-{vp_rms.max():.0f} km/s")
    absA = np.abs(A)
    res = {}
    for h in g_hypotheses(zc[0]).keys():
        prod = np.array([absA[i] * g_hypotheses(zc[i])[h] for i in range(len(zc))])
        res[h] = prod
    best = min(res, key=lambda h: abs(np.log10(np.median(res[h]))) +
               np.log10(res[h].max() / res[h].min()))
    for h, prod in res.items():
        flag = "  <== best" if h == best else ""
        print(f"[velocity:{lab}]   {h:26s} A*g = median {np.median(prod):.3g}, "
              f"spread max/min {prod.max() / prod.min():.2f}{flag}")
    ok = abs(np.log10(np.median(res[best]))) < 0.15 and res[best].max() / res[best].min() < 1.5 \
        and np.median(np.abs(r)) > 0.7
    print(f"[velocity:{lab}] VERDICT: {best}"
          f"{'' if ok else '  (WEAK: A*g not ~1 and flat, or low correlation -- inspect the plot)'}")
    return dict(z=zc, r=r, A=A, prod=res, best=best)


ORIENT = {
    "identity": lambda x: x, "transpose": lambda x: x.T,
    "flip_x": lambda x: x[::-1, :], "flip_y": lambda x: x[:, ::-1],
    "rot180": lambda x: x[::-1, ::-1], "T_flip_x": lambda x: x.T[::-1, :],
    "T_flip_y": lambda x: x.T[:, ::-1], "T_rot180": lambda x: x.T[::-1, ::-1],
}


def corr(a, b):
    a = np.asarray(a, np.float64) - np.mean(a); b = np.asarray(b, np.float64) - np.mean(b)
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else np.nan


def angular_checks(L4, A4):
    zr, cr, dr, _, vr = L4; za, ca, da, _, va = A4
    kr, ka = int(np.argmin(zr)), int(np.argmin(za))
    print(f"[angular] orientation: angular face z={za[ka]:.3f} vs rectilinear z={zr[kr]:.3f}; "
          f"name = operation applied to the ANGULAR face")
    out = {}
    for name, R_, A_ in (("density", dr, da), ("velocity", vr, va)):
        rr = {k: corr(R_[..., kr], f(A_[..., ka])) for k, f in ORIENT.items()}
        best = max(rr, key=lambda k: abs(rr[k]) if np.isfinite(rr[k]) else -1)
        print(f"[angular]   {name:8s} best={best} ({rr[best]:+.3f}) | " +
              "  ".join(f"{k}={v:+.2f}" for k, v in rr.items()))
        out[name] = rr
    rms_r = np.sqrt(np.mean(vr.astype(np.float64) ** 2, axis=(0, 1)))
    rms_a = np.sqrt(np.mean(va.astype(np.float64) ** 2, axis=(0, 1)))
    ratio = rms_a / np.interp(ca, cr, rms_r)   # match slices by comoving distance
    print(f"[angular] velocity rms angular/rectilinear: median {np.median(ratio):.3f} "
          f"(range {ratio.min():.3f}-{ratio.max():.3f}); ~1 => same units as v4 rectilinear")
    out["rms_ratio"] = (za, ratio)
    return out


# ============================================================================
# stage: plot
# ============================================================================
def plot_stage(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": False, "font.size": 9})

    d = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "native_lightcones_v*.npz"))):
        d.update({k: v for k, v in np.load(p).items()})
        print("loaded", p)
    labs = [l for l in LABELS if f"{l}__z" in d]
    if not labs:
        raise SystemExit("no native_lightcones_v*.npz in " + a.out_dir)
    L = float(d["BOX_LEN"]); N = int(d["HII_DIM"]); cell = L / N
    LC = {lab: sorted_lc(d, lab) for lab in labs}

    dens_rep, vel_rep = {}, {}
    for lab in labs:
        z, chi, dens, _, v = LC[lab]
        dens_rep[lab] = density_report(lab, z, dens)
        if lab.endswith("rect"):
            vel_rep[lab] = velocity_regression(lab, z, chi, dens, v, L, N)
    ang = angular_checks(LC["v4_rect"], LC["v4_ang"]) if {"v4_rect", "v4_ang"} <= set(labs) else None

    # ---------------- LOS slices, one figure per lightcone ----------------
    for lab in labs:
        z, chi, dens, xhi, v = LC[lab]
        dl = dens - 1.0 if dens_rep[lab]["one_plus"] else dens
        fig, axs = plt.subplots(3, 1, figsize=(12, 7.5), constrained_layout=True)
        for ax, (arr, title, cmap, sym) in zip(axs, (
                (dl, r"$\delta$", "RdBu_r", True),
                (xhi, r"$x_{\rm HI}$", "Greys", False),
                (v, r"$v_{\rm los}$, raw units (see units figure)", "RdBu_r", True))):
            sl = arr[:, N // 2, :]
            if sym:
                m = float(np.nanpercentile(np.abs(sl), 99)); vmin, vmax = -m, m
            else:
                vmin, vmax = 0.0, 1.0
            if lab.endswith("rect"):
                y = (np.arange(N) + 0.5) * cell
                im = ax.pcolormesh(chi, y, sl, cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading="nearest", rasterized=True)
                ax.set_ylabel("transverse [Mpc]")
            else:
                th = (np.arange(N) - (N - 1) / 2) * cell / chi.min()   # assumed pixel angle
                X, TH = np.meshgrid(chi, th)
                im = ax.pcolormesh(X, X * TH, sl, cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading="nearest", rasterized=True)
                ax.set_ylabel(r"$\chi\,\theta$ [Mpc]")
            for k in range(1, int((chi.max() - chi.min()) / L) + 1):
                ax.axvline(chi.min() + k * L, color="k", ls=":", lw=1)
            ax.set_title(title, loc="left")
            fig.colorbar(im, ax=ax, pad=0.01, aspect=12)
            sec = ax.secondary_xaxis("top", functions=(
                lambda c, chi=chi, z=z: np.interp(c, chi, z),
                lambda zz, chi=chi, z=z: np.interp(zz, z, chi)))
            sec.set_xlabel("z", fontsize=8); sec.tick_params(labelsize=7)
        axs[-1].set_xlabel(r"comoving distance $\chi$ [Mpc]   (dotted: every $L_{\rm box}$ from $\chi_{\rm min}$)")
        fig.suptitle(f"Native {LABELS[lab]} lightcone, {L:.0f} Mpc / {N}³, seed {int(d['random_seed'])}")
        fig.savefig(os.path.join(a.out_dir, f"native_lightcones_los_{lab}.png"), dpi=150)
        plt.close(fig)

    # ---------------- faces ----------------
    zs = [zt for zt in (6.0, 8.0, 10.0) if all(LC[l][0].min() <= zt <= LC[l][0].max() for l in labs)]
    fig, axs = plt.subplots(len(labs), len(zs), figsize=(3.4 * len(zs), 3.2 * len(labs)),
                            constrained_layout=True, squeeze=False)
    for i, lab in enumerate(labs):
        z, chi, dens, _, _ = LC[lab]
        dl = dens - 1.0 if dens_rep[lab]["one_plus"] else dens
        for j, zt in enumerate(zs):
            k = int(np.argmin(np.abs(z - zt))); face = dl[..., k]
            m = float(np.nanpercentile(np.abs(face), 99)); ax = axs[i, j]
            if lab.endswith("rect"):
                ext = [0, L, 0, L]; unit = "Mpc"
            else:
                t = N * cell / chi.min() * 180 / np.pi; ext = [-t / 2, t / 2, -t / 2, t / 2]
                unit = "deg (assumed)"
            im = ax.imshow(face.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-m, vmax=m)
            ax.set_title(f"{LABELS[lab]}, z={z[k]:.2f}", fontsize=8); ax.set_xlabel(unit, fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8)
    sup = "Density faces"
    if ang:
        r = ang["density"]; b = max(r, key=lambda k: abs(r[k]))
        sup += f".  v4 angular vs rectilinear at z_min: best orientation '{b}', corr {r[b]:+.2f}"
    fig.suptitle(sup)
    fig.savefig(os.path.join(a.out_dir, "native_lightcones_faces.png"), dpi=150)
    plt.close(fig)

    # ---------------- units ----------------
    col = {"v4_rect": "#1f77b4", "v4_ang": "#d62728", "v3_rect": "#2ca02c"}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for lab, rep in vel_rep.items():
        if rep is None:
            continue
        axs[0].plot(rep["z"], rep["prod"][rep["best"]], "o-", color=col[lab],
                    label=f"{LABELS[lab]}: {rep['best']}")
        axs[1].plot(rep["z"], rep["r"], "o-", color=col[lab], label=LABELS[lab])
    axs[0].axhline(1, color="k", ls=":", lw=1); axs[0].set_yscale("log")
    axs[0].set_title("velocity: A·g for the best hypothesis\n(=1 at all z if the units are right)")
    axs[1].set_title("corr(raw v, linear v from density)"); axs[1].set_ylim(-1.05, 1.05)
    if ang:
        za, ratio = ang["rms_ratio"]
        axs[1].plot(za, ratio, color=col["v4_ang"], lw=1, ls="--",
                    label="v4 angular rms / v4 rect rms")
    for lab in labs:
        z = LC[lab][0]; r = dens_rep[lab]
        axs[2].plot(z, r["std"], color=col[lab], lw=2, label=f"{LABELS[lab]} std")
        axs[2].plot(z, r["mean"], color=col[lab], lw=1, ls="--", label=f"{LABELS[lab]} mean")
        axs[3].plot(z, r["min"], color=col[lab], lw=2, label=LABELS[lab])
    axs[2].set_title(r"density $\delta$: per-slice std and mean")
    axs[3].axhline(-1, color="k", ls=":", lw=1); axs[3].set_title(r"density $\delta$: per-slice min ($\geq-1$)")
    for ax in axs:
        ax.set_xlabel("z"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.out_dir, "native_lightcones_units.png"), dpi=150)
    plt.close(fig)
    print("wrote figures to", a.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code", choices=("v3", "v4"), help="which py21cmfast to run (run stage)")
    p.add_argument("--stage", choices=("run", "plot", "all"), default="all")
    p.add_argument("--HII_DIM", type=int, default=64)
    p.add_argument("--BOX_LEN", type=float, default=400.0)
    p.add_argument("--z_min", type=float, default=5.0)
    p.add_argument("--z_max", type=float, default=12.0)
    p.add_argument("--random_seed", type=int, default=37)
    p.add_argument("--HII_EFF_FACTOR", type=float, default=30.0)
    p.add_argument("--cache_dir", default=None,
                   help="default: data/cache_v4_periodicity_quick (v4, reuses script 38) "
                        "or data/cache_v3_native_viz (v3)")
    p.add_argument("--out_dir", default="data/plots/39_native_lightcone_viz")
    p.add_argument("--n_threads", type=int, default=16)
    p.add_argument("--n_nodes", type=int, default=30)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.stage in ("run", "all"):
        if a.code is None:
            raise SystemExit("--code v3 or --code v4 is required for the run stage")
        if a.cache_dir is None:
            a.cache_dir = ("data/cache_v4_periodicity_quick" if a.code == "v4"
                           else "data/cache_v3_native_viz")
        npz = os.path.join(a.out_dir, f"native_lightcones_{a.code}.npz")
        (run_v4 if a.code == "v4" else run_v3)(a, npz)
    if a.stage in ("plot", "all"):
        plot_stage(a)
