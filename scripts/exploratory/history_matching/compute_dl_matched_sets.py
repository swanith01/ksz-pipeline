#!/usr/bin/env python
"""
compute_dl_matched_sets.py -- D_ell(kSZ) via the TRUSTED direct/coeval + Limber
method (ksz_pipeline.coeval.limber.compute_cell), for each history-matched
(zeta, log10 Tvir, R_BUBBLE_MAX) solution from solve_history*.py.

Unlike the stitched-lightcone method, this does NOT tile a single box along
an extended line-of-sight window, so it is NOT subject to the box-periodicity
artifact documented in HANDOFF_ksz_report.md ("Box periodicity is real"). It
IS still subject to ordinary finite-box effects: missing large-scale
(k < 2*pi/L) power, and grid/Nyquist resolution at high k. At this script's
default (box_len=400, hii_dim=64, matching the history solver), Nyquist is
k = pi/dx = 0.50 Mpc^-1; ell=3000 at chi_proxy=7800 Mpc (fiducial.yaml's
value) maps to k = ell/chi = 0.385 Mpc^-1 -- 77% of Nyquist, so treat
D_ell above ell~2000 as qualitative-only here. Re-run at --box-len 800
--hii-dim 512 (fiducial.yaml's production setting) for a trustworthy number
(19% of Nyquist at ell=3000 there).

Calls the pipeline's own qperp_power and compute_cell exactly as
scripts/02_make_ksz_coeval_boxes.py does, so results are directly comparable
to that script's trusted output. The one thing 02/fields.py does NOT do is
thread astro_params through run_coeval (it always uses 21cmFAST defaults) --
that's the actual new code here.

NOT YET VERIFIED against the real limber.py / momentum.py source -- written
from script 02's call pattern only. Check compute_cell's z-grid sensitivity
before trusting output; see this repo's chat history for context.

Usage (p21c_v3 env, on a compute node):
  python compute_dl_matched_sets.py --json solutions_400_64.json solutions_400_64_v2.json \
      --zsnap-config /user1/swanith/ksz-pipeline/configs/fiducial.yaml \
      --ksz-pipeline-path /user1/swanith/ksz-pipeline/src
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

from morph_compare import load_sets   # same loader used by the P_ee comparison


def run_coeval_fields_astro(p21c, velocity_conversion_factor, z, HII_DIM, BOX_LEN,
                            cache_dir, N_THREADS, random_seed, zeta, tvir, rmax):
    """Like ksz_pipeline.coeval.fields.run_coeval_fields, but with astro_params
    threaded through -- fields.py's own version always uses 21cmFAST defaults."""
    kw = dict(HII_EFF_FACTOR=float(zeta), ION_Tvir_MIN=float(tvir))
    if rmax is not None:
        kw["R_BUBBLE_MAX"] = float(rmax)
    astro = p21c.AstroParams(**kw)
    coeval = p21c.run_coeval(
        redshift=float(z),
        user_params={"HII_DIM": int(HII_DIM), "BOX_LEN": float(BOX_LEN),
                     "N_THREADS": int(N_THREADS)},
        astro_params=astro,
        random_seed=random_seed,
        write=True,
        direc=cache_dir,
    )
    fac = velocity_conversion_factor(z)
    return (coeval.density, coeval.xH_box,
            coeval.lowres_vx * fac * 1e5, coeval.lowres_vy * fac * 1e5,
            coeval.lowres_vz * fac * 1e5)


def run_one_set(p21c, qperp_power, velocity_conversion_factor, label, s, zs,
                cache_dir, threads, seed, box_len, hii_dim):
    results = {}
    for z in zs:
        t0 = time.time()
        delta, xH, vx, vy, vz = run_coeval_fields_astro(
            p21c, velocity_conversion_factor, z, hii_dim, box_len, cache_dir,
            threads, seed, s["zeta"], s["tvir"], s["rmax"])
        k_q, P_q, P_std = qperp_power(delta, xH, vx, vy, vz, box_len)
        results[z] = dict(k=k_q, Pqperp=P_q, Pstd=P_std, xH_mean=float(xH.mean()))
        print(f"  {label:20s} z={z:5.1f}  <xH>={xH.mean():.3f}  "
              f"[{time.time() - t0:5.1f}s]", flush=True)
    return results


def make_plot(all_dl, outdir):
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    labs = list(all_dl)
    rmax_vals = [15.0 if all_dl[l]["rmax"] is None else all_dl[l]["rmax"] for l in labs]
    norm = mpl.colors.Normalize(vmin=min(rmax_vals), vmax=max(rmax_vals))
    cmap = mpl.colormaps["viridis"]

    style = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
             "mathtext.fontset": "cm", "font.size": 15, "axes.labelsize": 16,
             "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
             "ytick.right": True, "legend.fontsize": 10, "savefig.dpi": 300,
             "savefig.bbox": "tight"}

    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        for lab, rmax in zip(labs, rmax_vals):
            r = all_dl[lab]
            color = cmap(norm(rmax))
            leg = (rf"$R_{{\rm max}}$={rmax:g} Mpc, $\zeta$={r['zeta']:.1f}, "
                  rf"$\log_{{10}}T_{{\rm vir}}$={r['tvir']:.2f}")
            ax.errorbar(r["ell"], r["Dl"], yerr=r["sigma"], color=color,
                        label=leg, lw=1.8, capsize=2, elinewidth=0.8)
            d3000 = float(np.interp(3000, r["ell"], r["Dl"]))
            print(f"  {lab:20s} D_3000 = {d3000:.4f} uK^2")

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Multipole $\ell$")
        ax.set_ylabel(r"$D_\ell^{\rm kSZ}\ [\mu{\rm K}^2]$")
        ax.axvline(3000, color="gray", ls=":", lw=1)
        ax.legend(fontsize=9, loc="upper left")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$R_{\rm BUBBLE\_MAX}$ [Mpc]")

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(outdir, f"dl_matched_sets.{ext}"))
        plt.close(fig)
    print(f"Saved {outdir}/dl_matched_sets.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--max-resid", type=float, default=0.02)
    ap.add_argument("--zsnap-config", required=True,
                    help="path to fiducial.yaml or quicktest.yaml -- z_snapshots "
                         "comes from its coeval_ksz section")
    ap.add_argument("--ksz-pipeline-path", required=True,
                    help="path to ksz-pipeline/src, added to sys.path so "
                         "ksz_pipeline is importable")
    ap.add_argument("--box-len", type=float, default=400.0)
    ap.add_argument("--hii-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--direc", default="/user1/swanith/hist_match/cache")
    ap.add_argument("--outdir", default="dl_out")
    a = ap.parse_args()

    sys.path.insert(0, a.ksz_pipeline_path)
    import py21cmfast as p21c
    from ksz_pipeline.coeval.fields import velocity_conversion_factor
    from ksz_pipeline.coeval.limber import compute_cell
    from ksz_pipeline.coeval.momentum import qperp_power

    sets = load_sets(a.json, a.max_resid)
    if not sets:
        raise SystemExit(f"No solutions with max_abs_dx <= {a.max_resid} in {a.json}")
    print("Parameter sets:")
    for lab, s in sets.items():
        print(f"  {lab:20s} zeta={s['zeta']:.3f}  log10Tvir={s['tvir']:.4f}  "
              f"resid={s['resid']:.4f}")

    with open(a.zsnap_config) as f:
        cfg = yaml.safe_load(f)
    zs = cfg["coeval_ksz"]["z_snapshots"]
    print(f"z_snapshots from {a.zsnap_config}: {len(zs)} points, "
          f"{max(zs):.1f} down to {min(zs):.1f}")

    dx = a.box_len / a.hii_dim
    print(f"box={a.box_len:g} Mpc / {a.hii_dim}^3  (dx={dx:.3f} Mpc, "
          f"k_Nyquist={np.pi / dx:.3f} Mpc^-1)")

    os.makedirs(a.outdir, exist_ok=True)
    all_dl = {}
    for lab, s in sets.items():
        rq = run_one_set(p21c, qperp_power, velocity_conversion_factor, lab, s,
                         zs, a.direc, a.threads, a.seed, a.box_len, a.hii_dim)
        ells, D_ell, sigma_D, C_ell, sigma_C, (ZS_asc, tau), (_, xe) = \
            compute_cell(rq)
        all_dl[lab] = dict(ell=ells, Dl=D_ell, sigma=sigma_D,
                           zeta=s["zeta"], tvir=s["tvir"], rmax=s["rmax"])
        safe = lab.replace(" ", "_").replace("(", "").replace(")", "")
        np.savez(os.path.join(a.outdir, f"dl_{safe}.npz"),
                 ell=ells, Dl=D_ell, sigma_Dl=sigma_D, tau=tau, xe=xe, ZS_asc=ZS_asc)

    make_plot(all_dl, a.outdir)


if __name__ == "__main__":
    main()
