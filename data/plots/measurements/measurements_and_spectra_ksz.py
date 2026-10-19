#!/usr/bin/env python3
"""
Measurements_and_Spectra_kSZ.py
================================

Script version of ``Measurements_and_Spectra_kSZ.ipynb``.

Figures produced (all written to ``--outdir``, default ``./figures``):

  sed        spectral_responses_sz.pdf   SZ spectral responses (kSZ, tSZ, tkSZ, rtSZ)
             kSZ_SED.pdf                 kSZ as a Doppler-shifted blackbody
  total      total_kSZ_history.pdf       total kSZ power at l=3000 vs. publication year
  patchy     patchy_kSZ_history.pdf      patchy (reionization) kSZ at l=3000 vs. year
  tsz        tSZ_measurements.pdf        tSZ power at l=3000 vs. year (ported unchanged)

Usage
-----
    python measurements_and_spectra_ksz.py                 # all figures
    python measurements_and_spectra_ksz.py total patchy    # only these
    python measurements_and_spectra_ksz.py total --labels  # annotate every point
    python measurements_and_spectra_ksz.py --list          # print the data tables

Every kSZ data point below carries its reference and arXiv ID. Values were
checked against the papers in October 2026; see the ``note`` field for the
exact model case quoted when a paper gives several numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Plot style (matches the Times-like, inward-tick style of the group figures)
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.linewidth": 1.2,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "legend.fontsize": 20,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "savefig.bbox": "tight",
}

COLORS = {
    "ACT": "#0a6cf5",
    "SPT": "#f50a0a",
    "Planck": "#14c814",
    "Independent": "#c412f0",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Point:
    year: float
    exp: str             # key into COLORS
    value: float         # muK^2 at l = 3000
    lo: float | None = None   # 68% lower error (None for an upper limit)
    hi: float | None = None   # 68% upper error
    upper: bool = False       # True -> 95% CL upper limit
    label: str = ""           # short author label for annotations
    ref: str = ""             # arXiv ID / journal
    note: str = ""            # which model case is quoted
    variant: bool = False     # alternative model choice of the same analysis
    verified: bool = True     # value confirmed against the paper
    dx: float = 0.0           # horizontal offset for same-year points
    label_pos: tuple | None = None  # (dx, dy, rotation, ha) override for the annotation


# Total kSZ power D_3000^kSZ (homogeneous + patchy)
TOTAL_KSZ: list[Point] = [
    Point(2010, "SPT", 13.0, upper=True, label="Hall+", ref="arXiv:0912.4315",
          note="SPT 2008 data, 95% CL"),
    Point(2011, "ACT", 8.0, upper=True, label="Dunkley+", ref="arXiv:1009.0866",
          note="kSZ amplitude free, independent of tSZ, 95% CL", dx=-0.12),
    Point(2011, "SPT", 6.5, upper=True, label="Shirokoff+", ref="arXiv:1012.4788",
          note="95% CL", dx=0.12),
    Point(2012, "SPT", 6.7, upper=True, label="Reichardt+", ref="arXiv:1111.0932",
          note="with free tSZ-CIB correlation; < 2.8 if correlation set to zero",
          dx=-0.12),
    Point(2012, "SPT", 6.1, upper=True, label="Zahn+", ref="arXiv:1111.6386",
          note="CSF+patchy template, free l-independent tSZ-CIB correlation "
               "(< 2.7 with xi=0, < 4.1 with rising xi_l)", dx=0.12),
    Point(2013, "ACT", 8.6, upper=True, label="Sievers+", ref="arXiv:1301.0824",
          note="ACT+WMAP7, LCDM, 95% CL", dx=-0.2),
    Point(2013, "ACT", 5.0, upper=True, label="Dunkley+ (ACT+SPT)",
          ref="arXiv:1301.0776",
          note="ACT+SPT, 0 < xi < 0.2; ACT alone < 8.6, broader 0 < xi < 0.5 gives < 6.9",
          dx=0.2),
    Point(2013, "Independent", 5.3, lo=2.4, hi=2.2, label="Addison+",
          ref="arXiv:1210.6697 (MNRAS 436, 1896)",
          note="SPT+ACT+Herschel-SPIRE+Planck; value as quoted by Crawford+14 Sec. 6.3.4",
          dx=0.0),
    Point(2014, "SPT", 2.9, lo=1.6, hi=1.6, label="Crawford+",
          ref="arXiv:1303.3535 (ApJ 784, 143)",
          note="R12 spectra + tSZ-bispectrum prior, free tSZ-CIB corr., A_kSZ>0 prior; "
               "< 5.6 (95%); 2.6 +/- 1.8 without the positivity prior"),
    # Could not confirm the number in the Planck 2013 papers (table not
    # retrievable). Off by default; enable with --include-unverified.
    Point(2014, "Planck", 5.3, lo=1.9, hi=2.8, label="Planck 2013",
          ref="arXiv:1303.5076 (?)",
          note="UNVERIFIED; not in Planck Int. XXIV (only 2D contours). Possibly a "
               "mix-up with Addison+13 (also 5.3)",
          verified=False),
    Point(2015, "SPT", 2.9, lo=1.3, hi=1.3, label="George+", ref="arXiv:1408.3161",
          note="with tSZ-bispectrum prior; spectra-only 95% limit < 5.4"),
    Point(2016, "Planck", 2.6, upper=True, label="Planck Int. XLVII",
          ref="arXiv:1605.03507",
          note="CSF & patchy template, lollipop+PlanckTT+VHL (ACT/SPT), 95% CL"),
    Point(2021, "SPT", 3.0, lo=1.0, hi=1.0, label="Reichardt+", ref="arXiv:2002.06197",
          note="SPT-SZ + SPTpol"),
    Point(2022, "Independent", 3.3, lo=0.3, hi=0.5, label="Gorce+",
          ref="arXiv:2202.08698 (A&A 662, A122)",
          note="re-analysis of SPT (R21) bandpowers with Planck priors, free cosmology"),
    Point(2025, "ACT", 1.48, lo=1.10, hi=0.71, label="Beringue+", ref="arXiv:2506.06274",
          note="ACT DR6 only (Louis+25 quote < 4 at 95% CL; P-ACT 2.0 +/- 0.9)"),
    Point(2026, "SPT", 1.98, lo=0.87, hi=0.87, label="Chaubal+",
          ref="arXiv:2601.20551",
          note="fiducial: free CIB + l^alpha G15 SZ"),
    Point(2026, "SPT", 3.96, lo=0.82, hi=0.82, label="Agora templates",
          ref="arXiv:2601.20551", note="Agora SZ + tSZ-CIB templates",
          variant=True, dx=0.22),
    Point(2026, "SPT", 2.5, lo=1.1, hi=1.1, label="free CIB+SZ",
          ref="arXiv:2601.20551", note="spline kSZ/tSZ/CIB/tSZ-CIB",
          variant=True, dx=-0.22),
]

# Patchy (reionization) kSZ power D_3000^pkSZ
PATCHY_KSZ: list[Point] = [
    Point(2012, "SPT", 4.9, upper=True, label="Zahn+", ref="arXiv:1111.6386",
          note="free tSZ-CIB correlation; < 2.1 with xi = 0", dx=0.15),
    Point(2012, "SPT", 5.7, upper=True, label="Reichardt+\n(all kSZ patchy)",
          ref="arXiv:1111.0932",
          note="Table 3: Z11 patchy-only template, free tSZ-CIB correlation; "
               "all kSZ assumed patchy, no homogeneous part subtracted", dx=-0.15,
          label_pos=(-0.1, 0.15, 0, "right")),
    Point(2013, "ACT", 7.1, upper=True, label="Sievers+", ref="arXiv:1301.0824",
          note="total < 8.6 minus assumed homogeneous 1.5 (Shaw+12 scaling), "
               "homogeneous uncertainty not propagated"),
    Point(2015, "SPT", 4.4, upper=True, label="George+", ref="arXiv:1408.3161",
          note="spectra only, after subtracting homogeneous kSZ"),
    Point(2021, "SPT", 2.5, upper=True, label="Reichardt+", ref="arXiv:2002.06197",
          note="with tSZ-bispectrum prior, best-estimate homogeneous kSZ; "
               "68%: 1.1 +1.0 -0.7"),
    Point(2022, "Independent", 1.58, upper=True, label="Gorce+",
          ref="arXiv:2202.08698", note="free cosmology"),
]


# ---------------------------------------------------------------------------
# kSZ history plots
# ---------------------------------------------------------------------------
def _draw_point(ax, p: Point, annotate: bool) -> None:
    color = COLORS[p.exp]
    x = p.year + p.dx
    if p.upper:
        ax.vlines(x, 0, p.value, color=color, alpha=0.35, lw=4, zorder=2)
        ax.scatter(x, p.value, marker="v", s=260, color=color,
                   edgecolor="black", lw=1.2, zorder=5)
    else:
        face = "white" if p.variant else color
        alpha = 0.55 if p.variant else 1.0
        ax.errorbar(x, p.value, yerr=[[p.lo], [p.hi]], fmt="none",
                    ecolor=color, elinewidth=2.5, capsize=6, capthick=2,
                    alpha=alpha, zorder=4)
        ax.scatter(x, p.value, marker="o", s=260, facecolor=face,
                   edgecolor=color if p.variant else "black",
                   lw=2.0 if p.variant else 1.2, alpha=alpha, zorder=6)
    if annotate and p.label and not p.variant:
        y = p.value + (0.35 if p.upper else (p.hi or 0) + 0.25)
        ldx, ldy, rot, ha = p.label_pos or (0.08, 0.0, 45, "left")
        ax.annotate(p.label, xy=(x + ldx, y + ldy), rotation=rot, ha=ha,
                    va="bottom", fontsize=18, color=color)


def _legend(ax, points: list[Point], loc="upper right") -> None:
    handles, seen = [], set()
    order = ["ACT", "SPT", "Planck", "Independent"]
    for exp in order:
        for upper in (False, True):
            if not any(p.exp == exp and p.upper == upper and not p.variant for p in points):
                continue
            key = (exp, upper)
            if key in seen:
                continue
            seen.add(key)
            marker = "v" if upper else "o"
            cl = "95% CL" if upper else "68% CL"
            handles.append(ax.scatter([], [], marker=marker, s=260,
                                      color=COLORS[exp], edgecolor="black",
                                      lw=1.2, label=f"{exp} ({cl})"))
    if any(p.variant for p in points):
        handles.append(ax.scatter([], [], marker="o", s=260, facecolor="white",
                                  edgecolor=COLORS["SPT"], lw=2.0, alpha=0.55,
                                  label="SPT-3G alt. models"))
    ax.legend(handles=handles, loc=loc, handlelength=1.2, borderpad=0.6)


def plot_history(points, ylabel, outfile, ylim, xlim, annotate,
                 ystep=5, variant_labels=False) -> None:
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 10.5))
        for p in points:
            _draw_point(ax, p, annotate)
        if variant_labels:
            for p in points:
                if p.variant:
                    ax.annotate(p.label, xy=(p.year + p.dx, p.value - (p.lo or 0) - 0.15),
                                ha="center", va="top", fontsize=14,
                                color=COLORS[p.exp], alpha=0.8)
        ax.set_xlabel("Year of publication", fontsize=58)
        ax.set_ylabel(ylabel, fontsize=50)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(int(np.ceil(xlim[0])), int(xlim[1]) + 1, 2))
        ax.set_yticks(np.arange(0, ylim[1], ystep))
        ax.tick_params(labelsize=36)
        _legend(ax, points)
        fig.savefig(outfile, dpi=200)
        fig.savefig(Path(outfile).with_suffix(".png"), dpi=200)
        plt.close(fig)
        print(f"Saved: {outfile}")


def make_total(outdir: Path, annotate: bool, include_unverified: bool) -> None:
    pts = [p for p in TOTAL_KSZ if p.verified or include_unverified]
    plot_history(
        pts,
        ylabel=r"$D^{\,\mathrm{total\text{-}kSZ}}_{\ell=3000}\ \ (\mu\mathrm{K}^2)$",
        outfile=outdir / "total_kSZ_history.pdf",
        ylim=(-0.4, 16.5), xlim=(2009.2, 2026.9), annotate=annotate,
    )


def make_patchy(outdir: Path) -> None:
    plot_history(
        PATCHY_KSZ,
        ylabel=r"$D^{\,\mathrm{pkSZ}}_{\ell=3000}\ \ (\mu\mathrm{K}^2)$",
        outfile=outdir / "patchy_kSZ_history.pdf",
        ylim=(-0.4, 9.0), xlim=(2009.8, 2026.9), annotate=True, ystep=2,
    )


def print_tables() -> None:
    for name, pts in (("TOTAL kSZ", TOTAL_KSZ), ("PATCHY kSZ", PATCHY_KSZ)):
        print(f"\n{name}")
        for p in pts:
            val = (f"< {p.value}" if p.upper
                   else f"{p.value} +{p.hi} -{p.lo}")
            flag = "" if p.verified else "  [UNVERIFIED]"
            alt = " (alt. model)" if p.variant else ""
            print(f"  {p.year}  {p.exp:<11} {val:<18} {p.label}{alt}  "
                  f"{p.ref}  -- {p.note}{flag}")


# ---------------------------------------------------------------------------
# SZ spectral responses (from notebook cells 0-1)
# ---------------------------------------------------------------------------
T_CMB = 2.725           # K
H_PL = 6.62607015e-34   # J s
K_B = 1.380649e-23      # J/K
C_LIGHT = 299792458.0   # m/s


def _D(g, x):
    """D = -x d/dx applied numerically."""
    return -x * np.gradient(g, x)


def make_sed(outdir: Path) -> None:
    nu = np.linspace(30e9, 900e9, 2000)
    x = H_PL * nu / (K_B * T_CMB)

    G = x * np.exp(x) / np.expm1(x) ** 2          # D fbar
    Y2 = _D(G, x) - 3.0 * G                        # (D^2 - 3D) fbar
    Y3 = _D(Y2, x)
    Y4 = _D(Y3, x)
    A = 0.4 * G - Y2 + 1.4 * Y3                    # tkSZ
    rt = -0.3 * Y2 - 2.1 * Y3 + 0.7 * Y4           # rtSZ (truncated)

    trim = slice(20, -20)                          # drop gradient edge effects
    f = nu[trim] / 1e9
    curves = {
        r"CMB + kSZ ($\nu^3 G$)": (nu**3 * G)[trim],
        r"tSZ ($\nu^3 Y_2$)": (nu**3 * Y2)[trim],
        r"tkSZ ($\nu^3 A$)": (nu**3 * A)[trim],
        r"rtSZ ($\nu^3\,$rtSZ)": (nu**3 * rt)[trim],
    }
    norm = np.max(np.abs(curves[r"tSZ ($\nu^3 Y_2$)"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax in (a1, a2):
        ax.axhline(0, color="k", lw=0.6)
    for lab, y in curves.items():
        a1.plot(f, y / norm, lw=1.6, label=lab)
        a2.plot(f, y / np.max(np.abs(y)), lw=1.6, label=lab.split(" (")[0])
    a1.set_ylabel(r"$\Delta I_\nu$ (normalised to $|$tSZ$|_{\max}$)")
    a1.set_title("Common normalisation")
    a2.set_ylabel(r"$\Delta I_\nu$ (each normalised to its max)")
    a2.set_title("Shape comparison")
    for ax in (a1, a2):
        ax.set_xlabel("Frequency [GHz]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "spectral_responses_sz.pdf")
    plt.close(fig)
    print(f"Saved: {outdir / 'spectral_responses_sz.pdf'}")

    # kSZ as a Doppler-shifted blackbody
    B = (2 * H_PL * nu**3 / C_LIGHT**2) / np.expm1(x)
    v_r = 300e3                                    # m/s
    x_s = H_PL * nu / (K_B * T_CMB * (1 - v_r / C_LIGHT))
    B_s = (2 * H_PL * nu**3 / C_LIGHT**2) / np.expm1(x_s)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nu / 1e9, B / B.max(), label="CMB blackbody (2.725 K)")
    ax.plot(nu / 1e9, B_s / B.max(), ls="--", label=r"Shifted ($v_r=300$ km/s)")
    ax.plot(nu / 1e9, 0.5e20 * (B_s - B), label=r"$0.5\times10^{20}\times$ kSZ difference")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("kSZ as a Doppler-shifted CMB blackbody")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "kSZ_SED.pdf")
    plt.close(fig)
    print(f"Saved: {outdir / 'kSZ_SED.pdf'}")


# ---------------------------------------------------------------------------
# tSZ history (notebook cell 6, ported unchanged; data not re-verified)
# ---------------------------------------------------------------------------
TSZ_DATA = [
    # (year, val, err-, err+, is_upper, exp, label_lines)
    (2011, 8.2, 2.9, 2.9, False, "ACT", ["Dunkley+", r"$A_{\rm kSZ}=A_{\rm tSZ}$"]),
    (2011, 3.3, 1.4, 1.4, False, "ACT", ["Dunkley+", "uniform prior", "tSZ-CIB"]),
    (2011, 5.3, None, None, True, "SPT", ["Shirokoff+"]),
    (2012, 3.65, 0.69, 0.69, False, "SPT", ["Reichardt+"]),
    (2013, 3.3, 1.4, 1.4, False, "ACT", ["Dunkley+ (ACT)"]),
    (2013, 4.1, 0.9, 0.9, False, "ACT", ["(SPT)"]),
    (2013, 4.0, 0.9, 0.9, False, "ACT", ["(ACT+SPT)"]),
    (2015, 4.08, 0.67, 0.58, False, "SPT", ["George+"]),
    (2021, 3.42, 0.54, 0.54, False, "SPT", ["Reichardt+"]),
]


def make_tsz(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    seen = set()
    for year, val, em, ep, upper, exp, lines in TSZ_DATA:
        col = COLORS[exp]
        if upper:
            ax.scatter(year, val, marker="v", color=col, s=80, zorder=5)
            ax.vlines(year, 0, val, color=col, alpha=0.12, lw=2)
            ax.annotate("\n".join(lines), xy=(year + 0.1, val + 0.4), fontsize=9.2,
                        rotation=45, va="bottom", ha="left", color=col)
        else:
            ax.errorbar(year, val, yerr=[[em], [ep]], fmt="^", color=col,
                        capsize=4, markersize=8, lw=1.5, zorder=6)
            ax.annotate("\n".join(lines), xy=(year + 0.12, val), fontsize=9.2,
                        rotation=45, va="center", ha="left", color=col)
        key = (exp, upper)
        if key not in seen:
            seen.add(key)
            ax.scatter([], [], marker="v" if upper else "^", color=col, s=80,
                       label=f"{exp} tSZ ({'95% CL' if upper else 'detection'})")
    ax.set_xlabel("Year of publication", fontsize=17)
    ax.set_ylabel(r"$D^{\mathrm{tSZ}}_{\ell=3000}\ \ (\mu K^2)$", fontsize=17)
    ax.set_xlim(2009.2, 2022.5)
    ax.set_xticks(np.arange(2010, 2023, 1))
    ax.tick_params(axis="x", rotation=45, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(-0.5, 12.0)
    ax.legend(loc="upper right", fontsize=13, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(outdir / "tSZ_measurements.pdf", dpi=200)
    plt.close(fig)
    print(f"Saved: {outdir / 'tSZ_measurements.pdf'}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", metavar="FIGURE",
                    help="any of: sed total patchy tsz (default: all)")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--labels", action="store_true",
                    help="annotate every point in the total-kSZ plot with its author")
    ap.add_argument("--include-unverified", action="store_true",
                    help="also plot points whose values could not be confirmed")
    ap.add_argument("--list", action="store_true", help="print data tables and exit")
    args = ap.parse_args()

    if args.list:
        print_tables()
        return

    allowed = {"sed", "total", "patchy", "tsz"}
    args.figures = args.figures or sorted(allowed)
    bad = set(args.figures) - allowed
    if bad:
        ap.error(f"unknown figure(s): {', '.join(sorted(bad))}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if "sed" in args.figures:
        make_sed(outdir)
    if "total" in args.figures:
        make_total(outdir, args.labels, args.include_unverified)
    if "patchy" in args.figures:
        make_patchy(outdir)
    if "tsz" in args.figures:
        make_tsz(outdir)


if __name__ == "__main__":
    main()
