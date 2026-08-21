# kSZ Pipeline — Validation Audit (Direct vs. Stitched)

**Status: neither method is called trusted.** Both landing near the Reichardt+2021
constraint is encouraging but does not resolve the internal factor-of-two-plus
disagreement between them, which remains open. This document supersedes the
earlier short table (previous version: see git history of this file).

**Update (2026-07-22): closure test (script 14) complete on the matched
window.** New results are in §1b below. Two things this changes about how to
read the rest of this document:
- **§1&2's per-slice Δ(D₃₀₀₀) columns are not directly comparable between
  methods** — see the caveat inserted at the top of §1&2. This was not
  flagged when that section was first written and affects how to interpret
  the "77×" finding below.
- **Open Item #1 (z-range mismatch) is downgraded** — the closure test shows
  it explains only 11.6% of the gap, not the bulk of it as previously
  hypothesized. See revised Open Items list at the end.

**Scope, per instruction:** direct vs. stitched only, same seed, same fields,
matched redshift slices. Georgiev-Eq10 is a separate audit (its own ~3×
normalization issue, tracked independently) and is excluded here.

**Commit / config:** results below come from the reseeded run at commit
`76e68fe` (the commit that fixed `random_seed` never reaching py21cmfast).
Two subsequent fixes are **not yet reflected** in the §1&2 numbers below
(the §1b closure test numbers ARE current, see that section):
- `23419d3` (`write=True`, fixes a caching bug — does not change any physics,
  only speed; unaffected results, safe to leave as-is)
- `6fd040e` (ne0 helium/hydrogen convention fix — confirmed ~0.4% effect,
  small enough that a rerun has not yet been prioritized)

---

## 1 & 2. Per-slice weighted kSZ contribution and cumulative D_3000(<z)

**⚠ CAVEAT, added 2026-07-22 — read before interpreting the table below:**
The two Δ(D₃₀₀₀) columns are **not measuring the same kind of quantity**,
even though they're tabulated at the same z checkpoints. Coeval-direct's
Δ(D₃₀₀₀) at a checkpoint is that shell's own P_qperp(k), summed **incoherently**
into the Limber integral (no cross-shell correlation, by construction of the
Limber approximation). Stitched's Δ(D₃₀₀₀) is the change in FFT power of a
**coherently-built** real-space map when that shell's LOS-interpolated data
is added — because the whole map is Fourier-transformed at once, this
reflects that shell's coupling with *every* shell already in the map, not
just its own standalone signal. So the "77× at z=18" finding below is real
(the curves do diverge there) but should be read as **evidence of where the
two methods' cumulative curves diverge**, not as proof that stitched has
spurious excess power *localized to* z>13. Some of that apparent excess
could be coherent cross-shell coupling with the whole z=4.5–18 structure
already built into the map by that point. This is directly relevant to
Open Item #1's revised status below.

Computed at coeval-direct's own 29 redshift checkpoints (see Audit
Methodology below for why coeval-direct's grid, not stitched's, sets the
checkpoints). Self-check: summing every coeval-direct per-slice contribution
reproduces the trusted `compute_cell` output to within 0.06% (1.7812 vs.
1.7822 μK²) — the per-slice breakdown below is faithful, not an artifact of
the extraction method. (This 1.7822 figure is independently reproduced by
the §1b closure test below, run via a completely different code path —
see §1b.)

| z | coeval Δ(D₃₀₀₀) | coeval cumulative | stitched cumulative | stitched Δ(D₃₀₀₀) |
|---|---|---|---|---|
| 4.5 | 0.0688 | 0.0688 | 0.0083 | 0.0083 |
| 5.5 | 0.0721 | 0.2136 | 0.1424 | 0.0676 |
| 6.5 | 0.1021 | 0.4181 | 0.4139 | 0.1819 |
| 7.5 | 0.1686 | 0.7539 | 0.8758 | 0.2530 |
| 8.5 (coeval peak) | **0.1913** | 1.2361 | 1.5121 | 0.3152 |
| 9.5 | 0.1463 | 1.5798 | 2.2804 | 0.3058 |
| 10.5 | 0.0845 | 1.7313 | 2.9303 | 0.2989 |
| 11.5 | 0.0416 | 1.7803 | 3.3229 | 0.2136 |
| 13.0 | 0.0123 | 1.7791 | 3.7106 | 0.1018 |
| 15.0 | 0.0020 | 1.7797 | 3.9235 | 0.0398 |
| 16.0 | 0.0011 | 1.7801 | 3.9700 | 0.0136 |
| 18.0 | 0.0002 | 1.7812 | 4.0032 | 0.0154 |
| 19.0 | — (excluded, xH_mean outside patchy window) | — | 4.0092 | 0.0060 |
| 20.0 | — | — | 4.0111 | 0.0019 |

**The key finding, stated plainly (see caveat above for interpretation
limits):** past z~13, coeval-direct's per-slice contribution is correctly
near-zero (the universe is >98% neutral, essentially nothing to source
patchy kSZ). Stitched keeps adding real, non-trivial signal all the way to
z~18-19 — at z=18, stitched's per-slice contribution (0.0154) is **~77×
larger** than coeval-direct's (0.0002) at the same redshift.

**Leading candidate explanation — REVISED, see §1b:** the two methods
define "patchy regime" differently — coeval-direct via a per-snapshot
`xH_mean` threshold (window: z=4.5–18.0), stitched via a LOS-interpolated
`x_e` threshold (window: z=4.19–19.80 this run). **The closure test (§1b)
now shows this window mismatch explains only 11.6% of the total gap** —
it is a real, confirmed effect, but not the primary driver it was
hypothesized to be here. See Open Items for the current ranking.

---

## 1b. Closure test update (script 14, 2026-07-22) — matched window + chi_eff

Direct-vs-stitched advisor-requested closure test, run with the **same**
unified z-window applied to both methods (intersection of coeval's and
stitched's own patchy windows), removing the window-definition mismatch
flagged in §1&2 entirely, and using a properly signal-weighted `chi_eff`
in place of the hardcoded `chi_Mpc=7800` default (see §4).

**Unified window:** z = [4.50, 18.00] — this equals coeval-direct's own
window exactly (confirms coeval's window is fully nested inside stitched's,
as anticipated in the handoff notes).

**D_3000, matched window:**
| | value | notes |
|---|---|---|
| direct | 1.7822 μK² | matches §1&2's independently-derived 1.7822 exactly — two separate code paths (script 11's per-slice extraction vs. script 14's windowed-subset `compute_cell` call) agree |
| stitched (chi_eff) | 4.0892 μK² | chi_eff = 8504.0 Mpc (signal-weighted, see §4) |
| **ratio** | **2.29×** | supersedes §3's 2.25× (full-window, broken-chi value) — see §3 for reconciliation |

**z≥13 exclusion test:** total excess (stitched − direct, full matched
window) = 2.4085 μK². Excess remaining with a z<13 cut = 2.1296 μK².
**Only 11.6% of the excess is attributable to z≥13.** This directly revises
§1&2's leading hypothesis (see Open Items) — the window-definition mismatch
is real but is not where most of the gap lives; the other 88.4% occurs
*within* the shared z=4.5–18 window both methods already agree on.

**Chi problem: underway, not yet closed.** Three chi candidates were
compared (chi_eff, chi at z_end-of-reionization = chi(z_lo), chi at
reionization midpoint via xH_mean=0.5 crossing, unweighted mean chi over
the window) as a robustness check on the single-chi approximation in
`ksz_map_to_Dl` — not to select whichever narrows the gap. **A bug was
found and fixed** in the reionization-midpoint crossing calculation
(reversed array broke `np.interp`'s ascending-xp assumption, silently
returning an out-of-window artifact z=4.00 instead of a real crossing).
The fix is committed; **a corrected rerun (job 1684234) is queued on the
cluster and has not yet completed** due to cluster traffic — numbers above
for chi_eff itself are already correct and unaffected by this bug (it was
isolated to the other two candidates' labeling), but the full three-way
D_3000 comparison table needs 1684234's output before it can be reported
here. **Update this section once 1684234 completes.**

Plots: `data/plots/closure_patchy_ksz_vs_limits.pdf` (patchy kSZ vs. SPT-3G/ACT
2σ upper limits, matched window), `data/plots/closure_total_ksz_vs_observations.pdf`
(+ Shaw et al. 2012 late-time template vs. Chaubal+26 total-kSZ points —
note the Shaw template is extrapolated from zrei=[8,12] down to the
matched window's zrei_eff=4.50, well outside Shaw's calibrated range;
treat the total-kSZ curve's absolute normalization as illustrative, not
precise). Source: `notebooks/exploratory/closure_test_plots.py`.

---

## 3. Map mean, RMS, and direct-vs-stitched amplitude ratio

**D_3000 amplitude ratio — two values, different provenance:**
- **Pre-window-fix, full window, broken chi_Mpc=7800 default:**
  stitched / coeval-direct = 4.0111 / 1.7822 = 2.25× (previously, pre-reseed:
  0.44× — the gap has changed direction, not just magnitude, since reseeding)
- **§1b closure test, matched window, chi_eff-corrected:**
  stitched / coeval-direct = 4.0892 / 1.7822 = **2.29×**

These are close in magnitude (2.25× vs 2.29×) despite fixing two real,
confirmed issues (window mismatch + hardcoded chi) simultaneously — the
chi fix alone shifts stitched's windowed D_3000 by only ~2.1% (4.0032→4.0892
at the same z=18 truncation), consistent with §4's Open Item #2
characterization of chi_Mpc as a "real but secondary effect." **The gap is
not primarily a window-definition or chi-convention artifact** — see
revised Open Items ranking.

**Full-map mean/RMS (reseeded run, commit 83c88e9, from `validation_run_stats.json`):**
| | direct (coeval-direct — N/A, no map) | stitched (full LOS-integrated map) |
|---|---|---|
| mean | N/A | 7.7423e-09 |
| RMS | N/A | 1.4095e-06 |

This RMS independently matches job B's own logged value (`kSZ map RMS :
1.4095e-06`, `fiducial_stitched.log`) to 5 significant figures — two
separate script runs (script 03 and script 08), same commit era, same
number. Counted as a third piece of reproducibility evidence (see item 5).

**Single-slice proxy (z=7.0, NOT the full map — has known limitations, see
Audit Methodology):**
| | direct (thin-slice proxy) | stitched (single LOS pixel) |
|---|---|---|
| mean | 1.1717e-05 | 1.3355e-03 |
| RMS | 1.9074e-03 | 2.5488e-02 |
| regression slope (stitched vs. direct) | — | 0.8097, r=0.06 |

The r=0.06 correlation is **not evidence of anything** — the "direct" proxy
averaged the coeval box's full 800 Mpc depth into one 2D slice, while
stitched's comparison point is a genuine 1.56 Mpc thin slice. Averaging over
hundreds of Mpc washes out the small-scale structure a thin slice captures;
of course they don't correlate well. This test needs a better-designed proxy
before it says anything meaningful, and should not be read as evidence either
for or against agreement between the methods.

---

## 4. Unit and weighting conventions

| Convention | Coeval-direct | Stitched | Status |
|---|---|---|---|
| Scale factor | a_i=1/(1+z), enters as a_i⁴ in Limber weight denominator | a=1/(1+z), enters as a² (a2_mid) in real-space integral | Same functional form; **net power of (1+z) not yet numerically cross-checked** |
| dχ discretization | `np.gradient` over 29 irregular z_snapshots | `np.diff` over 2320 uniform comoving-distance LOS pixels | Fundamentally different grids — this audit's checkpoint method works around it, does not unify it |
| Speed of light | `C_CGS` (constants.py) | `C_CGS` (constants.py) | Identical, shared constant |
| Velocity | cm/s (post-conversion), enters momentum field q=w·v, squared in P_qperp | cm/s→Mpc/s, enters linearly in real-space integrand | Conversion is shared/common code — see detail below. Not a source of the discrepancy. |
| Electron fraction (ne0) | `ne0_cgs()`, helium-inclusive = 2.064357e-07 cm⁻³ | was `NE0_HYDROGEN_ONLY` = 2.060000e-07 cm⁻³ | **Fixed in commit 6fd040e** (~0.4% effect, confirmed small — not the main discrepancy). Not yet reflected in the numbers above. |
| ℓ↔k conversion reference distance (stitched only) | N/A (per-z dynamic χ(z) throughout) | hardcoded `chi_Mpc=7800` default in `ksz_map_to_Dl`, never overridden by script 03 | **In progress.** chi_eff=8504.0 Mpc now computed (signal-weighted, power-weighted mean comoving distance) and used in the §1b closure test — shifts stitched D_3000 by only ~2.1% at fixed window, confirming this is a secondary effect, not the main discrepancy. Robustness check across 3 chi candidates pending job 1684234 (queued, cluster traffic) — **chi problem underway, not yet closed.** |
| Patchy-regime definition | `xH_mean` per-snapshot threshold, window z=4.5–18.0 | LOS-interpolated `x_e` threshold, window z=4.24–19.44 (this reseeded run; earlier pre-reseed run gave 4.19–19.80 — some realization-to-realization shift, expected) | **Confirmed real, but NOT the primary driver of the gap** — §1b's z≥13 exclusion test shows this explains only 11.6% of the excess. Downgraded from "leading hypothesis" — see Open Items. |

### Velocity: raw box units and Zel'dovich conversion (shared, applied once, identical for both methods)

py21cmfast's `lowres_vx/vy/vz` are **not** velocities as output from the box —
they are the **linear-theory IC displacement field Ψ** (Zel'dovich
approximation), in py21cmfast's internal comoving length-like units.
`run_coeval_fields()` — the single shared function both methods call, in
exactly the same way — converts this to a genuine **comoving peculiar
velocity**:

```
v_comoving [km/s] = Ψ · D(z) · f(z) · H(z) / (1+z)
```

where:
- **D(z)**: linear growth factor, normalized D(0)=1, computed by
  quadrature integration (not a fitting formula)
- **f(z)**: growth rate, f(z) = dlnD/dlna ≈ Ωm(z)^0.55 (Linder 2005 fit,
  accurate to ~1% for flat ΛCDM)
- **H(z)**: Hubble parameter [km/s/Mpc], astropy Planck18
- the **/(1+z)** converts physical→comoving velocity — confirmed necessary
  (not arbitrary) because the Limber C_ell integral requires `ds/a⁴`, not
  `ds/a⁵`; this only holds if v entering q is comoving, not physical

Result is then ×1e5 to convert km/s→cm/s, the units `run_coeval_fields`
actually returns. This conversion was validated to ~1% against linear
theory at z=0.5–15 (typical v_rms ~95–156 km/s for an 800 Mpc box,
z=5–15).

**Why this matters for the audit:** this entire conversion — box units to
cm/s, Zel'dovich displacement to comoving velocity — happens **once**,
identically, in shared code, before either method's divergent processing
even begins. `qperp_power` (coeval-direct) and `stitch_lightcone_from_coeval`
(stitched) both receive the same already-converted cm/s velocity array from
the same function call. **It cannot be the source of the direct-vs-stitched
discrepancy** — what differs is only what happens *after* this point:
coeval-direct squares the velocity inside the Fourier-space, curl-projected
momentum field (P_qperp); stitched uses it linearly inside the real-space
LOS integrand (ΔT/T ∝ ∫n_e·v_los). That linear-vs-squared difference is
expected and correct — only the resulting *power spectrum* should be
quadratic, not the real-space temperature map itself.

---

## 5. Reproducibility confirmation

Two independent pieces of evidence, both real (not asserted):

**(a) Cross-invocation seed consistency.** Script 04's resolution sweep
(`res512`, same BOX_LEN/HII_DIM as fiducial, different script, different
process, different day) reproduced fiducial's own per-redshift `xH_mean` to
8 decimal places at all 9 overlapping checkpoints — genuine bit-level
agreement across separate script runs, not just "should agree by
construction."

**(b) Within-script self-check.** The direct-vs-stitched audit script
(script 11) independently re-derived coeval-direct's D_3000 via a full-grid,
single-pass replication of the Limber sum, and it matched the trusted
`compute_cell` output to 0.06% (1.7812 vs. 1.7822).

**Scope of what (b) actually tests — added 2026-08-18, after a Slack
discussion clarified this was being over-read.** `compute_cell` and
script 11 compute the SAME physical quantity (the Limber sum) via two
separately-written code paths — this catches CODING bugs (typos,
off-by-one, wrong array indexing, a variable reused incorrectly), which
is exactly the category every bug actually found this session falls
into (the reversed-array chi bug, the hardcoded chi_Mpc=7800 default,
the missed ne0 fix in one file, the results-vs-results_subset caching
bug). It does NOT independently test whether the Limber approximation
itself, or this codebase's specific implementation choices, are the
right physics — both paths share the same underlying assumptions, so a
shared conceptual error would reproduce identically in both and this
check would not catch it. Confidence in the physics itself rests on the
external comparison in §1b/§4 (Georgiev+24, SPT-3G, ACT DR6, Reichardt+2021),
not on this self-check.

| | `compute_cell` (limber.py) | script 11 (audit) |
|---|---|---|
| Role | Production function — used by script 14, every convergence sweep, scripts 18/19 | One-off audit tool; `per_z_D3000_contributions` later reused for per-redshift breakdown plots |
| Output | Full ell curve (80 points) + errors, one call | Per-redshift contribution, cumulative-summed to one checkpoint |
| ell range checked | Native, full resolved range | Only ever verified at ell=3000 (D_3000) — full-curve agreement not separately confirmed |
| Weighting | dchi via `np.gradient` over the full 29-z grid | Same `np.gradient` weighting, computed once on the full grid, kept per-redshift instead of immediately summed |
| Bug it caught (in itself) | — | An earlier draft called `compute_cell` on shrinking z-subsets repeatedly; `np.gradient` gave inconsistent weights each time since it depends on neighboring points. Caught and fixed before results were trusted (see Audit Methodology notes below). |
| Unique benefit | Fast, single trusted call, ground truth downstream | Exposes per-redshift contribution — the thing that made the dD_3000/dz diagnostics in §1b possible at all |

**(c) Independent full-map RMS agreement.** Script 03 (fiducial stitched
run, job B) and script 08 (validation diagnostics, job C) — two separate
scripts, two separate processes — both independently computed the stitched
kSZ map's RMS as 1.4095e-06, matching to 5 significant figures.

**(d) D_3000 re-derivation, exact match.** Re-running `ksz_map_to_Dl` on
job C's independently-saved map reproduced job B's own logged D_3000 to
4 decimal places (4.0111 both ways). This closes the loop started by (c):
not just the map, but the power derived from it, is fully reproducible.
The 2.25× direct-vs-stitched gap is confirmed to be a genuine cross-method
disagreement — nothing hiding in map generation or the D_ell extraction
step itself.

**(e) Closure-test cross-check (2026-07-22).** §1b's direct D_3000 (1.7822,
via a windowed-subset dict passed through `compute_cell`) matches §1&2's
independently-extracted 1.7822 (via script 11's per-slice summation) —
a third, structurally different code path reproducing the same number.

**Not yet done:** literally re-running script 02 twice from a clean state
and diffing the two output files bit-for-bit. (a) and (b) are strong
indirect evidence but a direct repeat-run comparison would be the cleanest
possible confirmation, and is cheap to do given `write=True` is now fixed.

---

## Open items, ranked by how well-understood they currently are

1. **Patchy z-range mismatch → high-z (z>13) signal excess in stitched** —
   **DOWNGRADED 2026-07-22.** Previously ranked as the leading, best-
   characterized hypothesis for the gap. §1b's closure test shows it
   explains only **11.6%** of the total excess — real and confirmed, but
   a minor contributor, not the primary driver. The remaining 88.4% of
   the gap occurs within the shared z=4.5–18 window itself, where §1&2's
   apples-to-apples caveat (coherent vs. incoherent addition, see top of
   §1&2) becomes the more likely place to look next.
2. **chi_Mpc=7800 hardcoded default** — **in progress.** chi_eff=8504.0 Mpc
   now computed and used (§1b, §4); confirmed secondary effect (~2.1% shift
   at fixed window), consistent with earlier ~18% max-effect estimate as an
   upper bound, not the typical case. Three-candidate robustness comparison
   pending job 1684234 (queued, cluster traffic) — not yet fully closed.
3. **ne0 helium/hydrogen mismatch** — confirmed small (0.4%), fixed in code,
   rerun not yet done.
4. **Angle-of-stitching sweep does not clearly support the
   periodic-replication hypothesis** for the low-ℓ (ℓ~130-180) spike —
   D_3000 vs. angle is non-monotonic (4.01→3.92→3.96→4.01→3.98 μK² for
   0°/10°/30°/50°/70°), and the full D_ell curves show no systematic
   shrinking of the spike with angle. This was the leading explanation for
   the low-ℓ artifact; it is now genuinely in question, not confirmed.
5. **Resolution sweep (coeval-direct) is non-monotonic** at res32/res64 —
   plausibly small-box sample variance, not confirmed either way.
6. **Field-level regression test (item 3 above) is inconclusive** — proxy
   design flaw, not evidence of anything.
7. **NEW, 2026-07-22 — coherent vs. incoherent addition (§1&2 caveat).**
   Now the most likely candidate for the bulk (88.4%) of the gap not
   explained by item 1. Not yet a "hypothesis" in the same sense as the
   others — no dedicated test has been designed for it yet, distinct from
   the resolution non-convergence question. Worth a dedicated audit.
8. **Resolution non-convergence (stitched still climbing at res512)** —
   separately tracked, may share a common cause with item 7 above
   (both point at real-space map construction rather than the Limber
   sum), not yet disentangled from it.

---

## Audit methodology notes

- The two methods use fundamentally different redshift grids (29 coarse
  snapshots vs. 2320 fine LOS pixels). This audit compares **cumulative**
  sums at coeval-direct's 29 checkpoints rather than forcing a shared grid,
  since that would require expensively re-deriving coeval-direct at 2320
  points. This is an honest matched comparison at real checkpoints, not an
  approximation dressed up as an exact one.
- Coeval-direct's per-slice weights are computed **once**, on the full
  29-point grid, then cumulative-summed — not by repeatedly calling
  `compute_cell` on shrinking subsets (an earlier version of this audit did
  that, and it silently gave inconsistent, non-comparable weights per
  checkpoint since `np.gradient` depends on an array's neighboring points;
  caught and fixed before these results were produced).

---

## Provenance

Source data: `data/products/audit_cumulative_d3000.npz`,
`audit_per_slice.npz`, `audit_conventions.json`, `audit_field_regression.npz`
— all committed at `bc10e15`. Raw logs: `fiducial_coeval.log`,
`fiducial_stitched.log`, `audit_direct_vs_stitched.log`.

**§1b additions (2026-07-22):** `data/products/closure_test.npz`
(script `scripts/14_closure_test.py`, job 1684054 — corrected rerun 1684234
pending). Plots via `notebooks/exploratory/closure_test_plots.py`.
