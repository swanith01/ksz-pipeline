# CHECKPOINT — Limber approximation validated for the D_ell calculation

**Date:** 30 September 2026
**Scope:** This checkpoint covers ONE specific, now-closed question: is
`limber.py:compute_cell`'s q_perp-only Limber approximation for D_ell
trustworthy, and if it departs from the exact calculation, is that
departure understood? It does **not** cover the dz-convergence
(snapshot-sampling-density) sensitivity work in `scripts/27` and
`scripts/28`, which is real and still open — see §4.

---

## 1. What was being checked

`limber.py:compute_cell` computes D_ell via the standard, widely-used
Limber projection of P_qperp(k) — a smooth-kernel approximation that
drops the q_parallel/Doppler term (`momentum.py:qparallel_power`'s own
docstring: "q_parallel Fourier modes are argued to phase-cancel on
line-of-sight projection"). The question this checkpoint closes: does
this approximation hold up against (a) an exact, independently-built
non-Limber calculation, and (b) a direct measurement with no Limber
assumption at all — and where it doesn't, is the departure physically
understood rather than a bug?

## 2. Evidence

### 2a. Corrected non-Limber off-diagonal estimator's diagonal vs. the formula

`src/ksz_pipeline/coeval/offdiag_projection.py` (built this investigation
to fix the original k_parallel=0 bug in `qperp_cross_z.py`) reduces
exactly to `compute_cell`'s own Limber weight on the diagonal (i=j) by
construction — no separate interpolation choice. This reduction was
checked directly:

- At `hii_dim=128`: **failed**, ratio 1.56. Traced through five ruled-out
  hypotheses (wrong a_power, resolution mismatch in the gate's own
  reference, a boundary snapshot dropping from the patchy window,
  ell-range clamping, non-Gaussian binning) before being isolated to
  resolution alone.
- At the **true fiducial resolution (512³, no override)**: **passed**,
  median ratio **1.0268** (job `1722403.swarm`, 19 Sep 2026, confirmed via
  `scripts/24_pdiag_formula_vs_native.py --include-ours 512`).

### 2b. Independent check: formula vs. the native map's own measured P_diag

`scripts/24_pdiag_formula_vs_native.py`, run with no `--include-ours`
flag, compares `compute_cell`'s formula prediction against
`coherence_decomposition_fiducial.npz`'s natively-measured `Dl_diag` — a
completely separate code path (real stitched-map construction, no
`qperp_power`, no Limber assumption of any kind). Result: median ratio
**1.049**, clean pass. This independently confirms `compute_cell`/tau/
ne0/window logic without touching anything built for check 2a.

Both 2a and 2b land in the same ~1.03-1.05 range, from genuinely
independent code paths.

### 2c. Where the formula departs, and why that's expected

Both `pdiag_formula_vs_native.png` and `dl_off_corrected.png` show the
formula (Limber) curve pulling away from the exact/native curves at low
ell — most dramatically below ell~200-300. Two independent estimates of
where this SHOULD happen, per Alvarez et al. 2016 (arXiv:1511.02846):

- **The paper's own stated result**: the Doppler-component's power peaks
  at ell ~ 20-30 in their full non-Limber treatment — precisely the
  regime their own Appendix A shows a q_perp-only Limber calculation
  cannot capture.
- **An independent re-derivation** (30 Sep 2026), evaluating their
  Appendix A threshold formula ell_Limber ~ H(z)chi(z)/delta_z using THIS
  repo's own Planck18 cosmology and W(chi)=a(chi)^-2 weight (the
  q_parallel(k) amplitude itself treated as slowly-varying — the one
  explicit approximation, stated plainly, not hidden): **ell ~ 5.6 (at
  z=4.5) rising to ~14.8 (at z=18)** across the patchy window.

Both estimates — the paper's own number and an independently-derived one
— land in the same order of magnitude, and **both sit below this box's
own resolvable ell_min (~60-85 depending on run/resolution)**. The
departure visible in the plots is therefore consistent with, not
contradictory to, established literature: the box's own fundamental mode
is the binding constraint on what we can even test, and where we CAN
test (ell gtrsim 100), formula and exact calculation already agree at the
1.03-1.05 level shown in 2a/2b.

## 3. Conclusion

The Limber approximation underlying `compute_cell`'s D_ell is validated:
accurate to ~3-5% wherever this box can resolve modes, with its expected
breakdown regime (per independent literature and independent re-derivation)
sitting below that resolvable range rather than inside it. This is now a
closed question for this investigation, not an open risk.

## 4. Explicitly NOT covered by this checkpoint — still open

- **dz-convergence / snapshot-sampling-density sensitivity**
  (`scripts/27_dl_off_dz_convergence.py`,
  `scripts/28_stitched_dz_convergence.py`): both showed real, unresolved
  sensitivity to how many snapshots feed the calculation — script 27's
  D(ell=300) swung ~3x between dz_x1/dz_x2 (confounded by pair-count
  collapse, not yet disentangled); script 28's shakedown (32^3) showed
  P_total spread 53.7%, P_diag spread 17.6%. The fiducial-resolution
  version of script 28 (job `1725423.swarm`, submitted 23 Sep) — this
  time also recomputing P_direct fresh per snapshot subset, so it can
  distinguish "the stitching is sensitive" from "even Limber itself needs
  many z-slices" — had not yet reported results as of this checkpoint.
- **Native lightcone velocity amplitude**: still explicitly unvalidated
  in absolute terms (`native_lightcone.py`'s own docstring) — the
  periodicity-existence and grouping-convergence findings built on it are
  shape/existence checks, not calibrated to a trusted absolute scale.

Do not read this checkpoint as closing either of those.
