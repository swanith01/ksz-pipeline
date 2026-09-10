# HANDOFF: Reionization-History Dependence of P_diag/P_off/direct — AMBER

**Written by an assistant (Claude), not independently re-verified.** Treat
as a map for fast orientation — check `docs/validation_table.md` and the
real commit history in `github.com/swanith01/ksz-pipeline` before citing
anything from here directly.

---

## 1. The question this chat exists to answer

Everything so far (periodicity, grouping robustness, the q_perp cross-z
attempt) was run at ONE fixed reionization history — whatever 21cmFAST's
default astrophysics parameters produce. The next question: **how does
the direct-vs-stitched gap (and its P_diag/P_off decomposition) change as
the reionization history itself changes** — earlier vs. later midpoint,
longer vs. shorter duration, symmetric vs. asymmetric? This matters
directly to Girish's original request #4 and to the periodicity question
too: if the excess's *size* scales predictably with reionization duration
(similar to how Alvarez et al. 2016 found the Doppler term's amplitude
depends strongly on Δz), that's informative either way.

**Why AMBER, not another 21cmFAST run**: AMBER directly parametrizes the
reionization history via `z_mid`, `Δz`, `A_z` — a clean, physically
interpretable sweep. 21cmFAST would require indirectly hunting for
parameter combinations (`M_min`, `ζ_ion`, `λ_mfp`) that happen to produce
a given history, which is a much messier way to ask this specific
question.

---

## 2. What this chat inherits from the prior investigation (brief)

- Direct/Limber (`compute_cell`) is the validated baseline method — 0.06%
  independent re-derivation, literature/data-consistent.
- Stitched shows ~2.3x excess over direct at fiducial 21cmFAST settings.
- Box periodicity is a real, confirmed contributor (Delta-chi/L_box overlay).
- The P_diag/direct comparison is robust across radial-grouping choices.
- An attempt to isolate "real physics" directly (q_perp cross-z) gave an
  implausible result (would exceed observational upper limits) --
  informative failure, not a number to build on.
- Patchy-window convention: xH_mean thresholds at 1e-4 / (1-1e-4).

Full detail: `docs/validation_table.md`, and the separate report-writing
handoff (`docs/HANDOFF_ksz_report.md`) if a fuller narrative is useful.

---

## 3. Concrete task for this chat

1. **Get AMBER built and running.** Public repo (github.com/hytrac/amber),
   Fortran + OpenMP, needs Intel `ifort` + MKL -- may be step zero on the
   TIFR cluster if that toolchain isn't already present.
2. **Start small.** Don't replicate literature-scale runs (2048^3, full-sky
   Healpix Nside=32768) on a first pass -- that's a serious compute/storage
   commitment. A modest box, mirroring the "quicktest -> fiducial" pattern
   used throughout this whole investigation, is the sensible entry point.
3. **Understand AMBER's native output format** -- density, velocity,
   ionization fields on a grid, presumably at specifiable redshifts/times.
   Compare directly against what `coherence_decomposition.py` and
   `limber.py` currently expect (21cmFAST coeval-box conventions) to scope
   how much adaptation is needed.
4. **Reproduce the existing single-history result first**, as a sanity
   check, before sweeping -- i.e., pick one (z_mid, Delta_z, A_z)
   combination, run the same P_total/P_diag/P_off decomposition and the
   direct/Limber calculation, and confirm the qualitative picture (excess
   power in stitched-style construction, some periodicity-like sensitivity
   if the box is kept small) looks structurally similar before trusting a
   sweep.
5. **Then sweep.** Vary `z_mid` and `Delta_z` (the two parameters most
   directly analogous to "duration of reionization," which Alvarez et al.
   2016 already showed matters a lot for the Doppler term specifically --
   worth rereading that paper's section 2.2 figures for a sense of what
   shape to expect) and track how the direct-vs-stitched gap, and its
   P_diag/P_off split, moves.

---

## 4. Codes needed for full context, prioritized

**Essential -- upload these first:**
- This handoff
- `docs/validation_table.md`
- `src/ksz_pipeline/ksz/coherence_decomposition.py` (P_total/P_diag/P_off
  machinery, heavily commented, reusable core logic)
- `src/ksz_pipeline/coeval/limber.py` (`compute_cell` -- the direct/Limber
  calculation, needs to be rerun/compared across every history, not just
  reused as a fixed baseline)
- `src/ksz_pipeline/coeval/momentum.py` (`qperp_power`, needed if
  replicating any q_perp-based check across histories)

**Useful once adapting to AMBER's actual output format:**
- `scripts/14_closure_test.py` (full pipeline pattern -- window definition,
  chi_eff, direct-vs-stitched comparison -- the template to mirror)
- `scripts/17_coherence_decomposition_fiducial.py`,
  `scripts/22_qperp_cross_z.py` (most recent, most relevant driver
  patterns)
- `src/ksz_pipeline/ksz/stitch_from_coeval.py`,
  `src/ksz_pipeline/ksz/optical_depth.py` -- **note**: these are
  21cmFAST-specific plumbing (building a lightcone from coeval boxes,
  computing visibility/patchy-mask). If AMBER already outputs
  history-controlled fields more directly, much of this may not be
  needed at all -- worth checking AMBER's own docs before assuming this
  logic needs porting.

**Probably not needed:**
- The 21cmFAST-specific convergence-sweep scripts (04/05/11/13) --
  resolution/box-size sweeps for *our* pipeline specifically, not
  directly relevant to an AMBER-native parameter study.
- `HANDOFF_ksz_ATON.md` -- separate thread, different simulation, different
  open question (periodicity vs. genuine physics, not reionization-history
  dependence).

---

## 5. One honest note

The q_perp cross-z attempt earlier this investigation swung through three
wrong answers (0% -> 33% -> 936% of stitched's excess) before being
recognized as likely a channel-mismatch problem rather than a real
signal. Worth building in a comparably skeptical self-check habit here
too -- e.g., a sanity check against known kSZ amplitude scalings (Kaiser
1984's C_ell ~ <v^2>*tau^2 scaling, or Alvarez's own quadratic-in-tau fit,
equation 15 in arXiv:1511.02846) before trusting any single AMBER-derived
number, the same way the observational-upper-limit check caught the
q_perp result here.
