# HANDOFF: kSZ Stitched-vs-Direct Investigation — Report Writing

**This is a summary written by an assistant (Claude), not independently
re-verified against the live repo.** Use it to get oriented fast — for
anything going into an actual report, check `docs/validation_table.md`
and the real commit history in `github.com/swanith01/ksz-pipeline` before
citing a number from here.

---

## 1. The one-paragraph story

Two methods exist for computing the patchy-reionization kSZ angular power
spectrum from 21cmFAST: **direct** (Limber approximation, validated,
literature-consistent) and **stitched** (a real-space lightcone map).
Stitched shows ~2.3× more power than direct at ℓ=3000. This investigation
methodically ruled out several mundane explanations (window mismatch,
χ-convention, q_parallel), then found and quantitatively confirmed **box
periodicity** as a real contributor — the finite 800 Mpc simulation volume
getting repeated to fill a much longer line-of-sight window produces a
genuine, characteristic artifact. An attempt to isolate the remaining
"real physics" contribution directly (via cross-redshift q_perp
correlation) produced an implausible result — one that would exceed
current observational upper limits by 5–10× — which is itself informative:
it rules out that specific number as physical, and motivates an
independent, non-periodic simulation (ATON) as the next real step,
tracked separately.

---

## 2. Findings, ranked by confidence, with plot pointers

### High confidence — solid enough to state as results

| Finding | Evidence | Plot |
|---|---|---|
| Direct/Limber (`compute_cell`) is validated | 0.06% independent re-derivation, reproducible to several sig figs, matches literature (Georgiev+24) and data (SPT-3G, ACT, Reichardt+2021) | `closure_patchy_ksz_vs_limits.pdf` |
| Window-definition mismatch explains only 11.6% of the direct-vs-stitched gap | Direct calculation on matched window | — (numbers in `docs/validation_table.md`) |
| **Box periodicity is real** | Δχ/L_box overlay: 800 Mpc and 400 Mpc runs show the *same* oscillating peak/trough pattern at the same normalized separation; amplitude roughly doubles when box halves | `dchi_over_Lbox_overlay.pdf` — **probably the single most important plot in the whole investigation** |
| P_diag/direct match (8.5%) is robust, not a coincidence of choosing 26 radial groups | 1.3% spread across grouping choices n=15–40; full-spectrum overlay shows the SAME agreement, tightest right at ℓ=3000 | `pdiag_grouping_convergence.pdf`, `pdiag_plateau_overlay.pdf` |
| q_parallel does not explain the low-ℓ divergence | Shape test fails cleanly (not just wrong normalization) | — |

### Medium confidence — real, but with caveats worth stating explicitly

| Finding | Caveat |
|---|---|
| Low-ℓ excess is *consistent with* finite-box sample variance | Doesn't fully rule out periodicity also contributing at low ℓ — the two effects weren't cleanly separated at this end of the spectrum | `ell_over_ellmin_overlay.pdf` |
| χ-convention (single reference distance vs. pairwise) | Small, ~2–4% effect either way — not a major driver | `pairwise_chi_check.png` (not yet PDF, low priority) |

### Low confidence / informative failure — worth reporting honestly, not as a result

| Finding | What actually happened |
|---|---|
| q_perp cross-redshift correlation, attempted as a periodicity-free isolation of "real physics" | Result swung wildly across three fix attempts (0% → 33% → 936% of stitched's P_off) rather than converging — and the final number, checked against direct's own value, implies a total patchy kSZ signal ~5–10× above current SPT-3G/ACT upper limits. **This is not a physics result** — most likely a channel mismatch (q_perp is a different physical quantity than stitched's v_parallel-based construction), the same failure mode as the earlier q_parallel test. Reported as a negative/informative result, not a finding. | `qperp_cross_z.pdf` |

---

## 3. Suggested report structure (a starting point, not a mandate)

1. **Setup**: the two methods, why they should in principle agree, why
   they don't (~2.3× at ℓ=3000).
2. **Validating the baseline**: direct/Limber's own credibility (§2, high
   confidence row 1).
3. **Ruling out the boring explanations**: window, χ-convention,
   q_parallel — brief, since these are negative results establishing the
   gap is real, not the main content.
4. **The periodicity result**: the centerpiece. Δχ/L_box overlay,
   explained carefully (what "normalized by box length" means and why it
   is the discriminating test).
5. **Robustness checks**: the grouping plateau, showing the comparison
   itself is well-defined and not sensitive to an arbitrary choice.
6. **The attempted physics isolation, and why it doesn't work yet**:
   q_perp cross-z, reported honestly as informative-but-inconclusive, with
   the observational-limit sanity check as the reason it's flagged rather
   than trusted.
7. **Open questions / next steps**: ATON (tracked in a separate handoff,
   `HANDOFF_ksz_ATON.md`), the full spherical/Bessel-function non-Limber
   treatment (scoped, not built — see Alvarez et al. 2016, arXiv:1511.02846
   Appendix for the exact formula if this gets picked up later).

---

## 4. Getting figures into the report

All plots above should now exist as PDF in `data/plots/` after running
`notebooks/exploratory/make_report_pdfs.py` (pure local replotting from
already-downloaded `.npz` files, no cluster needed — see that script's own
comments for which file feeds which plot). Two files were already PDF
before this pass (`closure_patchy_ksz_vs_limits.pdf`,
`closure_total_ksz_vs_observations.pdf`) and needed no regeneration.

If a LaTeX-specific font/size adjustment is wanted later (matching a
journal template, consistent font sizing across all figures, etc.), that's
a small, mechanical follow-up on top of `make_report_pdfs.py` — happy to
help with that once the report's actual figure requirements are known.

---

## 5. One honest note for whoever writes this up

Several numbers in this investigation were wrong before they were right —
a missing physical prefactor, a reversed array, a silently-wrong CLI
override, a self-check that was itself flawed. All were caught and fixed,
and the final findings above reflect the corrected versions. Worth stating
plainly in the report's own methods section that this was an iterative
process with real course-corrections, rather than presenting the final
numbers as if they were obtained cleanly on the first attempt — that's
both more honest and, frankly, more scientifically convincing than a
suspiciously clean narrative would be.
