# kSZ Pipeline — Validation Table

Commit `9c1b3fe` | `configs/fiducial.yaml` (BOX_LEN=800 Mpc, HII_DIM_coeval=512, N_THREADS=32, seed=37)

| | Coeval-direct | Stitched | Georgiev-Eq10 recon |
|---|---|---|---|
| Script | `02_make_ksz_coeval_boxes.py` | `03_stitched_lightcone_crosscheck.py` | `02_make_ksz_coeval_boxes.py` |
| Field shape | (512,512,512) × 29 snapshots | (512,512,2320) | N/A — power-spectrum level only |
| Units | δ (dimensionless), x_HI (fraction), v (cm/s) | same, v converted to Mpc/s | N/A |
| z-range (patchy) | 4.5–18.0 | 4.19–19.80 | 4.5–18.0 |
| Map mean | N/A | 1.6910e-6 | N/A |
| Map RMS | N/A | 1.7305e-6 (original `fiducial_stitched.log`, commit `9c1b3fe`) **or** 2.5976e-6 (rerun via `08_validation_diagnostics.py`, commit `2e90da7`) — unresolved 50% discrepancy, see flags | N/A |
| D_3000 [μK²] | 1.8441 | 0.80689 | 5.1245 |
| D_ell ratio vs. coeval-direct | — | 0.4375 at ℓ=3000; flat 0.34–0.44 at ℓ=1800–13000; spike to ~62× at ℓ≈130–180 | 0.20–0.38 across ℓ; 0.27–0.37 pre-Limber across z=4–20 |
| Reshape/ordering check | — | `ksz_map.shape == (512,512)`, no reshape | N/A |

**Flags:**
- z-range differs between coeval-direct (`xH_mean` threshold, `coeval_reion.npz`) and stitched (LOS-interpolated `x_e` threshold, `fiducial_stitched.log`).
- Stitched map RMS disagrees by ~50% between the original fiducial run (`1.7305e-6`, commit `9c1b3fe`) and a rerun of the identical calculation via `08_validation_diagnostics.py` (`2.5976e-6`, commit `2e90da7`). Per-redshift `<xH>` values between the two runs also don't match exactly (e.g. z=6.0: `0.082` vs `0.0837`). Neither commit touched any code on this path — root cause not yet identified.

Source: `fiducial_coeval.log`, `fiducial_stitched.log`, committed `.npz` products at commit `9c1b3fe`.
