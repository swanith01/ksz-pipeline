# kSZ Pipeline — Validation Table

Commit `9c1b3fe` | `configs/fiducial.yaml` (BOX_LEN=800 Mpc, HII_DIM_coeval=512, N_THREADS=32, seed=37)

| | Coeval-direct | Stitched | Georgiev-Eq10 recon |
|---|---|---|---|
| Script | `02_make_ksz_coeval_boxes.py` | `03_stitched_lightcone_crosscheck.py` | `02_make_ksz_coeval_boxes.py` |
| Field shape | (512,512,512) × 29 snapshots | (512,512,2320) | N/A — power-spectrum level only |
| Units | δ (dimensionless), x_HI (fraction), v (cm/s) | same, v converted to Mpc/s | N/A |
| z-range (patchy) | 4.5–18.0 | 4.19–19.80 | 4.5–18.0 |
| Map mean | N/A | not recorded | N/A |
| Map RMS | N/A | 1.7305e-6 | N/A |
| D_3000 [μK²] | 1.8441 | 0.80689 | 5.1245 |
| D_ell ratio vs. coeval-direct | — | 0.4375 at ℓ=3000; flat 0.34–0.44 at ℓ=1800–13000; spike to ~62× at ℓ≈130–180 | 0.20–0.38 across ℓ; 0.27–0.37 pre-Limber across z=4–20 |
| Reshape/ordering check | — | `ksz_map.shape == (512,512)`, no reshape | N/A |

**Flags:**
- z-range differs between coeval-direct (`xH_mean` threshold, `coeval_reion.npz`) and stitched (LOS-interpolated `x_e` threshold, `fiducial_stitched.log`).
- Stitched map mean was never printed or saved — only RMS and the input-field mean `1+δ=0.9997` are on record.

Source: `fiducial_coeval.log`, `fiducial_stitched.log`, committed `.npz` products at commit `9c1b3fe`.
