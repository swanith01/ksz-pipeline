# Handoff: wrap-cycle lightcone stitching (ksz-pipeline → ksz-lae-xcorr)

Source branch: `main` (NOT `amber-integration`, which predates the fix and has
no `wrap_cycle` code). Built from commit `0c3b578`
("oct5: simple two-panel Nikolic comparison figure"). The wrap-cycle code itself
was added in `41157a5` ("add per-wrap-cycle rotation (wrap_cycle_seed),
additive only"); `323ef1e` added `astro_params`/`flag_options` passthrough.

Everything below was read from the code at that commit. Items marked
**NOT VERIFIED** or **TO FILL** are not established by the repo.

---

## a. The algorithm

**Problem.** A coeval box of side `BOX_LEN` is periodic. The lightcone is built
by walking along the LOS (third array axis) and, at each target redshift,
reading slab `box[:, :, y_cell]`. Once the LOS distance exceeds `BOX_LEN`,
`y_cell` wraps and the *same* slab is read again. A single fixed rotation does
not help: the same `(ir, jr)` applied at the same `y_cell` returns a
bit-identical slab (verified in the code's own notes). The rotation must change
between visits to the same box.

**Wrap cycle.** `wrap_cycle_index(z, z0, cell_size, ngrid)`
```
d     = chi(z) - chi(z0)                 # comoving Mpc, astropy Planck18
cycle = int(d // (cell_size * ngrid))    # cell_size*ngrid == BOX_LEN
```
- One cycle = exactly one `BOX_LEN` of comoving distance along the LOS.
- Origin `z0 = np.min(z_arr)` (set inside `stitch_lightcone_from_coeval`). In
  script 17 `z_arr = build_los_z_grid(z_min, z_max, cell_size)` with
  `z_min=4.0, z_max=20.0`, so `z0 = 4.0`: cycle 0 starts at the LOW-z end and
  cycle index increases with z. It is measured over the whole z grid, not the
  matched analysis window z=[4.5, 18].
- Over z=4→20 the LOS length is 3624.8 Mpc. Cycles touched: 800 Mpc box → 0–4
  (5 cycles); 500 Mpc → 0–7 (window z≤18 reaches cycle 6); 400 Mpc → 0–9
  (window reaches cycle 8).
- `y_cell = round(d/cell_size) % ngrid` (`comoving_pixel`). Because
  `build_los_z_grid` samples at exact multiples of `cell_size`, the `y_cell`
  wrap points and the cycle-index increments coincide (checked for the
  400 Mpc/256 grid, first four boundaries: positions 256, 512, 768, 1024).
  `cycle` uses a floor on a float, so this alignment relies on `d/BOX_LEN`
  not landing a hair below an integer. NOT VERIFIED beyond the cases above.

**Angle.** `cycle_angle(cycle_idx, seed)`
```
rng   = np.random.default_rng((int(seed), int(cycle_idx)))
angle = float(rng.uniform(0.0, 360.0))      # degrees
```
- RNG: NumPy `Generator` (default PCG64), seeded by the 2-integer sequence
  `(wrap_cycle_seed, cycle_idx)`. No shared state, no caching needed: the same
  pair always gives the same angle.
- Cycles are **independent** draws. There is no smoothing, no correlation
  between consecutive cycles, no constraint on angle differences.
- Cycle 0 is also rotated (it is not the identity).
- Reference values (numpy 2.4.4; use them to check a reimplementation, but
  NumPy does not formally guarantee Generator streams across versions):

  | seed | cycle 0 | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|---|
  | 123 | 245.6467 | 114.6296 | 131.6559 | 151.2682 | 61.4123 | 192.6411 |
  | 101 | 339.6717 | 14.5908 | 51.8010 | 60.4586 | 214.4531 | 138.8351 |

**Sharing across snapshots and fields.** In `stitch_field`, for each target
redshift `z_arr[n]` one `cycle` and one rotation are computed, and that same
rotation is applied to every snapshot box whose slab is read at that `y_cell`
before interpolating across snapshots in redshift. The angle is keyed on LOS
position only, never on the snapshot. `stitch_lightcone_from_coeval` passes the
same `wrap_cycle_seed` to the density, xH and v_z stitches, so all three fields
share the rotation at every position. The rotation coordinates are cached per
cycle (`coords_by_cycle`).

**Default.** `wrap_cycle_seed=None` runs the original code path, unchanged:
single `angle_deg` (script 17 always passes `0.0`), nearest-index rotation. With
`angle_deg=0` that is plain periodic box reuse. This is the "UNFIXED" control.
When `wrap_cycle_seed` is set, `angle_deg` is ignored.

## b. How a rotated slab is sampled

- **Wrap-cycle path: bilinear interpolation** in the transverse plane:
  `scipy.ndimage.map_coordinates(box[:, :, y_cell], [ic, jc], order=1, mode='wrap')`
  with continuous coordinates
  `ic = cos(a)*i - sin(a)*j`, `jc = sin(a)*i + cos(a)*j`
  (`i, j = np.meshgrid(arange(ngrid), arange(ngrid), indexing='ij')`;
  `a = deg2rad(angle)`). Rotation is about index origin (0,0), not the box
  centre, and acts on the two transverse axes only.
- **Fixed-angle path: nearest-index** `np.round(...) % ngrid` on both axes
  (`rotated_indices`, `get_slab`). The code notes this aliases 10–31% of pixels
  at generic angles (zero at multiples of 90°), which is why the wrap-cycle
  path uses bilinear.
- Across snapshots: `scipy.interpolate.interp1d(snap_z, slabs, axis=-1,
  bounds_error=False, fill_value="extrapolate")`, evaluated at `z_arr[n]`.
  (Linear extrapolation outside the snapshot range. Some docstrings say
  "flatly"; the code does linear.)
- **CAVEAT, found while writing this, NOT in the original docs:** scipy's
  `mode='wrap'` is *not* a true period-`n` wrap. I tested it (scipy 1.17.1) on
  `a = arange(8)`: `x=8.0` returns `1.0` (period `n-1 = 7`), whereas
  `mode='grid-wrap'` returns `0.0` (period 8) and gives the correct
  last-to-first blend at `x=7.5`. The code uses `'wrap'`, so for rotated
  coordinates outside `[0, n-1]` the box is wrapped with period `n-1`, a
  one-pixel-per-wrap distortion at the seams. A reimplementation should use
  `mode='grid-wrap'` unless it needs to reproduce our numbers bit for bit.
  The numerical effect on D_ell was NOT measured.
- **v_z needs no transformation.** The rotation acts only on the two transverse
  index axes, i.e. it is a rotation about the LOS axis. v_z is the LOS
  component, a scalar under that rotation, so it is stitched exactly like
  density and xH. Confirmed from the code: `run_coeval_fields` returns
  `vx, vy, vz` but `stitch_lightcone_from_coeval` keeps only `vz` and discards
  `vx, vy`. If the other assistant ever uses the transverse components, those
  would have to be rotated as a vector; ours never are.

## c. Where it is implemented (`src/ksz_pipeline/ksz/stitch_from_coeval.py`)

| line | signature |
|---|---|
| 81 | `build_los_z_grid(z_min, z_max, cell_size, z_oversample=4000)` — uniform comoving spacing `= cell_size` |
| 129 | `comoving_distance_mpc(z)` — astropy Planck18 |
| 133 | `comoving_pixel(z, z0, cell_size, ngrid)` → `int(round(d/cell_size)) % ngrid` |
| 139 | `rotated_indices(ngrid, angle_deg)` — nearest-index, old path |
| 157 | `get_slab(box, y_cell, ir, jr)` |
| 196 | `wrap_cycle_index(z, z0, cell_size, ngrid)` |
| 208 | `cycle_angle(cycle_idx, seed)` |
| 222 | `rotated_coords(ngrid, angle_deg)` — continuous, unwrapped |
| 241 | `get_slab_bilinear(box, y_cell, ic, jc)` |
| 254 | `stitch_field(snapshot_boxes, snap_z, z_arr, z0, cell_size, ngrid, angle_deg=0.0, wrap_cycle_seed=None)` → `(ngrid, ngrid, len(z_arr))` float32 |
| 334 | `stitch_lightcone_from_coeval(z_snapshots, z_arr, HII_DIM, BOX_LEN, cache_dir, angle_deg=0.0, N_THREADS=None, random_seed=None, wrap_cycle_seed=None, astro_params=None, flag_options=None)` → dict `density` (RAW delta, caller adds 1), `xH_box`, `velocity_z` [cm/s], `z_arr`, `pos_axis` |

Related: `src/ksz_pipeline/coeval/fields.py::run_coeval_fields` (box loader,
conventions in section g); `src/ksz_pipeline/ksz/coherence_decomposition.py`
(`compute_ksz_map_per_slice` l.71, `decompose_p_total_diag_off` l.118,
`group_slices_by_snapshot` l.188); driver
`scripts/17_coherence_decomposition_fiducial.py` (call at l.392; CLI flag
`--wrap-cycle-seed`). Wrap-cycle is `--source coeval` only; for `--source
native` the flag is ignored with a warning.

## d. The "P_diag vs direct, fiducial resolution" figure with wrap_cycle_seed=123

Committed figure: `plots/30-sep/coherence_decomposition_fiducial_wrapcycle123.png`
(left panel title "P_diag vs direct, fiducial resolution"; right panel
"P_off vs radial separation: real vs shifted, wrap_cycle_seed=123"). Baseline
(no rotation) counterpart: `plots/30-sep/coherence_decomposition_fiducial_baseline.png`.

**Reproduction.**
```
python scripts/17_coherence_decomposition_fiducial.py \
    --config configs/fiducial.yaml --source coeval --wrap-cycle-seed 123
```
(`--shift-seed` default 42; no `--box-len/--hii-dim` override;
`--random-seed` not passed, so the cosmic seed is the config's
`random_seed: 37`; `--skip-direct` not passed.)
Config `configs/fiducial.yaml`: BOX_LEN 800, HII_DIM_coeval 512
(dx = 1.5625 Mpc), 29 z_snapshots 20→4, z_min 4, z_max 20, N_THREADS 32,
HII_EFF_FACTOR 30. Matched window z=[4.5, 18], chi_eff = 8504.03 Mpc (from
`closure_test.npz`, script 14).

**Box size: your brief says 400 Mpc, the repo evidence says 800 Mpc. TO CONFIRM.**
- A 400 Mpc run (`--box-len 400 --hii-dim 256`) gets the tag `_box400`, so its
  default output is `coherence_decomposition_box400_wrapcycle123.{npz,png}`.
  The committed file has no `box400` in the name.
- The committed figure's stitched curves start at ell ≈ 100. The 800 Mpc
  decomposition grid starts at ell = 102.6; the 400 Mpc grid starts at 144.4
  (both from the tracked old-method `.npz` files).
- The title string "fiducial resolution" is the generic coeval title for any
  box size, so it does not distinguish the two.
- The committed filename contains `_fiducial_`, which the script itself would
  not write for a wrap-cycle run (it writes `coherence_decomposition_wrapcycle123.png`);
  the file was evidently renamed when copied into `plots/30-sep/`.
- Verify on swarm with the `box_len` key stored in the new-style `.npz`:
  `python -c "import numpy as np; d=np.load('data/products/coherence_decomposition_wrapcycle123.npz'); print(d['box_len'], d['hii_dim'], d['wrap_cycle_seed'])"`
  If that file does not exist, try `..._box400_wrapcycle123.npz`.

**D_3000 numbers [μK²]**

Old method, no rotation (`angle_deg=0`, nearest-index), from the tracked
`data/products/coherence_decomposition_fiducial.npz` (800 Mpc; this file
predates `box_len`/`wrap_cycle_seed` keys, so "no rotation" is inferred from
the script default at the time, NOT stored):

| | direct | stitched P_total | stitched P_diag | P_off |
|---|---|---|---|---|
| 800 Mpc / 512³, old method | 1.782 | 4.089 | 1.631 | 2.458 |
| 400 Mpc / 256³, old method (`coherence_decomposition_box400.npz`) | 1.524 | 6.401 | 1.219 | 5.182 |

Old-method P_total/direct = 2.29 at 800 Mpc (the former "~2× stitched vs
direct" disagreement).

Wrap-cycle, seed 123 — **TO FILL.** The `.npz` for this run is not in the git
history, only the PNG, so I cannot give exact values. Run on swarm from the
repo root and paste the result here:
```
python - <<'EOF'
import numpy as np
d = np.load('data/products/coherence_decomposition_wrapcycle123.npz')
g = lambda k: float(np.interp(3000, d['ell_dec'], d[k]))
print('box', float(d['box_len']), 'hii', int(d['hii_dim']), 'seed', int(d['wrap_cycle_seed']))
print('direct', float(d['d3000_direct']))
print('P_total', g('Dl_total'), 'P_diag', g('Dl_diag'), 'P_off', g('Dl_off'))
print('P_total shifted control', float(np.interp(3000, d['ell_shift'], d['Dl_total_shift'])))
EOF
```
```
box ___  hii ___  seed 123
direct ___   P_total ___   P_diag ___   P_off ___   P_total (shifted control) ___
```
(Qualitatively, in the figure P_total with rotation sits close to P_diag and
direct near ell ~ 1000–3000, unlike the old method where P_total is ~2.3× direct.)

## e. Other wrap_cycle_seeds

- **Fiducial astrophysics (HII_EFF_FACTOR model):** only seed 123 has a
  committed figure. No other seed's outputs are in the repo. I cannot say
  whether others were run and not committed.
- **Nikolić et al. 2023 replication** (500 Mpc / 256³, mass-dependent-ζ
  astro params, `configs/nikolic_mesinger_nophotoncons.yaml`), seeds 101–106,
  where `wrap_cycle_seed = --random-seed`. D_3000 [μK²] at the same seeds and
  initial conditions, stitching rotated (FIXED) vs plain box reuse (UNFIXED):

  | seed | FIXED total | diag | off | UNFIXED total | diag | off |
  |---|---|---|---|---|---|---|
  | 101 | 0.5818 | 0.6051 | −0.0233 | 1.6724 | 0.5557 | 1.1167 |
  | 102 | 0.4658 | 0.4669 | −0.0011 | 1.2673 | 0.4451 | 0.8222 |
  | 103 | 0.5208 | 0.5548 | −0.0340 | 1.6362 | 0.5669 | 1.0693 |
  | 104 | 0.4160 | 0.4055 | 0.0105 | 1.2807 | 0.4193 | 0.8615 |
  | 105 | 0.6073 | 0.6183 | −0.0110 | 1.8433 | 0.6121 | 1.2312 |
  | 106 | 0.4109 | 0.4328 | −0.0220 | 1.2519 | 0.4320 | 0.8199 |
  | mean | 0.500 ± 0.083 | 0.514 | −0.013 | 1.492 ± 0.257 | 0.505 | +0.987 |

  Unfixed/fixed total = 2.98 (per seed 2.7–3.1); P_diag nearly identical, the
  difference is entirely P_off. Products:
  `data/products/coherence_decomposition_wrapcycle10{1-6}_nikolic*.npz` (fixed;
  seed 101's plain name holds an INVALID photon-conservation run, the valid
  one is `..._nikolic_nophotoncons.npz`) and
  `coherence_decomposition_seed10{1-6}_nikolic.npz` (unfixed). Summary:
  `data/products/nikolic_closing_summary.npz`.

## f. Caveats and open questions

Behaviour at ell < 400 and ell > 4000, from the old-method fiducial `.npz`
(800 Mpc, no rotation). The wrap-cycle seed-123 values need the `.npz`
(section d); what I can say about it is read by eye from the PNG, ±15%.

| ell | direct | P_total | P_diag | P_off | diag/direct | total/direct |
|---|---|---|---|---|---|---|
| 150 | 0.071 | 0.696 | 9.166 | −8.470 | 128 | 9.7 |
| 200 | 0.216 | 0.820 | 10.504 | −9.684 | 49 | 3.8 |
| 300 | 0.443 | 1.080 | 4.890 | −3.810 | 11 | 2.4 |
| 400 | 0.713 | 1.607 | 3.074 | −1.467 | 4.3 | 2.3 |
| 1000 | 2.249 | 5.526 | 2.334 | 3.193 | 1.04 | 2.5 |
| 3000 | 1.782 | 4.084 | 1.630 | 2.454 | 0.91 | 2.3 |
| 4000 | 1.524 | 3.491 | 1.358 | 2.134 | 0.89 | 2.3 |
| 6000 | 1.122 | 2.481 | 0.982 | 1.499 | 0.88 | 2.2 |
| 10000 | 0.602 | 1.122 | 0.502 | 0.620 | 0.83 | 1.9 |
| 20000 | 0.206 | 0.204 | 0.169 | 0.035 | 0.82 | 1.0 |

- **ell < 400:** stitched P_diag is far above direct (4× at ell=400, ~50× at
  200, >100× at 150), and P_off is large and negative so P_total is only
  partly above direct. In the seed-123 figure P_diag has the same spike
  (≈ 10–25 μK² at ell ~ 100–250) while P_total is flat around 1.5–2 μK². The
  cause is NOT understood. The 800 Mpc box's fundamental mode is
  ell ≈ 2π·chi_eff/L ≈ 67, so the lowest bins have very few modes. Treat
  ell < ~400 as unvalidated; all quoted results are at ell = 3000.
- **ell > 4000:** under the old method, P_diag/direct drifts 0.89 → 0.82 and
  P_off goes to ~0 by ell ≈ 2×10⁴. In the seed-123 figure the stitched curves
  at ell ≈ 10⁴ sit visibly lower relative to direct (roughly half, read by
  eye) than in the old-method table. Hypothesis, NOT tested: bilinear
  rotation smooths the field. At ell = 3000 the effect is small (Nikolić
  P_diag 0.514 fixed vs 0.505 unfixed, ≈ 2%), but k_perp grows with ell, so
  it would matter more at ell ≳ 10⁴. A direct test: stitch with a 90°
  rotation (exact, no interpolation) vs a generic angle and compare P_diag.
- **`mode='wrap'` period n−1** (section b): not measured.
- **Rotation about the index origin**, not the box centre: no effect expected
  on statistics, but note it for a reimplementation.
- **Independent angles per cycle** may leave residual correlations between
  cycles through the *same* box's low-k modes at the cycle boundary; the
  shifted-control comparison (right panel) is the only test we have.
- **No tests** exist for the stitching or wrap-cycle code (see section h).
- **PHOTON_CONS** is invalid through this pipeline (per-snapshot
  `run_coeval`); no result here uses it. Nikolić replication caveats are in
  `README.md`.
- Seed-123 box size is unconfirmed (section d).

## g. Velocity and cosmology conventions

- Box loading: `coeval/fields.py::run_coeval_fields(z, HII_DIM, BOX_LEN,
  cache_dir, N_THREADS=None, random_seed=None, astro_params=None,
  flag_options=None)` calls `py21cmfast.run_coeval` (v3.3.1, `write=True`).
  Returns `delta` (RAW δ, mean ≈ 0, NOT 1+δ), `xH_box`, and `vx, vy, vz`.
- **v3 conversion** (`coeval/velocity.py::velocity_conversion_factor`):
  ```
  v_comoving [km/s] = lowres_v{x,y,z} * D(z) * f(z) * H(z) / (1+z)
  ```
  then `*1e5` → cm/s in `run_coeval_fields`. `lowres_v*` is 21cmFAST's initial
  displacement field Ψ. `D(z)` is the linear growth factor from
  `D ∝ H(z) ∫_z^∞ (1+z')/H(z')³ dz'` normalised to 1 at z=0; `f(z) =
  Ω_m(z)^0.55` (Linder). `H(z)` is astropy Planck18. The result is the
  COMOVING peculiar velocity (the ds/a⁴ factor in the C_ell integral needs
  comoving v). Validated to ~1% against linear theory at z=0.5–15, and the
  dz-convergence run lands on Reichardt's D_pkSZ.
- **Density convention:** the stitched `density` is raw δ. Callers add 1.0
  exactly once before `compute_ksz_map` (script 17: `density_1plus = 1.0 +
  stitched['density']`). `x_HII = 1 − xH`; `v_los_Mpc_s = velocity_z / MPC_CM`
  with `MPC_CM = 3.0856775814913673e24`.
- **Cosmology, post-processing:** `astropy.cosmology.Planck18` everywhere
  (`H(z)`, comoving distance, `Ω_m(z)`, mean electron density via
  `ne0_cgs(Y_He=0.24, include_He=True)` in `utils/constants.py`).
- **Cosmology, simulation:** no `cosmo_params` is set in any config, so
  21cmFAST uses its own py21cmfast 3.3.1 defaults, which are close to but not
  identical to astropy Planck18 (documented in `MIGRATION.md` and
  `utils/constants.py`). Exact values NOT recorded here; print
  `py21cmfast.CosmoParams()` in the p21c_v3 env. The Nikolić replication does
  NOT match Planck 2020 either.
- **D_ell definition:** D_ell = ell(ell+1)C_ell/2π in μK²; D_3000 by
  `np.interp(3000, ell_dec, Dl_*)` from the saved arrays.

## h. Tests

There are **no tests for the stitching or wrap-cycle code.** `tests/` contains
only `test_offdiag_projection.py` (six tests of the off-diagonal projection
estimator on synthetic fields: mixed correlator vs configuration space,
zero-separation = k_par integral, window integral = Fourier form, window
overlap normalisation, Limber reduction in the median, separation suppresses
cross terms). None of them import `stitch_from_coeval`.

What I did check, in a scratch session without py21cmfast (the module imports
only numpy/scipy/astropy at the top), not as committed tests:
`cycle_angle` determinism and the reference angles above; cycle counts per box
size; `y_cell`-wrap/cycle-boundary alignment at the 400 Mpc/256 grid; the
scipy `mode='wrap'` period behaviour. The synthetic checks were not turned
into a test file.
