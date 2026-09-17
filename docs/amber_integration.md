# AMBER integration

**Design:** AMBER supplies *fields* only. Every kSZ method (`compute_cell`,
`qperp_power`, `coherence_decomposition`) stays the repo's existing,
validated code, so any change across a reionization-history sweep is a
change in the simulation input, never in the method.

## Files

| path | what |
|---|---|
| `external/amber_patch/0001-dump-fields-for-ksz-pipeline.patch` | 1 call + 1 subroutine in AMBER's `cmbreion.f90`: dumps `rho2`, `mom2` at each CMB shell midpoint (the same fields AMBER's own Limber uses). Applies cleanly to upstream `main` (checked 2026-09-10). Not compiled against AMBER here (no MKL/HEALPix in the dev sandbox); the byte layout *is* tested with gfortran. |
| `scripts/make_amber_input.py` | Writes `input.txt` + `linpowspec.txt` + `amber_run.json` for one (z_mid, Δz, A_z), cosmology pinned to astropy Planck18 (the cosmology `limber.py` hardcodes). |
| `src/ksz_pipeline/amber/io.py` | Readers for `zre_*.dat`, `fields_*.dat`, `power_*.txt`, `cl_ksz.txt`. |
| `src/ksz_pipeline/amber/adapter.py` | Unit/convention mapping; `results_qperp_from_amber()` produces `compute_cell`'s input dict directly. |
| `src/ksz_pipeline/coeval/momentum.py` | Refactor: `qperp_power`'s body moved verbatim into `_qperp_power_from_q`; new `qperp_power_from_momentum`. `qperp_power` output is bit-identical to before. |
| `external/amber_patch/healpix_stub.f90` | Stand-in for the 4 HEALPix/FITS modules AMBER `use`s, so it builds with no HEALPix/cfitsio. Full-sky maps disabled (keep `Map = no`); every other output is unaffected. Stubs `stop` with a clear message if called. |
| `external/amber_patch/Makefile.ksz`, `build_amber.sh` | One-command build: clones AMBER at the pinned commit, applies the patch, adds the stub, builds with ifort+MKL. |
| `scripts/30_amber_validation_gate.py` | The validation gate below, automated. Run on every new AMBER config. |
| `src/ksz_pipeline/coeval/limber.py` | `compute_cell` gains optional `tau0=` (default `None` = previous behaviour exactly). |
| `tests/test_amber_adapter.py` | 11 tests (see below). |

## Workflow

```bash
# once: build AMBER (clones + patches + builds; no admin rights needed)
bash external/amber_patch/build_amber.sh          # -> ~/amber/src/amber.x
# TIFR: Intel oneAPI 2021.1.1 lives at /apps/intel/oneapi but has NO module
# file, so source it directly (build_amber.sh does this itself):
#   source /apps/intel/oneapi/setvars.sh

# per history
python scripts/make_amber_input.py --out runs/amber_q01 \
    --L 256 --N 128 --zmid 8 --zdel 4 --zasy 3
cd runs/amber_q01 && mkdir -p output/{cosmo,reion,grf,lpt,esf,mesh,cmb}
amber.x < input/input.txt > output/log.txt
```

```python
from ksz_pipeline.amber import io, adapter
from ksz_pipeline.coeval.limber import compute_cell
zre = io.read_zre(io.find_zre_file('output/reion'), N)
res = adapter.results_qperp_from_amber(io.list_field_files('output/cmb'),
                                       zre, N, L_box_mpc_h=256, h=0.6766)
ells, D_ell, *_ = compute_cell(res)
```

## Conventions (each checked against AMBER source)

- **Lengths / k:** AMBER Mpc/h, h/Mpc → repo Mpc, 1/Mpc.
- **Velocity:** AMBER `mom2 = (1+δ)v` in km/s, with v the *proper* peculiar
  velocity (`unit%vel = a·L/t`). This is the same physical quantity the
  repo already uses: validation_table §4's `Ψ·D·f·H/(1+z)` equals `a·dx/dt`,
  even though §4 labels it "comoving". So the mapping is ×1e5 only.
  **Do not apply an extra (1+z).** Consider relabelling §4.
- **Ionization:** binary; a cell is ionized iff `z < zre` (AMBER's own test).
- **Electrons:** AMBER `ne0 = nH + 2nHe` times `x_e = (X+Y/4)/(X+Y/2)`
  gives nH + nHe, the same as `ne0_cgs()`. With X=0.76, Y=0.24 and Planck18
  Ω_b this reproduces `ne0_cgs()` = 2.064357e-7 to 1.5e-7 (checked by the
  input script on every run).
- **P_qq → P_qperp:** ×h⁻³ ×1e10 ×½ ×x_e⁻², k ×h (`amber_pqq_to_repo`).
  Exact mode by mode to 1e-10.
- **Cosmology:** χ(z) from AMBER's H(z) with the generated parameters
  matches astropy Planck18 to 6e-6 over z = 4–20.
- **x_HI means:** AMBER's (z_mid, Δz, A_z) are *mass-weighted*;
  `compute_cell`'s patchy window and τ use the *volume-weighted* `xH_mean`.
  The adapter stores both (`xH_mean`, `xH_mass`). Label sweeps by AMBER's
  mass-weighted parameters, and don't read z_mid off an `xH_mean = 0.5`
  crossing.

## Validation gate (run on every new AMBER config)

```bash
python scripts/30_amber_validation_gate.py --run runs/amber_q01
```

AMBER computes its own P_qq and Limber C_ℓ from the same fields, so this
compares two independent codes on one set of fields. Results of the first
real run (L=64 Mpc/h, N=64, z_mid=8, Δz=4, A_z=3, gfortran sandbox build,
2026-09-10):

| gate | what | result |
|---|---|---|
| 1 | per-z P_qperp, ours vs AMBER, **same bins** (no interpolation) | max dev 4.7e-4 |
| 2a | rebuild `cl_ksz.txt` from AMBER's own power files | max dev 0.10% (ℓ≥2000) |
| 2b | `compute_cell` vs that formula over the patchy window | D_3000 ratio 1.017 |

Two things this gate caught, both worth knowing:

- **AMBER interpolates P_qq with a natural cubic spline** (`mkl.f90
  spline_cubic`, `DF_PP_NATURAL`/`DF_BC_FREE_END`), not linearly. Using
  linear interpolation gave 3.4% errors at low ℓ on a small box, where
  k=ℓ/r lands in the first couple of k bins. Anything reproducing AMBER's
  Limber sum must spline.
- **τ below the patchy window is history-dependent.** `compute_cell`
  previously always took it from `analytic_tau_below()`, one fixed
  history. Across a (z_mid, Δz) sweep the true value moves, and a wrong
  τ₀ rescales D_ℓ by exp(−2Δτ) — 13% in the test run. `compute_cell` now
  accepts `tau0=`, and `adapter.tau_below_amber()` reads AMBER's own
  `tau.txt`. **Use it for every AMBER run.** The default is unchanged, so
  the 21cmFAST path is unaffected.

Also reported: **24% of AMBER's D_3000 comes from below the patchy
window** (post-reionization kSZ). `compute_cell` excludes it by design.
That fraction itself varies with the history, so quote patchy-only numbers
consistently across the sweep, and don't compare them directly to
published total-kSZ D_3000 values.

## Build notes (verified 2026-09-10)

Tested against AMBER commit `9703f51`. The build was rehearsed in a Linux
sandbox with **gfortran 11 + pip MKL** (no ifort available there): all 25
objects compiled and a full L=64/N=64 run completed, producing `zre`,
`power_z=*.txt`, `cl_ksz.txt`, `tau.txt` and 28 `fields_*.dat` snapshots.
Two portability points found, both handled for ifort already:

- AMBER uses Intel's `form='binary'` in `mesh.f90` etc. gfortran needs
  `access='stream',form='unformatted'`; ifort does not. (The dump patch
  already uses the portable form.)
- `helper.f90` calls IFPORT's `qsort`. ifort supplies it; a gfortran build
  would need a shim.

So on TIFR use ifort via `build_amber.sh`. A gfortran fallback is possible
but needs those two edits — ask if oneAPI ever becomes unavailable.

## Not yet integrated

- **Stitched path.** Needs `stitch_from_coeval.py` (not reviewed yet).
  `adapter.velocity_from_momentum` gives (δ, v) for code that insists on
  them, and reports the fraction of cells where rho2 ≤ 0 makes v undefined.
  Separately, because AMBER's ionization is `zre > z`, the lightcone x_HII
  can be built *exactly* per LOS pixel without interpolating between
  snapshots. That changes the stitched method, though, so do it as an
  explicit variant, never silently.
- **Sweep driver** (script 30+), mirroring script 14/17 patterns.
