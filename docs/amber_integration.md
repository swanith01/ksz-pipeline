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
| `tests/test_amber_adapter.py` | 9 tests (see below). |

## Workflow

```bash
# once: build AMBER with the patch
git clone https://github.com/hytrac/amber && cd amber
git apply /path/to/ksz-pipeline/external/amber_patch/0001-dump-fields-for-ksz-pipeline.patch
cd src && make clean; make          # needs ifort (or port), MKL, HEALPix, cfitsio

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

## First validation gate (before any sweep)

AMBER computes its own Limber C_ℓ from the same fields (`cl_ksz.txt`) and
its own per-z P_qq (`power_z=*.txt`). Two independent codes, one physics:

1. Per z: `amber_pqq_to_repo(power_*.txt)` vs `res[z]['Pqperp']`. These
   should agree to the binning level (~1%), since the fields are identical.
2. Integrated: `compute_cell(res)` vs AMBER's `cl_ksz.txt`. Expect
   agreement at the few-% level, not exactly: the z grids differ, τ uses
   volume- vs mass-weighted x_e, and `compute_cell` applies the patchy
   window.

A failure at step 1 is a bug. A failure at step 2 beyond ~5% needs an
explanation before anything is built on it.

## Not yet integrated

- **Stitched path.** Needs `stitch_from_coeval.py` (not reviewed yet).
  `adapter.velocity_from_momentum` gives (δ, v) for code that insists on
  them, and reports the fraction of cells where rho2 ≤ 0 makes v undefined.
  Separately, because AMBER's ionization is `zre > z`, the lightcone x_HII
  can be built *exactly* per LOS pixel without interpolating between
  snapshots. That changes the stitched method, though, so do it as an
  explicit variant, never silently.
- **Sweep driver** (script 30+), mirroring script 14/17 patterns.
