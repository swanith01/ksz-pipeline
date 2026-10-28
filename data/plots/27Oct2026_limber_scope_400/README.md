# 27 Oct 2026 - Limber-scope test, L = 400 Mpc / 256^3 (dx = 1.5625 Mpc)

Made with `scripts/45_limber_scope_dchi.py` from `data/products/theta_groups_box400{,_wrapcycle123,_native}.npz`
(script 17 `--save-theta-width 50`; 63 uniform groups, ~49.5 Mpc wide, matched window z = 4.5-18, chi_eff = 8504 Mpc).
D_trunc = D_diag + cross-power of group pairs with |dchi| < L/2; D_far = pairs with |dchi| >= L/2 (includes box replicas).
Direct = coeval Limber at the same box, D_3000 = 1.524 uK^2.

| lightcone | l band | direct | total | diag | trunc | far | trunc/dir | total/dir |
|---|---|---|---|---|---|---|---|---|
| stitched, no rotation | 300-1000 | 0.876 | 3.243 | 4.183 | 0.340 | 2.902 | 0.39 | 3.70 |
| stitched, no rotation | 2000-4000 | 1.550 | 6.043 | 1.235 | 1.247 | 4.796 | 0.80 | 3.90 |
| stitched, wrap-cycle 123 | 300-1000 | 0.876 | 0.752 | 4.797 | 1.829 | -1.077 | 2.09 | 0.86 |
| stitched, wrap-cycle 123 | 2000-4000 | 1.550 | 1.210 | 1.305 | 1.205 | 0.005 | 0.78 | 0.78 |
| native v3 rectilinear | 300-1000 | 0.876 | 0.007 | 0.003 | 0.001 | 0.006 | 0.00 | 0.01 |
| native v3 rectilinear | 2000-4000 | 1.550 | 0.041 | 0.007 | 0.007 | 0.034 | 0.00 | 0.03 |

- l ~ 3000: box replicas carry 79% (stitched, no rotation) and 83% (native v3) of D_l; wrap-cycle removes them (far = 0.005).
  Dropping far pairs gives the same D_trunc with and without rotation (1.25 vs 1.21). Near cross-shell power is +1% / -8% of diag,
  so no non-Limber residual is resolvable within L/2 at l ~ 3000. D_trunc ~ 0.8 x direct: likely the known stitched-vs-direct
  resolution gap, not Limber (check at 800 Mpc).
- l = 300-1000: strong cancellation between neighbouring shells (LOS-velocity modes) extends beyond L/2 = 200 Mpc at this box
  size, so physics and replicas cannot be separated at low l here. Needs the 800 Mpc runs.
- Native v3 amplitude ~180x low in power: native velocity conversion (raw/H0 in native_lightcone.py) suspected, pending script 39.
  Spike positions and the far fraction do not depend on the velocity units.

**Correction (28 Oct 2026):** the native v3 rows above used the OLD loader (density-1 and velocity/H0, both wrong). With the fixed loader
(cb3aad1) native v3 at l = 2000-4000 gives total 7.543, trunc 1.527 (= 0.98 x direct), far/total 0.80. See ../28Oct2026_limber_scope_all/.
