# 28 Oct 2026 - Limber-scope test, all lightcones (script 45)

Inputs: data/products/theta_groups_{box400,box400_wrapcycle123,box400_native,box400_nativev4,box400_nativev4ang,fiducial,wrapcycle123}.npz
(script 17 --save-theta-width 50; 63 groups ~49.5 Mpc; window z = 4.5-18; chi_eff = 8504 Mpc; dx = 1.5625 Mpc at both boxes).
Native v3 uses the FIXED loader (cb3aad1: density is raw delta, v_pec = raw/(1+z), measured by script 39).
Native v4 = py21cmfast 4.1 with its own astro defaults: compare it by internal ratios, not absolute D_l.
D_trunc = D_diag + pairs with |dchi| < L/2; far = pairs with |dchi| >= L/2 (contains box replicas).

| lightcone | L | l band | direct | total | diag | trunc | far | trunc/dir | total/dir | far/total | near/diag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stitched, no rotation | 400 | 300-1000 | 0.876 | 3.243 | 4.183 | 0.340 | 2.902 | 0.39 | 3.70 | 0.90 | -0.92 |
| stitched, no rotation | 400 | 2000-4000 | 1.550 | 6.043 | 1.235 | 1.247 | 4.796 | 0.80 | 3.90 | 0.79 | 0.01 |
| stitched, wrap-cycle | 400 | 300-1000 | 0.876 | 0.752 | 4.797 | 1.829 | -1.077 | 2.09 | 0.86 | -1.43 | -0.62 |
| stitched, wrap-cycle | 400 | 2000-4000 | 1.550 | 1.210 | 1.305 | 1.205 | 0.005 | 0.78 | 0.78 | 0.00 | -0.08 |
| native v3 rectilinear | 400 | 300-1000 | 0.876 | 3.676 | 4.960 | 0.303 | 3.373 | 0.35 | 4.20 | 0.92 | -0.94 |
| native v3 rectilinear | 400 | 2000-4000 | 1.550 | 7.543 | 1.577 | 1.527 | 6.016 | 0.98 | 4.87 | 0.80 | -0.03 |
| native v4 rectilinear | 400 | 300-1000 | 0.876 | 1.774 | 3.995 | 0.003 | 1.771 | 0.00 | 2.02 | 1.00 | -1.00 |
| native v4 rectilinear | 400 | 2000-4000 | 1.550 | 5.257 | 1.060 | 1.022 | 4.235 | 0.66 | 3.39 | 0.81 | -0.04 |
| native v4 angular | 400 | 300-1000 | 0.876 | 0.671 | 4.304 | -0.154 | 0.826 | -0.18 | 0.77 | 1.23 | -1.04 |
| native v4 angular | 400 | 2000-4000 | 1.550 | 1.000 | 1.004 | 0.974 | 0.026 | 0.63 | 0.65 | 0.03 | -0.03 |
| stitched, no rotation | 800 | 300-1000 | 1.246 | 3.398 | 4.668 | 1.174 | 2.224 | 0.94 | 2.73 | 0.65 | -0.75 |
| stitched, no rotation | 800 | 2000-4000 | 1.829 | 4.247 | 1.649 | 1.721 | 2.526 | 0.94 | 2.32 | 0.59 | 0.04 |
| stitched, wrap-cycle | 800 | 300-1000 | 1.246 | 1.549 | 5.191 | 1.865 | -0.315 | 1.50 | 1.24 | -0.20 | -0.64 |
| stitched, wrap-cycle | 800 | 2000-4000 | 1.829 | 1.670 | 1.739 | 1.704 | -0.034 | 0.93 | 0.91 | -0.02 | -0.02 |

Reading (no further investigation yet):
- l ~ 3000: box replicas carry ~80% of D_l in every unrotated rectilinear lightcone at 400 Mpc (stitched, native v3, native v4), 59% at 800 Mpc.
  Native v4 ANGULAR carries only 3% at l ~ 3000 (ray drift decorrelates small scales) but is still replica-dominated at low l. Wrap-cycle: ~0.
- l ~ 3000: near/diag within -8%..+4% for all seven -> physical cross-shell P_off ~ 0 -> Limber holds at l ~ 3000 (below the box scale).
- Amplitude with replicas removed (trunc/direct, l ~ 3000): native v3 (fixed) 0.98; stitched 0.78-0.80 (400) -> 0.93-0.94 (800). The stitched
  gap sits in diag and shrinks with box size; cause open. Native v4 0.63-0.66 is a different astro model/realisation, not comparable in absolute terms.
- l = 300-1000: strong near-shell cancellation everywhere (near/diag -0.6..-1.0); at 400 Mpc it extends beyond L/2, so replicas and physics cannot be
  separated. 800 Mpc no-rotation gives trunc/dir 0.94; wrap-cycle 1.50 (rotation also removes real cancellation). Low l needs a larger box.
