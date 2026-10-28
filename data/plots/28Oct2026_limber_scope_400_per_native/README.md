# 28 Oct 2026 - Limber-scope test at L = 400 Mpc, one native lightcone per folder

Same analysis as ../28Oct2026_limber_scope_all/ (script 45), split for readability: each subfolder shows coeval-direct Limber,
stitched (no rotation), stitched (wrap-cycle 123) and ONE native lightcone. Numbers are identical to the combined run.

- native_v3_rect/  : 21cmFAST v3.3.1 run_lightcone, fixed loader (cb3aad1). trunc/direct = 0.98 at l = 2000-4000; far/total 0.80.
- native_v4_rect/  : 21cmFAST v4.1 RectilinearLightconer. far/total 0.81, near/diag -0.04. Absolute amplitude (0.66 x v3 Limber)
                     not comparable: v4 astro defaults / realisation differ.
- native_v4_ang/   : 21cmFAST v4.1 AngularLightconer.like_rectilinear, apodised patch. far/total 0.03 at l ~ 3000 but 1.23 at
                     l = 300-1000 (replicas only suppressed on small scales); interpolation smoothing above l ~ 5000.

Files: limber_scope_Dl.png (D_l, |dchi| < L/2 vs all pairs vs Limber), limber_scope_dchi.png (cross-power vs |dchi|/L, / D_direct),
limber_scope_dchi_selfnorm.png (same, / each lightcone's own D_diag), limber_scope.npz.

**Update (28 Oct 2026, afternoon):** native_v4_rect/ and native_v4_ang/ now compare the v4 lightcones with a v4 coeval-direct
Limber reference (scripts/46_v4_coeval_limber.py; same inputs/realisation as the v4 lightcones; file v4_coeval_direct_box400.npz
here). v4 coeval velocity units confirmed (v_pec = velocity/(1+z): linear-fit slope 1.001, corr +1.00). v4 reionizes later than v3
(midpoint z = 7.12) and its Limber D_3000 = 1.369 uK^2 (v3: 1.524). Against its own Limber, trunc/direct at l = 2000-4000 is
0.75 (v4 rect) and 0.71 (v4 ang), similar to stitched v3 at this box (0.78-0.80; 0.93 at 800 Mpc). Native v3 (0.98) is the outlier.
