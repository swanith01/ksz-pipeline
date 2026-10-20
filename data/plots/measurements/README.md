# kSZ measurement history

Plots of published kinematic Sunyaev–Zel'dovich (kSZ) power at ℓ = 3000 and the reionization-duration limits derived from it, against year of publication, plus the SZ spectral-response figures from the original notebook.

| File | What it is |
|---|---|
| `measurements_and_spectra_ksz.py` | Script that makes all figures and holds the data table |
| `total_kSZ_history.pdf` | Total kSZ power, D₃₀₀₀ᵏˢᶻ (homogeneous + patchy) |
| `patchy_kSZ_history.pdf` | Patchy (reionization) kSZ power, D₃₀₀₀ᵖᵏˢᶻ |
| `reionization_duration_history.pdf` | Top: Δz_re limits (kSZ power and trispectrum). Bottom: kSZ-trispectrum and patchy-screening amplitudes relative to a fiducial reionization model |

## Running

```bash
python3 measurements_and_spectra_ksz.py                 # all figures -> ./figures/
python3 measurements_and_spectra_ksz.py total patchy    # only these
python3 measurements_and_spectra_ksz.py total --labels  # author name on every point
python3 measurements_and_spectra_ksz.py --list          # print the data table
python3 measurements_and_spectra_ksz.py total --include-unverified
```

Other figure names: `duration` (reionization-duration figure), `sed` (SZ spectral responses, kSZ SED) and `tsz` (tSZ history; data not re-checked).

## Data plotted

kSZ powers are in μK² at ℓ = 3000. 68% intervals unless marked "<" (95% upper limit). Values were checked against the papers in October 2026. When a paper gives several numbers, the **Case** column says which one is plotted.

### Total kSZ

| Year | Exp. | Value | Paper | Case |
|---|---|---|---|---|
| 2010 | SPT | < 13 | Hall+ 2010 | 2008 data |
| 2011 | ACT | < 8.0 | Dunkley+ 2011 | kSZ amplitude free |
| 2011 | SPT | < 6.5 | Shirokoff+ 2011 | |
| 2012 | SPT | < 6.7 | Reichardt+ 2012 | free tSZ–CIB correlation (< 2.8 without) |
| 2012 | SPT | < 6.1 | Zahn+ 2012 | CSF+patchy template, free ℓ-independent ξ (< 2.7 with ξ = 0) |
| 2013 | ACT | < 8.6 | Sievers+ 2013 | ACT + WMAP7, ΛCDM |
| 2013 | ACT+SPT | < 5.0 | Dunkley+ 2013 | 0 < ξ < 0.2 (< 6.9 for 0 < ξ < 0.5) |
| 2013 | Independent | 5.3 (+2.2/−2.4) | Addison+ 2013 | SPT+ACT+SPIRE+Planck; value as quoted by Crawford+ 2014 §6.3.4 |
| 2014 | SPT | 2.9 ± 1.6 | Crawford+ 2014 | R12 spectra + tSZ-bispectrum prior, A_kSZ > 0 (< 5.6 at 95%; 2.6 ± 1.8 without prior) |
| 2015 | SPT | 2.9 ± 1.3 | George+ 2015 | with tSZ-bispectrum prior (spectra only: < 5.4) |
| 2016 | Planck | < 2.6 | Planck Int. XLVII | CSF & patchy template; Planck + ACT/SPT high-ℓ |
| 2021 | SPT | 3.0 ± 1.0 | Reichardt+ 2021 | SPT-SZ + SPTpol |
| 2022 | Independent | 3.3 (+0.5/−0.3) | Gorce+ 2022 | reanalysis of R21 bandpowers, free cosmology |
| 2025 | ACT | 1.48 (+0.71/−1.10) | Beringue+ 2025 | ACT DR6 only (Louis+ 2025: < 4 at 95%) |
| 2026 | SPT | 1.98 ± 0.87 | Chaubal+ 2026 | SPT-3G D1 fiducial: free CIB + ℓᵅ G15 SZ |
| 2026 | SPT (alt.) | 3.96 ± 0.82 | Chaubal+ 2026 | Agora templates (open marker) |
| 2026 | SPT (alt.) | 2.5 ± 1.1 | Chaubal+ 2026 | free CIB + free SZ spline (open marker) |

### Patchy kSZ

| Year | Exp. | Value | Paper | Case |
|---|---|---|---|---|
| 2012 | SPT | < 5.7 | Reichardt+ 2012 | all kSZ assumed patchy: Z11 patchy-only template, free tSZ–CIB correlation, no homogeneous part subtracted (Table 3) |
| 2012 | SPT | < 4.9 | Zahn+ 2012 | free tSZ–CIB correlation (< 2.1 with ξ = 0) |
| 2013 | ACT | < 7.1 | Sievers+ 2013 | total < 8.6 minus assumed homogeneous 1.5 ± 0.5 (Shaw+ 2012 scaling); uncertainty not propagated |
| 2015 | SPT | < 4.4 | George+ 2015 | spectra only, homogeneous kSZ subtracted |
| 2021 | SPT | < 2.5 | Reichardt+ 2021 | with tSZ-bispectrum prior (68%: 1.1 +1.0/−0.7) |
| 2022 | Independent | < 1.58 | Gorce+ 2022 | free cosmology |

Note: Reichardt+ 2012 assumes all kSZ is patchy, so its limit is looser by construction than the homogeneous-subtracted limits (Zahn+ 2012 uses the same data).

### Duration of reionization (top panel)

All are 95% upper limits. **Papers define Δz between different ionized fractions**, shown in brackets on the plot, so limits with different brackets are not directly comparable: a 20–99% or 10–99% duration is longer than a 25–75% one for the same history.

| Year | Exp. | Δz_re | Definition | Paper | Method / case |
|---|---|---|---|---|---|
| 2012 | SPT | < 7.9 | 20–99% | Zahn+ 2012 | kSZ power, free ℓ-independent tSZ–CIB correlation (same case as patchy < 4.9); < 4.4 with ξ = 0, < 6.2 with rising ξ_ℓ |
| 2015 | SPT | < 5.4 | 20–99% | George+ 2015 | kSZ power, homogeneous kSZ accounted for |
| 2016 | Planck | < 2.8 | 10–99% | Planck Int. XLVII | τ (low-ℓ) + ACT/SPT kSZ, redshift-symmetric model, z_end > 6 prior; < 4.8 uniform prior |
| 2021 | SPT | < 4.1 | 25–75% | Reichardt+ 2021 | kSZ power + tSZ-bispectrum prior; 68%: 1.1 (+1.6/−0.7); < 5.4 without prior |
| 2022 | Independent | < 4.3 | 25–75% | Gorce+ 2022 | SPT kSZ + Planck (< 4.8 at 99%) |
| 2024 | SPT | < 4.5 | 25–75% | Raghunathan+ 2024 | **kSZ trispectrum** (SPTpol + SPT-3G + Herschel-SPIRE), Gunn–Peterson prior |
| 2025 | ACT | < 4.4 | 25–75% | Beringue+ 2025 | kSZ power, ACT DR6, z_mid = 8, no low-z kSZ (conservative); < 2.5 or < 0.7 with low-z kSZ; AMBER Δz₉₀ < 7.7 |
| 2026 | SPT | < 4.2 | 25–75% | Chaubal+ 2026 | kSZ power, SPT-3G D1 fiducial, 1.76 μK² homogeneous subtracted; median 0.8 (+1.5/−0.6); AMBER Δz₉₀ < 7.0 |

**Missing: Sievers+ 2013.** The paper turns its patchy limit (< 7.1) into a z̄–Δz constraint using Battaglia+ 2012, but I could not retrieve that section, so no number is plotted yet. There is a placeholder comment in `DURATION` in the script.

### Higher-order statistics with no Δz (bottom panel)

These do not quote a duration, so the plot shows their amplitude relative to the paper's own fiducial reionization model (1 = that model). The two fiducials differ.

| Year | Exp. | Amplitude | Paper | Probe / fiducial |
|---|---|---|---|---|
| 2024 | ACT | 77 ± 33 | MacCrann+ 2024 | kSZ trispectrum, ACT DR6 + Planck; relative to Alvarez (2016), z_mid = 10. Lensing-hardened value; foreground-dominated, so read as an upper limit (≈ 50× recent simulations). No Δz constraint. |
| 2026 | ACT | < 450 (A_τ = 180 ± 157) | Kramer+ 2026 | Anisotropic (patchy) screening, ACT DR6; relative to AMBER z_mid = 8, Δz = 4. < 370 with ℓ_max = 2000. CIB contamination is the limiting systematic. |

### Not plotted

- **Planck 2014, 5.3 (+2.8/−1.9)** carried over from the old notebook. No source found. Planck Int. XXIV shows A_kSZ only in contour plots; Planck 2013 XVI's foreground table was not checked. Same central value as Addison+ 2013, so possibly a copy error. Off unless `--include-unverified`.
- **SPT-3G D1 patchy kSZ:** shown only as a posterior plot (their Fig. 8), no number quoted.
- **Raghunathan+ 2024** (kSZ trispectrum) is on the duration plot only; it constrains Δz⁵⁰_re < 4.5 directly, not a kSZ power.
- **Reichardt+ 2012** gives no Δz; the reionization interpretation is in the companion Zahn+ 2012.
- **Coulton+ 2024** (ACT patchy screening × unWISE) probes gas around low-redshift galaxies, not reionization. No detection; ⟨τ⟩ < 0.033 arcmin² (blue) and < 0.057 arcmin² (green) in a 6′ disk.
- **Earlier screening/τ-fluctuation limits:** Gluscevic+ 2013 (WMAP-7, several orders of magnitude above models) and Namikawa 2018 (Planck 2015 temperature trispectrum, rules out bubble size R ≳ 10 Mpc at x_e ≈ 0.5). Neither is expressed as an amplitude relative to a fiducial model, so not plotted.

## References

### Plotted

1. **Hall+ 2010** — N. R. Hall et al., *Angular Power Spectra of the Millimeter Wavelength Background Light from Dusty Star-forming Galaxies with the South Pole Telescope*, ApJ 718, 632 (2010). [arXiv:0912.4315](https://arxiv.org/abs/0912.4315)
2. **Dunkley+ 2011** — J. Dunkley et al., *The Atacama Cosmology Telescope: Cosmological Parameters from the 2008 Power Spectra*, ApJ 739, 52 (2011). [arXiv:1009.0866](https://arxiv.org/abs/1009.0866)
3. **Shirokoff+ 2011** — E. Shirokoff et al., *Improved Constraints on Cosmic Microwave Background Secondary Anisotropies from the Complete 2008 South Pole Telescope Data*, ApJ 736, 61 (2011). [arXiv:1012.4788](https://arxiv.org/abs/1012.4788)
4. **Reichardt+ 2012** — C. L. Reichardt et al., *A Measurement of Secondary Cosmic Microwave Background Anisotropies with Two Years of South Pole Telescope Observations*, ApJ 755, 70 (2012). [arXiv:1111.0932](https://arxiv.org/abs/1111.0932)
5. **Zahn+ 2012** — O. Zahn et al., *Cosmic Microwave Background Constraints on the Duration and Timing of Reionization from the South Pole Telescope*, ApJ 756, 65 (2012). [arXiv:1111.6386](https://arxiv.org/abs/1111.6386)
6. **Sievers+ 2013** — J. L. Sievers et al., *The Atacama Cosmology Telescope: Cosmological parameters from three seasons of data*, JCAP 10 (2013) 060. [arXiv:1301.0824](https://arxiv.org/abs/1301.0824)
7. **Dunkley+ 2013** — J. Dunkley et al., *The Atacama Cosmology Telescope: likelihood for small-scale CMB data*, JCAP 07 (2013) 025. [arXiv:1301.0776](https://arxiv.org/abs/1301.0776)
8. **Addison+ 2013** — G. E. Addison, J. Dunkley & J. R. Bond, *Constraining thermal dust emission in distant galaxies with number counts and angular power spectra*, MNRAS 436, 1896 (2013). [arXiv:1210.6697](https://arxiv.org/abs/1210.6697)
9. **Crawford+ 2014** — T. M. Crawford et al., *A Measurement of the Secondary-CMB and Millimeter-wave-foreground Bispectrum using 800 deg² of South Pole Telescope Data*, ApJ 784, 143 (2014). [arXiv:1303.3535](https://arxiv.org/abs/1303.3535)
10. **George+ 2015** — E. M. George et al., *A Measurement of Secondary Cosmic Microwave Background Anisotropies from the 2500 Square-degree SPT-SZ Survey*, ApJ 799, 177 (2015). [arXiv:1408.3161](https://arxiv.org/abs/1408.3161)
11. **Planck Int. XLVII** — Planck Collaboration, *Planck intermediate results. XLVII. Planck constraints on reionization history*, A&A 596, A108 (2016). [arXiv:1605.03507](https://arxiv.org/abs/1605.03507)
12. **Reichardt+ 2021** — C. L. Reichardt et al., *An Improved Measurement of the Secondary Cosmic Microwave Background Anisotropies from the SPT-SZ + SPTpol Surveys*, ApJ 908, 199 (2021). [arXiv:2002.06197](https://arxiv.org/abs/2002.06197)
13. **Gorce+ 2022** — A. Gorce, M. Douspis & L. Salvati, *Retrieving cosmological information from small-scale CMB foregrounds. II. The kinetic Sunyaev–Zel'dovich effect*, A&A 662, A122 (2022). [arXiv:2202.08698](https://arxiv.org/abs/2202.08698)
14. **Beringue+ 2025** — B. Beringue et al., *The Atacama Cosmology Telescope: DR6 Power Spectrum Foreground Model and Validation*, JCAP 2025, 082. [arXiv:2506.06274](https://arxiv.org/abs/2506.06274)
15. **Chaubal+ 2026** — P. Chaubal, N. Huang, C. L. Reichardt et al., *SPT-3G D1: A Measurement of Secondary Cosmic Microwave Background Anisotropy Power*, submitted to JCAP. [arXiv:2601.20551](https://arxiv.org/abs/2601.20551)
16. **Raghunathan+ 2024** — S. Raghunathan et al., *First Constraints on the Epoch of Reionization Using the Non-Gaussianity of the Kinematic Sunyaev-Zel'dovich Effect from the South Pole Telescope and Herschel-SPIRE Observations*, PRL 133, 121004 (2024). [arXiv:2403.02337](https://arxiv.org/abs/2403.02337)
17. **MacCrann+ 2024** — N. MacCrann et al., *The Atacama Cosmology Telescope: Reionization kSZ trispectrum methodology and limits*, MNRAS 532, 4247 (2024). [arXiv:2405.01188](https://arxiv.org/abs/2405.01188)
18. **Kramer+ 2026** — D. M. Kramer et al., *The Atacama Cosmology Telescope: Constraints on the anisotropic screening and birefringence effects with DR6*, preprint (2026). [arXiv:2608.27458](https://arxiv.org/abs/2608.27458)

### Cited in notes or checked

19. **Louis+ 2025** — T. Louis et al., *The Atacama Cosmology Telescope: DR6 power spectra, likelihoods and ΛCDM parameters*, JCAP 2025, 062. [arXiv:2503.14452](https://arxiv.org/abs/2503.14452)
20. **Calabrese+ 2014** — E. Calabrese et al., *Precision epoch of reionization studies with next-generation CMB experiments*, JCAP 08 (2014) 010. [arXiv:1406.4794](https://arxiv.org/abs/1406.4794). Forecast paper, no measurement. Source of the homogeneous-kSZ scaling 1.65 (σ₈/0.8)^4.46 μK² used by SPT-3G D1.
21. **Planck Int. XXIV** — Planck Collaboration, *Planck intermediate results. XXIV. Constraints on variation of fundamental constants*, A&A 580, A22 (2015). A_kSZ appears only in contour plots; no value quoted.
22. **Planck 2013 XVI** — Planck Collaboration, *Planck 2013 results. XVI. Cosmological parameters*, A&A 571, A16 (2014). [arXiv:1303.5076](https://arxiv.org/abs/1303.5076). Possible source of the unverified Planck 2014 point; not yet checked.
23. **Coulton+ 2024** — W. R. Coulton et al., *The Atacama Cosmology Telescope: Detection of Patchy Screening of the Cosmic Microwave Background*, preprint (2024). [arXiv:2401.13033](https://arxiv.org/abs/2401.13033)
24. **Gluscevic+ 2013** — V. Gluscevic, M. Kamionkowski & D. Hanson, *Patchy screening of the cosmic microwave background by inhomogeneous reionization*, PRD 87, 047303 (2013). [arXiv:1210.5507](https://arxiv.org/abs/1210.5507)
25. **Namikawa 2018** — T. Namikawa, *Constraints on patchy reionization from Planck CMB temperature trispectrum*, PRD 97, 063505 (2018). [arXiv:1711.00058](https://arxiv.org/abs/1711.00058)
26. **Kramer+ 2025** — D. M. Kramer et al., forecast for kSZ and patchy screening with AMBER simulations. [arXiv:2501.07623](https://arxiv.org/abs/2501.07623). Source of the AMBER Δz₉₀ relation used by ACT DR6 and SPT-3G D1.
