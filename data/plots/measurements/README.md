# kSZ measurement history

Plots of published kinematic Sunyaev–Zel'dovich (kSZ) power at ℓ = 3000 against year of publication, plus the SZ spectral-response figures from the original notebook.

| File | What it is |
|---|---|
| `measurements_and_spectra_ksz.py` | Script that makes all figures and holds the data table |
| `total_kSZ_history.pdf` | Total kSZ power, D₃₀₀₀ᵏˢᶻ (homogeneous + patchy) |
| `patchy_kSZ_history.pdf` | Patchy (reionization) kSZ power, D₃₀₀₀ᵖᵏˢᶻ |

## Running

```bash
python3 measurements_and_spectra_ksz.py                 # all figures -> ./figures/
python3 measurements_and_spectra_ksz.py total patchy    # only these
python3 measurements_and_spectra_ksz.py total --labels  # author name on every point
python3 measurements_and_spectra_ksz.py --list          # print the data table
python3 measurements_and_spectra_ksz.py total --include-unverified
```

Other figure names: `sed` (SZ spectral responses, kSZ SED) and `tsz` (tSZ history; data not re-checked).

## Data plotted

All values are in μK² at ℓ = 3000. 68% intervals unless marked "<" (95% upper limit). Values were checked against the papers in October 2026. When a paper gives several numbers, the **Case** column says which one is plotted.

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

### Not plotted

- **Planck 2014, 5.3 (+2.8/−1.9)** carried over from the old notebook. No source found. Planck Int. XXIV shows A_kSZ only in contour plots; Planck 2013 XVI's foreground table was not checked. Same central value as Addison+ 2013, so possibly a copy error. Off unless `--include-unverified`.
- **SPT-3G D1 patchy kSZ:** shown only as a posterior plot (their Fig. 8), no number quoted.
- **Raghunathan+ 2024** (kSZ trispectrum) constrains Δz⁵⁰_re < 4.5 directly, not a kSZ power.

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

### Cited in notes or checked

16. **Louis+ 2025** — T. Louis et al., *The Atacama Cosmology Telescope: DR6 power spectra, likelihoods and ΛCDM parameters*, JCAP 2025, 062. [arXiv:2503.14452](https://arxiv.org/abs/2503.14452)
17. **Calabrese+ 2014** — E. Calabrese et al., *Precision epoch of reionization studies with next-generation CMB experiments*, JCAP 08 (2014) 010. [arXiv:1406.4794](https://arxiv.org/abs/1406.4794). Forecast paper, no measurement. Source of the homogeneous-kSZ scaling 1.65 (σ₈/0.8)^4.46 μK² used by SPT-3G D1.
18. **Planck Int. XXIV** — Planck Collaboration, *Planck intermediate results. XXIV. Constraints on variation of fundamental constants*, A&A 580, A22 (2015). A_kSZ appears only in contour plots; no value quoted.
19. **Planck 2013 XVI** — Planck Collaboration, *Planck 2013 results. XVI. Cosmological parameters*, A&A 571, A16 (2014). [arXiv:1303.5076](https://arxiv.org/abs/1303.5076). Possible source of the unverified Planck 2014 point; not yet checked.
20. **Raghunathan+ 2024** — S. Raghunathan et al., *First Constraints on the Epoch of Reionization Using the Non-Gaussianity of the Kinematic Sunyaev-Zel'dovich Effect from the South Pole Telescope and Herschel-SPIRE Observations*, PRL 133, 121004 (2024). [arXiv:2403.02337](https://arxiv.org/abs/2403.02337)
