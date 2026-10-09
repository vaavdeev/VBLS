# VBLS — Verification Scripts

Supplementary materials for the monograph
**Viability Boundaries of Living Systems: A Phenomenological Thermodynamic Theory**
Vasiliy Avdeev, 2026

All scripts require Python 3.9+ with NumPy, SciPy, Matplotlib, Pandas.
Calibrated VBLS human parameters: σ₀ = 0.0162, α = 0.030 yr⁻¹ (April 2026).

---

## Directory structure

| Folder | What is verified | Data source | Section in monograph |
|---|---|---|---|
| [MAN/](MAN/) | Full system (S) calibration for human lifespan; parameter sensitivity; α sensitivity; E-channel ceiling under R-channel suppression; g(E) functional-form sensitivity | WHO 2023 longevity data | §17.1, Appendix D.7 |
| [Whittemore/](Whittemore/) | Interspecies telomere shortening; universal residual R* at death; T_max prediction across 9 species | Whittemore et al. (2019) PNAS | §17.2 |
| [Cagan/](Cagan/) | Somatic mutation accumulation as one proxy of Σ(t); rate × T_max invariant across 15 species | Cagan et al. (2022) Nature 604:517 | §17.3 |
| [Escala/](Escala/) | Four allometric scaling tests (Kleiber, longevity, Rubner invariant) on 94 non-human mammal species; human exceptionalism anomaly | Escala (2022) Sci. Rep. 12:2407 | Appendix D.1 |
| [HUNT/](HUNT/) | η_abs(t) calibration against VO₂max decline in humans (HUNT3/HUNT4 cohorts) | Loe et al. (2013); Stensvold et al. (2020) | Appendix D.2 |
| [InterspeciesAllometry/](InterspeciesAllometry/) | Full lifecycle curves E(t), R(t) for 11 species under two scaling conventions; initial-dip analysis | AnAge Build 14 + physiological literature | Appendix D.3 |
| [Bertalanffy/](Bertalanffy/) | Mass-based fish growth: VBLS vs. classical von Bertalanffy growth function, Atlantic cod | ICES DATRAS NS-IBTS | Appendix D.4 |
| [AnAge/](AnAge/) | Cross-species T_max allometry on 318/111 Mammalia (AnAge Build 14); allometric anomalies; attenuation-bias Monte Carlo | Tacutu et al. (2018) AnAge Build 14 | §17.4, Appendix D.6 |
| [Alpha_optimum/](Alpha_optimum/) | The memory-sensitivity optimum α\* and its relation to the switch of the dominant death channel from R to E (exact when both lifetime branches are monotone); monotonicity of the branches; the sufficient threshold α &lt; α_guar; counterexamples without saturated restoration; stability and residual checks | Calibrated (S) parameters for *Homo sapiens* (§17.1) | Appendix G, §8.2.1 |
| [Paper1/](Paper1/) | Numbers and figures for the article on the permanent-trace principle: illustrative trajectory and explicit bound; memory-sensitivity scan; permanent versus decaying memory kernels | Illustrative parameter set (§17.1) | §9.5, §12.1.2, Appendix A |

Each folder has its own `README.md` with the exact script→figure mapping and run instructions.

## Root-level files

| File | Description |
|---|---|
| `anage_data.csv` | AnAge Build 14 (semicolon-delimited) — input for `lifetime_near_Mmin.py`. |
| `lifetime_near_Mmin.py` / `.png` | Numerical verification of t_B vs. repair capacity ρ near the viability boundary ℳ_min (§15.2.2, Appendix A remark). |

## Citation

If you use these scripts, please cite the monograph:

Avdeev, V. (2026). *Viability Boundaries of Living Systems: A Phenomenological Thermodynamic Theory*. Zenodo. https://doi.org/10.5281/zenodo.20161329

## License

CC BY-NC-ND 4.0 — see [../LICENSE](../LICENSE). Matches the `SPDX-License-Identifier: CC-BY-NC-ND-4.0` header in every script.
