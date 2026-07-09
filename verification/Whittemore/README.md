# Whittemore — Telomere shortening verification

**Monograph section:** §17.2
**VBLS prediction tested:** universal residual telomere fraction R* at death; T_max prediction via two death channels (R-channel and E-channel); power law T_max ~ ν*⁻ᵅ reproduced from first principles.

**Data source:** Whittemore K., Vera E., Martínez-Nevado E., Sanpera C., Blasco M.A. (2019). Telomere shortening rate predicts species life span. *Proceedings of the National Academy of Sciences* 116(30):15122–15127. DOI: [10.1073/pnas.1902452116](https://doi.org/10.1073/pnas.1902452116)
9 species: Mouse, Goat, Reindeer, Audouin's gull, Griffon vulture, Bottlenose dolphin, American flamingo, Sumatran elephant, Human.
Measurements: initial telomere length L₀ (kb), shortening rate (bp/yr), T_max, T_avg.

**Note on the source PDF:** the publisher's article PDF and Supplementary Appendix are not redistributed in this repository (PNAS copyright). The two tables needed for reproducibility have been extracted into `telomer_Table_S1_clean.csv` / `telomer_Table_S2_clean.csv`; the full paper is available at the DOI above.

---

## Scripts

| File | What it does |
|---|---|
| `telomere_verification.py` | Main script. Four figures: (1) universal R* bar chart + cluster zones; (2) predicted vs observed T_max scatter (two channels); (3) power law T_max ~ ν*⁻ᵅ, VBLS vs Whittemore regression; (4) sensitivity analysis / tornado chart. Data embedded in script. |
| `lifecycle_telomere.py` | Illustrative lifecycle trajectory for the telomere system mapped onto VBLS phases (Fig. 17.2.5). |
| `telomere_robustness.py` | Small-sample robustness of the n=7 cluster-1 result (reviewer point RV-3, §19.2.4): leave-one-out, nonparametric bootstrap 95% CI, and a leverage diagnostic. Prints to stdout; no figures. |

## Data

| File | Description |
|---|---|
| `telomer_Table_S1_clean.csv` | Table S1 from Whittemore (2019): species, L₀, shortening rate, T_max, T_avg |
| `telomer_Table_S2_clean.csv` | Table S2: residual telomere fractions at observed end-of-life |

## Figures

| File | Figure in monograph |
|---|---|
| `telomere_fig1_Rstar.png/.pdf` | Fig. 17.2.1 — Universal R* |
| `telomere_fig2_pred_vs_obs.png/.pdf` | Fig. 17.2.2 — Predicted vs observed T_max |
| `telomere_fig3_powerlaw.png/.pdf` | Fig. 17.2.3 — Power law reproduction |
| `telomere_fig4_sensitivity.png/.pdf` | Fig. 17.2.4 — Sensitivity analysis |
| `lifecycle_telomere.png/.pdf` | Fig. 17.2.5 — Lifecycle phases illustration |

## Run

```bash
python telomere_verification.py
python lifecycle_telomere.py
```
Output files written to current directory.

---

## Small-sample robustness (`telomere_robustness.py`)

Addresses the small-sample concern for the cluster-1 result reported in §19.2.4
(R²_log = 0.976, MAPE = 12.7% on n = 7). The model is Variant B:

    t_B = (1 - R*_univ) / nu*,   R*_univ = mean residual telomere fraction over the cluster

`nu*` is taken directly from the data (not fitted), so the model carries a single
fitted scalar (`R*_univ`) against 7 points.

**What it computes (stdout only, no figures):**

- **Point estimate** (n = 7): reproduces R*_univ = 0.446, MAPE = 12.7%, R²_log = 0.976.
- **Leave-one-out (LOO):** drops each species, refits `R*_univ` on the remaining 6, reports
  R²_log and MAPE. Result: R²_log ∈ [0.944, 0.987], MAPE ∈ [10.3%, 14.7%] — no single
  species drives the fit; dropping the mouse (extreme `nu*`, an order of magnitude above the
  rest) lowers R²_log only to 0.944.
- **Nonparametric bootstrap** (B = 2·10⁴, resample 7 species with replacement): 95% percentile
  CIs — R²_log [0.929, 0.999], MAPE [5.8%, 19.8%]. Degenerate resamples (<2 distinct points)
  are skipped and counted.
- **Leverage diagnostic:** removes both log-log endpoints (mouse and human) and refits on the
  5 mid-range species → R²_log = 0.980, confirming the correlation is not endpoint-driven.

**Note.** R²_log is invariant to R* (since t_B ∝ 1/nu* ⇒ log t_B = const − log nu*), hence
R²_log = corr(log nu*, log T_max)² — essentially Whittemore's own rate–lifespan correlation.
The VBLS-specific contribution is the universal threshold R* that sets the absolute calibration
(MAPE). The bootstrap uses a fixed seed (20260616) for reproducibility.

```bash
python telomere_robustness.py
```
