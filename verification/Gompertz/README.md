# Gompertz — the Gompertz law on the full system (S)

**Monograph section:** §16.2.4 (Fig. 16.2)
**What is computed:** populations on the full field of system (S) with the canonical human calibration (Section 17.1.1). Individuals share the state of the canonical trajectory at age 50 and differ in the repair reserve R at that age (and, in one population, in the wear rates). System (S) is deterministic and does not generate these spreads: they are assumptions of the computation. The figure checks the statement of Section 16.2.4: with a rate nu_eff common to the population, the mortality rate is Gompertzian if and only if the reserve is Weibull-distributed, with slope b = k_W*nu_eff. No external dataset.

---

## Scripts

| File | What it does |
|---|---|
| `gompertz_full_system.py` | Fig. 16.2. Three populations of 10^5 individuals integrated on the full field (both cut-offs of g(E), max(0, 1 - Sigma) in the restoration term): (W) Weibull spread of the reserve at age 50, shape k_W = 2; (L) log-normal spread with the same mean and standard deviation of ln R; (W+z) as (W) plus a gamma-distributed factor on nu_0 and nu (CV 0.3). Panels: (A) reserve of five individuals, full system vs late approximation; (B) distributions of death times; (C) life-table mortality rate with a Gompertz fit and the late approximation; (D) a spread of nu_eff slows the growth of mortality. Prints the fitted slope, local slopes by age and the share of restoration in the reserve balance. |
| `gompertz_comparison_4panels.py` | Former Fig. 16.2, superseded on 15.10.2026. An illustrative population outside system (S): the R-channel death time of an individual is the minimum of 50 independent normal "pathway exhaustion times", the E-channel death time is exponential; panel D is drawn analytically, with the plateau set by hand. Kept for reference. |

## Figures

| File | Figure in monograph |
|---|---|
| `gompertz_full_system.png` | Fig. 16.2 — the Gompertz law on the full system (S) and the spread of the reserve |
| `gompertz_comparison_4panels.png` | none (former Fig. 16.2) |

## Result of the run (15.10.2026)

- Weibull spread: Gompertz slope b = 0.095 per year over ages 56-90, rms deviation of ln(mu) from the straight line 0.021; local slope 0.089-0.097; k_W*nu_eff grows from 0.067 at age 50 to 0.092 at the canonical death time.
- Log-normal spread: the local slope falls from 0.165 at age 55 to 0.072 at age 80 (concave log-hazard).
- Spread of nu_eff (CV 0.3): the local slope falls from 0.10 to 0.01 by age 90; plateau near 0.1 per year at ages 85-110.
- Restoration is 4-8% of the wear of the reserve at ages 50-77 on the canonical trajectory.

## Run

```bash
python gompertz_full_system.py
```
Self-contained: no external data files required; about two minutes on a laptop.
