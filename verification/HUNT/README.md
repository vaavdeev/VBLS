# HUNT — the Σ(t) trajectory against VO₂max data

**Monograph section:** Appendix D.2 (within-species VO₂max verification); calibration anchored to §17.1  
**What is compared:** the damage increment ΔΣ(t) of system (S) at the canonical human calibration (§17.1; α = 0.030 yr⁻¹ is fixed by that calibration, not by these data) is mapped onto the longitudinal VO₂max decline of HUNT4 men by VO₂max/VO₂max₀ = 1 − k·ΔΣ^β, with k and β fitted to the same seven points. The comparison tests the shape of the trajectory, not an independent prediction (App. D.2.4–D.2.6, D.2.10). HUNT3 is cross-sectional (N=3816, ages 20–90), HUNT4 longitudinal (N=1431, 10-year follow-up).

**Data sources:**  
— HUNT3: Loe et al. (2013) PLoS ONE 8(5):e64319  
— HUNT4: Letnes et al. (2020) Progress in Cardiovascular Diseases 63(6):730–737; decade values from Fig. 3 of the review by Letnes, Nes & Wisløff (2023), Int. J. Cardiol. Cardiovasc. Risk Prev. 16:200171  
— FRIEND registry: Kaminsky et al. (2015) Mayo Clinic Proceedings 90(11):1515–1523

Note: HUNT data are not redistributed here. Scripts reconstruct the published curves via parametric fits to published summary statistics.

---

## Scripts

| File | What it does |
|---|---|
| `hunt_verification.py` | Main mapping. Solves system (S) (σ₀ = 0.0162); maps VO₂max/VO₂max₀ = 1 − k·(ΔΣ)^β, with k and β fitted jointly to all seven points of the male trajectory (scipy.optimize). Output: k = 2.45, β = 2.30, RMSE = 0.027 (App. D.2.5). |
| `hunt_aic.py` | AICc model comparison: the mapping through ΔΣ(t) (VBLS) vs simple exponential / time-power / power-law, and the direct identification of η_abs with VO₂max (M0, no fitted parameters) (App. D.2.9); the memory mechanism itself is not tested by it: M0 gives RMSE 0.058, as the exponential does, yet by AICc it is on a par with the mapping through ΔΣ(t), so the data do not separate the shares of the two components (App. D.2.4). |
| `inflow_sensitivity.py` | Role of the assumed inflow shape (App. D.2.5, §17.3.6): the canonical Φ_in with and without its post-60 decline — dΣ/dt at 70 yr and at t_B as a fraction of its maximum (0.48 and 0.33 vs 0.68 and 0.60) and the mapping refitted to the male trajectory (k = 2.45, β = 2.30, RMSE 0.027 vs k = 1.58, β = 1.89, RMSE 0.014). Console only. |
| `superposition.py` | Extended model: M1 (VBLS) / M2 (+HR channel) / M3 (+activity groups; the group levels −15/−9/−5 % per decade are assumed inputs whose source in the published HUNT results has not been established), and the t_death→σ₀ calibration analysis (App. D.2.11–D.2.12). Activity scale factors re-tuned for canonical σ₀ (June 2026). |

## Figures

| File | Figure in monograph |
|---|---|
| `hunt_verification.png` / `.pdf` | Fig. D.2.2 — HUNT3/HUNT4 fit + VBLS η_abs(t), Σ(t) |
| `hunt_aic.png` | Fig. D.2.1 — AICc comparison + the shape of the η_abs profile (form only) |
| `superposition.png` / `.pdf` | Fig. D.2.3 — M1/M2/M3 superposition + activity groups |
| `calibration_tdeath.png` / `.pdf` | Fig. D.2.4 — t_death→σ₀ calibration; VO₂max fit is robust to the choice |

## Key calibrated parameters (output)

σ₀ = 0.0162, α = 0.030 yr⁻¹  (canonical human calibration, identical to MAN/).
These values are used in all other human-calibrated scripts (MAN/, Cagan/).

## Run

```bash
python hunt_verification.py
python hunt_aic.py
python superposition.py
python inflow_sensitivity.py
```
