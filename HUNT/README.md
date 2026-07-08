# HUNT — η_abs(t) calibration against VO₂max data

**Monograph section:** Appendix D.2 (within-species VO₂max verification); calibration anchored to §17.1  
**VBLS prediction tested:** the theoretical trajectory of η_abs(t) — assimilation efficiency — is mapped onto the empirical VO₂max decline in humans. The HUNT3 cross-sectional dataset (N=3816, ages 20–90) and HUNT4 longitudinal dataset (N=1431, 10-year follow-up) constrain the memory degradation rate α = 0.030 yr⁻¹.

**Data sources:**  
— HUNT3: Loe et al. (2013) PLoS ONE 8(5):e64319  
— HUNT4: Stensvold et al. (2020) Progress in Cardiovascular Diseases 63:85–91  
— FRIEND registry: Kaminsky et al. (2019) Mayo Clinic Proceedings (ages 70–89)

Note: HUNT data are not redistributed here. Scripts reconstruct the published curves via parametric fits to published summary statistics.

---

## Scripts

| File | What it does |
|---|---|
| `hunt_verification.py` | Main mapping. Solves system (S) (σ₀ = 0.0162); maps VO₂max/VO₂max₀ = 1 − k·(ΔΣ)^β via scipy.optimize. Output: k = 2.43, β = 2.30, RMSE = 0.016 (App. D.2.5). |
| `hunt_aic.py` | AICc model comparison: VBLS memory mapping vs simple exponential / time-power / power-law (App. D.2.9). |
| `superposition.py` | Extended model: M1 (VBLS) / M2 (+HR channel) / M3 (+PAI), and the t_death→σ₀ calibration analysis (App. D.2.11–D.2.12). PAI scale factors re-tuned for canonical σ₀ (June 2026). |

## Figures

| File | Figure in monograph |
|---|---|
| `hunt_verification.png` / `.pdf` | App. D.2 — HUNT3/HUNT4 fit + VBLS η_abs(t), Σ(t) |
| `hunt_aic.png` | Fig. D.2.9 — AICc comparison + η_abs memory profile |
| `superposition.png` / `.pdf` | Fig. D.2.2 — M1/M2/M3 superposition + PAI stratification |
| `calibration_tdeath.png` / `.pdf` | t_death→σ₀ calibration; VO₂max fit is robust to the choice |

## Key calibrated parameters (output)

σ₀ = 0.0162, α = 0.030 yr⁻¹  (canonical human calibration, identical to MAN/).
These values are used in all other human-calibrated scripts (MAN/, Cagan/).

## Run

```bash
python hunt_verification.py
python hunt_aic.py
python superposition.py
```
