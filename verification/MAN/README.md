# MAN — Human lifespan calibration

**Monograph section:** §17.1 (main calibration), Appendix D.7 (g(E) sensitivity)
**VBLS prediction tested:** full numerical integration of the coupled system (S) with human parameters; calibration target t_B = 75–83 yr (WHO 2023, developed countries); restoration-potential function g(E) confirmed to equal 1 throughout the normal aging trajectory; parameter sensitivity analysis (local tornado + global Monte Carlo); the counter-intuitive result α↑ → t_B↑ (§8.2.1); E-channel longevity ceiling under R-channel suppression (§17.0 falsification criterion Ф10); sensitivity of t_B to the functional form of g(E) (Appendix D.7).

Calibrated parameters: σ₀ = 0.0162, α = 0.030 yr⁻¹, ν₀ = 0.006, ν = 0.060, ρ = 0.002, E* = 0.04, R* = 0.15, E₀ = 0.10.

---

## Scripts

| File | What it does |
|---|---|
| `human_lifespan.py` | Integrates the full system (S) for human parameters. Death event: R(t) = R* or E(t) = E* (whichever first). Produces the main lifecycle figure with all state variables and the R(Σ) phase portrait. |
| `human_sensitivity.py` | Full parameter sensitivity analysis: tornado chart (each parameter ±10%/±20%/±50%), sensitivity bars (|Δt_B|/t_B at ±10%), 2D iso-lifespan heatmap, Monte Carlo (all parameters ±20% simultaneously). |
| `alpha_sensitivity.py` | Shows how the memory degradation rate α shapes the human trajectory. Counter-intuitive result: faster η_abs decay → slower Σ accumulation → delayed R-channel death, with an internal optimum α*≈0.10 yr⁻¹. Four panels: t_B vs α; η_abs(t), Σ(t), R(t) for multiple α values. |
| `E_channel_ceiling.py` | Console/numeric computation of the E-channel longevity ceiling under full R-channel suppression (§17.0, criterion Ф10): t_B ≈ 246 yr (bounded inflow) / ≈ 136 yr (human Φ_in profile with senescent decline). |
| `E_channel_ceiling_fig.py` | Generates the two-panel figure from `E_channel_ceiling.py`'s model: channel-switch vs. R-suppression factor, and E-channel ceiling vs. σ₀ (with Greenland shark / *Arctica islandica* reference lines). |
| `g_sensitivity.py` | Appendix D.7: tests whether replacing the linear restoration-potential function g(E) with a Hill-type sigmoid (n=2, 4) changes t_B on the real human trajectory. Result: Δt_B < 0.5 yr (<0.6%), death channel unchanged. |
| `regime_sensitivity.py` | Numeric test (not embedded as a monograph figure) cited in §8.3.1/§9.5/§19.2.5: shows that μ, λ, E* have zero local sensitivity in the R-dominant regime, but 14–21% sensitivity once the R-channel is artificially suppressed (four-fold reduction of ν₀, ν or σ₀→0.25σ₀), confirming the E-channel is not structurally unobservable. |

## Figures

| File | Figure in monograph |
|---|---|
| `human_lifespan_EN.png` | Fig. 17.1a — full human lifecycle: E(t), Σ(t), η_abs(t), R(t), Φ balance, R(Σ) phase portrait |
| `human_sensitivity_EN.png` | Fig. 17.1b — parameter sensitivity: tornado + heatmap + Monte Carlo |
| `alpha_sensitivity.png/.pdf` | Fig. 17.1c — α sensitivity: t_B vs α and trajectory panels |
| `E_channel_ceiling.png` | Fig. 17.1d — E-channel ceiling under R-channel suppression |
| `g_sensitivity.png` | Fig. D.7.1 — g(E) functional-form sensitivity |
| `regime_sensitivity.png` | Not embedded in the monograph (script's printed numbers are cited directly in the text) |

## Run

```bash
python human_lifespan.py         # Fig. 17.1a
python human_sensitivity.py      # Fig. 17.1b
python alpha_sensitivity.py      # Fig. 17.1c
python E_channel_ceiling.py      # console output for Fig. 17.1d
python E_channel_ceiling_fig.py  # Fig. 17.1d
python g_sensitivity.py          # Fig. D.7.1
python regime_sensitivity.py     # console output only, no figure
```
