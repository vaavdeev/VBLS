# Alpha_optimum — the memory-sensitivity optimum α\* as a channel crossover

**Monograph section:** Appendix G; §8.2.1 (Hypothesis H1 and the channel-switch characterization of α\*); §19.3.6 (the evolutionary reading)

**What is tested:** whether, on the human calibration, the lifespan optimum α\* coincides with the point at which the dominant death channel switches from R to E. Formally

    t_B(α) = min( t_B^R(α), t_B^E(α) ).

If t_B^R increases in α and t_B^E decreases, the minimum of the two branches attains its maximum where they meet, so an interior optimum reduces to two monotonicities. The monotonicities are assumptions: Appendix G gives an example within the class where both branches are non-monotone and the maximum lies inside the R-branch, at a kink of the vector field.

**Parameters:** the calibrated system (S) for *Homo sapiens* (§17.1) — μ = 1.50, λ = 0.30, σ₀ = 0.0162, ν₀ = 0.006, ν = 0.060, ρ = 0.002, R\* = 0.15, E\* = 0.04.

---

## Scripts

| Script | What it does | Output |
|---|---|---|
| `alpha_crossover.py` | The central test: computes both lifetime branches over a range of α, locates their intersection, the first grid node where the E-channel fires and arg max t_B(α) | `alpha_crossover.png`; on the human calibration the three coincide at α = 0.1346 to the grid step (≈0.0012) — one point found three ways, not independent estimates |
| `monotonicity_check.py` | Check of the α-sensitivity sign pattern of the coupled system: ∂E/∂α ≤ 0, ∂Σ/∂α ≤ 0, ∂η/∂α ≤ 0, ∂R/∂α ≥ 0, pointwise in α by finite differences, from t = 0 (the first year of life included), with tight integrator tolerances | console report; the pattern holds on the whole range 0.010–0.135 |
| `alpha_guar_check.py` | Sturm bound α_guar for the auxiliary equation Ẅ + k(t)W = Σ̇ of Appendix G.4 (the adiabatically reduced equation for W with the export response −αΦ_cap u_R dropped); also how far the bound covers the decreasing E-branch, and the Brent refinement of α\* | console report: α_guar ≈ 0.085; E-branch covered up to α ≈ 0.04; α\* = 0.1344, t_B^max = 93.71 yr |
| `rho_residual_check.py` | Diagnostic of the apparent Σ/R sign violation at α ∈ [0.070, 0.075] that appears with loose tolerances. The name is historical: the violation persists at ρ → 0 and with g ≡ 1 and vanishes with tight tolerances, so it is an integration artifact at the kink of (7.1) near t ≈ 3 yr, not the ρ·g′ restoration term | console report |
| `stability_check.py` | Stability of α\* under grid resolution and integration horizon; isolates whether an earlier α\* ≈ 0.10 was a coarse-grid artifact | console report |
| `reproductive_vitality.py` | The trade-off of §19.3.6: the lifespan gain from α_human → α\* is post-reproductive, while the memory feedback producing it lowers reproductive vitality — hence selection settles below α\* | console report |
| `h1_counterexamples.py` | Two systems of the class that delimit Hypothesis H1, integrated in the exact field of Appendix A.1 (no [0,1] projection). Example 1 (remark to Lemma G.1): restoration not saturated, t_B(α) decreases from α = 0, no optimum. Example 2 (Appendix G.2): conditions (a)–(c) hold, but both branches are non-monotone and the maximum (t_B ≈ 711 at α ≈ 0.31) lies inside the R-branch, at a kink of the field; the R → E switch is near α ≈ 1.0 | console report |

## Running

Python 3 with NumPy and SciPy. Each script is standalone:

    python alpha_crossover.py

`alpha_crossover.py` writes `alpha_crossover.png`; the remaining scripts report to the console.

## Note on the result

The initial growth of t_B(α) is proved rigorously, in the full system, when restoration is saturated (Lemma G.1). For finite α < α_guar, positivity of the sensitivity is proved for an auxiliary equation — adiabatically reduced, with a small restoration term in g′(E) and without the export response to α (the term −αΦ_cap u_R); this is an analytical argument for the rising branch, not a proof of its monotonicity for (S). The decrease of t_B^E and the location of α\* are established numerically, and the existence of α\* remains Hypothesis H1. The particular value of α\* is calibration-dependent and is deliberately not fixed as a parameter of the theory. The structural statement is conditional: with monotone branches, an optimum, if it exists, coincides with the channel switch; on the human calibration this holds, and the calibrated α lies below α\*.
