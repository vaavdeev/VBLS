# Alpha_optimum — the memory-sensitivity optimum α\* as a channel crossover

**Monograph section:** Appendix G; §8.2.1 (the Proposition); §19.3.6 (the evolutionary reading)

**VBLS prediction tested:** the lifespan optimum α\* is not an isolated feature of the curve t_B(α) but the point at which the dominant death channel switches from R to E. Formally

    t_B(α) = min( t_B^R(α), t_B^E(α) ),

with t_B^R increasing in α and t_B^E decreasing. The minimum of an increasing and a decreasing branch attains its maximum where they meet, so the existence of an interior optimum reduces to two monotonicities rather than being postulated.

**Parameters:** the calibrated system (S) for *Homo sapiens* (§17.1) — μ = 1.50, λ = 0.30, σ₀ = 0.0162, ν₀ = 0.006, ν = 0.060, ρ = 0.002, R\* = 0.15, E\* = 0.04.

---

## Scripts

| Script | What it does | Output |
|---|---|---|
| `alpha_crossover.py` | The central test: computes both lifetime branches over a range of α, locates their intersection, and checks that it coincides with arg max t_B(α) | `alpha_crossover.png`; agreement to within 1% on the human calibration |
| `monotonicity_check.py` | Level-2 check of the α-sensitivity sign pattern of the coupled system: ∂E/∂α ≤ 0, ∂Σ/∂α ≤ 0, ∂η/∂α ≤ 0, ∂R/∂α ≥ 0, pointwise in α by finite differences; reports where any monotonicity breaks | console report |
| `alpha_guar_check.py` | Sturm bound α_guar for the monotonicity closure of Appendix G.4, replacing an earlier WKB accumulated-phase argument on the forced oscillator Ẅ + k(t)W = Σ̇ | console report |
| `rho_residual_check.py` | Diagnostic: re-runs the sign-pattern check for decreasing ρ to test whether the small residual violation in Σ/R monotonicity is the ρ·g′ restoration term; compliance → 100% as ρ → 0 confirms it | console report |
| `stability_check.py` | Stability of α\* under grid resolution and integration horizon; isolates whether an earlier α\* ≈ 0.10 was a coarse-grid artifact | console report |
| `reproductive_vitality.py` | The trade-off of §19.3.6: the lifespan gain from α_human → α\* is post-reproductive, while the memory feedback producing it lowers reproductive vitality — hence selection settles below α\* | console report |

## Running

Python 3 with NumPy and SciPy. Each script is standalone:

    python alpha_crossover.py

`alpha_crossover.py` writes `alpha_crossover.png`; the remaining scripts report to the console.

## Note on the result

The status of the conclusion is level 1 in the sense used in the monograph — the same as the R-channel of the finite-lifetime theorem, which is proved under the hypothesis H_Σ. It rests on a separation of timescales and on a sufficient-condition threshold, not on all α. Beyond that threshold the monotonicities are confirmed numerically. The particular value of α\* is calibration-dependent and is deliberately not fixed as a parameter of the theory: what carries meaning is the structural statement — an optimum exists, the death channel switches there, and the calibrated α for real species lies below it.
