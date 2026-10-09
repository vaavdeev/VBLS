# Paper1 — numbers and figures for the article on the permanent-trace principle

Script for the companion article (in preparation) that isolates the structural property behind the finiteness theorem of the monograph (Appendix A): a permanent record of damage exposure in the assimilation efficiency.

| File | Content |
|---|---|
| `paper1_numbers.py` | Integrates the exact vector field of the canonical system (Appendix A.1): the restoration term carries max(0, 1 − Σ), g(E) is clipped to [0, 1], and the state is not projected onto [0, 1]. Memory is written in kernel form, η = η₀·exp(−αH) with dH/dt = Σ − κH; κ = 0 is the permanent trace (K ≡ 1), κ > 0 an exponentially decaying kernel. |
| `paper1_numbers.out.txt` | Output of the last run: the illustrative trajectory (t_B, death channel, branch times on the continuation, Σ_crit, the explicit E-channel bound); the memory-sensitivity scan t_B(α) with both branches and the maximiser α*; permanent versus decaying kernel in a constant environment. |
| `paper1_fig1.png` | (a) The illustrative trajectory and its continuation past t_B; (b) t_B(α) with the reserve and potential branches. |
| `paper1_fig2.png` | Permanent trace (K ≡ 1) versus exponentially decaying kernel (κ = 0.01) in a constant environment: assimilation efficiency, potential and reserve. |

**Run:** `python paper1_numbers.py` (Python 3 with NumPy, SciPy and Matplotlib; about a minute and a half). The figures are written next to the script.

**Parameters:** the illustrative parameter set of the monograph (Section 17.1); time in model units.

**Related sections of the monograph:** Section 9.5 (the structural form of the theorem), Section 12.1.2 (the kernel dichotomy), Appendix A (Corollary 3, uniformity in the environment).
