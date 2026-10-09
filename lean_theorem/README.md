# VBLS — Formal proof of the finite lifetime theorem (Lean 4 + Mathlib)

Machine-checked formalization of the central result of the VBLS monograph (Appendix A): every system of class $(\mathcal{S})$ that has accumulated damage at least once has a finite lifetime $t_B < +\infty$; for every trajectory, either $\Sigma\equiv0$ or $t_B<+\infty$.

## Status

**Sorry-free.** Both channels are closed:

- **Main theorem** (`finite_lifespan_from_axioms`, E-channel) — closed under axioms A1–A4 and the non-degeneracy condition (N): $\Sigma(t_1) > 0$ at some moment $t_1$, with the memory solution (A.1) taken as the hypothesis `hMemSol`. Axiom A5 (existence and continuation of solutions) is not needed: the statement holds for any trajectory with the stated right derivatives (see the table below). Since $\Sigma\ge0$ is non-decreasing, (N) fails only for $\Sigma\equiv0$, so for every trajectory either damage never leaves a trace or $t_B<+\infty$. Axiom A6 (the admissible box, in particular the upper bound $\Sigma \le 1$) is deliberately **not** among its hypotheses — that bound is a scale normalization, never used by the E-channel derivation (see the monograph's Appendix A.5, Remark 6).
- **R-channel corollary** (`finite_death_R`, Lemma A.4) — stated for the full equation (4.1), with restoration $\rho\,g(E)\max(0,1-\Sigma)(1-R)$ for any $g(E)\le1$; closed under the explicit condition that $\Sigma$ exceeds $\Sigma_{crit}$ (a margin inequality plus an eventual bound $\Sigma \ge s_0$), which is not derivable from A1–A6 alone; $\Sigma\to1$ is a special case. It does not assume $\Sigma\le1$; it uses only $R\le1$.
- **Derivatives.** Axiom A2 allows piecewise-continuous, right-continuous fluxes, at whose jumps the solution has only a right derivative; all ODE hypotheses are therefore stated with right derivatives (`HasDerivWithinAt` on $[t,\infty)$), which covers the continuous case as well.
- **What is and is not formalized.** The formal conclusions are existence statements: `finite_lifespan_from_axioms` gives a moment $t_B>t_A$ at which the state lies outside the viability domain; `finite_death_E` and `finite_death_R` give moments after $t_A$ with $E\le E^*$ and $R\le R^*$. Not formalized: existence and continuation of solutions (A5); the derivation of the memory solution (A.1) from equation (5.1), which enters as the hypothesis `hMemSol` (`rhsEta` is declared but unused); the equalities $E(t_B^E)=E^*$ and $R(t_B^R)=R^*$; the time bounds of Lemmas A.3, A.4 and of the theorem; and the explicit parameter bound on $t_B$.

`#print axioms` on both results returns only the standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`.

## Correspondence with the monograph

Lean checks the formal statements below; this table records how they relate to Appendix A of the monograph.

| Monograph (Appendix A) | Lean | Note |
|---|---|---|
| A1 (birth) | `AxiomA1` | |
| A2 (bounded fluxes) | `AxiomA2` | bound on $\Phi_{in}$; the theorem needs no bound on $\Phi_{cap}$ |
| A3, A4 (dissipation) | `DissipFn` | A4 is the field `hMinDiss`; in the monograph it follows from A3 and $E^*>0$. `DissipFn` asks for strict monotonicity on all of $\mathbb R$, while A5 extends $F_{diss}$ by zero for $E<0$ — see below |
| A5 (existence, uniqueness, continuation) | not formalized | the theorems hold for any trajectory with the stated right derivatives |
| A6 (admissible box) | `AxiomA6` (defined, unused) | not a hypothesis of any theorem |
| Condition (N) | `hND : ∃ t1, 0 < (traj t1).damage` | |
| Solution (A.1) of the memory equation | `hMemSol` | $\eta=\eta_0e^{-\alpha I}$ with $I'=\Sigma$, for $t\ge t_A$; a hypothesis — its derivation from (5.1) is not formalized |
| Lemma A.4, $\Sigma>\Sigma_{crit}$ | `s0`, `hmargin`, `hSigLB` | level $s_0\le1$ above $\Sigma_{crit}$, reached eventually; `hmargin` reads $\rho(1-s_0)(1-R^*)<(\nu_0+\nu s_0)R^*$, which is equivalent to $s_0>\Sigma_{crit}$ because the denominator $\nu R^*+\rho(1-R^*)$ of $\Sigma_{crit}$ is positive |
| Explicit parameter bound on $t_B$ (remark after Lemma A.2a) | not formalized | `finite_death_E` proves existence of the crossing time |
| Time bounds of Lemmas A.3, A.4 and of the theorem; equalities $E(t_B^E)=E^*$, $R(t_B^R)=R^*$ | not formalized | the formal statements give existence of the crossing times only |

## Applying the theorems to a solution on $t\ge t_A$

The hypotheses `hEDeriv`, `hRDeriv`, `hRle1`, `hI_deriv`, `hSigma_mono` and `AxiomA2` are imposed for all $t\in\mathbb R$, whereas the monograph's solution is defined for $t\ge t_A$; only `hMemSol` is restricted to $t\ge t_A$. A solution is brought under the hypotheses by an extension to $t<t_A$. The conclusions concern moments after $t_A$ and do not depend on the extension.

- *E-channel* (`finite_lifespan_from_axioms`, `finite_death_E`). For $t<t_A$ set $E\equiv E_0$, $\Sigma\equiv\Sigma_0$, $\Phi_{in}\equiv\Phi_{max}$ and $\eta\equiv F_{diss}(E_0)/\Phi_{max}$. The right-hand side of (6.1) then vanishes there, $\Sigma$ stays continuous and non-decreasing, and $\eta$ for $t<t_A$ is not constrained by `hMemSol`.
- *R-channel* (`finite_death_R`). For $t<t_A$ set $R\equiv R_0$, $g\equiv0$ and $\Sigma\equiv-\nu_0/\nu$. The right-hand side of (4.1) then vanishes, and the lemma imposes no other condition on $\Sigma$ or $g$ there. This extension is formal ($\Sigma<0$), and in the typical case it cannot be avoided. If $R_0>\rho/(\rho+\nu_0)$, i.e. the reserve starts above its damage-free plateau (as it does for $R_0$ near 1), no extension with $R\le1$, $g\le1$ and $\Sigma\ge0$ exists: while $R$ stays above the plateau, $dR/dt\le(\rho+\nu_0)\bigl(\rho/(\rho+\nu_0)-R\bigr)<0$, so backwards in time the excess over the plateau grows at least exponentially and $R$ exceeds 1 in finite time. Only for $R_0\le\rho/(\rho+\nu_0)$ is an extension with physically admissible values possible: $\Sigma\equiv0$, $R\equiv R_0$ and the constant $g=\nu_0R_0/\bigl(\rho(1-R_0)\bigr)\le1$.
- *Dissipation.* `DissipFn` asks for $F$ strictly increasing on all of $\mathbb R$, whereas axiom A5 extends $F_{diss}$ by zero for $E<0$. Since $E\ge0$ on solutions, $F$ may be continued to $E<0$ by any strictly increasing function without changing the trajectory.

Restricting the hypotheses to $t\ge t_A$ (`Set.Ici tA`) would make these extensions unnecessary.

## Proof strategy

Four coupled ODEs (system $(\mathcal{S})$): integrity potential $E(t)$, structural damage fraction $\Sigma(t)\ge0$ (normalized scale $[0,1]$; `damage` in the code), assimilation efficiency $\eta(t)\in(0,1]$ (`eta`), repair reserve $R(t)\in[0,1]$ (`rep`).

| Lean name | Monograph correspondence | Statement |
|---|---|---|
| `damage_nondecreasing` | Lemma A.1 | $\Sigma(t)$ is non-decreasing (from $\max(0,\cdot)\ge0$ in the ODE; via `mono_of_rightDeriv_nonneg`) |
| `eta_tends_zero` | Lemma A.2 | $\eta(t)\to0$ as $t\to+\infty$ ($\eta$ decays like $e^{-\alpha\int\Sigma}$, and $\int\Sigma\to+\infty$) |
| `I_diverges_of_Sigma_mono_pos_at` | Lemma A.2 | $\int\Sigma\to+\infty$ under (N): $\Sigma(t_1)>0$ at some $t_1$ |
| `finite_T1` | Lemma A.2a | $\exists$ finite $T_1$: $\eta(t)<\delta$ for all $t\ge T_1$ (standard $\varepsilon$–$\delta$ limit) |
| `finite_death_E` | Lemma A.3 | $\exists$ finite $t_{BE}$: $E(t_{BE})\le E^*$ — once $\eta$ is small, $dE/dt\le-\varepsilon^*<0$ |
| `finite_lifespan_from_axioms` | **Main theorem** | $t_B\le t_{BE}<+\infty$ |
| `finite_death_R` | Lemma A.4 (R-channel) | $\exists$ finite $t_{BR}$: $R(t_{BR})\le R^*$, once $\Sigma$ exceeds $\Sigma_{crit}$ |

Naming convention (Lean 4 requires ASCII field names): `damage` = $\Sigma$, `eta` = $\eta_{abs}$, `rep` = $R$.

## Requirements

- **Lean 4.14.0** — pinned in [`lean-toolchain`](lean-toolchain); if you use [elan](https://github.com/leanprover/elan), the correct toolchain is selected automatically.
- **Mathlib4** pinned to a specific commit compatible with Lean 4.14.0 (see [`lakefile.lean`](lakefile.lean)); transitive dependencies (aesop, batteries, Qq, import-graph, proofwidgets, ...) are resolved automatically by Lake from [`lake-manifest.json`](lake-manifest.json).
- First build fetches Mathlib4 over the network — this is a large download.

## Build / verify

```bash
lake exe cache get      # fetch prebuilt Mathlib .olean cache (recommended, much faster)
lake env lean VBLS_theorem.lean
```

**Note:** `lake build` does not compile this file correctly in this project (the `lean_lib` target's `roots` is set without a matching `srcDir` layout) — always use `lake env lean VBLS_theorem.lean` directly to compile and see errors/warnings.

## Files

| File | Description |
|---|---|
| `VBLS_theorem.lean` | The formalization itself — axioms A1–A6 (A6 defined for reference, unused), the four ODE right-hand sides, Lemmas A.1–A.4, the main theorem, and the `#print axioms` audits. |
| `lakefile.lean` | Lake package definition; pins the Mathlib4 commit. |
| `lean-toolchain` | Pins the Lean 4 version. |
| `lake-manifest.json` | Resolved versions of all transitive dependencies. |

## License

CC BY-NC-ND 4.0 — see [../LICENSE](../LICENSE).
