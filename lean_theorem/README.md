# VBLS — Formal proof of the finite lifetime theorem (Lean 4 + Mathlib)

Machine-checked formalization of the central result of the VBLS monograph (Appendix A): every system of class $(\mathcal{S})$ has a finite lifetime $t_B < +\infty$.

## Status

**Sorry-free.** Both channels are closed:

- **Main theorem** (`finite_lifespan_from_axioms`, E-channel) — closed from axioms A1–A5 plus the non-degeneracy condition $\Sigma(t_A) > 0$. Axiom A6 (the upper bound $\Sigma \le 1$) is deliberately **not** among its hypotheses — that bound is a scale normalization, never used by the E-channel derivation (see the monograph's Appendix A.5, Remark 6).
- **R-channel corollary** (`finite_death_R`, Lemma A.4) — closed under the explicit condition $\Sigma_\infty > \Sigma_{crit}$ (a margin inequality plus an eventual bound $\Sigma \ge s_0$), not derivable from A1–A6 alone; $\Sigma\to1$ is a special case of this condition.

`#print axioms` on both results returns only the standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`.

## Proof strategy

Four coupled ODEs (system $(\mathcal{S})$): integrity potential $E(t)$, structural damage fraction $\Sigma(t)\in[0,1]$ (`damage` in the code), assimilation efficiency $\eta(t)\in(0,1]$ (`eta`), repair reserve $R(t)\in[0,1]$ (`rep`).

| Lean name | Monograph correspondence | Statement |
|---|---|---|
| `damage_nondecreasing` | Lemma A.1 | $\Sigma(t)$ is non-decreasing (from $\max(0,\cdot)\ge0$ in the ODE) |
| `eta_tends_zero` | Lemma A.2 | $\eta(t)\to0$ as $t\to+\infty$ ($\eta$ decays like $e^{-\alpha\int\Sigma}$, and $\int\Sigma\to+\infty$) |
| `finite_T1` | Lemma A.2a | $\exists$ finite $T_1$: $\eta(t)<\delta$ for all $t\ge T_1$ (standard $\varepsilon$–$\delta$ limit) |
| `finite_death_E` | Lemma A.3 | $\exists$ finite $t_{BE}$: $E(t_{BE})\le E^*$ — once $\eta$ is small, $dE/dt<-\varepsilon^*<0$ |
| `finite_lifespan_from_axioms` | **Main theorem** | $t_B\le t_{BE}<+\infty$ |
| `finite_death_R` | Lemma A.4 (R-channel) | $\exists$ finite $t_{BR}$: $R(t_{BR})\le R^*$, under $\Sigma_\infty>\Sigma_{crit}$ |

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
| `VBLS_theorem.lean` | The formalization itself — axioms A1–A6, the four ODE right-hand sides, Lemmas A.1–A.4, the main theorem, and the `#print axioms` audits. |
| `lakefile.lean` | Lake package definition; pins the Mathlib4 commit. |
| `lean-toolchain` | Pins the Lean 4 version. |
| `lake-manifest.json` | Resolved versions of all transitive dependencies. |

## License

CC BY-NC-ND 4.0 — see [../LICENSE](../LICENSE).
