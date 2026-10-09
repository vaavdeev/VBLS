/-
  ══════════════════════════════════════════════════════════════════════════════
  VBLS — Viability Boundaries of Living Systems
  Formal verification in Lean 4 + Mathlib

  THEOREM: Every system of class (S) that has accumulated damage at least once
  (condition (N)) has a finite lifespan t_B < +∞.

  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY.md (Appendix A)
  Toolchain:  Lean v4.14.0, Mathlib v4.14.0
  ══════════════════════════════════════════════════════════════════════════════

  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
  ══════════════════════════════════════════════════════════════════════════════

  WHAT THIS FILE PROVES
  ─────────────────────
  A biological or minimal living system is modelled by four coupled ODEs
  (the system (S)) over variables:
      E(t)   — integrity potential (phenomenological potential of self-maintenance)
      Σ(t)   — structural damage fraction ≥ 0, scale [0, 1]  (upper bound = scale
               normalization, NOT dynamically enforced; see Axiom A6 caveat)
      η(t)   — assimilation efficiency (0, 1]
      R(t)   — repair reserve capacity      [0, 1]

  The main theorem says: under axioms A1–A5 and the non-degeneracy condition
  (N) — Σ(t₁) > 0 at some moment t₁ — the system exits its viability region
  V = {E > E*, R > R*} in finite time t_B.  Equivalently: for every solution,
  either Σ ≡ 0 (damage never left a trace) or t_B < +∞.  Finite lifespan is a
  mathematical theorem, not an empirical observation.  (Axiom A6 — the
  admissible box, in particular the upper bound Σ ≤ 1 — is deliberately NOT
  among the hypotheses; see below.)

  PROOF STRATEGY  (chain of four lemmas)
  ───────────────────────────────────────
  A.1  Σ(t) is non-decreasing         (from max(0,·) ≥ 0 in the ODE)
  A.2  η(t) → 0  as t → +∞           (η decays like exp(-α·∫Σ), and ∫Σ → +∞)
  A.2a ∃ finite T₁: η(t) < δ ∀ t≥T₁  (standard ε–δ limit)
  A.3  ∃ finite t_BE: E(t_BE) ≤ E*    (KEY: once η is small, dE/dt ≤ -ε* < 0)
  ──►  Main theorem: t_B ≤ t_BE < +∞  □

  The R-channel (Lemma A.4) — R(t) hitting R* — is a separate corollary
  requiring that Σ exceed, at some moment, the critical level
  Σ_crit = [ρ(1−R*) − ν₀R*] / [νR* + ρ(1−R*)]; it is NOT needed for the main
  theorem.

  DERIVATIVES
  ───────────
  Axiom A2 allows piecewise-continuous, right-continuous fluxes.  At a jump of
  a flux the solution has only a right derivative, so every ODE hypothesis
  below is stated with HasDerivWithinAt on [t, ∞) (right derivatives); the
  continuous-flux case is included.

  STATUS
  ──────
  Main theorem `finite_lifespan_from_axioms` is closed from A1–A5 + the
  non-degeneracy condition (N): ∃ t₁, Σ(t₁) > 0 — with no remaining sorry.
  Axiom A6 is NOT among its hypotheses: the upper bound Σ ≤ 1 is a scale
  normalization and is never used by the E-channel (monograph App. A.5, R.6).
  The R-channel `finite_death_R` is ALSO closed (no sorry): it is stated for
  the full equation (4.1) — restoration ρ·g(E)·max(0, 1−Σ)·(1−R) with any
  g(E) ∈ [0, 1] — under the explicit condition Σ(t) ≥ s0 > Σ_crit eventually,
  which is not derivable from A1–A6; Σ → 1 is its special case.  It does not
  assume Σ ≤ 1.  The whole file is sorry-free; both `#print axioms` audits
  return only the standard Lean axioms.

  NOT FORMALIZED
  ──────────────
  · Existence, uniqueness and continuation of solutions (axiom A5): the
    theorems are statements about any trajectory with the stated derivatives.
  · The derivation of the memory solution (A.1) from equation (5.1): it
    enters as the hypothesis `hMemSol`; `rhsEta` is declared but not used.
  · The equalities E(t_B^E) = E* and R(t_B^R) = R* at the crossing moments
    and the time bounds of Lemmas A.3, A.4 and of the main theorem:
    `finite_death_E` and `finite_death_R` prove existence of a crossing
    time only.
  · The explicit parameter bound on t_B (remark after Lemma A.2a of the
    monograph).

  DOMAIN OF THE HYPOTHESES
  ────────────────────────
  The derivative, monotonicity and flux hypotheses are imposed for all t ∈ ℝ,
  whereas the monograph's solution is defined for t ≥ t_A.  A solution is
  brought under them by an extension to t < t_A (README, "Applying the
  theorems to a solution on t ≥ t_A").  `DissipFn` asks for F strictly
  increasing on all of ℝ, while axiom A5 extends F_diss by zero for E < 0;
  since E ≥ 0 on solutions, F may be continued to E < 0 by any strictly
  increasing function without affecting the trajectory.

  NAMING CONVENTION (Lean 4 requires ASCII field names)
  ──────────────────────────────────────────────────────
    damage  = Σ   (accumulated structural damage fraction)
    eta     = η   (assimilation efficiency, η_abs)
    rep     = R   (repair reserve)
-/

import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.ODE.PicardLindelof
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.MeasureTheory.Integral.FundThmCalculus
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import Mathlib.Order.Filter.Basic
import Mathlib.Analysis.Calculus.MeanValue

open Real Filter MeasureTheory intervalIntegral

-- ══════════════════════════════════════════════════════════════════════════════
-- §1.  PARAMETERS OF SYSTEM (S)
-- ══════════════════════════════════════════════════════════════════════════════

/--
  Kinetic parameters of system (S).  All are strictly positive real numbers.

  Physical meaning (see monograph §4, §5, §7):
    s0  (σ₀) — fraction of dissipated energy that irreversibly converts to
                structural damage; calibrated from somatic mutation rates.
    a   (α)  — rate at which accumulated damage degrades η_abs; the "memory"
                coefficient linking ∫Σ dτ to efficiency loss.
    v0  (ν₀) — baseline decay rate of repair reserve R: spontaneous wear of
                the reserve under thermal noise (monograph §4.6).
    v   (ν)  — damage-dependent acceleration of repair decay (Σ amplifies
                the loss of R beyond the baseline rate).
    rho (ρ)  — restoration intensity of R at full potential; an energetic
                order-of-magnitude estimate gives ρ ≲ F_diss(E₀) / (E₀ - E*).
                Condition (C4): ρ ≲ ν₀, so
                the reserve has no high plateau (monograph §4.6).  (C4)
                restricts rates, not the outcome: the reserve may settle at
                a low level above R*.
-/
structure VBLSParams where
  s0   : ℝ   -- σ₀ : damage production coefficient
  a    : ℝ   -- α  : memory degradation coefficient
  v0   : ℝ   -- ν₀ : baseline repair reserve decay rate
  v    : ℝ   -- ν  : damage-dependent repair decay rate
  rho  : ℝ   -- ρ  : repair restoration rate
  hs0  : 0 < s0
  ha   : 0 < a
  hv0  : 0 < v0
  hv   : 0 < v
  hrho : 0 < rho

/--
  Viability thresholds and external flux bounds.

    Estar  (E*)     — minimum integrity potential for active self-maintenance.
                      Below E* the system cannot sustain repair.
    Rstar  (R*)     — critical threshold for the repair reserve, fixed by the
                      definition of the viability domain; it does not enter the
                      right-hand sides and marks no bifurcation (monograph §9.2, §9.6).
    eps    (ε*)     — half of F_diss(E*); used in Axiom A4 and in the key
                      estimate of Lemma A.3: once η < ε*/Φ_max, dE/dt < -ε*.
    Phimax (Φ_max)  — uniform upper bound on the input resource flux Φ_in(t);
                      comes from Axiom A2 (bounded environment).
-/
structure VBLSThresholds where
  Estar   : ℝ
  Rstar   : ℝ
  eps     : ℝ   -- ε*: defined by F_diss(E*) = 2ε* > 0  (Axiom A4)
  Phimax  : ℝ   -- Φ_max: upper bound on input flux  (Axiom A2)
  hEstar  : 0 < Estar
  hRstar  : 0 < Rstar
  heps    : 0 < eps
  hPhimax : 0 < Phimax

-- ══════════════════════════════════════════════════════════════════════════════
-- §2.  STATE VECTOR AND VIABILITY CONDITION
-- ══════════════════════════════════════════════════════════════════════════════

/--
  The state vector of system (S) at time t.
  All four variables are dimensionless (normalised to system-characteristic scales).
-/
structure VBLSState where
  E      : ℝ        -- integrity potential (phenomenological non-equilibrium potential)
  damage : ℝ        -- Σ: structural damage fraction ≥ 0 (scale [0, 1]; Σ ≤ 1 not enforced)
  eta    : ℝ        -- η_abs: assimilation efficiency ∈ (0, 1]
  rep    : ℝ        -- R: repair reserve ∈ [0, 1]

/--
  A state is viable if both thresholds are satisfied simultaneously.
  Death is defined as the first exit from this region:
      t_B = inf { t ≥ t_A | ¬ isViable(state(t)) }
-/
def isViable (s : VBLSState) (th : VBLSThresholds) : Prop :=
  s.E > th.Estar ∧ s.rep > th.Rstar

-- ══════════════════════════════════════════════════════════════════════════════
-- §3.  AXIOMS A1–A6
-- ══════════════════════════════════════════════════════════════════════════════
/-
  Axioms A1–A6 are the standing assumptions of the monograph's theorem; the
  main theorem below uses A1–A5 and condition (N), and A6 follows from A5.
  The class (S) itself is fixed by Definition 2.1 of the monograph.  The
  axioms are NOT universal physical laws.

  In Lean each axiom is a Prop (or structure) that appears as a hypothesis
  in the relevant lemmas.  Axiom A4 follows from A3 and the positivity of the
  threshold E* (monograph Definition 2.1; field `hEstar` here), not from A1;
  it is named explicitly because it drives the key inequality in Lemma A.3.
-/

/--
  Axiom A1 — Viable start.
  At the initial moment t_A the system is inside the viability region:
  E > E*, R > R*, and η > 0.  This excludes systems that are born dead.
-/
def AxiomA1 (s0 : VBLSState) (th : VBLSThresholds) : Prop :=
  s0.E > th.Estar ∧ s0.rep > th.Rstar ∧ s0.eta > 0

/--
  Axiom A2 — Bounded external fluxes.
  The input resource flux Φ_in(t) is non-negative and uniformly bounded
  above by Φ_max.  This models a finite, non-ideal environment.
  No matter how good the environment, it cannot prevent the death of a system
  that has accumulated damage at least once (see Corollary 1).
-/
def AxiomA2 (PhiIn : ℝ → ℝ) (th : VBLSThresholds) : Prop :=
  ∀ t : ℝ, 0 ≤ PhiIn t ∧ PhiIn t ≤ th.Phimax

/--
  Axioms A3 + A4 — Dissipation function F_diss.

  A3: F_diss is continuous, strictly increasing, and F_diss(0) = 0.
      This encodes: higher integrity potential → faster internal dissipation.
      (In the canonical form μE² + λE, the linear term λE is the Onsager-type
      relaxation near E = 0 — a structural guide, monograph §6.8.)

  A4 (named ε*): F_diss(E) ≥ 2ε* > 0 whenever E ≥ E*.
      This says: even at the viability threshold E*, the system still dissipates
      at a positive rate.  Biologically: a dying system keeps producing damage.
      In Lean: hMinDiss encodes A4; the value 2ε* = F_diss(E*) > 0 follows from
      A3 (StrictMono, F(0)=0) and the positivity of the threshold E* > 0
      (monograph Definition 2.1; field `hEstar`), not from A1.
-/
structure DissipFn (th : VBLSThresholds) where
  F        : ℝ → ℝ
  hCont    : Continuous F
  hZero    : F 0 = 0
  hMono    : StrictMono F
  hMinDiss : ∀ E : ℝ, E ≥ th.Estar → F E ≥ 2 * th.eps

/--
  Axiom A6 — Invariant domain (well-posedness / normalization).
  The trajectory is taken inside the physically meaningful box for all t:
      E ≥ 0,  damage ∈ [0,1],  eta ∈ (0,1],  rep ∈ [0,1].

  CAVEAT (monograph Appendix A.5, Remark 6).  The UPPER bound `damage ≤ 1` is a
  NORMALIZATION of scale, not a property enforced by the dynamics: the Σ-ODE
  `max(0, σ₀·F − R·Φ_cap)` has NO saturating factor, so on the face Σ = 1 the
  field can still point outward (dΣ/dt ≥ 0).  Hence the box is NOT positively
  invariant at Σ = 1.  The other faces are: with the clipped restoration factor
  max(0, 1−Σ) of equation (4.1), dR/dt = ρ·g(E)·max(0, 1−Σ) ≥ 0 at R = 0 and
  dR/dt = −ν₀ − νΣ < 0 at R = 1, so 0 ≤ R ≤ 1 holds whatever Σ does
  (monograph axioms A5, A6).

  Crucially, the finite-lifespan theorem does NOT use `damage ≤ 1`: the E-channel
  lemmas use only monotonicity of Σ and a positive value Σ(t₁) > 0.
  Accordingly `finite_death_E`, `finite_lifespan`, `finite_lifespan_from_axioms`
  carry NO AxiomA6 hypothesis, and neither does `finite_death_R`, which needs
  only R ≤ 1.  `AxiomA6` is kept as a definition for reference.
-/
def AxiomA6 (traj : ℝ → VBLSState) : Prop :=
  ∀ t : ℝ,
    (traj t).E ≥ 0 ∧
    0 ≤ (traj t).damage ∧ (traj t).damage ≤ 1 ∧
    0 < (traj t).eta    ∧ (traj t).eta    ≤ 1 ∧
    0 ≤ (traj t).rep    ∧ (traj t).rep    ≤ 1

-- ══════════════════════════════════════════════════════════════════════════════
-- §4.  RIGHT-HAND SIDES OF SYSTEM (S)
-- ══════════════════════════════════════════════════════════════════════════════
/-
  The four ODEs of system (S) (monograph equations 6.1, 7.1, 5.1, 4.1).
  Each right-hand side is defined as a pure function of the current state
  and external inputs; the ODE is then expressed as HasDerivAt in lemmas.

  NOTE: rhsRep is the full equation (4.1),
      dR/dt = -ν₀R - νΣR + ρ·g(E)·max(0, 1-Σ)·(1-R),
  where the value gE of the restoration-potential function
  g(E) = max(0, min(1, (E-E*)/(E₀-E*))) ∈ [0,1]  (monograph formula 4.2)
  enters as an argument; `finite_death_R` uses only 0 ≤ g ≤ 1.  The clip
  max(0, 1-Σ) extends restoration by zero beyond the normalized scale
  (monograph §4.5).  The E-channel proof does not use rhsRep at all.
-/

-- All four RHS functions take (p, th, fd) explicitly with the same calling
-- convention, regardless of which parameters each one actually uses.

/-- ODE for E(t) — integrity potential.  Equation (6.1).
    Inflow: η·Φ_in(t).  Outflow: F_diss(E).
    When η → 0 (memory collapse), the inflow vanishes and E decays to E*. -/
def rhsE (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (PhiIn : ℝ → ℝ) (eta E : ℝ) (t : ℝ) : ℝ :=
  eta * PhiIn t - fd.F E

/-- ODE for Σ(t) — structural damage.  Equation (7.1).
    Σ can only increase (max(0,·) ensures non-negativity of the derivative).
    Production: σ₀·F_diss(E).  Export (repair): R·Φ_cap(t).
    When export ≥ production, dΣ/dt = 0 (damage is controlled but not reversed).
    NOTE: there is NO (1−Σ) saturation factor, so this RHS does not bound Σ
    above by 1; the upper bound Σ ≤ 1 is a scale normalization (Axiom A6 caveat). -/
def rhsDamage (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (PhiCap : ℝ → ℝ) (E rep : ℝ) (t : ℝ) : ℝ :=
  max 0 (p.s0 * fd.F E - rep * PhiCap t)

/-- ODE for η(t) — assimilation efficiency.  Equation (5.1).
    η decays proportionally to the current damage level Σ.
    Analytical solution: η(t) = η₀·exp(-α·∫_{tA}^t Σ(τ) dτ).
    This permanent memory is what makes the lifespan finite once damage has
    left a trace. -/
def rhsEta (p : VBLSParams) (damage eta : ℝ) : ℝ :=
  -p.a * damage * eta

/-- ODE for R(t) — repair reserve.  Equation (4.1).
    Decay: -ν₀·R (spontaneous wear) and -ν·Σ·R (damage-accelerated).
    Restoration: +ρ·g(E)·max(0, 1-Σ)·(1-R) — blocked when Σ ≥ 1 (full damage),
    when R = 1 (full reserve) and when E ≤ E* (g = 0); gE is the value g(E). -/
def rhsRep (p : VBLSParams) (gE damage rep : ℝ) : ℝ :=
  -p.v0 * rep - p.v * damage * rep + p.rho * gE * max 0 (1 - damage) * (1 - rep)

-- ══════════════════════════════════════════════════════════════════════════════
-- §5.  AUXILIARY LEMMAS (the proof chain A.1 → A.2 → A.2a → A.3)
-- ══════════════════════════════════════════════════════════════════════════════

/-  ────────────────────────────────────────────────────────────────────────────
    LEMMA A.1  —  Σ(t) is non-decreasing.

    Why it matters: once we know Σ never goes back down, we can lower-bound
    the integral ∫Σ dτ by Σ(t₁)·(t - t₁) → +∞, where Σ(t₁) > 0 by the
    non-degeneracy condition (N).  This drives Lemma A.2.

    Proof idea: the right-hand side of the Σ-ODE is max(0,·) ≥ 0, so the
    right derivative is always non-negative → `mono_of_rightDeriv_nonneg`.
    ──────────────────────────────────────────────────────────────────────────-/

/-- A continuous function with a non-negative right derivative everywhere is
    monotone.  Proof: compare -f with the constant -f(a) on [a, b] using
    Mathlib's right-derivative comparison lemma. -/
lemma mono_of_rightDeriv_nonneg {f f' : ℝ → ℝ}
    (hf : Continuous f)
    (hderiv : ∀ x : ℝ, HasDerivWithinAt f (f' x) (Set.Ici x) x)
    (hnn : ∀ x : ℝ, 0 ≤ f' x) :
    Monotone f := by
  intro a b hab
  have key : ∀ x ∈ Set.Icc a b, -f x ≤ -f a :=
    image_le_of_deriv_right_le_deriv_boundary
      (f := fun y => -f y) (f' := fun y => -f' y)
      (B := fun _ => -f a) (B' := fun _ => 0)
      hf.neg.continuousOn
      (fun x _ => (hderiv x).neg)
      le_rfl
      continuousOn_const
      (fun x _ => hasDerivWithinAt_const x _ (-f a))
      (fun x _ => by simp only; linarith [hnn x])
  have hb := key b ⟨hab, le_rfl⟩
  linarith

lemma damage_nondecreasing
    (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (traj : ℝ → VBLSState)
    (PhiCap : ℝ → ℝ)
    (hCont  : Continuous (fun t => (traj t).damage))
    (hDeriv : ∀ t : ℝ,
        HasDerivWithinAt (fun t => (traj t).damage)
                   (rhsDamage p th fd PhiCap (traj t).E (traj t).rep t) (Set.Ici t) t) :
    Monotone (fun t => (traj t).damage) :=
  mono_of_rightDeriv_nonneg
    (f' := fun t => rhsDamage p th fd PhiCap (traj t).E (traj t).rep t)
    hCont hDeriv
    (fun t => by simp only [rhsDamage]; exact le_max_left 0 _)

/-  ────────────────────────────────────────────────────────────────────────────
    LEMMA A.2  —  η(t) → 0 as t → +∞.

    Why it matters: η → 0 is the key consequence of memory accumulation.
    Once η is negligibly small, the energy inflow η·Φ_in is dominated by
    dissipation F_diss(E), forcing E to decay (see Lemma A.3).

    Proof idea:
      · η(t) = η₀·exp(-α·I(t))  where  I(t) = ∫_{tA}^t Σ(τ) dτ
      · hIdiv says I(t) → +∞  (the KEY step: in finite_lifespan_from_axioms
        it is derived from Lemma A.1 and the non-degeneracy condition (N)
        by `I_diverges_of_Sigma_mono_pos_at`)
      · exp(-α·I(t)) → 0  because  -α·I(t) → -∞  and  exp(x) → 0 as x → -∞
    ──────────────────────────────────────────────────────────────────────────-/
lemma eta_tends_zero
    (p : VBLSParams)
    (eta0 : ℝ) (heta0 : 0 < eta0)
    (I : ℝ → ℝ)
    (hIdiv : Tendsto I atTop atTop) :
    Tendsto (fun t => eta0 * Real.exp (-p.a * I t)) atTop (nhds 0) := by
  have ha := p.ha
  -- Step 1: α·I(t) → +∞  (product of positive constant and divergent sequence)
  have h1 : Tendsto (fun t => p.a * I t) atTop atTop :=
    hIdiv.const_mul_atTop ha
  -- Step 2: -α·I(t) → -∞
  have h2 : Tendsto (fun t => -(p.a * I t)) atTop atBot :=
    tendsto_neg_atTop_atBot.comp h1
  -- Step 3: exp(-α·I(t)) → 0  (standard: exp → 0 at -∞)
  have h3 : Tendsto (fun t => Real.exp (-(p.a * I t))) atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp h2
  -- Step 4: scalar multiplication by η₀ > 0  (limit: η₀ · 0 = 0)
  have h4 : Tendsto (fun t => eta0 * Real.exp (-(p.a * I t))) atTop (nhds (eta0 * 0)) :=
    h3.const_mul eta0
  simp only [mul_zero] at h4
  exact h4.congr (fun t => by ring_nf)

/-  ────────────────────────────────────────────────────────────────────────────
    LEMMA A.2a  —  For any δ > 0, there exists a finite T₁ such that
                   η(t) < δ for all t ≥ T₁.

    Why it matters: this is the standard ε–δ unpacking of "η → 0".
    We need an explicit finite T₁ so that Lemma A.3 can write a concrete
    upper bound  t_BE ≤ T₁ + max 0 (E(T₁) - E*)/ε*  (in the text; the Lean
    statement of `finite_death_E` gives existence of t_BE only).

    Proof: direct application of Metric.tendsto_atTop (the definition of
    convergence in metric spaces), then extract T₁ and use |x - 0| = |x|.
    ──────────────────────────────────────────────────────────────────────────-/
lemma finite_T1
    (etaFn : ℝ → ℝ)
    (hconv : Tendsto etaFn atTop (nhds 0))
    (delta : ℝ) (hdelta : 0 < delta) :
    ∃ T1 : ℝ, ∀ t : ℝ, t ≥ T1 → etaFn t < delta := by
  -- Unpack the definition of Tendsto via the metric on ℝ
  rw [Metric.tendsto_atTop] at hconv
  obtain ⟨T1, hT1⟩ := hconv delta hdelta
  exact ⟨T1, fun t ht => by
    have h := hT1 t ht
    -- dist(x, 0) = |x|; since etaFn(t) > 0 (physically), |etaFn(t)| < δ gives etaFn(t) < δ
    rw [Real.dist_eq, abs_lt] at h
    linarith [h.1]⟩

/-  ────────────────────────────────────────────────────────────────────────────
    LEMMA A.3  —  ∃ finite t_BE: E(t_BE) ≤ E*.   [E-CHANNEL DEATH]

    This is the MAIN MATHEMATICAL TASK, and it is fully closed (no sorry).

    Why it matters: this is the PRIMARY proof of finite lifespan.  It does NOT
    require the R-channel or the Σ_crit condition — and it does NOT require Axiom A6
    (the invariant box, in particular the upper bound Σ ≤ 1): it works from
    A1–A5 alone.

    Proof sketch:
      1. Set δ = ε*/Φ_max > 0.
      2. By Lemma A.2a: ∃ finite T₁ such that η(t) < δ for all t ≥ T₁.
      3. KEY ESTIMATE: for t ≥ T₁ while E(t) ≥ E*:
            dE/dt = η(t)·Φ_in(t) - F_diss(E(t))
                  < δ·Φ_max - 2ε*
                  = ε* - 2ε* = -ε* < 0
         So E decreases at rate ≥ ε* per unit time.
      4. LINEAR COMPARISON: E(t) ≤ E(T₁) - ε*·(t - T₁) for t ≥ T₁.
         Existence of a crossing time t* ∈ (T₁, T₁ + (E(T₁)-E*)/ε* + 1] with
         E(t*) ≤ E* is extracted by contradiction using Mathlib's
         `image_le_of_deriv_right_le_deriv_boundary`.
      5. Explicit bound (monograph, Appendix A; NOT formalized here — the
         statement below proves existence only):
            t_BE ≤ t_A + (1/αΣ_min)·ln⁺(η₀·Φ_max/ε*) + (Ē − E*)/ε*,
         where Ē = max(E₀, E₊) and F_diss(E₊) = η₀·Φ_max bounds E along the
         trajectory.  E₀ alone is not an upper bound: E rises above its
         starting value during the growth phase, so E(T₁) ≤ E₀ fails in general.
    ──────────────────────────────────────────────────────────────────────────-/
lemma finite_death_E
    (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (traj : ℝ → VBLSState)
    (PhiIn : ℝ → ℝ) (tA : ℝ)
    (hA1 : AxiomA1 (traj tA) th)
    (hA2 : AxiomA2 PhiIn th)
    -- NOTE: AxiomA6 (in particular the upper bound Σ ≤ 1) is deliberately ABSENT.
    -- The E-channel uses only flux boundedness (A2) and the decay of η (hEtaConv,
    -- ultimately the monotonicity of Σ).  Formal counterpart of monograph
    -- Appendix A.5, Remark 6: the normalization Σ ≤ 1 is NOT used by the theorem.
    (hECont  : Continuous (fun t => (traj t).E))
    -- right derivative (piecewise-continuous fluxes, axiom A2)
    (hEDeriv : ∀ t : ℝ,
        HasDerivWithinAt (fun t => (traj t).E)
                   (rhsE p th fd PhiIn (traj t).eta (traj t).E t) (Set.Ici t) t)
    (hEtaConv : Tendsto (fun t => (traj t).eta) atTop (nhds 0)) :
    ∃ tBE : ℝ, tA < tBE ∧ (traj tBE).E ≤ th.Estar := by
  -- Choose δ = ε*/Φ_max so that η < δ implies η·Φ_in < ε*
  have hδpos : (0 : ℝ) < th.eps / th.Phimax := div_pos th.heps th.hPhimax
  -- Apply Lemma A.2a: get finite T₁ with η(t) < δ for t ≥ T₁
  obtain ⟨T1, hT1⟩ := finite_T1 _ hEtaConv _ hδpos
  -- KEY ESTIMATE: while t ≥ T₁ and E(t) ≥ E*, the right derivative is ≤ -ε*
  have hdneg : ∀ t : ℝ, t ≥ T1 → (traj t).E ≥ th.Estar →
      rhsE p th fd PhiIn (traj t).eta (traj t).E t ≤ -th.eps := by
    intro t ht hE
    rw [rhsE]
    have heta_lt : (traj t).eta < th.eps / th.Phimax := hT1 t ht
    have hPhi    : PhiIn t ≤ th.Phimax              := (hA2 t).2
    have hPhiNN  : 0 ≤ PhiIn t                       := (hA2 t).1
    -- (No use of η ≥ 0 / AxiomA6: `mul_le_mul` below needs only 0 ≤ Φ_in and
    --  0 ≤ ε*/Φ_max, both already available.  Σ ≤ 1 is never invoked.)
    -- F_diss(E(t)) ≥ 2ε*  by Axiom A4 (hMinDiss), since E(t) ≥ E*
    have hFdiss  : fd.F (traj t).E ≥ 2 * th.eps      := fd.hMinDiss _ hE
    -- Upper bound on energy inflow: η·Φ_in ≤ (ε*/Φ_max)·Φ_max = ε*
    have h1 : (traj t).eta * PhiIn t ≤ th.eps := by
      calc (traj t).eta * PhiIn t
          ≤ (th.eps / th.Phimax) * th.Phimax :=
            mul_le_mul heta_lt.le hPhi hPhiNN (le_of_lt hδpos)
        _ = th.eps := div_mul_cancel₀ th.eps (ne_of_gt th.hPhimax)
    -- Combine: dE/dt = η·Φ_in - F_diss ≤ ε* - 2ε* = -ε*
    linarith
  -- Effective start: T0 = max (tA + 1) T1.  Two properties used below:
  --   · T0 > tA strictly  (so tA < tBE is automatic from T0 > tA)
  --   · T0 ≥ T1           (so hT1 and hdneg still apply for t ≥ T0)
  set T0 := max (tA + 1) T1 with hT0_def
  have hT0_gt_tA : tA < T0 := lt_of_lt_of_le (by linarith) (le_max_left _ _)
  have hT0_ge_T1 : T1 ≤ T0 := le_max_right _ _
  -- Case split on whether E has already crossed E* by T0.
  by_cases hET0 : (traj T0).E ≤ th.Estar
  · -- CASE 1: E(T0) ≤ E* already.  Use T0 as witness.
    exact ⟨T0, hT0_gt_tA, hET0⟩
  · -- CASE 2: E(T0) > E*.  Linear decay forces E to cross E* in (T0, tBE_max].
    --   We do NOT commit upfront to the explicit `T0 + (E(T0)−E*)/ε + 1` as the
    --   witness — once E first hits E*, it may oscillate back above E*, so the
    --   specific endpoint need not satisfy E ≤ E*.  Instead we prove existence
    --   of SOME crossing time t in (T0, tBE_max] via Mathlib's right-derivative
    --   comparison lemma `image_le_of_deriv_right_le_deriv_boundary`.
    push_neg at hET0  -- hET0 : (traj T0).E > th.Estar
    set tBE_max := T0 + ((traj T0).E - th.Estar) / th.eps + 1 with htBE_def
    have hgap_nn : 0 ≤ ((traj T0).E - th.Estar) / th.eps :=
      div_nonneg (by linarith) (le_of_lt th.heps)
    have hT0_lt_tBE : T0 < tBE_max := by rw [htBE_def]; linarith
    -- Linear bound B(t) = E(T0) − ε*·(t − T0); B(tBE_max) = E* − ε*.
    have hB_deriv : ∀ x : ℝ,
        HasDerivAt (fun t : ℝ => (traj T0).E - th.eps * (t - T0)) (-th.eps) x := by
      intro x
      have h1 : HasDerivAt (fun t : ℝ => t - T0) 1 x := (hasDerivAt_id x).sub_const T0
      have h2 : HasDerivAt (fun t : ℝ => th.eps * (t - T0)) th.eps x := by
        simpa using h1.const_mul th.eps
      simpa using (hasDerivAt_const x (traj T0).E).sub h2
    have hB_cont : Continuous (fun t : ℝ => (traj T0).E - th.eps * (t - T0)) := by
      continuity
    -- Existence of crossing time, proved by contradiction.
    have hexists : ∃ t : ℝ, t ∈ Set.Ioc T0 tBE_max ∧ (traj t).E ≤ th.Estar := by
      by_contra hneg
      push_neg at hneg
      -- hneg : ∀ t ∈ Ioc T0 tBE_max, (traj t).E > th.Estar
      -- Apply image_le_of_deriv_right_le_deriv_boundary on [T0, tBE_max].
      have happly : ∀ x ∈ Set.Icc T0 tBE_max,
          (traj x).E ≤ (traj T0).E - th.eps * (x - T0) :=
        image_le_of_deriv_right_le_deriv_boundary
          hECont.continuousOn
          (fun x _ => hEDeriv x)
          (by show (traj T0).E ≤ (traj T0).E - th.eps * (T0 - T0); ring_nf; exact le_refl _)
          hB_cont.continuousOn
          (fun x _ => (hB_deriv x).hasDerivWithinAt)
          (by
            intro x hx
            have hxT0 : T0 ≤ x := hx.1
            have hxLt : x < tBE_max := hx.2
            have hx_ge_T1 : T1 ≤ x := le_trans hT0_ge_T1 hxT0
            have hE_ge : (traj x).E ≥ th.Estar := by
              rcases lt_or_eq_of_le hxT0 with hxgt | hxeq
              · exact le_of_lt (hneg x ⟨hxgt, le_of_lt hxLt⟩)
              · rw [← hxeq]; exact le_of_lt hET0
            exact hdneg x hx_ge_T1 hE_ge)
      -- B(tBE_max) = E* − ε* (purely arithmetic).
      have hB_tBE : (traj T0).E - th.eps * (tBE_max - T0) = th.Estar - th.eps := by
        rw [htBE_def]
        have hne : th.eps ≠ 0 := ne_of_gt th.heps
        field_simp
        ring
      -- Specialise the comparison at tBE_max.
      have hE_tBE_le : (traj tBE_max).E ≤ th.Estar - th.eps := by
        have := happly tBE_max ⟨le_of_lt hT0_lt_tBE, le_refl _⟩
        rw [hB_tBE] at this; exact this
      -- But hneg(tBE_max) gives E(tBE_max) > E*; contradicts ε* > 0.
      have hE_tBE_gt : (traj tBE_max).E > th.Estar :=
        hneg tBE_max ⟨hT0_lt_tBE, le_refl _⟩
      linarith [th.heps]
    -- Extract the witness.
    obtain ⟨t_w, ⟨hT0_lt_tw, _⟩, hE_le⟩ := hexists
    exact ⟨t_w, lt_trans hT0_gt_tA hT0_lt_tw, hE_le⟩

/-  ────────────────────────────────────────────────────────────────────────────
    HELPER LEMMA  —  I(t) → +∞ from monotonicity of Σ and non-degeneracy.

    This closes the "hIdiv" gap in the main theorem `finite_lifespan`:
    instead of taking `Tendsto I atTop atTop` as an external hypothesis, the
    divergence of the integral of Σ follows from
       (i)  the monotonicity of Σ (proved as Lemma A.1 / `damage_nondecreasing`)
       (ii) the non-degeneracy condition Σ(tA) > 0
       (iii) the analytical relation I'(t) = Σ(t), I(tA) = 0.

    Mathematical argument:
      Let J(t) := I(t) − Σ(tA)·(t − tA).
      J'(t) = Σ(t) − Σ(tA) ≥ 0 for t ≥ tA  (by monotonicity).
      So J is non-decreasing on [tA, ∞), J(tA) = 0, hence J(t) ≥ 0,
      i.e. I(t) ≥ Σ(tA)·(t − tA) → +∞  since Σ(tA) > 0.
    ──────────────────────────────────────────────────────────────────────────-/
lemma I_diverges_of_Sigma_mono_nondeg
    (sigF : ℝ → ℝ) (I : ℝ → ℝ) (tA : ℝ)
    (hsigmono   : Monotone sigF)
    (hI_deriv : ∀ t : ℝ, HasDerivAt I (sigF t) t)
    (hI_tA    : I tA = 0)
    (hND      : 0 < sigF tA) :
    Tendsto I atTop atTop := by
  -- Auxiliary J(t) := I(t) − Σ(tA)·(t − tA).
  -- J'(t) = Σ(t) − Σ(tA), which is ≥ 0 on [tA, ∞) by monotonicity.
  have hJ_deriv : ∀ t : ℝ,
      HasDerivAt (fun s : ℝ => I s - sigF tA * (s - tA)) (sigF t - sigF tA) t := by
    intro t
    have h1 : HasDerivAt I (sigF t) t := hI_deriv t
    have h2 : HasDerivAt (fun s : ℝ => s - tA) 1 t := (hasDerivAt_id t).sub_const tA
    have h3 : HasDerivAt (fun s : ℝ => sigF tA * (s - tA)) (sigF tA) t := by
      simpa using h2.const_mul (sigF tA)
    exact h1.sub h3
  -- J is differentiable everywhere → continuous everywhere.
  have hJ_diff : Differentiable ℝ (fun s : ℝ => I s - sigF tA * (s - tA)) :=
    fun t => (hJ_deriv t).differentiableAt
  -- J is non-decreasing on [tA, ∞).
  have hJ_mono_on :
      MonotoneOn (fun s : ℝ => I s - sigF tA * (s - tA)) (Set.Ici tA) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici tA)
      hJ_diff.continuous.continuousOn hJ_diff.differentiableOn
    intro x hx
    rw [interior_Ici] at hx  -- hx : x ∈ Set.Ioi tA, i.e. tA < x
    rw [(hJ_deriv x).deriv]
    have : sigF tA ≤ sigF x := hsigmono (le_of_lt hx)
    linarith
  -- For t ≥ tA: J(t) ≥ J(tA) = 0, so I(t) ≥ Σ(tA)·(t − tA).
  have h_bound : ∀ t : ℝ, tA ≤ t → sigF tA * (t - tA) ≤ I t := by
    intro t ht
    have hJ_ge : (I tA - sigF tA * (tA - tA)) ≤ (I t - sigF tA * (t - tA)) :=
      hJ_mono_on (Set.left_mem_Ici) ht ht
    have : I tA - sigF tA * (tA - tA) = 0 := by rw [hI_tA]; ring
    linarith
  -- Linear bound diverges: Σ(tA)·(t − tA) → +∞.
  have h_linear : Tendsto (fun t : ℝ => sigF tA * (t - tA)) atTop atTop := by
    have h_sub : Tendsto (fun t : ℝ => t - tA) atTop atTop := by
      rw [Filter.tendsto_atTop_atTop]
      intro b
      exact ⟨b + tA, fun a ha => by linarith⟩
    exact Tendsto.const_mul_atTop hND h_sub
  -- Conclude I → +∞ by eventual monotonic comparison.
  apply tendsto_atTop_mono' atTop _ h_linear
  filter_upwards [Filter.eventually_ge_atTop tA] with t ht
  exact h_bound t ht

/-  ────────────────────────────────────────────────────────────────────────────
    HELPER LEMMA  —  I(t) → +∞ under the non-degeneracy condition (N):
                     Σ(t₁) > 0 at SOME moment t₁.

    Monotonicity leaves exactly two possibilities: Σ ≡ 0, or Σ(t₁) > 0 at some
    t₁ — and then the integral diverges.  Apply the previous lemma to the
    shifted antiderivative I(t) − I(t₁), which vanishes at t₁, and add back
    the constant I(t₁).  No condition at the birth moment t_A is needed.
    ──────────────────────────────────────────────────────────────────────────-/
lemma I_diverges_of_Sigma_mono_pos_at
    (sigF : ℝ → ℝ) (I : ℝ → ℝ) (t1 : ℝ)
    (hsigmono : Monotone sigF)
    (hI_deriv : ∀ t : ℝ, HasDerivAt I (sigF t) t)
    (hND      : 0 < sigF t1) :
    Tendsto I atTop atTop := by
  have hshift : Tendsto (fun t => I t - I t1) atTop atTop :=
    I_diverges_of_Sigma_mono_nondeg sigF (fun t => I t - I t1) t1 hsigmono
      (fun t => (hI_deriv t).sub_const (I t1)) (by simp) hND
  have hsum : Tendsto (fun t => (I t - I t1) + I t1) atTop atTop :=
    hshift.atTop_add (tendsto_const_nhds (x := I t1))
  exact hsum.congr (fun t => by ring)

/-  ────────────────────────────────────────────────────────────────────────────
    LEMMA A.4  —  ∃ finite t_BR: R(t_BR) ≤ R*.   [R-CHANNEL DEATH]

    A SEPARATE COROLLARY, NOT needed for the main theorem.  It holds once Σ(t)
    exceeds, at some moment, the critical level
        Σ_crit = [ρ(1−R*) − ν₀R*] / [νR* + ρ(1−R*)],
    the root of ρ(1−Σ)(1−R*) = (ν₀+νΣ)R*.  That condition is EXPLICIT in the
    parameters and is NOT a consequence of axioms A1–A6 (monograph §9.3,
    FAQ 1.3, 3.10): monotonicity gives only the existence of a limit
    Σ_∞ ∈ [Σ₀, +∞], not its position relative to Σ_crit.  The regime Σ → 1 is
    the special case s0 ↑ 1.

    Stated for the full equation (4.1): restoration ρ·g(E)·max(0, 1−Σ)·(1−R)
    with any g(E) ≤ 1 (passed as the function of time gfun; in the monograph
    g(E) ∈ [0, 1]), and with right derivatives.  Σ is NOT assumed ≤ 1; only R ≤ 1 is used (the face
    R = 1 is invariant, monograph axioms A5, A6).

    Proof (port of monograph Appendix A.4).  Encode "Σ exceeds Σ_crit" as a
    level s0 ≤ 1 with margin ρ(1−s0)(1−R*) < (ν₀+ν·s0)R* and an eventual lower
    bound Σ(t) ≥ s0 for t ≥ Tₛ (for monotone Σ this is Σ(Tₛ) ≥ s0).  Put
    c := (ν₀+ν·s0)R* − ρ(1−s0)(1−R*) > 0.  For t ≥ Tₛ and R* ≤ R ≤ 1,
        g·max(0, 1−Σ)·(1−R) ≤ (1−s0)(1−R*),
    hence
        dR/dt ≤ −ν₀R* − ν·s0·R* + ρ(1−s0)(1−R*)  =  −c  <  0.
    A linear-comparison argument (identical in shape to `finite_death_E`) then
    yields a finite crossing time t_BR with R(t_BR) ≤ R*.
    ──────────────────────────────────────────────────────────────────────────-/
lemma finite_death_R
    (p : VBLSParams) (th : VBLSThresholds)
    (traj : ℝ → VBLSState) (tA : ℝ)
    -- value of the restoration-potential function g(E(t)) ∈ [0, 1];
    -- only the upper bound g ≤ 1 is needed
    (gfun : ℝ → ℝ) (hg1 : ∀ t : ℝ, gfun t ≤ 1)
    -- invariance of the face R = 1 (no assumption on Σ ≤ 1)
    (hRle1   : ∀ t : ℝ, (traj t).rep ≤ 1)
    (hRCont  : Continuous (fun t => (traj t).rep))
    -- right derivative (piecewise-continuous fluxes, axiom A2)
    (hRDeriv : ∀ t : ℝ,
        HasDerivWithinAt (fun t => (traj t).rep)
                   (rhsRep p (gfun t) (traj t).damage (traj t).rep) (Set.Ici t) t)
    -- Σ-threshold regime.  s0 is a level above the critical value Σ_crit: the
    -- margin inequality below is exactly s0 > Σ_crit.  NOT derivable from A1–A6.
    (s0 : ℝ) (hs0_nonneg : 0 ≤ s0) (hs0_le1 : s0 ≤ 1)
    (hmargin : p.rho * (1 - s0) * (1 - th.Rstar) < (p.v0 + p.v * s0) * th.Rstar)
    (hSigLB : ∃ Ts : ℝ, ∀ t : ℝ, t ≥ Ts → s0 ≤ (traj t).damage) :
    ∃ tBR : ℝ, tA < tBR ∧ (traj tBR).rep ≤ th.Rstar := by
  -- Decay margin c := (ν₀ + ν·s0)·R* − ρ·(1−s0)·(1−R*) > 0  (positivity is s0 > Σ_crit).
  set c : ℝ := (p.v0 + p.v * s0) * th.Rstar - p.rho * (1 - s0) * (1 - th.Rstar) with hc_def
  have hc_pos : 0 < c := by rw [hc_def]; linarith [hmargin]
  -- Eventual lower bound Σ(t) ≥ s0 for t ≥ Tₛ.
  obtain ⟨Ts, hTs⟩ := hSigLB
  -- KEY ESTIMATE: while t ≥ Tₛ and R(t) ≥ R*, the right derivative is ≤ −c.
  have hdneg : ∀ t : ℝ, t ≥ Ts → (traj t).rep ≥ th.Rstar →
      rhsRep p (gfun t) (traj t).damage (traj t).rep ≤ -c := by
    intro t ht hR
    rw [rhsRep]
    have hsig_ge : s0 ≤ (traj t).damage := hTs t ht
    have hR_le1 : (traj t).rep ≤ 1 := hRle1 t
    -- wear terms
    have hb1 : p.v0 * th.Rstar ≤ p.v0 * (traj t).rep :=
      mul_le_mul_of_nonneg_left hR p.hv0.le
    have hsigR : s0 * th.Rstar ≤ (traj t).damage * (traj t).rep :=
      mul_le_mul hsig_ge hR th.hRstar.le (le_trans hs0_nonneg hsig_ge)
    have hb2 : p.v * (s0 * th.Rstar) ≤ p.v * ((traj t).damage * (traj t).rep) :=
      mul_le_mul_of_nonneg_left hsigR p.hv.le
    -- restoration term: g·max(0, 1−Σ)·(1−R) ≤ (1−s0)·(1−R*)
    have hm_nn : 0 ≤ max 0 (1 - (traj t).damage) := le_max_left _ _
    have hm_le : max 0 (1 - (traj t).damage) ≤ 1 - s0 :=
      max_le (by linarith) (by linarith)
    have h1R_nn : 0 ≤ 1 - (traj t).rep := by linarith
    have h1R_le : 1 - (traj t).rep ≤ 1 - th.Rstar := by linarith
    have hgm : gfun t * max 0 (1 - (traj t).damage) ≤ 1 * (1 - s0) :=
      mul_le_mul (hg1 t) hm_le hm_nn (by norm_num)
    have hprod : gfun t * max 0 (1 - (traj t).damage) * (1 - (traj t).rep)
                   ≤ (1 - s0) * (1 - th.Rstar) := by
      have h := mul_le_mul hgm h1R_le h1R_nn (by linarith)
      simpa using h
    have hb3 : p.rho * (gfun t * max 0 (1 - (traj t).damage) * (1 - (traj t).rep))
                 ≤ p.rho * ((1 - s0) * (1 - th.Rstar)) :=
      mul_le_mul_of_nonneg_left hprod p.hrho.le
    -- combine: dR/dt = −ν₀R − νΣR + ρ·g·max(0,1−Σ)·(1−R) ≤ −c
    nlinarith [hb1, hb2, hb3, hc_def]
  -- Linear-comparison existence (mirror of finite_death_E).
  set T0 := max (tA + 1) Ts with hT0_def
  have hT0_gt_tA : tA < T0 := lt_of_lt_of_le (by linarith) (le_max_left _ _)
  have hT0_ge_Ts : Ts ≤ T0 := le_max_right _ _
  by_cases hRT0 : (traj T0).rep ≤ th.Rstar
  · exact ⟨T0, hT0_gt_tA, hRT0⟩
  · push_neg at hRT0
    set tBR_max := T0 + ((traj T0).rep - th.Rstar) / c + 1 with htBR_def
    have hgap_nn : 0 ≤ ((traj T0).rep - th.Rstar) / c :=
      div_nonneg (by linarith) (le_of_lt hc_pos)
    have hT0_lt_tBR : T0 < tBR_max := by rw [htBR_def]; linarith
    have hB_deriv : ∀ x : ℝ,
        HasDerivAt (fun t : ℝ => (traj T0).rep - c * (t - T0)) (-c) x := by
      intro x
      have h1 : HasDerivAt (fun t : ℝ => t - T0) 1 x := (hasDerivAt_id x).sub_const T0
      have h2 : HasDerivAt (fun t : ℝ => c * (t - T0)) c x := by
        simpa using h1.const_mul c
      simpa using (hasDerivAt_const x (traj T0).rep).sub h2
    have hB_cont : Continuous (fun t : ℝ => (traj T0).rep - c * (t - T0)) :=
      continuous_const.sub (continuous_const.mul (continuous_id.sub continuous_const))
    have hexists : ∃ t : ℝ, t ∈ Set.Ioc T0 tBR_max ∧ (traj t).rep ≤ th.Rstar := by
      by_contra hneg
      push_neg at hneg
      have happly : ∀ x ∈ Set.Icc T0 tBR_max,
          (traj x).rep ≤ (traj T0).rep - c * (x - T0) :=
        image_le_of_deriv_right_le_deriv_boundary
          hRCont.continuousOn
          (fun x _ => hRDeriv x)
          (by show (traj T0).rep ≤ (traj T0).rep - c * (T0 - T0); ring_nf; exact le_refl _)
          hB_cont.continuousOn
          (fun x _ => (hB_deriv x).hasDerivWithinAt)
          (by
            intro x hx
            have hxT0 : T0 ≤ x := hx.1
            have hxLt : x < tBR_max := hx.2
            have hx_ge_Ts : Ts ≤ x := le_trans hT0_ge_Ts hxT0
            have hR_ge : (traj x).rep ≥ th.Rstar := by
              rcases lt_or_eq_of_le hxT0 with hxgt | hxeq
              · exact le_of_lt (hneg x ⟨hxgt, le_of_lt hxLt⟩)
              · rw [← hxeq]; exact le_of_lt hRT0
            exact hdneg x hx_ge_Ts hR_ge)
      have hB_tBR : (traj T0).rep - c * (tBR_max - T0) = th.Rstar - c := by
        rw [htBR_def]
        have hne : c ≠ 0 := ne_of_gt hc_pos
        field_simp
        ring
      have hR_tBR_le : (traj tBR_max).rep ≤ th.Rstar - c := by
        have := happly tBR_max ⟨le_of_lt hT0_lt_tBR, le_refl _⟩
        rw [hB_tBR] at this; exact this
      have hR_tBR_gt : (traj tBR_max).rep > th.Rstar :=
        hneg tBR_max ⟨hT0_lt_tBR, le_refl _⟩
      linarith [hc_pos]
    obtain ⟨t_w, ⟨hT0_lt_tw, _⟩, hR_le⟩ := hexists
    exact ⟨t_w, lt_trans hT0_gt_tA hT0_lt_tw, hR_le⟩

-- ══════════════════════════════════════════════════════════════════════════════
-- §6.  MAIN THEOREM
-- ══════════════════════════════════════════════════════════════════════════════

/--
  ══════════════════════════════════════════════════════════════════════════════
  THEOREM  (Finite Lifespan of Living Systems)

  Hypotheses: system (S) satisfies axioms A1–A5 (A6 not needed); the solution trajectory is
  given; η has the analytical form η(t) = η₀·exp(-α·I(t)) where I(t) = ∫Σ.

  Conclusion: ∃ finite t_B > t_A such that the system is not viable at t_B.

  Proof chain:
    1. hMemSol + hIdiv  →  eta_tends_zero  →  η(t) → 0
    2. η(t) → 0         →  finite_death_E  →  ∃ t_BE: E(t_BE) ≤ E*
    3. E(t_BE) ≤ E*     →  ¬ isViable     □

  Abstract form: the divergence hIdiv — "I(t) → +∞" — is taken as a hypothesis
  here; `finite_lifespan_from_axioms` below derives it from the monotonicity
  of Σ (Lemma A.1) and the non-degeneracy condition (N).
  ══════════════════════════════════════════════════════════════════════════════
-/
theorem finite_lifespan
    (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (traj : ℝ → VBLSState)
    (PhiIn : ℝ → ℝ) (tA : ℝ)
    (hA1    : AxiomA1 (traj tA) th)
    (hA2    : AxiomA2 PhiIn th)
    -- AxiomA6 removed: Σ ≤ 1 (and the whole invariant box) is not needed for the
    -- E-channel — see `finite_death_E` and the Axiom A6 caveat above.
    (hECont : Continuous (fun t => (traj t).E))
    (eta0   : ℝ) (heta0 : 0 < eta0)
    (I      : ℝ → ℝ)
    -- η has the analytical form from solving the memory ODE (5.1)
    (hMemSol : ∀ t : ℝ, t ≥ tA →
        (traj t).eta = eta0 * Real.exp (-p.a * I t))
    -- I(t) = ∫_{tA}^t Σ dτ → +∞  (derived in `finite_lifespan_from_axioms`).
    (hIdiv  : Tendsto I atTop atTop)
    -- right derivative (piecewise-continuous fluxes, axiom A2)
    (hEDeriv : ∀ t : ℝ,
        HasDerivWithinAt (fun t => (traj t).E)
                   (rhsE p th fd PhiIn (traj t).eta (traj t).E t) (Set.Ici t) t) :
    ∃ tB : ℝ, tA < tB ∧ ¬ isViable (traj tB) th := by
  -- Step 1: deduce η(t) → 0 from the analytical form and hIdiv
  have hEtaConv : Tendsto (fun t => (traj t).eta) atTop (nhds 0) :=
    (eta_tends_zero p eta0 heta0 I hIdiv).congr' (by
      -- Match η(t) to the formula η₀·exp(-α·I(t)) for t ≥ tA
      filter_upwards [Filter.eventually_ge_atTop tA] with t ht
      exact (hMemSol t ht).symm)
  -- Step 2: apply Lemma A.3 (E-channel) to get t_BE with E(t_BE) ≤ E*
  obtain ⟨tBE, hpos, hEle⟩ :=
    finite_death_E p th fd traj PhiIn tA hA1 hA2 hECont hEDeriv hEtaConv
  -- Step 3: at t_BE the system is not viable (E ≤ E* violates E > E*)
  exact ⟨tBE, hpos, fun hv => by linarith [hv.1]⟩

/--
  ══════════════════════════════════════════════════════════════════════════════
  THEOREM  (Finite Lifespan from axioms — central closed form)

  Same conclusion as `finite_lifespan`, but with `hIdiv` derived internally
  from monotonicity of damage (Lemma A.1) + the non-degeneracy condition (N):
  Σ(t₁) > 0 at SOME moment t₁.  The result follows from A1–A5 + (N) ALONE — no
  external assumption on the divergence of the memory integral, and no
  condition at the birth moment t_A (a system that starts with Σ₀ = 0 and
  export exceeding production, as Barrier II requires, is covered as soon as
  damage leaves a trace).  Since Σ ≥ 0 is non-decreasing, (N) fails only for
  Σ ≡ 0: for every trajectory, either damage never leaves a trace or t_B < ∞.

  Hypotheses replacing `hIdiv`:
    · hI_deriv     : I'(t) = Σ(t) for all t  (I is an antiderivative of Σ)
    · hSigma_mono  : Σ is monotone non-decreasing  (output of `damage_nondecreasing`)
    · hND          : ∃ t₁, Σ(t₁) > 0        (non-degeneracy (N))
  ══════════════════════════════════════════════════════════════════════════════
-/
theorem finite_lifespan_from_axioms
    (p : VBLSParams) (th : VBLSThresholds) (fd : DissipFn th)
    (traj : ℝ → VBLSState)
    (PhiIn : ℝ → ℝ) (tA : ℝ)
    (hA1    : AxiomA1 (traj tA) th)
    (hA2    : AxiomA2 PhiIn th)
    -- AxiomA6 removed: Σ ≤ 1 (and the whole invariant box) is not needed for the
    -- E-channel — see `finite_death_E` and the Axiom A6 caveat above.
    (hECont : Continuous (fun t => (traj t).E))
    (eta0   : ℝ) (heta0 : 0 < eta0)
    (I      : ℝ → ℝ)
    -- Analytical form of η coming from the memory ODE.
    (hMemSol : ∀ t : ℝ, t ≥ tA →
        (traj t).eta = eta0 * Real.exp (-p.a * I t))
    -- I is an antiderivative of Σ (for instance I(t) = ∫_{tA}^t Σ dτ).
    (hI_deriv : ∀ t : ℝ, HasDerivAt I ((traj t).damage) t)
    -- Damage is monotone non-decreasing (this is `damage_nondecreasing`).
    (hSigma_mono : Monotone (fun t => (traj t).damage))
    -- Non-degeneracy (N): damage has left a trace at some moment t₁.
    (hND : ∃ t1 : ℝ, 0 < (traj t1).damage)
    -- right derivative (piecewise-continuous fluxes, axiom A2)
    (hEDeriv : ∀ t : ℝ,
        HasDerivWithinAt (fun t => (traj t).E)
                   (rhsE p th fd PhiIn (traj t).eta (traj t).E t) (Set.Ici t) t) :
    ∃ tB : ℝ, tA < tB ∧ ¬ isViable (traj tB) th := by
  -- Derive `hIdiv` from monotonicity of Σ + non-degeneracy (N).
  obtain ⟨t1, ht1⟩ := hND
  have hIdiv : Tendsto I atTop atTop :=
    I_diverges_of_Sigma_mono_pos_at
      (fun t => (traj t).damage) I t1 hSigma_mono hI_deriv ht1
  -- Apply the abstract `finite_lifespan`.
  exact finite_lifespan p th fd traj PhiIn tA hA1 hA2 hECont
        eta0 heta0 I hMemSol hIdiv hEDeriv

-- ══════════════════════════════════════════════════════════════════════════════
-- §7.  TYPE CHECKS  (verify that all declarations have the expected signatures)
-- ══════════════════════════════════════════════════════════════════════════════

#check @finite_lifespan_from_axioms  -- ★ MAIN THEOREM: closed form (A1–A5 + (N))
#check @finite_lifespan              -- abstract form (with hIdiv as hypothesis)
#check @finite_death_E               -- Lemma A.3 (E-channel) — no sorry
#check @finite_death_R               -- Lemma A.4 (R-channel), full eq. (4.1), Σ > Σ_crit
#check @eta_tends_zero               -- Lemma A.2
#check @damage_nondecreasing         -- Lemma A.1
#check @finite_T1                    -- Lemma A.2a
#check @mono_of_rightDeriv_nonneg    -- helper: monotonicity from right derivatives
#check @I_diverges_of_Sigma_mono_nondeg  -- helper: divergence from Σ(tA) > 0
#check @I_diverges_of_Sigma_mono_pos_at  -- helper: divergence from (N), Σ(t₁) > 0

-- Axiom audit: every result must depend ONLY on Lean's standard axioms
-- (propext, Classical.choice, Quot.sound) and NOT on `sorryAx`.  This is the
-- machine-checked guarantee that BOTH channels are complete (no hidden sorry)
-- and that the E-channel does not secretly rely on AxiomA6 / the bound Σ ≤ 1.
#print axioms finite_lifespan_from_axioms  -- E-channel (main theorem), A6-free
#print axioms finite_death_R               -- R-channel (Lemma A.4), under Σ > Σ_crit

/-
  ══════════════════════════════════════════════════════════════════════════════
  STATUS SUMMARY
  ══════════════════════════════════════════════════════════════════════════════

  Closed (no sorry):
    ✓ mono_of_rightDeriv_nonneg         — helper (right derivatives)
    ✓ damage_nondecreasing              — Lemma A.1
    ✓ eta_tends_zero                    — Lemma A.2
    ✓ finite_T1                         — Lemma A.2a
    ✓ I_diverges_of_Sigma_mono_nondeg   — derives Tendsto I atTop atTop from Σ(tA) > 0
    ✓ I_diverges_of_Sigma_mono_pos_at   — the same from (N): Σ(t₁) > 0 at some t₁
    ✓ finite_death_E                    — Lemma A.3 (E-channel)
    ✓ finite_lifespan                   — main theorem (abstract form)
    ✓ finite_lifespan_from_axioms       — main theorem (closed form: A1–A5 +
                                          (N) → finite lifespan)
    ✓ finite_death_R                    — Lemma A.4 (R-channel), full equation (4.1),
                                          under Σ > Σ_crit

  Open (sorry):
    — none.  The file is fully sorry-free; both `#print axioms` audits above
      return only [propext, Classical.choice, Quot.sound] (no sorryAx).
    Note: Σ > Σ_crit (the margin ρ(1−s0)(1−R*) < (ν₀+ν·s0)R* with eventual
    Σ ≥ s0) for the R-channel is an explicit HYPOTHESIS of finite_death_R,
    not an axiom and not a consequence of A1–A6; Σ → 1 is its special case.

  ══════════════════════════════════════════════════════════════════════════════
-/
