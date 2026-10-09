"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================
  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0.
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================

VBLS — Sturm bound alpha_guar for the auxiliary equation of Appendix G.4.

THE AUXILIARY EQUATION
----------------------
With W := d/d(alpha)[alpha * int Sigma], the exact equation in the adiabatically reduced
system is  Ẅ + k(t) W = Σ̇ − α Φ_cap u_R,  k(t) = α σ₀ Φ_in(t) η_abs(t).  The last term is
the response of the export R·Φ_cap to α; the adiabatic reduction does not remove it.
Dropping it gives the auxiliary equation (a forced oscillator)
      Ẅ + k(t) W = Σ̇,   W(t_A)=0, Ẇ(t_A)=Σ₀,
for which this script evaluates the sufficient condition below. It is an argument for the
rising branch, not a proof of its monotonicity for (S) (Appendix G.4, G.5).

STURM COMPARISON (no slow-variation assumption on k)
----------------------------------------------------
k(t) ≤ k̄ := α σ₀ Φ_max η₀, with Φ_max = sup_t Φ_in(t), η₀ = η_abs(t_A). By the Sturm comparison
theorem, the first zero after s of a solution of Ẅ + k W = 0 with W(s)=0, Ẇ(s)=1 comes no
earlier than s + π/√k̄. On an interval shorter than π/√k̄ the equation is disconjugate, so
φ(t) > 0 and the causal Green's function H(t,s) > 0 for t > s. With Σ̇ ≥ 0,
      W(t) = Σ₀ φ(t) + ∫_{t_A}^t H(t,s) Σ̇(s) ds ≥ 0   (> 0 if Σ₀ > 0)   on (t_A, t_B],
provided the life interval is shorter than the first half-period:
      t_B − t_A < π/√k̄   ⟺   α < α_guar := π² / [σ₀ Φ_max η₀ (t_B − t_A)²].

SELF-CONSISTENCY
----------------
t_B on the right is the very lifespan being bounded, so α_guar is a fixed-point condition,
not a closed form. The script evaluates, for each α, the actual t_B(α) and
      q(α) := √(α σ₀ Φ_max η₀) · (t_B(α) − t_A),
and reports the largest α with q(α) < π. Substituting a constant t_B from 73–93 yr instead
gives thresholds from 0.12 down to 0.07. The same quantity with the E-branch time t_B^E(α)
(trajectory continued past death) shows how far the condition covers the decreasing branch.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            rho=0.002, R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]      # E0, Σ0, η0, R0
E_RANGE = IC[0] - BASE['E_star']
ETA0    = IC[2]
T_A     = 0.0
T_MAX   = 400.0


def Phi_in(t):  return (1.0 - np.exp(-t/10.0)) * np.exp(-max(0.0, t-60.0)/30.0)
def Phi_cap(t): return 0.003


def rhs(t, y, alpha):
    E, S, eta, R = y
    E = max(E, 0.0); S = np.clip(S, 0, 1); eta = np.clip(eta, 0, 1); R = np.clip(R, 0, 1)
    F = BASE['mu']*E**2 + BASE['lam']*E
    g = max(0.0, min(1.0, (E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t) - F,
            max(0.0, BASE['sigma_0']*F - R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R - BASE['nu']*S*R + BASE['rho']*g*(1-S)*(1-R)]


def tB(alpha):
    def eE(t, y): return y[0]-BASE['E_star']
    def eR(t, y): return y[3]-BASE['R_star']
    eE.terminal = eR.terminal = True; eE.direction = eR.direction = -1
    s = solve_ivp(lambda t, y: rhs(t, y, alpha), [0, T_MAX], IC, method='RK45',
                  events=[eE, eR], rtol=1e-8, atol=1e-10, max_step=0.2)
    tE = s.t_events[0][0] if s.t_events[0].size else np.inf
    tR = s.t_events[1][0] if s.t_events[1].size else np.inf
    return float(min(tE, tR)) if (tE < np.inf or tR < np.inf) else T_MAX


def tE_branch(alpha):
    """E-branch time: first crossing of E* after 2 yr on the trajectory continued past death."""
    s = solve_ivp(lambda t, y: rhs(t, y, alpha), [0, T_MAX], IC, method='RK45',
                  rtol=1e-8, atol=1e-10, max_step=0.2)
    m = (s.t >= 2.0) & (s.y[0] <= BASE['E_star'])
    return float(s.t[m][0]) if m.any() else np.inf


# Φ_max = sup_t Φ_in(t)  (peak of the intake profile, near t=60)
tt = np.linspace(0.0, 120.0, 20001)
PHI_MAX = float(max(Phi_in(t) for t in tt))

# α_guar as a closed expression, evaluated at a reference t_B (for display only)
def alpha_guar_closed(t_B):
    return np.pi**2 / (BASE['sigma_0'] * PHI_MAX * ETA0 * (t_B - T_A)**2)

# Sturm quantity q(α) = √(α σ₀ Φ_max η₀) · (t_B(α) − t_A); condition q < π
def q_of(alpha):
    return np.sqrt(alpha * BASE['sigma_0'] * PHI_MAX * ETA0) * (tB(alpha) - T_A)

print("=" * 70)
print("  VBLS — Sturm bound α_guar for the auxiliary equation (Appendix G.4)")
print("=" * 70)
print(f"  Φ_max = sup_t Φ_in(t) = {PHI_MAX:.4f}   (at t≈60)")
print(f"  σ₀ = {BASE['sigma_0']}, η₀ = {ETA0}, t_A = {T_A}")
print("-" * 70)

# closed-form α_guar at representative finite lifespans
for t_B_ref in (73.0, 77.0, 93.0):
    print(f"  closed-form α_guar at t_B={t_B_ref:>5.0f} yr : {alpha_guar_closed(t_B_ref):.4f}")
print("-" * 70)

# self-consistent α_guar: largest α with q(α) < π
A = np.linspace(0.005, 0.20, 400)
Q = np.array([q_of(a) for a in A])
below = A[Q < np.pi]
alpha_guar_sc = float(below.max()) if below.size else np.nan
print(f"  self-consistent α_guar (largest α with q(α)<π) = {alpha_guar_sc:.4f}")
print(f"    q(α) at that point = {q_of(alpha_guar_sc):.4f}  (π = {np.pi:.4f})")
print("-" * 70)

# coverage of the decreasing E-branch: q_E(α) with t_B^E(α) on the continued trajectory
A_E = np.arange(0.005, 0.0801, 0.005)
Q_E = np.array([np.sqrt(a * BASE['sigma_0'] * PHI_MAX * ETA0) * (tE_branch(a) - T_A) for a in A_E])
cov = A_E[Q_E < np.pi]
print(f"  E-branch covered (q_E(α) < π) up to α ≈ {cov.max():.3f}  (grid step 0.005);"
      f"  q_E(0.030) = {np.interp(0.030, A_E, Q_E):.3f}")
print("-" * 70)

# where the numerical optimum sits, for comparison
f = lambda a: -tB(a)
r = minimize_scalar(f, bracket=(0.125, 0.135, 0.145), method='brent', options={'xtol': 1e-5})
print(f"  numerical optimum α* = {r.x:.4f}  (t_B_max = {-r.fun:.2f} yr)")
print(f"  calibrated α = {0.030}  ->  q = {q_of(0.030):.3f} < π : "
      f"{'INSIDE bound' if q_of(0.030) < np.pi else 'OUTSIDE'}")
print(f"    margin α_guar/α_cal = {alpha_guar_sc/0.030:.1f}×")
print("=" * 70)
print("  Summary: α_guar is a sufficient condition for the auxiliary equation")
print("  (export response −αΦ_cap·u_R dropped), self-consistent in t_B; the calibrated")
print("  α=0.030 lies inside. It covers the rising R-branch, not the decreasing E-branch")
print("  near α*; beyond α_guar the signs are checked numerically (monotonicity_check.py).")
print("=" * 70)
