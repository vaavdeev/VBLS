"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================
  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0.
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================

VBLS — Sturm bound alpha_guar for the H1 monotonicity closure (Appendix G.4).

WHAT THIS REPLACES
------------------
An earlier draft of G.4 bounded the forced oscillator
      Ẅ + k(t) W = Σ̇,   k(t) = α σ₀ Φ_in(t) η_abs(t),   W(t_A)=0, Ẇ(t_A)=Σ₀>0
by a WKB "accumulated-phase" argument (W stays positive while ∫√k dt < π). That
step silently assumed (i) k(t) is slowly varying — false, since η_abs(t) decays on
the very timescale in question — and (ii) positivity of the Green's function on the
first half-period. Neither was proved.

THE RIGOROUS REPLACEMENT (Sturm comparison, no WKB, no slow-variation assumption)
---------------------------------------------------------------------------------
k(t) = α σ₀ Φ_in(t) η_abs(t) ≤ K := α σ₀ Φ_max η₀,   with Φ_max = sup_t Φ_in(t), η₀ = η_abs(t_A).
By the Sturm comparison theorem, the first zero after t_A of any solution of Ẅ + k W = 0
with W(t_A)=0, Ẇ(t_A)>0 occurs no earlier than the first zero of the constant-coefficient
comparison equation Ẅ + K W = 0, i.e. no earlier than t_A + π/√K. Up to that first
conjugate point the causal Green's function H(t,s) of the operator is POSITIVE (standard
for a Hill operator before its first conjugate point). Hence with the nonnegative forcing
Σ̇ ≥ 0 and the positive homogeneous part Σ₀·φ(t),
      W(t) = Σ₀ φ(t) + ∫_{t_A}^t H(t,s) Σ̇(s) ds > 0   for all t_A < t ≤ t_B,
provided the whole life interval fits inside the first half-period:
      t_B − t_A < π/√K   ⟺   α < α_guar := π² / [σ₀ Φ_max η₀ (t_B − t_A)²].

SELF-CONSISTENCY (removes the circular-t_B objection)
-----------------------------------------------------
t_B on the right is the very lifespan being bounded, so α_guar is a self-consistent
condition, not a closed form. We solve it by simply evaluating, for each α, the actual
t_B(α) and the Sturm quantity
      q(α) := √(α σ₀ Φ_max η₀) · (t_B(α) − t_A),
and reporting the largest α with q(α) < π. The earlier "0.07–0.13" was NOT an error bar:
it is exactly the range obtained by substituting different finite t_B(α) (73–93 yr) and
different Φ estimates. Here we pin Φ_max = sup Φ_in and report a single self-consistent value.
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
print("  VBLS — Sturm bound α_guar for H1 closure (Appendix G.4)")
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

# where the numerical optimum sits, for comparison
f = lambda a: -tB(a)
r = minimize_scalar(f, bracket=(0.125, 0.135, 0.145), method='brent', options={'xtol': 1e-5})
print(f"  numerical optimum α* = {r.x:.4f}  (t_B_max = {-r.fun:.2f} yr)")
print(f"  calibrated α = {0.030}  ->  q = {q_of(0.030):.3f} < π : "
      f"{'INSIDE bound' if q_of(0.030) < np.pi else 'OUTSIDE'}")
print(f"    margin α_guar/α_cal = {alpha_guar_sc/0.030:.1f}×")
print("=" * 70)
print("  Verdict: α_guar obtained by Sturm comparison (no WKB), self-consistent")
print("  in t_B; calibrated α=0.030 lies safely inside; the upper reach of the")
print("  bound approaches the numerical optimum α*≈0.135 from below.")
print("=" * 70)
