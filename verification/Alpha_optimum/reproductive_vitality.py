"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================
  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0.
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================

VBLS — REPRODUCTIVE VITALITY vs LIFESPAN trade-off across alpha (§19.3.6).

Purpose
-------
Section 19.3.6 argues that natural selection does NOT drive alpha to the
lifespan optimum alpha*, because:

  (a) the lifespan gain from alpha_human -> alpha* is entirely POST-reproductive
      (the system clears the reproductive window t < 45 yr at every alpha in the
      admissible range, so the number of reproductive years is unchanged), and

  (b) raising alpha toward alpha* LOWERS reproductive vitality: both the
      integral of the integrity potential over the reproductive window,
      J = \int_0^45 E dt, and the terminal value E(45) decrease.

The numbers quoted in §19.3.6 were computed at the OLD alpha* ~ 0.10.
Appendix G characterises alpha* as the R->E channel crossover and gives
alpha* ~ 0.1346, t_B^max ~ 93.3 yr. This script recomputes (a) and (b) at the
corrected alpha*, using the SAME calibration as alpha_crossover.py.

Outputs (console):
  1. t_B at baseline alpha and at alpha*
  2. J = \int_0^45 E dt and E(45) at both, with relative drops
  3. min over alpha in [0.005, 0.18] of t_B  -> checks the "t_B > 45 always" claim
  4. monotonicity of J(alpha) and E(45; alpha) over the same range
"""

import numpy as np
from scipy.integrate import solve_ivp

# ============================================================================
# CANONICAL PARAMETERS — identical to alpha_crossover.py (Homo sapiens)
# ============================================================================
BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, alpha=0.030,
            nu_0=0.006, nu=0.060, rho=0.002, R_star=0.15, E_star=0.04)
IC      = [0.10, 0.01, 0.96, 0.99]      # E0, Σ0, η0, R0
E0      = IC[0]
E_RANGE = E0 - BASE['E_star']           # 0.06
T_MAX   = 400.0
T_REPRO = 45.0                          # end of the reproductive window


def Phi_in(t):
    return (1.0 - np.exp(-t / 10.0)) * np.exp(-max(0.0, t - 60.0) / 30.0)


def Phi_cap(t):
    return 0.003


def _rhs(t, y, alpha):
    E, S, eta, R = y
    E   = max(E,   0.0)
    S   = np.clip(S,   0.0, 1.0)
    eta = np.clip(eta, 0.0, 1.0)
    R   = np.clip(R,   0.0, 1.0)
    F   = BASE['mu'] * E**2 + BASE['lam'] * E
    g_E = max(0.0, min(1.0, (E - BASE['E_star']) / E_RANGE))
    return [
        eta * Phi_in(t) - F,
        max(0.0, BASE['sigma_0'] * F - R * Phi_cap(t)),
        -alpha * S * eta,
        -BASE['nu_0'] * R - BASE['nu'] * S * R + BASE['rho'] * g_E * (1.0 - S) * (1.0 - R),
    ]


def run_terminal(alpha):
    """Physical run: first threshold crossing wins. Returns (t_B, channel)."""
    def ev_E(t, y): return y[0] - BASE['E_star']
    def ev_R(t, y): return y[3] - BASE['R_star']
    ev_E.terminal = True;  ev_E.direction = -1
    ev_R.terminal = True;  ev_R.direction = -1
    sol = solve_ivp(lambda t, y: _rhs(t, y, alpha), [0, T_MAX], IC,
                    method='RK45', events=[ev_E, ev_R],
                    rtol=1e-8, atol=1e-10, max_step=0.2)
    tE = float(sol.t_events[0][0]) if sol.t_events[0].size else np.inf
    tR = float(sol.t_events[1][0]) if sol.t_events[1].size else np.inf
    if tE == np.inf and tR == np.inf:
        return T_MAX, 'none'
    return (tE, 'E') if tE < tR else (tR, 'R')


def repro_vitality(alpha, n_grid=4501):
    """
    Integrate over the reproductive window [0, T_REPRO] on a dense uniform grid.
    Returns (J, E_at_45) where J = \int_0^45 E dt  (units: E-units * yr).
    """
    t_eval = np.linspace(0.0, T_REPRO, n_grid)
    sol = solve_ivp(lambda t, y: _rhs(t, y, alpha), [0.0, T_REPRO], IC,
                    method='RK45', t_eval=t_eval,
                    rtol=1e-10, atol=1e-12, max_step=0.05)
    E = sol.y[0]
    J = float(np.trapezoid(E, sol.t)) if hasattr(np, 'trapezoid') else float(np.trapz(E, sol.t))
    return J, float(E[-1])


# ============================================================================
# 1. Locate alpha* on the same grid used by alpha_crossover.py
# ============================================================================
ALPHA = np.linspace(0.005, 0.30, 240)
tB_scan = np.array([run_terminal(a)[0] for a in ALPHA])
i_star = int(np.argmax(tB_scan))
ALPHA_STAR = float(ALPHA[i_star])
TB_STAR = float(tB_scan[i_star])

A_HUMAN = BASE['alpha']
tB_human, ch_human = run_terminal(A_HUMAN)
_, ch_star = run_terminal(ALPHA_STAR)

print("=" * 72)
print("  VBLS — reproductive vitality vs lifespan across alpha  (§19.3.6)")
print("=" * 72)
print(f"  alpha_human = {A_HUMAN:.4f}  ->  t_B = {tB_human:6.2f} yr   (channel {ch_human})")
print(f"  alpha*      = {ALPHA_STAR:.4f}  ->  t_B = {TB_STAR:6.2f} yr   (channel {ch_star})")
print(f"  lifespan gain               = {TB_STAR - tB_human:+.2f} yr")

# ============================================================================
# 2. Reproductive vitality at both alphas
# ============================================================================
J_h, E45_h = repro_vitality(A_HUMAN)
J_s, E45_s = repro_vitality(ALPHA_STAR)

dJ = (J_s - J_h) / J_h * 100.0
dE = (E45_s - E45_h) / E45_h * 100.0

print("-" * 72)
print(f"  J = int_0^45 E dt   at alpha_human = {J_h:.5f}")
print(f"                      at alpha*      = {J_s:.5f}   -> {dJ:+.2f} %")
print(f"  E(45)               at alpha_human = {E45_h:.5f}")
print(f"                      at alpha*      = {E45_s:.5f}   -> {dE:+.2f} %")

# ============================================================================
# 3. Does the system always clear the reproductive window on [0.005, 0.18]?
# ============================================================================
A_LO, A_HI = 0.005, 0.18
A_RANGE = np.linspace(A_LO, A_HI, 200)
tB_range = np.array([run_terminal(a)[0] for a in A_RANGE])
i_min = int(np.argmin(tB_range))

print("-" * 72)
print(f"  scan alpha in [{A_LO}, {A_HI}]:")
print(f"    min t_B = {tB_range[i_min]:.2f} yr  at alpha = {A_RANGE[i_min]:.4f}")
print(f"    t_B > {T_REPRO:.0f} yr for all alpha in range : "
      f"{'YES' if tB_range.min() > T_REPRO else 'NO'}")

# ============================================================================
# 4. Monotone decline of reproductive vitality in alpha?
# ============================================================================
A_V = np.linspace(A_LO, A_HI, 40)
JV, EV = [], []
for a in A_V:
    j, e = repro_vitality(a, n_grid=2251)
    JV.append(j); EV.append(e)
JV, EV = np.array(JV), np.array(EV)

print("-" * 72)
print(f"  J(alpha)     monotonically decreasing on range : "
      f"{'YES' if np.all(np.diff(JV) < 0) else 'NO'}")
print(f"  E(45; alpha) monotonically decreasing on range : "
      f"{'YES' if np.all(np.diff(EV) < 0) else 'NO'}")
print(f"  J   range: {JV[0]:.5f} (alpha={A_V[0]:.3f})  ->  {JV[-1]:.5f} (alpha={A_V[-1]:.3f})")
print(f"  E45 range: {EV[0]:.5f} (alpha={A_V[0]:.3f})  ->  {EV[-1]:.5f} (alpha={A_V[-1]:.3f})")
print("=" * 72)
print("  Numbers to quote in §19.3.6:")
print(f"    alpha* = {ALPHA_STAR:.3f} ;  t_B: {tB_human:.0f} -> {TB_STAR:.0f} yr")
print(f"    J drops by {abs(dJ):.0f} % ;  E(45) drops by {abs(dE):.0f} %")
print("=" * 72)
