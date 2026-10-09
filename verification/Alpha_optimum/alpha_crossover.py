"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================
  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0.
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================

VBLS — Alpha-optimum CROSSOVER test (Appendix G.2–G.3).

Question tested: on the calibrated Homo sapiens parameter set, does the lifespan optimum α*
coincide with the point at which the dominant death channel switches from R to E? Formally,
        t_B(α) = min( t_B^R(α), t_B^E(α) ).
If t_B^R increases in α (higher α -> lower E -> lower F_diss -> slower Σ -> slower ν·Σ·R
depletion) and t_B^E decreases (higher α -> faster η_abs decay -> earlier E starvation), the
minimum of the two branches peaks at their crossing, so α* == α_switch. Both monotonicities are
assumptions: Appendix G.2 gives an example within the class where they fail and the maximum
lies inside the R-branch (h1_counterexamples.py, example 2).

The script (parameters identical to alpha_sensitivity.py) reports:
    1. α*       = argmax_α t_B(α)              (physical, terminal events)
    2. α_switch = first grid node where the E-channel fires
    3. α_cross  = crossing of the branches t_B^E(α) = t_B^R(α)
With monotone branches these are one point found three ways, not independent estimates;
agreement is resolved to the grid step (≈0.0012). The diagnostic branches t_B^R(α), t_B^E(α)
are integrated on the full horizon (past the first death, shown only to visualise the crossing).

Output: alpha_crossover.png  + console verdict.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# CANONICAL PARAMETERS — identical to alpha_sensitivity.py (Homo sapiens)
# ============================================================================
BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, alpha=0.030,
            nu_0=0.006, nu=0.060, rho=0.002, R_star=0.15, E_star=0.04)
IC      = [0.10, 0.01, 0.96, 0.99]      # E0, Σ0, η0, R0
E0      = IC[0]
E_RANGE = E0 - BASE['E_star']           # 0.06
T_MAX   = 400.0                         # long horizon: E-channel ceiling can be large
T_GUARD = 2.0                           # ignore threshold crossings in first 2 yr (Phase-I dip)


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


# ============================================================================
# (1) PHYSICAL run — two terminal events, report which channel fired
# ============================================================================
def run_terminal(alpha):
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


# ============================================================================
# (2) DIAGNOSTIC run — no terminal stop; first-crossing time of EACH threshold
#     (past the first death the trajectory is non-physical; illustrative only)
# ============================================================================
def run_diagnostic(alpha):
    sol = solve_ivp(lambda t, y: _rhs(t, y, alpha), [0, T_MAX], IC,
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=0.2,
                    dense_output=False)
    t, E, R = sol.t, sol.y[0], sol.y[3]
    m = t >= T_GUARD

    def first_cross(x):
        below = m & (x <= 0.0)
        return float(t[below][0]) if below.any() else np.inf

    tE = first_cross(E - BASE['E_star'])
    tR = first_cross(R - BASE['R_star'])
    return tE, tR


# ============================================================================
# SCAN
# ============================================================================
ALPHA = np.linspace(0.005, 0.30, 240)

tB   = np.empty_like(ALPHA)
chan = []
for i, a in enumerate(ALPHA):
    tB[i], c = run_terminal(a)
    chan.append(c)
chan = np.array(chan)

tE_diag = np.empty_like(ALPHA)
tR_diag = np.empty_like(ALPHA)
for i, a in enumerate(ALPHA):
    tE_diag[i], tR_diag[i] = run_diagnostic(a)

# α*  = argmax t_B
i_star = int(np.argmax(tB))
alpha_star, tB_max = ALPHA[i_star], tB[i_star]

# α_switch = first α at which firing channel becomes 'E' (R→E), scanning upward
switch_idx = None
for i in range(1, len(ALPHA)):
    if chan[i] == 'E' and chan[i-1] == 'R':
        switch_idx = i
        break
alpha_switch = ALPHA[switch_idx] if switch_idx is not None else np.nan

# α_cross = where diagnostic branches cross (t_B^E == t_B^R), finite region
d = tE_diag - tR_diag
alpha_cross = np.nan
finite = np.isfinite(d)
idx = np.where(finite[:-1] & finite[1:] & (np.sign(d[:-1]) != np.sign(d[1:])))[0]
if idx.size:
    j = idx[0]
    # linear interpolation of the zero of d between j and j+1
    a0, a1, d0, d1 = ALPHA[j], ALPHA[j+1], d[j], d[j+1]
    alpha_cross = a0 - d0 * (a1 - a0) / (d1 - d0)

# ============================================================================
# VERDICT
# ============================================================================
print("=" * 66)
print("  VBLS — α-optimum crossover test (Homo sapiens calibration)")
print("=" * 66)
print(f"  baseline α           = {BASE['alpha']:.3f}  ->  t_B = {run_terminal(BASE['alpha'])[0]:.1f} yr")
print(f"  α*  (argmax t_B)     = {alpha_star:.4f}  ->  t_B_max = {tB_max:.1f} yr")
print(f"  α_switch (R→E)       = {alpha_switch:.4f}")
print(f"  α_cross (t_E==t_R)   = {alpha_cross:.4f}")
print("-" * 66)
if np.isfinite(alpha_switch):
    step = ALPHA[1] - ALPHA[0]
    nsteps = abs(alpha_star - alpha_switch) / step
    print(f"  |α* - α_switch|      = {abs(alpha_star-alpha_switch):.4f}  ({nsteps:.0f} grid step(s) of {step:.4f})")
    print(f"     -> on this calibration the optimum {'coincides with' if nsteps <= 1 else 'is NOT at'}"
          f" the R→E channel switch (to the grid step)")
else:
    print("  channel does not switch in the scanned range — inspect range/model")
print("-" * 66)
# channel composition around the peak
for lbl, i in [("just below α*", max(0, i_star-8)), ("at α*", i_star), ("just above α*", min(len(ALPHA)-1, i_star+8))]:
    print(f"  {lbl:>15}: α={ALPHA[i]:.3f}  channel={chan[i]}  t_B={tB[i]:.1f}")
print("=" * 66)

# ============================================================================
# FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# diagnostic branches (clip inf to T_MAX for plotting)
tE_p = np.where(np.isfinite(tE_diag), tE_diag, np.nan)
tR_p = np.where(np.isfinite(tR_diag), tR_diag, np.nan)
ax.plot(ALPHA, tR_p, color='#2980b9', lw=1.6, ls='--', label=r'$t_B^R(\alpha)$  (R-channel branch, ↑)')
ax.plot(ALPHA, tE_p, color='#c0392b', lw=1.6, ls='--', label=r'$t_B^E(\alpha)$  (E-channel branch, ↓)')

# physical t_B, colored by firing channel
isR = chan == 'R'
ax.plot(ALPHA[isR], tB[isR], 'o', ms=3.0, color='#2980b9', label=r'$t_B=\min$  (R fires)')
ax.plot(ALPHA[~isR & (chan != 'none')], tB[~isR & (chan != 'none')], 'o', ms=3.0,
        color='#c0392b', label=r'$t_B=\min$  (E fires)')

ax.axvline(alpha_star, color='#27ae60', lw=1.8, ls='-',
           label=fr'$\alpha^*={alpha_star:.3f}$ (argmax $t_B$)')
if np.isfinite(alpha_switch):
    ax.axvline(alpha_switch, color='#8e44ad', lw=1.5, ls=':',
               label=fr'$\alpha_{{switch}}={alpha_switch:.3f}$ (R→E)')
ax.axvline(BASE['alpha'], color='#7f8c8d', lw=1.0, ls='-.', alpha=0.7,
           label=fr"baseline $\alpha={BASE['alpha']}$")

ax.set_xlabel(r'$\alpha$ (yr$^{-1}$)')
ax.set_ylabel(r'$t_B$ (yr)')
ax.set_title(r'VBLS: is the lifespan optimum $\alpha^*$ the R→E channel crossover?')
ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
ax.set_ylim(0, min(np.nanmax(tB) * 1.6, T_MAX))
ax.grid(alpha=0.25)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpha_crossover.png')
plt.savefig(out, bbox_inches='tight', dpi=170)
plt.close()
print(f"Saved: {out}")
