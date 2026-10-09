"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================

  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================
VBLS — sign-pattern check of the alpha-sensitivity of system (S) (Appendix G.3).

Tests, pointwise in alpha, the sign pattern P
    u_E = dE/d(alpha) <= 0,  u_Sigma = dSigma/d(alpha) <= 0,
    u_eta = d(eta)/d(alpha) <= 0,  u_R = dR/d(alpha) >= 0,
by finite differences between neighbouring alpha (step 0.005) on a common time grid,
from t = 0 (the first year of life, where restoration is not saturated, is included).
Same Homo sapiens calibration as alpha_sensitivity.py.

Analytical status (Appendix G.4): given u_E <= 0 and a small restoration term in g'(E),
the pattern reduces to u_eta <= 0, i.e. d/d(alpha)[alpha * int Sigma] >= 0. At alpha -> 0 with
saturated restoration the pattern is proved in the full system (Lemma G.1); for finite alpha
there is only an argument for an auxiliary equation. This script is the numerical check.

Tolerances matter: with loose tolerances an apparent violation of size ~1e-7 appears for
Sigma and R near t ~ 3 yr, where damage production first exceeds export and the field (7.1)
has a kink; it disappears with the tolerances used here (see rho_residual_check.py).
"""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            rho=0.002, R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0] - BASE['E_star']
T_MAX = 130.0
RTOL, ATOL, MAX_STEP = 1e-11, 1e-13, 0.02

_trapz = getattr(np, 'trapezoid', None) or np.trapz


def Phi_in(t):  return (1.0 - np.exp(-t/10.0)) * np.exp(-max(0.0, t-60.0)/30.0)
def Phi_cap(t): return 0.003


def rhs(t, y, alpha):
    E, S, eta, R = y
    E = max(E, 0.0); S = np.clip(S, 0, 1); eta = np.clip(eta, 0, 1); R = np.clip(R, 0, 1)
    F = BASE['mu']*E**2 + BASE['lam']*E
    g = max(0.0, min(1.0, (E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t)-F,
            max(0.0, BASE['sigma_0']*F - R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R - BASE['nu']*S*R + BASE['rho']*g*(1-S)*(1-R)]


def tB_and_traj(alpha):
    def eE(t, y): return y[0]-BASE['E_star']
    def eR(t, y): return y[3]-BASE['R_star']
    eE.terminal = eR.terminal = True; eE.direction = eR.direction = -1
    s = solve_ivp(lambda t, y: rhs(t, y, alpha), [0, T_MAX], IC, method='RK45',
                  events=[eE, eR], rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True)
    tE = s.t_events[0][0] if s.t_events[0].size else np.inf
    tR = s.t_events[1][0] if s.t_events[1].size else np.inf
    tB = min(tE, tR) if (tE < np.inf or tR < np.inf) else T_MAX
    return tB, s


# alpha grid on the R-limited side, up to the optimum
ALPHA = np.linspace(0.010, 0.135, 26)
t_eval = np.linspace(0.0, T_MAX, 13001)   # 0.01-yr sampling

traj, tBs = {}, {}
for a in ALPHA:
    tB, s = tB_and_traj(a)
    tBs[a] = tB
    tt = t_eval[t_eval <= tB]
    traj[a] = (tt, s.sol(tt))

labels = ['E', 'Sigma', 'eta', 'R']
# expected: E, Sigma, eta decrease in alpha; R increases
want_decrease = {'E': True, 'Sigma': True, 'eta': True, 'R': False}
TOL = 1e-9

print("="*70)
print("  VBLS sign-pattern check (pattern P over alpha, Homo sapiens)")
print("="*70)
worst = {k: (1.0, None) for k in labels}
any_break = {k: False for k in labels}

for i in range(len(ALPHA)-1):
    a0, a1 = ALPHA[i], ALPHA[i+1]
    (t0, Y0), (t1, Y1) = traj[a0], traj[a1]
    n = min(len(t0), len(t1))
    A, B = Y0[:, :n], Y1[:, :n]
    for k, lab in enumerate(labels):
        ok = (A[k] >= B[k] - TOL) if want_decrease[lab] else (A[k] <= B[k] + TOL)
        frac = float(np.mean(ok)) if n else 1.0
        if frac < worst[lab][0]:
            worst[lab] = (frac, (a0, a1))
        if frac < 1.0 - 1e-6:
            any_break[lab] = True

print(f"  alpha range: [{ALPHA[0]:.3f}, {ALPHA[-1]:.3f}]  (alpha* ~ 0.135), t from 0")
print(f"  solver: RK45, rtol={RTOL:g}, atol={ATOL:g}, max_step={MAX_STEP}")
print(f"  baseline t_B(0.030) = {tBs[min(ALPHA, key=lambda a: abs(a-0.030))]:.1f} yr")
print("-"*70)
print(f"  {'var':>6} {'expected':>10} {'holds?':>8} {'min compliance':>16}  where")
print("-"*70)
for lab in labels:
    exp = 'decr in a' if want_decrease[lab] else 'incr in a'
    frac, where = worst[lab]
    verdict = 'YES' if not any_break[lab] else 'breaks'
    wtxt = '' if where is None else f'alpha in [{where[0]:.3f},{where[1]:.3f}]'
    print(f"  {lab:>6} {exp:>10} {verdict:>8} {frac*100:>14.2f}%  {wtxt}")
print("-"*70)
if not any(any_break.values()):
    print("  => the sign pattern P holds numerically on the whole range, first year included.")
    print("     Analytically: Lemma G.1 (alpha -> 0, saturated restoration); for finite")
    print("     alpha < alpha_guar only an argument for the auxiliary equation (Appendix G.4).")
else:
    print("  => the pattern breaks somewhere: inspect that alpha-window and the tolerances.")
print("="*70)

# cross-check via the master scalar alpha * int Sigma (should be non-decreasing in alpha)
print("  Cross-check via the master scalar  alpha * int Sigma dt  (non-decreasing in alpha):")
Iint = np.array([a * _trapz(traj[a][1][1], traj[a][0]) for a in ALPHA])
dI = np.diff(Iint)
print(f"    alpha * int Sigma : {'monotone non-decreasing' if np.all(dI >= -1e-9) else 'NOT monotone'} "
      f"(min increment = {dI.min():+.3e})")
print("="*70)
