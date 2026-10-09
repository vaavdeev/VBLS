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
VBLS — two systems of the class (S) that delimit Hypothesis H1 (Section 8.2.1, Appendix G).

Example 1 (Appendix G.4, remark to Lemma G.1): restoration is NOT saturated.
    Conditions (a) and (c) of H1 hold, condition (b) fails: after the inflow declines, the
    potential settles below E0, so g(E) < 1. A larger alpha lowers E and g(E), restoration
    weakens, and t_B(alpha) decreases from alpha = 0 onwards: there is no maximum at alpha > 0.

Example 2 (Appendix G.2, "Without monotone branches"): conditions (a)-(c) hold and an
    optimum exists, but both branches t_B^R(alpha), t_B^E(alpha) are non-monotone and the
    maximum of t_B lies inside the R-branch, at a kink of the vector field (damage production
    ceases to exceed export), not at the R -> E channel switch.

The vector field is the exact system (S) of Appendix A.1, without the [0,1] projection used in
the calibration scripts:
    dE/dt     = eta * Phi_in(t) - F(E),           F(E) = 1.5 E^2 + 0.3 E (zero for E < 0)
    dSigma/dt = max(0, sigma_0 F(E) - R Phi_cap)
    deta/dt   = -alpha * Sigma * eta
    dR/dt     = -nu_0 R - nu Sigma R + rho g(E) max(0, 1 - Sigma) (1 - R)
    g(E)      = max(0, min(1, (E - E*) / (E0 - E*)))

Run: python h1_counterexamples.py   (console output only, about one minute)
"""

import numpy as np
from scipy.integrate import solve_ivp

COMMON = dict(mu=1.5, lam=0.3, sigma_0=0.0162, nu_0=0.006, R_star=0.15,
              Phi_cap=0.01, E0=0.427, Sigma0=0.01, eta0=0.96, R0=0.99)

EX1 = dict(COMMON, nu=0.02, rho=0.0055, E_star=0.20,
           Phi_in=lambda t: 0.234 + 0.266 * np.exp(-max(0.0, t - 20.0) / 20.0),
           Phi_in_min=0.234, horizon=1500.0)

EX2 = dict(COMMON, nu=0.06, rho=0.004, E_star=0.04,
           Phi_in=lambda t: 0.5, Phi_in_min=0.5, horizon=1500.0)


def F(p, E):
    E = max(E, 0.0)
    return p['mu'] * E * E + p['lam'] * E


def rhs(t, y, p, alpha):
    E, S, eta, R = y
    Fd = F(p, E)
    g = max(0.0, min(1.0, (E - p['E_star']) / (p['E0'] - p['E_star'])))
    return [eta * p['Phi_in'](t) - Fd,
            max(0.0, p['sigma_0'] * Fd - R * p['Phi_cap']),
            -alpha * S * eta,
            -p['nu_0'] * R - p['nu'] * S * R + p['rho'] * g * max(0.0, 1.0 - S) * (1.0 - R)]


def run(p, alpha):
    """Trajectory continued past death; returns t_B^R, t_B^E (first crossings) and the solution."""
    y0 = [p['E0'], p['Sigma0'], p['eta0'], p['R0']]

    def eR(t, y): return y[3] - p['R_star']
    def eE(t, y): return y[0] - p['E_star']
    eR.direction = eE.direction = -1
    s = solve_ivp(lambda t, y: rhs(t, y, p, alpha), [0.0, p['horizon']], y0, method='RK45',
                  events=[eR, eE], rtol=1e-9, atol=1e-11, max_step=0.5, dense_output=True)
    tR = s.t_events[0][0] if s.t_events[0].size else np.inf
    tE = s.t_events[1][0] if s.t_events[1].size else np.inf
    return tR, tE, s


def t_B(p, alpha):
    tR, tE, s = run(p, alpha)
    return (tR, 'R') if tR <= tE else (tE, 'E')


def class_checks(p, name):
    F0 = F(p, p['E0'])
    window_lo = p['nu_0'] * p['R_star'] / (1 - p['R_star'])
    print('  %s: class conditions' % name)
    print('    Barrier I + (C4) window: %.5f < rho = %.4f <~ nu_0 = %.3f' % (window_lo, p['rho'], p['nu_0']))
    print('    Barrier II: R0*Phi_cap = %.4f > sigma_0*F(E0) = %.4f' % (p['R0'] * p['Phi_cap'], p['sigma_0'] * F0))
    print('    A1: E0 = %.3f > E* = %.3f,  R0 = %.2f > R* = %.2f' % (p['E0'], p['E_star'], p['R0'], p['R_star']))
    print('    E-channel at alpha = 0 needs eta0*Phi_in > F(E*) everywhere: %.4f vs %.4f' % (
        p['eta0'] * p['Phi_in_min'], F(p, p['E_star'])))


def saturation_at_zero(p):
    tR, tE, s = run(p, 0.0)
    T = min(tR, tE)
    tt = np.linspace(1e-6, T, 20001)
    E = s.sol(tt)[0]
    S_end = s.sol(T)[1]
    return T, ('R' if tR <= tE else 'E'), E.min(), S_end


def max_net_production(p, alpha):
    """max over the life span of sigma_0 F(E) - R Phi_cap (positive: Sigma grows somewhere)."""
    tR, tE, s = run(p, alpha)
    T = min(tR, tE)
    tt = np.linspace(0.0, T, 20001)
    E, S, eta, R = s.sol(tt)
    net = p['sigma_0'] * (p['mu'] * E * E + p['lam'] * E) - R * p['Phi_cap']
    return net.max(), S[-1]


# --------------------------------------------------------------------------------------
print('=' * 78)
print('  Example 1 — unsaturated restoration: no maximum of t_B(alpha) at alpha > 0')
print('=' * 78)
class_checks(EX1, 'Example 1')
T0, ch0, Emin0, S0end = saturation_at_zero(EX1)
print('    alpha = 0: t_B = %.3f (%s-channel), Sigma(t_B) = %.3f' % (T0, ch0, S0end))
print('    condition (b): min E on (t_A, t_B] = %.4f vs E0 = %.3f  ->  %s' % (
    Emin0, EX1['E0'], 'saturated' if Emin0 > EX1['E0'] else 'NOT saturated'))
print('  %10s %10s %8s' % ('alpha', 't_B', 'channel'))
for a in (0.0, 1e-4, 4.96e-4, 2.46e-3, 0.0122, 0.0605, 0.3):
    tb, ch = t_B(EX1, a)
    print('  %10.4g %10.3f %8s' % (a, tb, ch))

print()
print('=' * 78)
print('  Example 2 — conditions (a)-(c) hold, but the maximum lies inside the R-branch')
print('=' * 78)
class_checks(EX2, 'Example 2')
T0, ch0, Emin0, S0end = saturation_at_zero(EX2)
print('    alpha = 0: t_B = %.3f (%s-channel), Sigma(t_B) = %.3f' % (T0, ch0, S0end))
print('    condition (b): min E on (t_A, t_B] = %.4f vs E0 = %.3f  ->  %s' % (
    Emin0, EX2['E0'], 'saturated' if Emin0 > EX2['E0'] else 'NOT saturated'))
print('  %8s %10s %8s %10s %10s %18s' % ('alpha', 't_B', 'channel', 't_B^R', 't_B^E', 'max(prod-export)'))
for a in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.8, 1.6, 4.0):
    tR, tE, _ = run(EX2, a)
    net, _ = max_net_production(EX2, a)
    tb, ch = (tR, 'R') if tR <= tE else (tE, 'E')
    print('  %8.3f %10.2f %8s %10.2f %10.2f %18.2e' % (a, tb, ch, tR, tE, net))

A = np.arange(0.25, 0.45, 0.0025)
vals = [t_B(EX2, a) for a in A]
k = int(np.argmax([v[0] for v in vals]))
print('  fine scan: max t_B = %.2f at alpha = %.4f (%s-channel)' % (vals[k][0], A[k], vals[k][1]))
for a in (A[k] - 0.0025, A[k], A[k] + 0.0025):
    net, Send = max_net_production(EX2, a)
    print('    alpha = %.4f: max(prod - export) = %+.2e, Sigma(t_B) = %.4f' % (a, net, Send))
A2 = np.arange(0.80, 1.20, 0.01)
sw = next((a for a in A2 if t_B(EX2, a)[1] == 'E'), np.nan)
print('  first E-channel death on [0.80, 1.20] (step 0.01): alpha = %.2f, t_B = %.2f' % (sw, t_B(EX2, sw)[0]))
print('=' * 78)
