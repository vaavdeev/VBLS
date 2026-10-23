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
inflow_sensitivity.py
VBLS — Appendix D.2.5, Section 17.3.6 — role of the assumed inflow shape

The canonical human calibration (Section 17.1.1) takes
    Phi_in(t) = (1 - exp(-t/10)) * exp(-max(0, t - 60)/30),   Phi_cap = 0.003.
The decline of Phi_in after 60 years is an input of the calibration, not a
property of system (S). This script compares the canonical inflow with the same
inflow without the post-60 decline:

  1. the rate dSigma/dt: age of its maximum, and its value at 70 years and at
     t_B as a fraction of the maximum;
  2. the mapping 1 - k * DeltaSigma(t)^beta refitted to the seven points of the
     HUNT4 male trajectory (least squares, as in hunt_verification.py), and the
     modelled VO2max decline per decade.

Console output only, no figure.
Data: HUNT4 men — decade declines read off figure 3 of Letnes, Nes & Wisloff (2023),
redrawing Letnes et al. (2020) Prog. Cardiovasc. Dis. 63(6):730-737.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ── Canonical human calibration (Section 17.1.1) ──────────────────────────────
MU, LAM = 1.50, 0.30
SIGMA0 = 0.0162
ALPHA = 0.030
NU0, NU = 0.006, 0.060
RHO = 0.002
E_STAR, R_STAR = 0.04, 0.15
E0 = 0.10
PHI_CAP = 0.003
Y0 = [E0, 0.01, 0.96, 0.99]

# ── HUNT4 male trajectory ─────────────────────────────────────────────────────
AGES = np.array([25, 35, 45, 55, 65, 75, 85], dtype=float)
D_MEN = np.array([5.0, 5.6, 10.4, 12.8, 15.5, 24.5])          # % per decade
OBS = np.cumprod(np.r_[1.0, 1.0 - D_MEN / 100.0])


def phi_in(t, decline):
    growth = 1.0 - np.exp(-t / 10.0)
    return growth * (np.exp(-max(0.0, t - 60.0) / 30.0) if decline else 1.0)


def rhs(t, y, decline):
    E = max(y[0], 0.0)
    S = max(y[1], 0.0)
    eta = min(max(y[2], 0.0), 1.0)
    R = min(max(y[3], 0.0), 1.0)
    F = MU * E**2 + LAM * E
    g = max(0.0, min(1.0, (E - E_STAR) / (E0 - E_STAR)))
    return [eta * phi_in(t, decline) - F,
            max(0.0, SIGMA0 * F - R * PHI_CAP),
            -ALPHA * S * eta,
            -NU0 * R - NU * S * R + RHO * g * max(0.0, 1.0 - S) * (1.0 - R)]


def run(decline, stop_at_death):
    """Integrate (S). With stop_at_death the run ends at the first threshold crossing;
    without it the trajectory is continued past t_B (needed for the HUNT ages up to 85,
    as in hunt_aic.py and hunt_verification.py)."""
    def hit_r(t, y, d):
        return y[3] - R_STAR
    hit_r.terminal, hit_r.direction = True, -1

    def hit_e(t, y, d):
        return y[0] - E_STAR
    hit_e.terminal, hit_e.direction = True, -1
    return solve_ivp(rhs, (0.0, 300.0 if stop_at_death else 140.0), Y0, args=(decline,),
                     events=[hit_r, hit_e] if stop_at_death else None,
                     dense_output=True, method='RK45', rtol=1e-8, atol=1e-10, max_step=0.2)


def fit_mapping(dsig):
    def pred(p):
        return np.maximum(1.0 - p[0] * dsig**p[1], 0.0)

    def loss(p):
        if p[0] <= 0.0 or p[1] <= 0.1:
            return 1e9
        return float(np.sum((pred(p) - OBS)**2))
    best = min((minimize(loss, x0, method='Nelder-Mead',
                         options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
                for x0 in ([2.5, 2.3], [1.0, 1.5], [5.0, 3.0], [10.0, 4.0])),
               key=lambda r: r.fun)
    return best.x, np.sqrt(best.fun / len(AGES)), pred(best.x)


print('Role of the post-60 decline of Phi_in (canonical calibration otherwise unchanged)')
print('HUNT4 men, decline per decade 25-35 ... 75-85 (%):', D_MEN.tolist())
for decline, name in ((True, 'canonical inflow (decline after 60)'),
                      (False, 'same inflow without the decline after 60')):
    sol = run(decline, stop_at_death=True)
    if len(sol.t_events[0]):
        t_b, channel = sol.t_events[0][0], 'R-channel'
    elif len(sol.t_events[1]):
        t_b, channel = sol.t_events[1][0], 'E-channel'
    else:
        t_b, channel = sol.t[-1], 'no death within 300 yr'
    tt = np.linspace(0.0, t_b, 6000)
    Y = sol.sol(tt)
    F = MU * Y[0]**2 + LAM * Y[0]
    rate = np.maximum(0.0, SIGMA0 * F - np.clip(Y[3], 0.0, 1.0) * PHI_CAP)
    i_max = int(np.argmax(rate))
    cont = run(decline, stop_at_death=False)
    s25 = float(cont.sol(25.0)[1])
    dsig = np.array([max(float(cont.sol(a)[1]) - s25, 0.0) for a in AGES])
    (k, beta), rmse, vo2 = fit_mapping(dsig)
    decades = 100.0 * (1.0 - vo2[1:] / vo2[:-1])
    print(f'\n{name}:')
    print(f'  t_B = {t_b:.1f} yr ({channel})')
    print(f'  dSigma/dt: maximum at {tt[i_max]:.1f} yr; '
          f'at 70 yr {np.interp(70.0, tt, rate) / rate[i_max]:.2f} of the maximum, '
          f'at t_B {rate[-1] / rate[i_max]:.2f}')
    print(f'  mapping refitted to HUNT4 men: k = {k:.2f}, beta = {beta:.2f}, RMSE = {rmse:.4f}')
    print('  modelled VO2max decline per decade (%):', np.round(decades, 1).tolist())
