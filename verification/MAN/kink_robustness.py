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
VBLS — Numerical robustness of t_B to the kink of the right-hand side
on the export switching surface  sigma_0 * F_diss(E) = R * Phi_cap.

The truncation max(0, .) in equation (7.1) makes the right-hand side
Lipschitz but not smooth on this surface. Piecewise smoothness without
step control at the kink can lower the order of accuracy, so the script
measures how strongly the predicted t_B depends on the way the kink is
handled (Section 17.1.1, "Numerical scheme at the kink").

Equations and parameters are identical to MAN/human_lifespan.py.
The script changes nothing in the verification runs; it only measures
the sensitivity of t_B to the treatment of the kink.

Four treatments are compared:
  1) adaptive step with an upper bound max_step (the one used in the monograph);
  2) the same without a step bound;
  3) event detection: integration is split at the crossing of the surface
     and restarted from it;
  4) smooth replacement max(0,x) -> (x + sqrt(x^2 + eps^2)) / 2.

Dependencies: numpy, scipy
Run:          python kink_robustness.py   (console output only, no files written)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- parameters
MU, LAM = 1.50, 0.30
SIGMA0 = 0.0162
ALPHA = 0.030
NU0, NU, RHO = 0.006, 0.060, 0.002
R_STAR, E_STAR = 0.15, 0.04
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0] - E_STAR
T_MAX = 140.0


def phi_in(t):
    return (1.0 - np.exp(-t / 10.0)) * np.exp(-max(0.0, t - 60.0) / 30.0)


def phi_cap(t):
    return 0.003


def rhs(t, y, eps=0.0):
    E, S, eta, R = y
    E = max(E, 0.0)
    S = np.clip(S, 0.0, 1.0)
    eta = np.clip(eta, 0.0, 1.0)
    R = np.clip(R, 0.0, 1.0)
    F = MU * E ** 2 + LAM * E
    g = max(0.0, min(1.0, (E - E_STAR) / E_RANGE))
    z = SIGMA0 * F - R * phi_cap(t)
    dS = max(0.0, z) if eps == 0.0 else 0.5 * (z + np.sqrt(z * z + eps * eps))
    return [eta * phi_in(t) - F,
            dS,
            -ALPHA * S * eta,
            -NU0 * R - NU * S * R + RHO * g * (1.0 - S) * (1.0 - R)]


def death(t, y):
    return min(y[3] - R_STAR, y[0] - E_STAR)


death.terminal, death.direction = True, -1


def switch(t, y):
    """Switching surface: production minus export."""
    E, S, eta, R = y
    return SIGMA0 * (MU * E ** 2 + LAM * E) - R * phi_cap(t)


switch.terminal = False


def run(method="RK45", rtol=1e-8, atol=1e-10, max_step=np.inf, eps=0.0,
        y0=None, t0=0.0, t_max=T_MAX, track_switch=False):
    ev = [death, switch] if track_switch else [death]
    sol = solve_ivp(lambda t, y: rhs(t, y, eps), [t0, t_max], y0 or IC,
                    method=method, events=ev, rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=True)
    tb = sol.t_events[0][0] if sol.t_events[0].size else None
    sw = sol.t_events[1] if track_switch else np.array([])
    return tb, sw, sol


SEP = "=" * 74
print(SEP)
print("  Robustness of t_B to the treatment of the kink max(0,.)")
print(SEP)

# --- how many times the trajectory crosses the switching surface ---
tb0, sw, sol0 = run(max_step=0.05, track_switch=True)
print("\n  Baseline run (RK45, rtol 1e-8, max_step 0.05)")
print("    t_B = %.6f yr" % tb0)
print("    crossings of the switching surface: %d %s"
      % (len(sw), ("at t = " + ", ".join("%.3f" % x for x in sw)) if len(sw) else ""))

# --- 1-2) grid of step and tolerance settings ---
print("\n  1-2. Adaptive step: dependence of t_B on solver settings")
print("    %-8s %-10s %-10s %s" % ("method", "rtol", "max_step", "t_B, yr"))
grid = []
for method in ("RK45", "LSODA", "Radau"):
    for rtol in (1e-6, 1e-8, 1e-10):
        for ms in (np.inf, 0.5, 0.05, 0.01):
            tb, _, _ = run(method=method, rtol=rtol, max_step=ms)
            grid.append(tb)
            print("    %-8s %-10.0e %-10s %.6f"
                  % (method, rtol, ("none" if ms == np.inf else "%.2f" % ms), tb))
grid = np.array([g for g in grid if g is not None])
print("\n    spread over the whole grid: %.2e yr (%.1e %%)"
      % (grid.max() - grid.min(), 100 * (grid.max() - grid.min()) / grid.mean()))

# --- 3) event detection: split at the crossing ---
print("\n  3. Event detection: split and restart from the surface")
if len(sw):
    t_sw = float(sw[0])
    y_sw = sol0.sol(t_sw)
    tb_split, _, _ = run(max_step=0.05, y0=list(y_sw), t0=t_sw)
    print("    crossing at t = %.6f, restart from it" % t_sw)
    print("    t_B = %.6f yr   (difference from baseline %.2e yr)"
          % (tb_split, abs(tb_split - tb0)))
else:
    print("    no crossings detected; no split needed")

# --- 4) smooth replacement of the truncation operator ---
print("\n  4. Smooth replacement max(0,x) -> (x + sqrt(x^2 + eps^2)) / 2")
print("    %-10s %-12s %s" % ("eps", "t_B, yr", "difference from baseline"))
for eps in (1e-3, 1e-4, 1e-5, 1e-6):
    tb_s, _, _ = run(max_step=0.05, eps=eps)
    print("    %-10.0e %-12.6f %.2e yr" % (eps, tb_s, abs(tb_s - tb0)))

print("\n" + SEP)
