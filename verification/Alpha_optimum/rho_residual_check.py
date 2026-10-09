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
VBLS — diagnostic of the apparent Sigma/R sign violation in the alpha-sensitivity check
(Appendix G.3). The file name is historical: the violation was first attributed to the
rho * g'(E) restoration term; this script shows that it is an integration artifact.

With loose tolerances (rtol 1e-9, max_step 0.1) the sign check finds a narrow violation for
Sigma and R in the window alpha in [0.070, 0.075]. The script measures it in three ways:
  (1) loose tolerances, decreasing rho (the original test);
  (2) loose tolerances, rho unchanged, g(E) replaced by 1 (removes the g'(E) term);
  (3) tight tolerances (rtol 1e-11, max_step 0.02), original model.
If (2) keeps the violation and (3) removes it, the violation is not the g'(E) term but an
artifact at the kink of (7.1), where damage production first exceeds export (t ~ 3 yr).
"""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0] - BASE['E_star']
T_MAX = 130.0


def Phi_in(t):  return (1-np.exp(-t/10))*np.exp(-max(0.0, t-60)/30)
def Phi_cap(t): return 0.003


def rhs(t, y, alpha, rho, g_one):
    E, S, eta, R = y
    E = max(E, 0.0); S = np.clip(S, 0, 1); eta = np.clip(eta, 0, 1); R = np.clip(R, 0, 1)
    F = BASE['mu']*E**2 + BASE['lam']*E
    g = 1.0 if g_one else max(0.0, min(1.0, (E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t)-F,
            max(0.0, BASE['sigma_0']*F-R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R-BASE['nu']*S*R+rho*g*(1-S)*(1-R)]


def traj(alpha, rho, g_one, tight):
    rtol, atol, ms = (1e-11, 1e-13, 0.02) if tight else (1e-9, 1e-11, 0.1)
    def eE(t, y): return y[0]-BASE['E_star']
    def eR(t, y): return y[3]-BASE['R_star']
    eE.terminal = eR.terminal = True; eE.direction = eR.direction = -1
    s = solve_ivp(lambda t, y: rhs(t, y, alpha, rho, g_one), [0, T_MAX], IC, method='RK45',
                  events=[eE, eR], rtol=rtol, atol=atol, max_step=ms, dense_output=True)
    tE = s.t_events[0][0] if s.t_events[0].size else np.inf
    tR = s.t_events[1][0] if s.t_events[1].size else np.inf
    return min(tE, tR, T_MAX), s


ALPHA = np.linspace(0.010, 0.135, 26)
labels = ['E', 'Sigma', 'eta', 'R']
want_dec = {'E': 1, 'Sigma': 1, 'eta': 1, 'R': 0}


def check(rho, g_one=False, tight=False):
    T = {}
    for a in ALPHA:
        tB, s = traj(a, rho, g_one, tight)
        T[a] = (tB, s)
    worst = {k: (1.0, None, None) for k in labels}
    for i in range(len(ALPHA)-1):
        (tb0, s0), (tb1, s1) = T[ALPHA[i]], T[ALPHA[i+1]]
        tt = np.arange(0.0, min(tb0, tb1), 0.01)
        A, B = s0.sol(tt), s1.sol(tt)
        for k, lab in enumerate(labels):
            d = (A[k]-B[k]) if want_dec[lab] else (B[k]-A[k])
            bad = d < -1e-9
            frac = 1.0 - float(np.mean(bad))
            if frac < worst[lab][0]:
                worst[lab] = (frac, (ALPHA[i], ALPHA[i+1]), (tt[bad].min(), tt[bad].max(), d.min()))
    return worst


print(f"{'case':>34} " + " ".join(f"{l:>8}" for l in labels) + "   (min compliance %)")
print("-"*80)
cases = [('loose, rho=0.002', dict(rho=0.002)),
         ('loose, rho=0.0006', dict(rho=0.0006)),
         ('loose, rho=0.0002', dict(rho=0.0002)),
         ('loose, rho=0', dict(rho=0.0)),
         ('loose, rho=0.002, g = 1', dict(rho=0.002, g_one=True)),
         ('tight, rho=0.002', dict(rho=0.002, tight=True))]
for name, kw in cases:
    w = check(**kw)
    print(f"{name:>34} " + " ".join(f"{w[l][0]*100:>8.2f}" for l in labels))
    for l in ('Sigma', 'R'):
        if w[l][1] is not None:
            (a0, a1), (ta, tb, dmin) = w[l][1], w[l][2]
            print(f"{'':>34}   {l}: alpha in [{a0:.3f},{a1:.3f}], t in [{ta:.2f},{tb:.2f}] yr, size {dmin:.1e}")
print("-"*80)
print("Note: rho = 0.0006, 0.0002 and 0 lie below the Barrier I window (rho > 0.00106), i.e.")
print("outside the viable class; they are kept only to reproduce the original test.")
print("If the 'g = 1' case keeps the violation and the 'tight' case removes it, the violation is")
print("an integration artifact at the kink of (7.1), not the rho * g'(E) restoration term.")
