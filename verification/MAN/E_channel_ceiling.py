"""
Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
SPDX-License-Identifier: CC-BY-NC-ND-4.0

VBLS — E-channel longevity ceiling under full R-channel suppression.

Question (author dialogue, 2026-06-22):
  Humans die by the R-channel (telomere depletion) at ~77 yr.
  If an organism were evolutionarily "perfect" on the R-channel
  (repair reserve never depletes), its lifespan would be set by the
  E-channel (integrity-potential exhaustion via memory decay of eta_abs).
  How long is that E-channel ceiling, from the §17.1 human calibration?

Method:
  Same canonical system (S) and parameters as human_lifespan.py.
  R-channel suppression = scale (nu_0, nu) by factor f in {1,...,0}.
  f=0  => dR/dt = +rho*g(E)*(1-Sigma)*(1-R) >= 0  => R never reaches R*,
          so death can only occur via E -> E*  (pure E-channel).

Two external-flux regimes:
  (A) "human"   : Phi_in declines after 60 yr, Phi_cap declines after 40 yr
                  (organism-aging clearance) — as in human_lifespan.py.
  (B) "plateau" : Phi_in held at plateau (no senescent decline), Phi_cap const.
                  This isolates the THEORETICAL E-channel (Lemma A.2/A.3):
                  death from memory decay of eta_abs at BOUNDED inflow,
                  not from an externally imposed decline of Phi_in.

Reported per run: t_B, death channel, and state at death.
Analytical reference: t ~ 1/(alpha*Sigma) e-folding of eta_abs (Lemma A.2).
"""

import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------------------
# Parameters (identical to §17.1 / human_lifespan.py)
# ----------------------------------------------------------------------------
mu, lam      = 1.50, 0.30
sigma_0      = 0.0162
alpha        = 0.030
nu_0_base    = 0.006
nu_base      = 0.060
rho          = 0.002
R_star       = 0.15
E_star       = 0.04
E0_init, Sigma0, eta0, R0 = 0.10, 0.01, 0.96, 0.99
E_range      = E0_init - E_star          # 0.06

# ----------------------------------------------------------------------------
# External fluxes
# ----------------------------------------------------------------------------
def Phi_in_human(t):
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline

def Phi_in_plateau(t):
    return 1.0 - np.exp(-t / 10.0)        # grows to 1, NO senescent decline

def Phi_cap_human(t):
    return 0.003 * np.exp(-max(0.0, t - 40.0) * 0.008)

def Phi_cap_const(t):
    return 0.003

# ----------------------------------------------------------------------------
# System (S), canonical F_diss = mu E^2 + lam E
# ----------------------------------------------------------------------------
def make_system(nu0, nuv, Phi_in, Phi_cap):
    def system(t, y):
        E, Sigma, eta, R = y
        E     = max(E, 0.0)
        Sigma = min(max(Sigma, 0.0), 1.0)
        eta   = min(max(eta,   0.0), 1.0)
        R     = min(max(R,     0.0), 1.0)
        F = mu * E**2 + lam * E
        g = max(0.0, min(1.0, (E - E_star) / E_range))
        dE     = eta * Phi_in(t) - F
        dSigma = max(0.0, sigma_0 * F - R * Phi_cap(t))
        deta   = -alpha * Sigma * eta
        dR     = -nu0 * R - nuv * Sigma * R + rho * g * (1.0 - Sigma) * (1.0 - R)
        return [dE, dSigma, deta, dR]
    return system

def run(nu0, nuv, Phi_in, Phi_cap, t_max):
    sys = make_system(nu0, nuv, Phi_in, Phi_cap)

    def ev_R(t, y): return y[3] - R_star
    def ev_E(t, y): return y[0] - E_star
    ev_R.terminal = ev_E.terminal = True
    ev_R.direction = ev_E.direction = -1

    sol = solve_ivp(sys, [0, t_max], [E0_init, Sigma0, eta0, R0],
                    method='LSODA', events=[ev_R, ev_E],
                    rtol=1e-9, atol=1e-12, dense_output=True, max_step=t_max/2000.0)

    tR = sol.t_events[0][0] if sol.t_events[0].size else np.inf
    tE = sol.t_events[1][0] if sol.t_events[1].size else np.inf
    if min(tR, tE) == np.inf:
        return None
    tB = min(tR, tE)
    channel = "R" if tR < tE else "E"
    E, S, eta, R = sol.sol(tB)
    return dict(tB=tB, channel=channel, E=E, Sigma=S, eta=eta, R=R)

# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------
def show(tag, r):
    if r is None:
        print(f"  {tag:<34}  no death within horizon (survives)")
        return
    print(f"  {tag:<34}  t_B = {r['tB']:>10.1f} yr  | {r['channel']}-channel "
          f"| E={r['E']:.3f} Σ={r['Sigma']:.3f} η={r['eta']:.4f} R={r['R']:.3f}")

print("=" * 100)
print("  VBLS — E-channel ceiling under R-channel suppression (params: §17.1)")
print("=" * 100)

print("\n[0] Baseline sanity (full R-channel, human fluxes):")
show("f=1.00  human fluxes", run(nu_0_base, nu_base, Phi_in_human, Phi_cap_human, 200))

print("\n[A] R-suppression sweep — HUMAN fluxes (Phi_in declines after 60):")
for f in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0]:
    show(f"f={f:<4}  (nu0={f*nu_0_base:.4f}, nu={f*nu_base:.4f})",
         run(f*nu_0_base, f*nu_base, Phi_in_human, Phi_cap_human, 5000))

print("\n[B] R-suppression sweep — PLATEAU fluxes (bounded Phi_in, const Phi_cap):")
print("    (this isolates the theoretical E-channel: memory decay at bounded inflow)")
for f in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0]:
    show(f"f={f:<4}  (nu0={f*nu_0_base:.4f}, nu={f*nu_base:.4f})",
         run(f*nu_0_base, f*nu_base, Phi_in_plateau, Phi_cap_const, 100000))

print("\n[C] Pure E-channel ceiling (f=0, R fully suppressed):")
rH = run(0.0, 0.0, Phi_in_human,   Phi_cap_human, 5000)
rP = run(0.0, 0.0, Phi_in_plateau, Phi_cap_const, 200000)
show("human fluxes (Phi_in decays)", rH)
show("plateau fluxes (bounded inflow)", rP)

print("\n[D] Analytical references (Lemma A.2, eta_abs e-folding ~ 1/(alpha*Sigma)):")
print(f"    1/(alpha*Sigma0={Sigma0})                = {1/(alpha*Sigma0):>10.0f} yr   (e-fold at start)")
for Spl in [0.05, 0.1, 0.2, 0.5]:
    print(f"    1/(alpha*Sigma_pl={Spl})              = {1/(alpha*Spl):>10.0f} yr")
print("    death needs ~ln(eta0*Phi/F_diss(E*)) e-foldings (several), so t_B is a few x these")
print("=" * 100)
