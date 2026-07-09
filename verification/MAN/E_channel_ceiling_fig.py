"""
Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
SPDX-License-Identifier: CC-BY-NC-ND-4.0

VBLS — E-channel longevity ceiling: figure.

Panel A: t_B vs R-suppression factor f (scaling nu_0, nu) — shows the
         R->E channel switch and the ceiling, for human and plateau fluxes.
Panel B: E-channel ceiling t_B^E vs damage-production rate sigma_0
         (full R suppression, bounded inflow) — lower sigma_0 -> higher ceiling.
         Reference lines: real R-escaped organisms (Greenland shark, ocean quahog).

Parameters: §17.1 human calibration.  Model identical to human_lifespan.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ---- §17.1 parameters --------------------------------------------------------
mu, lam   = 1.50, 0.30
sigma0_h  = 0.0162
alpha     = 0.030
nu0_b     = 0.006
nu_b      = 0.060
rho       = 0.002
R_star    = 0.15
E_star    = 0.04
E0, S0, eta0, R0 = 0.10, 0.01, 0.96, 0.99
Erange    = E0 - E_star

def Phi_in_human(t):
    return (1 - np.exp(-t/10)) * np.exp(-max(0.0, t-60)/30)
def Phi_in_plat(t):
    return 1 - np.exp(-t/10)
def Phi_cap_human(t):
    return 0.003*np.exp(-max(0.0, t-40)*0.008)
def Phi_cap_const(t):
    return 0.003

def run(nu0, nuv, sig0, Phi_in, Phi_cap, t_max):
    def sysf(t, y):
        E, S, eta, R = y
        E = max(E, 0.0); S = min(max(S,0.0),1.0)
        eta = min(max(eta,0.0),1.0); R = min(max(R,0.0),1.0)
        F = mu*E*E + lam*E
        g = max(0.0, min(1.0, (E-E_star)/Erange))
        return [eta*Phi_in(t)-F,
                max(0.0, sig0*F - R*Phi_cap(t)),
                -alpha*S*eta,
                -nu0*R - nuv*S*R + rho*g*(1-S)*(1-R)]
    def evR(t,y): return y[3]-R_star
    def evE(t,y): return y[0]-E_star
    evR.terminal=evE.terminal=True; evR.direction=evE.direction=-1
    sol = solve_ivp(sysf,[0,t_max],[E0,S0,eta0,R0], method='LSODA',
                    events=[evR,evE], rtol=1e-9, atol=1e-12,
                    dense_output=True, max_step=t_max/2000.0)
    tR = sol.t_events[0][0] if sol.t_events[0].size else np.inf
    tE = sol.t_events[1][0] if sol.t_events[1].size else np.inf
    if min(tR,tE)==np.inf: return None, None
    return (tR if tR<tE else tE), ("R" if tR<tE else "E")

# ---- Panel A data: suppression sweep ----------------------------------------
fs = np.array([1.0,0.7,0.5,0.35,0.25,0.15,0.1,0.05,0.02,0.0])
tA_h, chA_h, tA_p, chA_p = [], [], [], []
for f in fs:
    t,c = run(f*nu0_b, f*nu_b, sigma0_h, Phi_in_human, Phi_cap_human, 6000)
    tA_h.append(t); chA_h.append(c)
    t,c = run(f*nu0_b, f*nu_b, sigma0_h, Phi_in_plat, Phi_cap_const, 120000)
    tA_p.append(t); chA_p.append(c)

# ---- Panel B data: ceiling vs sigma_0 (full R suppression, plateau) ----------
sigs = np.geomspace(sigma0_h, 0.0008, 22)
tB_sig = []
for s in sigs:
    t,c = run(0.0, 0.0, s, Phi_in_plat, Phi_cap_const, 400000)
    tB_sig.append(t if t is not None else np.nan)
tB_sig = np.array(tB_sig)

# console echo
print("Panel A (plateau): f, t_B, channel")
for f,t,c in zip(fs,tA_p,chA_p): print(f"  f={f:<5} t_B={t:8.1f}  {c}")
print("Panel B: sigma0, t_B^E")
for s,t in zip(sigs,tB_sig): print(f"  sigma0={s:.4f}  t_B={t:8.1f}")

# ---- Figure ------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#f7f9fc')
fig.suptitle("VBLS — E-channel longevity ceiling under R-channel suppression "
             "(§17.1 human calibration)", fontsize=13, fontweight='bold')

# Panel A
def colmark(ch): return ['#27ae60' if c=='R' else '#c0392b' for c in ch]
axA.plot(fs, tA_p, '-', color='#34495e', lw=1.5, zorder=1, label='bounded inflow (clean E-channel)')
axA.scatter(fs, tA_p, c=colmark(chA_p), s=70, zorder=3, edgecolor='k', linewidth=0.4)
axA.plot(fs, tA_h, '--', color='#7f8c8d', lw=1.3, zorder=1, label='human fluxes (Φ_in declines)')
axA.scatter(fs, tA_h, c=colmark(chA_h), s=40, zorder=3, marker='s', edgecolor='k', linewidth=0.3)
axA.set_xlabel("R-suppression factor  f   (ν₀,ν → f·ν₀, f·ν)   [1 = normal, 0 = R fully suppressed]")
axA.set_ylabel("lifespan  t_B  (yr)")
axA.set_title("A. R→E channel switch and ceiling", fontsize=11)
axA.invert_xaxis()
axA.annotate("baseline\n77 yr (R)", xy=(1.0, tA_p[0]), xytext=(0.85, 95),
             fontsize=8, color='#27ae60', ha='center',
             arrowprops=dict(arrowstyle='->', color='#27ae60'))
axA.annotate(f"ceiling\n{tA_p[-1]:.0f} yr (E)", xy=(0.0, tA_p[-1]), xytext=(0.13, 215),
             fontsize=8, color='#c0392b', ha='center',
             arrowprops=dict(arrowstyle='->', color='#c0392b'))
from matplotlib.lines import Line2D
leg = [Line2D([],[],marker='o',color='w',markerfacecolor='#27ae60',markersize=9,label='R-channel'),
       Line2D([],[],marker='o',color='w',markerfacecolor='#c0392b',markersize=9,label='E-channel'),
       Line2D([],[],ls='-',color='#34495e',label='bounded inflow'),
       Line2D([],[],ls='--',color='#7f8c8d',label='human fluxes')]
axA.legend(handles=leg, fontsize=8, loc='center left')
axA.set_facecolor('#f7f9fc')

# Panel B
axB.plot(sigs, tB_sig, '-o', color='#c0392b', ms=4, lw=1.8)
axB.set_xscale('log'); axB.set_yscale('log')
axB.set_xlabel("damage-production rate  σ₀")
axB.set_ylabel("E-channel ceiling  t_B^E  (yr, full R suppression)")
axB.set_title("B. Ceiling vs damage rate  (lower σ₀ → longer ceiling)", fontsize=11)
axB.axvline(sigma0_h, color='#2980b9', ls=':', lw=1.4)
axB.annotate("human σ₀=0.0162\n(ceiling ≈246 yr)", xy=(sigma0_h, np.nanmin(tB_sig)),
             xytext=(sigma0_h*0.9, np.nanmin(tB_sig)*1.4),
             fontsize=8, color='#2980b9', ha='right')
for yr, name, col in [(392,'Greenland shark ≈390 yr','#16a085'),
                      (507,'ocean quahog ≈500 yr','#8e44ad')]:
    axB.axhline(yr, color=col, ls='--', lw=1.1)
    axB.text(sigs[-1], yr*1.03, name, fontsize=7.5, color=col, ha='left', va='bottom')
axB.set_facecolor('#f7f9fc')

plt.tight_layout(rect=[0,0,1,0.96])
out = "E_channel_ceiling.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f7f9fc')
print("\nFigure saved:", out)
