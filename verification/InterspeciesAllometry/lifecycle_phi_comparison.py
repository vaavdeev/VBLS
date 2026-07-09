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
VBLS — Lifecycle curves: comparison of two Phi_in variants

Phi_in A (original):  starts at 0 at t=0 — models systems where
                      the resource flow builds up gradually
                      (egg-laying species, unicellular organisms, seeds)

Phi_in B (phi_birth): starts at phi_birth > 0 — models live-bearing
                      mammals, where a maternal resource flow
                      (milk, yolk-sac reserves) is already present
                      at birth

The difference visually demonstrates that the "moment of birth" in
VBLS theory is a parameter, not a fixed point.

Scaling: T_ls (variant B from lifecycle_comparison.py)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         9,
    'axes.titlesize':    9.5,
    'axes.labelsize':    8.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.15,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

T_REF = 77.6
BASE = dict(
    mu=1.50, lam=0.30, sigma0=0.018, alpha=0.060,
    nu0=0.006, nu=0.060, rho=0.002,
    R_star=0.15, E_star=0.04,
)
Y0 = [0.10, 0.01, 0.96, 0.99]

# Scaling by T_ls
SPECIES = [
    ("Mouse",       4.0,   2.0),
    ("Rat",         3.8,   2.5),
    ("Rabbit",     13.0,   8.0),
    ("Dog",        24.0,  12.0),
    ("Cat",        30.0,  15.0),
    ("NMR",        31.0,  20.0),
    ("Pig",        27.0,  15.0),
    ("Horse",      57.0,  30.0),
    ("Chimpanzee", 59.0,  40.0),
    ("Elephant",   86.0,  60.0),
    ("Human",     122.5,  77.6),
]

# ── Two Phi_in variants ─────────────────────────────────────────────────────
PHI_BIRTH = 0.45   # initial flow for live-bearing mammals

def Phi_in_A(t_norm):
    """Original: starts from zero. Egg-laying species, unicellular organisms."""
    growth  = 1.0 - np.exp(-t_norm / 0.08)
    decline = np.exp(-max(0.0, t_norm - 0.55) / 0.30)
    return growth * decline

def Phi_in_B(t_norm):
    """With an initial flow phi_birth. Live-bearing mammals."""
    growth  = PHI_BIRTH + (1.0 - PHI_BIRTH) * (1.0 - np.exp(-t_norm / 0.08))
    decline = np.exp(-max(0.0, t_norm - 0.55) / 0.30)
    return growth * decline

def scale_params(T_ls):
    sc = T_REF / T_ls
    return dict(
        mu=BASE['mu'], lam=BASE['lam']*sc,
        sigma0=BASE['sigma0']*sc, alpha=BASE['alpha']*sc,
        nu0=BASE['nu0']*sc, nu=BASE['nu']*sc, rho=BASE['rho']*sc,
        R_star=BASE['R_star'], E_star=BASE['E_star'],
    )

def system(t, y, p, T_ls, phi_func):
    E,S,eta,R = y
    E=max(E,0.); S=np.clip(S,0.,1.); eta=np.clip(eta,0.,1.); R=np.clip(R,0.,1.)
    t_norm = t / T_ls
    diss = p['mu']*E**2 + p['lam']*E
    dE   = eta * phi_func(t_norm) - diss
    dS   = max(0., p['sigma0']*diss - R*0.003)
    deta = -p['alpha']*S*eta
    dR   = -p['nu0']*R - p['nu']*S*R + p['rho']*(1-S)*(1-R)
    return [dE,dS,deta,dR]

def run(T_ls, phi_func):
    p = scale_params(T_ls)
    def ev(t,y,p,T,phi): return min(y[3]-p['R_star'], y[0]-p['E_star'])
    ev.terminal=True; ev.direction=-1
    t_end = T_ls*1.15
    sol = solve_ivp(system,[0,t_end],Y0,args=(p,T_ls,phi_func),
                    method='RK45',t_eval=np.linspace(0,t_end,4000),
                    events=ev,rtol=1e-8,atol=1e-10,max_step=T_ls/500)
    td = sol.t_events[0][0] if sol.t_events[0].size>0 else sol.t[-1]
    mask = sol.t <= td
    return sol.t[mask], sol.y[:,mask], td

# ── Computation ──────────────────────────────────────────────────────────────
print("="*70)
print("  VBLS — Phi_in(0)=0 vs Phi_in(0)=phi_birth")
print(f"  phi_birth = {PHI_BIRTH}  |  scaling by T_ls")
print("="*70)
print(f"\n{'Species':<14} {'T_ls':>7} | {'t_B(A)':>8} {'ΔA%':>7} | {'t_B(B)':>8} {'ΔB%':>7} | {'Δt_B':>7}")
print("-"*65)

results = {}
for name, T_max, T_ls in SPECIES:
    tA, yA, tdA = run(T_ls, Phi_in_A)
    tB, yB, tdB = run(T_ls, Phi_in_B)
    dA = (tdA - T_ls)/T_ls*100
    dB = (tdB - T_ls)/T_ls*100
    dt = tdB - tdA
    print(f"{name:<14} {T_ls:>7.1f} | {tdA:>8.1f} {dA:>+7.1f}% | {tdB:>8.1f} {dB:>+7.1f}% | {dt:>+7.2f}")
    results[name] = (tA, yA, tdA, tB, yB, tdB, T_ls)

# ── Visualization ────────────────────────────────────────────────────────────
BG   = '#f7f9fc'
C_A  = '#e74c3c'    # Phi_in A — red (dip)
C_B  = '#2980b9'    # Phi_in B — blue (no dip)
C_R  = '#27ae60'    # R(t)

n = len(SPECIES)
ncols = 6
nrows = (n + ncols-1)//ncols

fig = plt.figure(figsize=(22, 9), facecolor=BG)

for idx, (name, T_max, T_ls) in enumerate(SPECIES):
    tA, yA, tdA, tB, yB, tdB, _ = results[name]
    EA, SA, etaA, RA = yA
    EB, SB, etaB, RB = yB

    ax = fig.add_subplot(nrows, ncols, idx+1)
    ax.set_facecolor(BG)

    # Variant A — red
    ax.plot(tA, EA, color=C_A, lw=2.0, alpha=0.85,
            label=f'E, Φ(0)=0')
    ax.plot(tA, RA, color=C_A, lw=1.5, ls='--', alpha=0.6,
            label=f'R, Φ(0)=0')
    ax.axvline(tdA, color=C_A, ls=':', lw=1.0, alpha=0.5)

    # Variant B — blue
    ax.plot(tB, EB, color=C_B, lw=2.0, alpha=0.85,
            label=f'E, Φ(0)={PHI_BIRTH}')
    ax.plot(tB, RB, color=C_B, lw=1.5, ls='--', alpha=0.6,
            label=f'R, Φ(0)={PHI_BIRTH}')
    ax.axvline(tdB, color=C_B, ls=':', lw=1.0, alpha=0.5)

    # Mark the dip for variant A
    if len(EA) > 10:
        dip_idx = np.argmin(EA[:int(len(EA)*0.15)])
        t_dip   = tA[dip_idx]
        E_dip   = EA[dip_idx]
        if E_dip < Y0[0] - 0.005:
            ax.annotate('', xy=(t_dip, E_dip),
                        xytext=(t_dip, Y0[0]),
                        arrowprops=dict(arrowstyle='->', color=C_A, lw=1.2))

    dt = tdB - tdA
    ax.set_title(
        f'{name}  T_ls={T_ls:.0f}y\n'
        f'$t_B$: {tdA:.1f}y → {tdB:.1f}y  (Δ={dt:+.1f}y)',
        fontsize=8, pad=2
    )
    ax.set_xlim(0, max(tdA, tdB)*1.08)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('yr', fontsize=7)

    if idx == 0:
        ax.legend(fontsize=6.5, loc='center right',
                  framealpha=0.85, ncol=1)

# Empty cells
for idx in range(n, nrows*ncols):
    fig.add_subplot(nrows, ncols, idx+1).set_visible(False)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('lifecycle_phi_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.savefig('lifecycle_phi_comparison.pdf', bbox_inches='tight', facecolor=BG)
print("\nFigures saved: lifecycle_phi_comparison.png / .pdf")
