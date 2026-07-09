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
VBLS — Comparison of two lifecycle-curve scaling variants

Variant A: scaled by T_max (biological maximum)
Variant B: scaled by T_ls  (mean lifespan)

Takeaway:
  - Variant A reproduces the upper bound of possible lifespan
  - Variant B predicts the observed mean lifespan
  - The difference visually shows that T_ls ≈ 0.5-0.7 × T_max

Sources for T_max / T_ls:
  AnAge Build 14: Mouse, Rat, Rabbit, Pig, Elephant, Human
  Literature: Dog, Cat, NMR, Horse, Chimpanzee
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

# ── Reference parameters (Human, T_ref = 77.6 yr — population calibration) ──
T_REF = 77.6   # mean human lifespan — calibration baseline

BASE = dict(
    mu=1.50, lam=0.30, sigma0=0.018, alpha=0.060,
    nu0=0.006, nu=0.060, rho=0.002,
    R_star=0.15, E_star=0.04,
)
Y0 = [0.10, 0.01, 0.96, 0.99]

# ── Species: (name, T_max, T_ls, source) ─────────────────────────────────────
# T_ls  — mean lifespan under controlled captive conditions
# T_max — verified biological maximum
SPECIES = [
    ("Mouse",       4.0,   2.0,  "AnAge / lab"),
    ("Rat",         3.8,   2.5,  "AnAge / lab"),
    ("Rabbit",     13.0,   8.0,  "AnAge / lit."),
    ("Dog",        24.0,  12.0,  "lit."),
    ("Cat",        30.0,  15.0,  "lit."),
    ("NMR",        31.0,  20.0,  "AnAge / lit."),
    ("Pig",        27.0,  15.0,  "AnAge / lit."),
    ("Horse",      57.0,  30.0,  "lit."),
    ("Chimpanzee", 59.0,  40.0,  "lit."),
    ("Elephant",   86.0,  60.0,  "AnAge / lit."),
    ("Human",     122.5,  77.6,  "AnAge / WHO"),
]

def Phi_in(t_norm):
    growth  = 1.0 - np.exp(-t_norm / 0.08)
    decline = np.exp(-max(0.0, t_norm - 0.55) / 0.30)
    return growth * decline

def Phi_cap():
    return 0.003

def scale_params(T_scale):
    """sc = T_REF / T_scale — scales all rate parameters"""
    sc = T_REF / T_scale
    return dict(
        mu=BASE['mu'], lam=BASE['lam']*sc,
        sigma0=BASE['sigma0']*sc, alpha=BASE['alpha']*sc,
        nu0=BASE['nu0']*sc, nu=BASE['nu']*sc, rho=BASE['rho']*sc,
        R_star=BASE['R_star'], E_star=BASE['E_star'],
    )

def system(t, y, p, T_scale):
    E,S,eta,R = y
    E=max(E,0.); S=np.clip(S,0.,1.); eta=np.clip(eta,0.,1.); R=np.clip(R,0.,1.)
    t_norm = t / T_scale
    diss = p['mu']*E**2 + p['lam']*E
    dE   = eta*Phi_in(t_norm) - diss
    dS   = max(0., p['sigma0']*diss - R*Phi_cap())
    deta = -p['alpha']*S*eta
    dR   = -p['nu0']*R - p['nu']*S*R + p['rho']*(1-S)*(1-R)
    return [dE,dS,deta,dR]

def run(T_scale):
    p = scale_params(T_scale)
    def ev(t,y,p,T): return min(y[3]-p['R_star'], y[0]-p['E_star'])
    ev.terminal=True; ev.direction=-1
    t_end = T_scale*1.15
    sol = solve_ivp(system,[0,t_end],Y0,args=(p,T_scale),
                    method='RK45',t_eval=np.linspace(0,t_end,4000),
                    events=ev,rtol=1e-8,atol=1e-10,max_step=T_scale/500)
    td = sol.t_events[0][0] if sol.t_events[0].size>0 else sol.t[-1]
    mask = sol.t <= td
    ch = "R" if float(np.interp(td,sol.t[mask],sol.y[3,mask]))<=p['R_star']+0.01 else "E"
    return sol.t[mask], sol.y[:,mask], td, ch

# ── Computing both variants ──────────────────────────────────────────────────
print("="*72)
print("  VBLS — Scaling comparison: T_max vs T_ls")
print(f"  Reference: Human T_ref = {T_REF} yr")
print("="*72)
print(f"\n{'Species':<14} {'T_max':>7} {'t_B(A)':>8} {'Δ_A%':>7} | {'T_ls':>7} {'t_B(B)':>8} {'Δ_B%':>7}")
print("-"*72)

res_A, res_B = {}, {}
for name, T_max, T_ls, src in SPECIES:
    tA, yA, tdA, chA = run(T_max)
    tB, yB, tdB, chB = run(T_ls)
    dA = (tdA - T_max) / T_max * 100
    dB = (tdB - T_ls)  / T_ls  * 100
    print(f"{name:<14} {T_max:>7.1f} {tdA:>8.1f} {dA:>+7.1f}% | {T_ls:>7.1f} {tdB:>8.1f} {dB:>+7.1f}%")
    res_A[name] = (tA, yA, tdA, T_max, chA)
    res_B[name] = (tB, yB, tdB, T_ls,  chB)

# ── Figure: 2 rows × n columns ────────────────────────────────────────────────
BG   = '#f7f9fc'
C_A  = '#2980b9'   # variant A — blue
C_B  = '#e74c3c'   # variant B — red
C_R  = '#27ae60'   # R(t)
C_S  = '#95a5a6'   # Σ(t) — gray, secondary

n    = len(SPECIES)
ncols= 6
nrows_sp = (n + ncols-1)//ncols   # rows needed per variant: 2
total_rows = 2*nrows_sp + 1        # variant A rows + variant B rows + summary panel

fig = plt.figure(figsize=(22, 16), facecolor=BG)

# Rows 0..nrows_sp-1: Variant A (T_max); rows nrows_sp..2*nrows_sp-1: Variant B (T_ls)
for row_variant, (res, variant_label, T_key, color) in enumerate([
    (res_A, 'A: T_max', 0, C_A),
    (res_B, 'B: T_ls',  1, C_B),
]):
    for i, (name, T_max, T_ls, src) in enumerate(SPECIES):
        row_local = i // ncols
        col_local = i % ncols
        fig_row = row_variant * nrows_sp + row_local
        subplot_idx = fig_row * ncols + col_local + 1
        ax = fig.add_subplot(total_rows, ncols, subplot_idx)
        ax.set_facecolor(BG)

        T_scale = T_max if T_key == 0 else T_ls
        t, y, td, T_sp, ch = res[name]
        E, S, eta, R = y

        ax.plot(t, R, color=C_R,  lw=2.0, label='R(t)')
        ax.plot(t, E, color=color, lw=1.5, ls='--', alpha=0.7, label='E(t)')
        ax.plot(t, S, color='#aaa', lw=1.0, ls=':', alpha=0.6)
        ax.axhline(BASE['R_star'], color=C_R, ls=':', lw=0.8, alpha=0.5)
        ax.axvline(td, color=color, ls='--', lw=1.2, alpha=0.7)
        ax.axvline(T_scale, color='#888', ls=':', lw=0.8, alpha=0.6)

        delta = (td - T_scale)/T_scale*100
        ax.set_title(
            f'{name}  ({variant_label})\n'
            f'T={T_scale:.0f}y → t_B={td:.1f}y  ({delta:+.0f}%)',
            fontsize=8, pad=2
        )
        ax.set_xlim(0, max(td, T_scale)*1.08)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('yr', fontsize=7)
        if col_local == 0:
            ax.set_ylabel(variant_label, fontsize=8, color=color, fontweight='bold')

# Summary panel — scatter t_B vs T_observed for both variants
ax_sum = fig.add_subplot(total_rows, 1, total_rows)
ax_sum.set_facecolor(BG)

T_maxs = [sp[1] for sp in SPECIES]
T_lss  = [sp[2] for sp in SPECIES]
tB_A   = [res_A[sp[0]][2] for sp in SPECIES]
tB_B   = [res_B[sp[0]][2] for sp in SPECIES]
names  = [sp[0] for sp in SPECIES]

ax_sum.scatter(T_maxs, tB_A, color=C_A, s=80, zorder=4,
               label=f'Variant A: sc=T_ref/T_max', edgecolors='white', lw=0.8)
ax_sum.scatter(T_lss,  tB_B, color=C_B, s=80, zorder=4,
               label=f'Variant B: sc=T_ref/T_ls',  edgecolors='white', lw=0.8, marker='s')

# Line of ideal agreement
all_T = T_maxs + T_lss
xmax  = max(all_T)*1.05
ax_sum.plot([0,xmax],[0,xmax], color='#aaa', lw=1.2, ls='--', label='Ideal t_B=T')

# Annotations
for i, name in enumerate(names):
    if name in ['Mouse','Human','Elephant','NMR']:
        ax_sum.annotate(name, (T_maxs[i], tB_A[i]),
                        xytext=(T_maxs[i]+1, tB_A[i]+2), fontsize=7.5, color=C_A)
        ax_sum.annotate(name, (T_lss[i],  tB_B[i]),
                        xytext=(T_lss[i]+1,  tB_B[i]-4), fontsize=7.5, color=C_B)

ax_sum.set_xlabel('T_observed (yr)', fontsize=9)
ax_sum.set_ylabel('t_B model (yr)', fontsize=9)
ax_sum.set_title(
    'Summary: model t_B vs observed T  |  11 species\n'
    'Systematic ~40% underestimate in Variant A is explained by calibration on human T_ls',
    fontsize=9, fontweight='bold'
)
ax_sum.legend(fontsize=9, framealpha=0.93)
ax_sum.set_xlim(0, xmax)
ax_sum.set_ylim(0, xmax)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('lifecycle_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.savefig('lifecycle_comparison.pdf', bbox_inches='tight', facecolor=BG)
print("\nFigures saved: lifecycle_comparison.png / .pdf")
