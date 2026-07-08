"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================

  Copyright (c) 2026 Vasiliy Avdeev.  All rights reserved.

  This work is licensed under the Creative Commons Attribution-NonCommercial-
  NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

  You are free to:
    Share — copy and redistribute the material in any medium or format
    The licensor cannot revoke these freedoms as long as you follow the
    license terms.

  Under the following terms:
    Attribution    — You must give appropriate credit, provide a link to
                     the license, and indicate if changes were made.  You
                     may do so in any reasonable manner, but not in any way
                     that suggests the licensor endorses you or your use.
    NonCommercial  — You may not use the material for commercial purposes.
    NoDerivatives  — If you remix, transform, or build upon the material,
                     you may not distribute the modified material.
    No additional restrictions — You may not apply legal terms or
                     technological measures that legally restrict others
                     from doing anything the license permits.

  Notices:
    You do not have to comply with the license for elements of the material
    in the public domain or where your use is permitted by an applicable
    exception or limitation.

    No warranties are given.  The license may not give you all of the
    permissions necessary for your intended use.  For example, other rights
    such as publicity, privacy, or moral rights may limit how you use the
    material.

  Full license text:        https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
  Human-readable summary:   https://creativecommons.org/licenses/by-nc-nd/4.0/

  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================
VBLS — Lifecycle curves for 9 species with telomere overlay  v2
======================================================================
For each species (subplot 3×3):

Left axis (normalized values 0..1):
  R(t)   repair reserve                 — purple, solid
  E(t)   integrity potential (norm.)    — blue, solid
  T(t)   telomere reserve (Whittemore)  — green, dashed
  R*     VBLS bifurcation threshold     — red, dotted
  E*     viability threshold (norm.)    — blue, dotted
  R*univ universal telomere threshold   — green, dotted
  ●/◆    threshold crossing moment; ◆ = first (= death)

Right axis:
  dS/dt  damage accumulation rate dSigma/dt — orange

Background zones: Phase I / II / III

Data: Whittemore et al. (2019) PNAS, Table S1/S2
Parameters: from mde_human_lifespan.py, scaled to LS_REF/T_max

Run: python lifecycle_telomere.py
Out: lifecycle_telomere.pdf / .png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         9.5,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8.5,
    'axes.spines.top':   False,
    'axes.grid':         True,
    'grid.alpha':        0.15,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})

COL_R    = '#8e44ad'
COL_E    = '#1a6faf'
COL_TEL  = '#27ae60'
COL_RSTAR= '#c0392b'
COL_DSIG = '#e67e22'
PH_COL   = ['#ddeeff', '#fff9e6', '#ffe6e6']

# ── Species data ──────────────────────────────────────────────────────────────
# (name, L0_kb, rate_bp_yr, T_max, res_pct_at_T_max, cluster)
SPECIES = [
    ("Mouse",              50.0,   7000, 4.0,    44.0, 'R'),
    ("Goat",               10.4,    363, 20.8,   27.4, 'R'),
    ("Audouin's gull",     35.0,    771, 25.0,   44.9, 'R'),
    ("Reindeer",           19.8,    531, 21.8,   41.5, 'R'),
    ("Griffon vulture",    19.8,    209, 41.4,   56.3, 'R'),
    ("Bottlenose dolphin", 90.7,    766, 51.6,   56.4, 'R'),
    ("American flamingo",  21.0,    105, 60.0,   70.0, 'E'),
    ("Sumatran elephant",  36.3,    109, 65.5,   80.3, 'E'),
    ("Human",              15.0,     71, 122.5,  42.0, 'R'),
]

nu_star    = {sp[0]: sp[2] / (sp[1] * 1000) for sp in SPECIES}
R_STAR_OBS = {sp[0]: sp[4] / 100 for sp in SPECIES}
cluster1   = [sp[0] for sp in SPECIES if sp[5] == 'R']
R_UNIV     = np.mean([R_STAR_OBS[n] for n in cluster1])

# ── VBLS: calibrated human params (mde_human_lifespan.py), scaled per species ──
LS_REF = 122.5

BASE_ABS = dict(mu=1.50, lam=0.30, sigma0=0.0162, alpha=0.030,
                nu0=0.006, nu=0.06, rho=0.002,
                Phi_max=3.0, Phi_cap=0.042,
                T_mature=20.0,
                R_star=0.15, E_star=0.04)

Y0 = [0.10, 0.01, 0.96, 0.99]

# Denominator of g(E) = (E − E*) / (E₀ − E*), formula 4.2
E_RANGE = Y0[0] - BASE_ABS['E_star']  # 0.10 − 0.04 = 0.06

def scaled_args(T_max):
    sc = LS_REF / T_max
    p  = BASE_ABS
    T_mat = p['T_mature'] * T_max / LS_REF
    return (p['mu'], p['lam']*sc, p['sigma0']*sc, p['alpha']*sc,
            p['nu0']*sc, p['nu']*sc, p['rho']*sc,
            p['Phi_max'], p['Phi_cap'], T_mat)

def rhs(t, y, mu, lam, s0, alpha, nu0, nu, rho, Phi_max, Phi_cap, T_mat):
    E, S, eta, R = y
    S   = np.clip(S,   0.0, 1.0)
    eta = np.clip(eta, 0.0, 1.0)
    R   = np.clip(R,   0.0, 1.0)
    phi  = Phi_max * min(1.0, t / max(T_mat, 1e-9))
    dE   = eta * phi - mu * E**2 - lam * E
    dS   = max(0.0, s0 * (mu * E**2 + lam * E) - R * Phi_cap)
    deta = -alpha * S * eta
    # Rosen closure function g(E), formula 4.2: gates repair restoration
    g_E  = max(0.0, min(1.0, (E - BASE_ABS['E_star']) / E_RANGE))
    dR   = -nu0 * R - nu * S * R + rho * g_E * (1.0 - S) * (1.0 - R)
    return [dE, dS, deta, dR]

def run_vbls(T_max):
    args = scaled_args(T_max)

    def ev_R(t, y, *a): return y[3] - BASE_ABS['R_star']
    ev_R.terminal  = True
    ev_R.direction = -1

    t_eval = np.linspace(0, T_max * 2.5, 1500)
    sol = solve_ivp(rhs, [0, T_max * 2.5], Y0, args=args,
                    method='RK45', t_eval=t_eval,
                    events=[ev_R], dense_output=True)
    return sol

def first_crossing(t, arr, thr, direction='down'):
    idx = np.where(arr <= thr)[0] if direction=='down' else np.where(arr >= thr)[0]
    return float(t[idx[0]]) if len(idx) else None

def find_phases(t, E_norm, R_norm, tB):
    t_peak  = float(t[np.argmax(E_norm)])
    col_idx = np.where((t > t_peak) & (R_norm < 0.50))[0]
    t_col   = float(t[col_idx[0]]) if len(col_idx) else tB * 0.75
    return t_peak, t_col

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(17, 15))
fig.subplots_adjust(hspace=0.55, wspace=0.45,
                    left=0.07, right=0.97, top=0.91, bottom=0.10)

print("Integrating VBLS for 9 species...")

for idx, (name, L0, rate, T_max, res_pct, cluster) in enumerate(SPECIES):
    row, ci = divmod(idx, 3)
    ax  = axes[row, ci]
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)

    # ── ODE ────────────────────────────────────────────────────────────
    sol   = run_vbls(T_max)
    t     = sol.t
    E, Sigma, eta, R = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    tB    = float(t[-1])

    E_max  = max(E.max(), 1e-9)
    E_norm = E / E_max
    R_norm = R

    t_peak, t_col = find_phases(t, E_norm, R_norm, tB)

    # ── Channel crossings ───────────────────────────────────────────────
    # R channel: use event time directly (more accurate than array scan)
    if sol.t_events and len(sol.t_events[0]) > 0:
        t_cross_R = float(sol.t_events[0][0])
    else:
        t_cross_R = first_crossing(t, R_norm, BASE_ABS['R_star'] * 1.02, 'down')
    E_star_norm = BASE_ABS['E_star'] / E_max
    t_cross_E = first_crossing(t, E_norm, E_star_norm * 1.02, 'down')

    if t_cross_R is not None and t_cross_E is not None:
        first_ch = 'R' if t_cross_R <= t_cross_E else 'E'
    elif t_cross_R is not None:
        first_ch = 'R'
    elif t_cross_E is not None:
        first_ch = 'E'
    else:
        first_ch = '?'

    rstr = f'{t_cross_R:.1f}' if t_cross_R is not None else '-'
    estr = f'{t_cross_E:.1f}' if t_cross_E is not None else '-'
    print(f"  {name:<22} T_max={T_max:5.1f}  tB={tB:.1f}  "
          f"t_R*={rstr}  t_E*={estr}  ch={first_ch}")

    # ── dSigma/dt — directly from RHS (no numerical noise) ──────────────
    sc_sp   = LS_REF / T_max
    s0sc    = BASE_ABS['sigma0'] * sc_sp
    lam_sc  = BASE_ABS['lam']    * sc_sp
    diss    = s0sc * (BASE_ABS['mu'] * E**2 + lam_sc * E)
    dS      = np.maximum(0.0, diss - R * BASE_ABS['Phi_cap'])
    dS_max  = max(dS.max(), 1e-12)
    dS_norm = dS / dS_max

    # ── Phase background ─────────────────────────────────────────────────
    ax.axvspan(0,       t_peak, alpha=0.22, color=PH_COL[0], zorder=0, lw=0)
    ax.axvspan(t_peak,  t_col,  alpha=0.22, color=PH_COL[1], zorder=0, lw=0)
    ax.axvspan(t_col,   tB,     alpha=0.22, color=PH_COL[2], zorder=0, lw=0)

    # ── dSigma/dt (right axis) ───────────────────────────────────────────
    ax2.plot(t, dS_norm, color=COL_DSIG, lw=1.4, alpha=0.65, zorder=2)
    ax2.set_ylim(-0.05, 1.50)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(['0', '', '1'], fontsize=7, color=COL_DSIG)
    ax2.tick_params(axis='y', colors=COL_DSIG, labelsize=7)
    ax2.spines['right'].set_edgecolor(COL_DSIG)
    ax2.spines['right'].set_alpha(0.45)
    if ci == 2:
        ax2.set_ylabel('$d\\Sigma/dt$ (norm.)', fontsize=7.5, color=COL_DSIG)

    # ── Main curves (left axis) ───────────────────────────────────────────
    ax.plot(t, R_norm, color=COL_R,  lw=2.1, zorder=4)
    ax.plot(t, E_norm, color=COL_E,  lw=1.7, zorder=4, alpha=0.80)

    # Telomere T(t)
    t_tel = np.linspace(0, tB * 1.02, 400)
    T_tel = np.clip(1.0 - nu_star[name] * t_tel, 0, 1)
    ax.plot(t_tel, T_tel, color=COL_TEL, lw=1.8, ls='--', zorder=5)
    T_obs = max(0.0, 1.0 - nu_star[name] * T_max)
    ax.scatter([T_max], [T_obs], color=COL_TEL, s=45, zorder=7,
               edgecolors='white', lw=0.8)

    # ── Thresholds ────────────────────────────────────────────────────────
    ax.axhline(BASE_ABS['R_star'], color=COL_RSTAR, lw=1.1, ls=':', alpha=0.85, zorder=3)
    ax.text(tB*0.02, BASE_ABS['R_star']+0.022, f"$R^*$={BASE_ABS['R_star']}",
            color=COL_RSTAR, fontsize=6.8, va='bottom')

    ax.axhline(E_star_norm, color=COL_E, lw=0.9, ls=':', alpha=0.60, zorder=3)
    ax.text(tB*0.48, E_star_norm+0.022, "$E^*$ (norm.)",
            color=COL_E, fontsize=6.8, va='bottom')

    ax.axhline(R_UNIV, color=COL_TEL, lw=0.9, ls=':', alpha=0.55, zorder=3)
    ax.text(tB*0.55, R_UNIV+0.022, f"$R^*_{{univ}}$={R_UNIV:.2f}",
            color=COL_TEL, fontsize=6.8, va='bottom')

    # ── Crossing markers ──────────────────────────────────────────────────
    def mark_cross(tx, yv, col, is_first, label_above=False):
        mkr  = 'D' if is_first else 'o'
        sz   = 75  if is_first else 45
        ax.scatter([tx], [yv], color=col, s=sz, zorder=9,
                   marker=mkr, edgecolors='white', lw=0.8,
                   alpha=1.0 if is_first else 0.65)
        tag = f'$t={tx:.1f}$ [1]' if is_first else f'$t={tx:.1f}$'
        dy  = 0.10 if label_above else -0.12
        ax.annotate(tag,
                    xy=(tx, yv), xytext=(tx - tB*0.12, yv + dy),
                    fontsize=6.8, color=col,
                    arrowprops=dict(arrowstyle='->', color=col, lw=0.7, alpha=0.7))

    if t_cross_R is not None:
        mark_cross(t_cross_R, BASE_ABS['R_star'], COL_RSTAR,
                   is_first=(first_ch=='R'), label_above=False)

    if t_cross_E is not None:
        mark_cross(t_cross_E, E_star_norm, COL_E,
                   is_first=(first_ch=='E'), label_above=True)

    # ── t_B and T_max verticals ───────────────────────────────────────────
    ax.axvline(tB,    color='#444',  lw=1.0, ls=':', zorder=5)
    ax.axvline(T_max, color=COL_TEL, lw=0.8, ls='--', alpha=0.40, zorder=4)
    ax.text(tB*0.98, 0.94, f'$t_B={tB:.1f}$',
            ha='right', va='top', fontsize=7.2, color='#333')

    # ── Phase labels ──────────────────────────────────────────────────────
    for xc, lbl, fc in [(t_peak/2,'I','#2980b9'),
                         ((t_peak+t_col)/2,'II','#b7950b'),
                         ((t_col+tB)/2,'III','#c0392b')]:
        ax.text(xc, 1.04, lbl, ha='center', va='bottom', fontsize=8,
                color=fc, fontweight='bold',
                transform=ax.get_xaxis_transform())

    # ── Channel badge (reflects actual ODE result, not hardcoded cluster) ───
    bc = '#1a6faf' if first_ch == 'R' else '#c0392b'
    bt = 'ch: R(t)' if first_ch == 'R' else 'ch: E(t)'
    ax.text(0.98, 0.06, f'[ODE] {bt}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5, color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=bc, ec='none', alpha=0.88))
    # Note when ODE result differs from expected channel (telomere analysis)
    if first_ch != cluster:
        ax.text(0.98, 0.15, f'expected: {cluster}(t) channel', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=6.5, color='#c0392b',
                style='italic')

    # ── Axis limits & labels ──────────────────────────────────────────────
    ax.set_xlim(-tB*0.02, tB*1.08)
    ax.set_ylim(-0.06, 1.12)
    ax.set_xlabel('Time (years)', fontsize=8.5)
    if ci == 0:
        ax.set_ylabel('Normalized value', fontsize=8.5)

    tc = '#1a6faf' if cluster=='R' else '#c0392b'
    ax.set_title(f'{name}   ($T_{{max}}={T_max}$ yr)',
                 fontsize=9.5, fontweight='bold', color=tc, pad=5)

# ── Shared legend ─────────────────────────────────────────────────────────────
leg = [
    mpatches.Patch(color=PH_COL[0], alpha=0.55, label='Phase I — Growth'),
    mpatches.Patch(color=PH_COL[1], alpha=0.55, label='Phase II — Plateau'),
    mpatches.Patch(color=PH_COL[2], alpha=0.55, label='Phase III — Degradation'),
    plt.Line2D([0],[0], color=COL_R,    lw=2.1,           label='$R(t)$ — repair reserve (VBLS)'),
    plt.Line2D([0],[0], color=COL_E,    lw=1.7, alpha=0.8, label='$E(t)$ — integrity potential (VBLS, norm.)'),
    plt.Line2D([0],[0], color=COL_TEL,  lw=1.8, ls='--',  label='$T(t)$ — telomere reserve (Whittemore 2019)'),
    plt.Line2D([0],[0], color=COL_DSIG, lw=1.4, alpha=0.7, label='$d\\Sigma/dt$ — damage accumulation rate (right axis)'),
    plt.Line2D([0],[0], color=COL_RSTAR, lw=1.1, ls=':',  label=f'$R^*={BASE_ABS["R_star"]}$ — VBLS threshold'),
    plt.Line2D([0],[0], color=COL_E,    lw=0.9, ls=':',   label='$E^*$ — viability threshold (norm.)'),
    plt.Line2D([0],[0], color=COL_TEL,  lw=0.9, ls=':',   label=f'$R^*_{{univ}}={R_UNIV:.2f}$ — telomere threshold'),
    plt.Line2D([0],[0], color='#555', marker='D', ms=6, ls='', label='[1] first channel = death event $t_B$'),
    plt.Line2D([0],[0], color='#555', marker='o', ms=5, ls='', label='second channel (non-limiting)'),
]

fig.legend(handles=leg, loc='lower center', ncol=4, fontsize=8.0,
           framealpha=0.93, bbox_to_anchor=(0.5, -0.03),
           title='Legend  |  [1] = first active channel = death event t_B',
           title_fontsize=8.5)

fig.suptitle(
    'Lifecycle curves for 9 species: VBLS + telomere dynamics + damage accumulation rate $d\\Sigma/dt$\n'
    'Whittemore et al. (2019) PNAS  |  '
    'Blue — $R(t)$ channel,  red — $E(t)$ channel  |  [1] = first active threshold',
    fontsize=11.0, fontweight='bold', y=0.975
)

plt.savefig('lifecycle_telomere.pdf', bbox_inches='tight')
plt.savefig('lifecycle_telomere.png', bbox_inches='tight', dpi=180)
plt.close()
print("\nSaved: lifecycle_telomere.pdf / .png")
