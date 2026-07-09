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
VBLS — Verification against Cagan et al. (2022) Nature
Somatic mutations as direct measurement of Sigma(t)
=================================================================

3 Figures:
  Fig 1 — Sigma(t) trajectories: Cagan data (scatter) + linear regression + VBLS curve
          for species with sufficient age coverage (3-column grid)
  Fig 2 — Universal Sigma*: mutation burden at end of observed life (bar chart)
          analogue of R*_univ from the Whittemore verification
  Fig 4 — Human detail: VBLS Sigma(t) over all 28 crypts
          + life phases + dSigma/dt on right axis

  Note: the rate x T_max power-law is owned exclusively by cagan_rateXtmax.py
  (15-species slope fit) and is intentionally not reproduced here.

Data: Cagan et al. (2022) Nature 604:517-524, Supplementary Table 4
Input file: Table_4.csv

Out: cagan_fig1 / fig2 / fig4 .pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.18,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'figure.dpi':        150,
})

COL_DATA  = '#2c3e50'   # data points — dark
COL_REG   = '#7f8c8d'   # linear regression — grey
COL_VBLS  = '#8e44ad'   # VBLS Sigma(t) — purple
COL_DSIG  = '#e67e22'   # dSigma/dt — orange
COL_UNIV  = '#c0392b'   # universal threshold — red
PH_COL    = ['#ddeeff', '#fff9e6', '#ffe6e6']

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('Table_4.csv', sep=';', na_values=['NA','na',''])
df = df.dropna(subset=['Age_years', 'Substitution_burden'])
df['Age_years']           = pd.to_numeric(df['Age_years'],           errors='coerce')
df['Substitution_burden'] = pd.to_numeric(df['Substitution_burden'], errors='coerce')
df['Substitution_rate_per_year'] = pd.to_numeric(df['Substitution_rate_per_year'], errors='coerce')
df = df.dropna(subset=['Age_years', 'Substitution_burden'])

# ── Species metadata (max lifespan from AnAge / literature) ──────────────────
TMAX = {
    'Human':                  122.5,
    'Mouse':                    4.0,
    'Rat':                      3.8,
    'Naked mole-rat':          37.0,
    'Dog':                     20.0,
    'Cat':                     17.0,
    'Horse':                   57.0,
    'Tiger':                   26.0,
    'Cow':                     30.0,
    'Rabbit':                  13.0,
    'Giraffe':                 39.5,
    'Lion':                    29.0,
    'Ferret':                  10.0,
    'Ring-tailed lemur':       27.0,
    'Black-and-white colobus': 30.0,
    'Harbour porpoise':        21.5,
}

# Species colours for Fig 1
SP_COLORS = {
    'Human':        '#1565C0',
    'Mouse':        '#F44336',
    'Rat':          '#E91E63',
    'Naked mole-rat':'#4CAF50',
    'Dog':          '#FF9800',
    'Cat':          '#9C27B0',
    'Horse':        '#00BCD4',
    'Tiger':        '#FF5722',
    'Cow':          '#607D8B',
    'Rabbit':       '#8BC34A',
    'Giraffe':      '#FFC107',
    'Lion':         '#795548',
    'Ferret':       '#03A9F4',
    'Ring-tailed lemur':       '#673AB7',
    'Black-and-white colobus': '#009688',
}

# ── VBLS parameters (calibrated for human, scaled by LS_REF/T_max) ───────────
LS_REF   = 122.5
BASE_ABS = dict(mu=1.50, lam=0.30, sigma0=0.0162, alpha=0.030,
                nu0=0.006, nu=0.06, rho=0.002,
                Phi_max=3.0, Phi_cap=0.042,
                T_mature=20.0, R_star=0.15, E_star=0.04)
Y0 = [0.10, 0.01, 0.96, 0.99]

# Denominator of g(E) = (E - E*) / (E0 - E*), formula 4.2
E_RANGE = Y0[0] - BASE_ABS['E_star']  # 0.10 - 0.04 = 0.06

def scaled_args(T_max):
    sc = LS_REF / T_max
    p  = BASE_ABS
    return (p['mu'], p['lam']*sc, p['sigma0']*sc, p['alpha']*sc,
            p['nu0']*sc, p['nu']*sc, p['rho']*sc,
            p['Phi_max'], p['Phi_cap'], p['T_mature']*T_max/LS_REF)

def rhs(t, y, mu, lam, s0, alpha, nu0, nu, rho, Phi_max, Phi_cap, T_mat):
    E, S, eta, R = y
    S=np.clip(S,0,1); eta=np.clip(eta,0,1); R=np.clip(R,0,1)
    phi  = Phi_max * min(1.0, t / max(T_mat, 1e-9))
    dE   = eta*phi - mu*E**2 - lam*E
    dS   = max(0.0, s0*(mu*E**2+lam*E) - R*Phi_cap)
    deta = -alpha*S*eta
    # Recovery-potential function g(E), formula 4.2: gates repair restoration
    g_E  = max(0.0, min(1.0, (E - BASE_ABS['E_star']) / E_RANGE))
    dR   = -nu0*R - nu*S*R + rho*g_E*(1-S)*(1-R)
    return [dE, dS, deta, dR]

def run_vbls(T_max, n=800):
    args = scaled_args(T_max)
    def ev_R(t,y,*a): return y[3] - BASE_ABS['R_star']
    ev_R.terminal = True; ev_R.direction = -1
    t_eval = np.linspace(0, T_max*2.0, n)
    sol = solve_ivp(rhs, [0, T_max*2.0], Y0, args=args,
                    t_eval=t_eval, method='RK45',
                    events=[ev_R], dense_output=True)
    return sol

def vbls_sigma_scaled(T_max, burden_at_end):
    """Return t, Sigma scaled to match observed mutation burden at t_B."""
    sol = run_vbls(T_max)
    t   = sol.t
    S   = sol.y[1]
    tB  = float(t[-1])
    S_end = max(S[-1], 1e-9)
    S_scaled = S / S_end * burden_at_end
    return t, S_scaled, tB

def vbls_dsigma(T_max):
    """Compute dSigma/dt directly from RHS."""
    sol  = run_vbls(T_max)
    t_s  = sol.t
    E    = sol.y[0]
    R    = sol.y[3]
    sc   = LS_REF / T_max
    s0sc = BASE_ABS['sigma0'] * sc
    lsc  = BASE_ABS['lam'] * sc
    diss = s0sc * (BASE_ABS['mu'] * E**2 + lsc * E)
    dSdt = np.maximum(0, diss - R * BASE_ABS['Phi_cap'])
    return t_s, dSdt

# ── Per-species mean burden at oldest age (Sigma* equivalent) ─────────────────
last_age  = df.groupby('Species')['Age_years'].max().rename('max_age')
df2       = df.merge(last_age, on='Species')
sigma_end = (df2[df2['Age_years'] == df2['max_age']]
             .groupby('Species')['Substitution_burden'].mean())

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Sigma(t) per species: data + linear regression + VBLS
# Species with age span > 1 year (multiple individuals at different ages)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 1...")

sp_span = df.groupby('Species')['Age_years'].apply(lambda x: x.max()-x.min())
sp_use  = sp_span[sp_span > 1.0].index.tolist()
sp_use  = [s for s in sp_use if s in TMAX and s != 'Harbour porpoise']
sp_use  = sorted(sp_use, key=lambda s: TMAX.get(s, 0))

n_sp = len(sp_use)
ncols = 3
nrows = (n_sp + ncols - 1) // ncols

fig1, axes1 = plt.subplots(nrows, ncols, figsize=(15, 4.5*nrows))
fig1.subplots_adjust(hspace=0.50, wspace=0.38,
                     left=0.07, right=0.97, top=0.91, bottom=0.07)
axes1_flat = axes1.flatten()

for i, sp in enumerate(sp_use):
    ax = axes1_flat[i]
    g  = df[df['Species']==sp].copy()
    T_max = TMAX[sp]
    col   = SP_COLORS.get(sp, '#555')

    # ── Data points (each crypt = one point) ──────────────────────────
    ax.scatter(g['Age_years'], g['Substitution_burden'],
               color=col, s=35, alpha=0.65, zorder=4,
               edgecolors='white', lw=0.5,
               label='Crypts (Cagan 2022)')

    # ── Mean per individual ────────────────────────────────────────────
    ind_mean = g.groupby(['Individual','Age_years'])['Substitution_burden'].mean().reset_index()
    ax.scatter(ind_mean['Age_years'], ind_mean['Substitution_burden'],
               color=col, s=80, zorder=5,
               edgecolors='white', lw=1.0, marker='D',
               label='Individual mean')

    # ── Linear regression ─────────────────────────────────────────────
    if len(ind_mean) >= 2:
        slope, intercept, r, p, _ = linregress(ind_mean['Age_years'],
                                               ind_mean['Substitution_burden'])
        t_reg = np.linspace(0, T_max * 0.60, 200)
        ax.plot(t_reg, intercept + slope*t_reg,
                color=COL_REG, lw=1.4, ls='--', zorder=3,
                label=f'Linear regression ($R^2={r**2:.2f}$)')

    # ── VBLS Sigma(t) scaled to observed end-burden ────────────────────
    end_b = sigma_end.get(sp, g['Substitution_burden'].max())
    t_vbls, S_vbls, tB = vbls_sigma_scaled(T_max, end_b)
    t_obs_max = g['Age_years'].max()
    mask = t_vbls <= t_obs_max * 1.15
    ax.plot(t_vbls[mask], S_vbls[mask],
            color=COL_VBLS, lw=2.2, zorder=5,
            label='VBLS $\\Sigma(t)$')
    # Dashed continuation to t_B
    mask2 = t_vbls >= t_obs_max * 1.10
    ax.plot(t_vbls[mask2], S_vbls[mask2],
            color=COL_VBLS, lw=1.4, ls=':', zorder=4, alpha=0.55)

    # ── t_B marker ────────────────────────────────────────────────────
    ax.axvline(tB, color='#999', lw=0.9, ls=':', zorder=2)
    ax.axvline(T_max, color=col, lw=0.8, ls='--', alpha=0.40, zorder=2)

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xlim(-T_max*0.02, max(tB*1.05, t_obs_max*1.3))
    ax.set_ylim(-50, max(g['Substitution_burden'].max()*1.25, end_b*1.15))
    ax.set_xlabel('Age (years)', fontsize=9)
    if i % ncols == 0:
        ax.set_ylabel('Mutations per genome', fontsize=9)
    ax.set_title(f'{sp}  ($T_{{max}}={T_max}$ yr)', fontsize=10,
                 fontweight='bold', color=col, pad=4)

    if i == 0:
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.92)

# Hide unused subplots
for j in range(i+1, len(axes1_flat)):
    axes1_flat[j].set_visible(False)

fig1.suptitle(
    'Somatic mutation accumulation: Cagan et al. (2022) data + VBLS $\\Sigma(t)$\n'
    'Dots = individual crypts;  ◆ = individual mean;  '
    '-- grey = linear regression;  — purple = VBLS;  : = extrapolation to $t_B$',
    fontsize=11, fontweight='bold', y=0.98)

plt.savefig('cagan_fig1_trajectories.pdf', bbox_inches='tight')
plt.savefig('cagan_fig1_trajectories.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: cagan_fig1_trajectories")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Universal Sigma*: mutation burden at end of observed life
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2...")

sp_ord = sigma_end.sort_values().index.tolist()
vals   = sigma_end[sp_ord].values
cols2  = [SP_COLORS.get(s, '#888') for s in sp_ord]
sigma_univ = sigma_end.mean()
sigma_std  = sigma_end.std()

fig2, ax2 = plt.subplots(figsize=(13, 5.5))
fig2.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.18)

bars = ax2.bar(range(len(sp_ord)), vals, color=cols2, alpha=0.78,
               edgecolor='white', lw=0.8, zorder=3)

ax2.axhline(sigma_univ, color=COL_UNIV, lw=1.8, ls='--', zorder=4,
            label=f'$\\Sigma^*_{{univ}}={sigma_univ:.0f}$ mut. (mean)')
ax2.axhspan(sigma_univ - sigma_std, sigma_univ + sigma_std,
            alpha=0.10, color=COL_UNIV, zorder=2,
            label=f'±1σ = [{sigma_univ-sigma_std:.0f}–{sigma_univ+sigma_std:.0f}]')

# Naked mole-rat annotation
nmr_idx = sp_ord.index('Naked mole-rat') if 'Naked mole-rat' in sp_ord else None
if nmr_idx is not None:
    ax2.annotate('Naked mole-rat\n(exception?)',
                 xy=(nmr_idx, vals[nmr_idx]),
                 xytext=(nmr_idx+0.8, vals[nmr_idx]+300),
                 fontsize=8, color='#4CAF50',
                 arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=0.9))

ax2.set_xticks(range(len(sp_ord)))
ax2.set_xticklabels(sp_ord, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Mutations per genome at end of observation', fontsize=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.93)
ax2.set_title(
    'Universal mutation accumulation threshold $\\Sigma^*$: 15 species (Cagan 2022)\n'
    'VBLS predicts: all species reach similar burden at end of life',
    fontsize=11, fontweight='bold')
ax2.text(0.99, -0.22, 'Data: Cagan et al. (2022) Nature, Supplementary Table 4',
         transform=ax2.transAxes, ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('cagan_fig2_sigma_star.pdf', bbox_inches='tight')
plt.savefig('cagan_fig2_sigma_star.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: cagan_fig2_sigma_star")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Human detail: VBLS Sigma(t) + all 28 crypts + phases + dSigma/dt
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4...")

hum = df[df['Species']=='Human'].copy()
T_max_h = TMAX['Human']
end_b_h = sigma_end.get('Human', hum['Substitution_burden'].max())

# VBLS solution
sol_h = run_vbls(T_max_h)
t_h   = sol_h.t
E_h   = sol_h.y[0]
S_h   = sol_h.y[1]
R_h   = sol_h.y[3]
tB_h  = float(t_h[-1])

# Scale Sigma to mutation burden units
S_end_h   = max(S_h[-1], 1e-9)
S_scaled_h = S_h / S_end_h * end_b_h

# dSigma/dt from RHS
sc_h   = LS_REF / T_max_h
s0sc_h = BASE_ABS['sigma0'] * sc_h
lsc_h  = BASE_ABS['lam'] * sc_h
diss_h = s0sc_h * (BASE_ABS['mu']*E_h**2 + lsc_h*E_h)
dSdt_h = np.maximum(0, diss_h - R_h * BASE_ABS['Phi_cap'])
dSdt_norm_h = dSdt_h / max(dSdt_h.max(), 1e-12)

# Phases
E_norm_h = E_h / max(E_h.max(), 1e-9)
t_peak_h = float(t_h[np.argmax(E_norm_h)])
col_idx_h = np.where((t_h > t_peak_h) & (R_h < 0.50))[0]
t_col_h   = float(t_h[col_idx_h[0]]) if len(col_idx_h) else tB_h*0.75

# Individual means
ind_h = (hum.groupby(['Individual','Age_years'])['Substitution_burden']
         .mean().reset_index().sort_values('Age_years'))

fig4, ax4 = plt.subplots(figsize=(11, 7))
fig4.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.11)
ax4r = ax4.twinx()
ax4r.spines['top'].set_visible(False)

# Phase backgrounds
ax4.axvspan(0,        t_peak_h, alpha=0.18, color=PH_COL[0], zorder=0, lw=0)
ax4.axvspan(t_peak_h, t_col_h,  alpha=0.18, color=PH_COL[1], zorder=0, lw=0)
ax4.axvspan(t_col_h,  tB_h,     alpha=0.18, color=PH_COL[2], zorder=0, lw=0)

# dSigma/dt right axis
ax4r.plot(t_h, dSdt_norm_h, color=COL_DSIG, lw=1.5, alpha=0.65, zorder=2)
ax4r.set_ylim(-0.05, 1.60)
ax4r.set_yticks([0, 0.5, 1.0])
ax4r.set_yticklabels(['0', '0.5', '1'], fontsize=8.5, color=COL_DSIG)
ax4r.tick_params(axis='y', colors=COL_DSIG)
ax4r.set_ylabel('$d\\Sigma/dt$ (norm., right axis)', fontsize=9.5, color=COL_DSIG)

# All crypt data points
ax4.scatter(hum['Age_years'], hum['Substitution_burden'],
            color=SP_COLORS['Human'], s=40, alpha=0.50, zorder=4,
            edgecolors='white', lw=0.5, label='Individual crypts (Cagan 2022)')

# Individual means
ax4.scatter(ind_h['Age_years'], ind_h['Substitution_burden'],
            color=SP_COLORS['Human'], s=110, zorder=6,
            edgecolors='white', lw=1.2, marker='D',
            label='Individual mean (n=6)')

# Individual labels
for _, row in ind_h.iterrows():
    ax4.annotate(f"{int(row['Age_years'])} yr",
                 xy=(row['Age_years'], row['Substitution_burden']),
                 xytext=(row['Age_years']+1.2, row['Substitution_burden']+80),
                 fontsize=8, color=SP_COLORS['Human'])

# Linear regression on individual means
slope_h, intercept_h, r_h, _, _ = linregress(ind_h['Age_years'],
                                              ind_h['Substitution_burden'])
t_reg_h = np.linspace(0, 122.5, 300)
ax4.plot(t_reg_h, intercept_h + slope_h*t_reg_h,
         color=COL_REG, lw=1.5, ls='--', zorder=3,
         label=f'Linear regression: {slope_h:.0f} mut/yr  ($R^2={r_h**2:.3f}$)')

# VBLS Sigma(t)
ax4.plot(t_h, S_scaled_h, color=COL_VBLS, lw=2.5, zorder=5,
         label='VBLS $\\Sigma(t)$ (scaled to data)')

# t_B and T_max
ax4.axvline(tB_h, color='#555', lw=1.1, ls=':', zorder=5)
ax4.axvline(T_max_h, color=SP_COLORS['Human'], lw=0.9, ls='--', alpha=0.35)
ax4.text(tB_h*0.99, ax4.get_ylim()[1]*0.97 if ax4.get_ylim()[1]>0 else 3500,
         f'$t_B={tB_h:.0f}$ yr', ha='right', fontsize=9, color='#444')

# Phase labels
y_top = end_b_h * 1.05
for xc, lbl, fc in [(t_peak_h/2, 'I Growth', '#2980b9'),
                    ((t_peak_h+t_col_h)/2, 'II Plateau', '#b7950b'),
                    ((t_col_h+tB_h)/2, 'III Degradation', '#c0392b')]:
    ax4.text(xc, y_top, lbl, ha='center', fontsize=9,
             color=fc, fontweight='bold')

ax4.set_xlim(-3, tB_h*1.08)
ax4.set_ylim(-50, end_b_h*1.20)
ax4.set_xlabel('Age (years)', fontsize=10.5)
ax4.set_ylabel('Mutations per genome ($\\Sigma$)', fontsize=10.5)
ax4.legend(fontsize=9, loc='upper left', framealpha=0.93)

ax4.set_title(
    'Human: somatic mutation accumulation (Cagan 2022) vs VBLS $\\Sigma(t)$\n'
    f'VBLS nonlinearity vs observed linearity  |  '
    f'Regression slope: {slope_h:.0f} mut/genome/yr  |  $t_B={tB_h:.0f}$ yr',
    fontsize=10.5, fontweight='bold')

fig4.text(0.99, 0.02, 'Data: Cagan et al. (2022) Nature 604:517-524',
          ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('cagan_fig4_human.pdf', bbox_inches='tight')
plt.savefig('cagan_fig4_human.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: cagan_fig4_human")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICATION SUMMARY — CAGAN 2022")
print("="*60)
print(f"Species with age span > 1yr:  {len(sp_use)}")
print(f"Total crypts (clean):         {len(df)}")
print(f"Universal Sigma*_univ:        {sigma_univ:.0f} ± {sigma_std:.0f} mutations")
print(f"Human regression slope:       {slope_h:.1f} mut/yr  R²={r_h**2:.3f}")
print("="*60)
print("\nAll figures saved.")
