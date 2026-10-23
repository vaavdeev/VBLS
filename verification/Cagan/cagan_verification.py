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
VBLS — Verification against Cagan et al. (2022) Nature
Somatic mutations as one observable proxy of Sigma(t)
=================================================================

3 Figures:
  Fig 1 — Sigma(t) trajectories: Cagan data (scatter) + linear regression + VBLS curve
          for species with sufficient age coverage (3-column grid). The curve has the
          model shape Sigma(t)/Sigma(t_B); its amplitude is fitted per species by least
          squares over the individual means, so the figure compares shape, not level.
  Fig 2 — Mutation burden at the end of observation (bar chart); a common damage
          level at death is a transfer assumption (App. D.5.1.5, item 3)
  Fig 4 — Human detail: VBLS Sigma(t) over all 28 crypts
          + life phases + dSigma/dt on right axis

  Note: the rate x T_max power-law is owned exclusively by cagan_rateXtmax.py
  (15-species slope fit) and is intentionally not reproduced here.

  T_max: maximum longevity from AnAge (../AnAge/anage_data.csv) for all species,
  without corrections (Section 17.3). The script also prints the model (S) sampled
  at the individual ages of Table 4: the zero crossings of the per-species
  regressions and the model exponent phi over the sampled windows
  (Sections 17.3.5.1, 17.3.5.2).

Data: Cagan et al. (2022) Nature 604:517-524, Supplementary Table 4
Input files: Table_4.csv; ../AnAge/anage_data.csv

Out: cagan_fig1 / fig2 / fig4 .pdf / .png
"""

import csv
import os

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

# ── Species metadata: maximum longevity from AnAge, all species, no corrections ──
ANAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AnAge', 'anage_data.csv')
# Cagan species -> AnAge genus and species (the domestic ferret is listed under Mustela putorius);
# the harbour porpoise has no age data in Table 4 and is not used
ANAGE_NAME = {
    'Human':                   ('Homo', 'sapiens'),
    'Mouse':                   ('Mus', 'musculus'),
    'Rat':                     ('Rattus', 'norvegicus'),
    'Naked mole-rat':          ('Heterocephalus', 'glaber'),
    'Dog':                     ('Canis', 'familiaris'),
    'Cat':                     ('Felis', 'catus'),
    'Horse':                   ('Equus', 'caballus'),
    'Tiger':                   ('Panthera', 'tigris'),
    'Cow':                     ('Bos', 'taurus'),
    'Rabbit':                  ('Oryctolagus', 'cuniculus'),
    'Giraffe':                 ('Giraffa', 'camelopardalis'),
    'Lion':                    ('Panthera', 'leo'),
    'Ferret':                  ('Mustela', 'putorius'),
    'Ring-tailed lemur':       ('Lemur', 'catta'),
    'Black-and-white colobus': ('Colobus', 'guereza'),
}


def anage_tmax(path=ANAGE_FILE):
    """Maximum longevity (yr) from AnAge for the species of ANAGE_NAME."""
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.reader(f, delimiter=';'))
    h = rows[0]
    ig, isp, iml = h.index('Genus'), h.index('Species'), h.index('Maximum longevity (yrs)')
    out = {}
    for name, (genus, species) in ANAGE_NAME.items():
        hit = [r for r in rows[1:] if r[ig] == genus and r[isp] == species]
        if len(hit) != 1:
            raise SystemExit(f'AnAge: {genus} {species} found {len(hit)} times')
        out[name] = float(hit[0][iml])
    return out


TMAX = anage_tmax()

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

# ── VBLS parameters: canonical human calibration (Section 17.1.1) ────────────
# Fluxes as in MAN/human_lifespan.py: Phi_in(t) = (1 - exp(-t/10)) exp(-max(0, t-60)/30),
# Phi_cap = 0.003. Per species, time is rescaled: every rate (mu, lam, the amplitude of Phi_in,
# Phi_cap, alpha, nu0, nu, rho) is multiplied by sc = LS_REF/T_max, the time constants of Phi_in
# are divided by sc, and sigma0 (a fraction, not a rate) is unchanged. Each species is the human
# calibration in its own time scale; for the human (sc = 1) this is the calibration itself.
LS_REF   = 122.5
BASE_ABS = dict(mu=1.50, lam=0.30, sigma0=0.0162, alpha=0.030,
                nu0=0.006, nu=0.06, rho=0.002,
                Phi_cap=0.003,
                tau_growth=10.0, t_decline=60.0, tau_decline=30.0,
                R_star=0.15, E_star=0.04)
Y0 = [0.10, 0.01, 0.96, 0.99]

# Denominator of g(E) = (E - E*) / (E0 - E*), formula 4.2
E_RANGE = Y0[0] - BASE_ABS['E_star']  # 0.10 - 0.04 = 0.06

def scaled_args(T_max):
    sc = LS_REF / T_max
    p  = BASE_ABS
    return (p['mu']*sc, p['lam']*sc, p['sigma0'], p['alpha']*sc,
            p['nu0']*sc, p['nu']*sc, p['rho']*sc,
            p['Phi_cap']*sc, p['tau_growth']/sc, p['t_decline']/sc, p['tau_decline']/sc, sc)

def rhs(t, y, mu, lam, s0, alpha, nu0, nu, rho, Phi_cap, tau_g, t_d, tau_d, amp):
    E, S, eta, R = y
    E=max(E,0.0); S=np.clip(S,0,1); eta=np.clip(eta,0,1); R=np.clip(R,0,1)
    phi  = amp * (1.0 - np.exp(-t/tau_g)) * np.exp(-max(0.0, t - t_d)/tau_d)
    dE   = eta*phi - mu*E**2 - lam*E
    dS   = max(0.0, s0*(mu*E**2+lam*E) - R*Phi_cap)
    deta = -alpha*S*eta
    # Restoration-potential function g(E), formula 4.2: gates repair restoration
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
                    events=[ev_R], dense_output=True,
                    rtol=1e-8, atol=1e-10, max_step=T_max/600)
    return sol

def vbls_sigma_fitted(T_max, ages, burdens):
    """Return t, the curve A * Sigma(t)/Sigma(t_B), t_B, A and the model shape at the given
    ages. The shape is the model's; the amplitude A is fitted by least squares through the
    origin over the individual means (ages, burdens)."""
    sol = run_vbls(T_max)
    tB  = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    S_B = max(sol.sol(tB)[1], 1e-9)
    m_i = sol.sol(np.minimum(ages, tB))[1] / S_B
    A   = float(np.sum(burdens * m_i) / np.sum(m_i**2))
    return sol.t, A * sol.y[1] / S_B, float(tB), A, m_i

def vbls_dsigma(T_max):
    """Compute dSigma/dt directly from RHS."""
    sol  = run_vbls(T_max)
    t_s  = sol.t
    E    = sol.y[0]
    R    = sol.y[3]
    sc   = LS_REF / T_max
    diss = BASE_ABS['sigma0'] * sc * (BASE_ABS['mu'] * E**2 + BASE_ABS['lam'] * E)
    dSdt = np.maximum(0, diss - R * BASE_ABS['Phi_cap'] * sc)
    return t_s, dSdt

# ── Per-species mean burden at oldest age (mutation proxy) ─────────────────
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
fit_rel_rmse = {}

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

    # ── VBLS Sigma(t): model shape, amplitude fitted to the individual means ──
    y_ind = ind_mean['Substitution_burden'].to_numpy()
    t_vbls, S_vbls, tB, amp, m_ind = vbls_sigma_fitted(
        T_max, ind_mean['Age_years'].to_numpy(), y_ind)
    fit_rel_rmse[sp] = np.sqrt(np.mean((y_ind - amp * m_ind)**2)) / np.mean(y_ind)
    if sp == 'Human':
        # Numbers of Section 17.3.6 (late and early age): model at the sampled ages
        print(f"  Human: amplitude A = {amp:.0f}; ages {ind_mean['Age_years'].round(1).tolist()}")
        print(f"    model {np.round(amp * m_ind).astype(int).tolist()}")
        print(f"    data  {np.round(y_ind).astype(int).tolist()}")
    t_obs_max = g['Age_years'].max()
    mask = t_vbls <= t_obs_max * 1.15
    ax.plot(t_vbls[mask], S_vbls[mask],
            color=COL_VBLS, lw=2.2, zorder=5,
            label='VBLS $\\Sigma(t)$, amplitude fitted')
    # Dashed continuation to t_B
    mask2 = t_vbls >= t_obs_max * 1.10
    ax.plot(t_vbls[mask2], S_vbls[mask2],
            color=COL_VBLS, lw=1.4, ls=':', zorder=4, alpha=0.55)

    # ── t_B marker ────────────────────────────────────────────────────
    ax.axvline(tB, color='#999', lw=0.9, ls=':', zorder=2)
    ax.axvline(T_max, color=col, lw=0.8, ls='--', alpha=0.40, zorder=2)

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xlim(-T_max*0.02, max(tB*1.05, t_obs_max*1.3))
    ax.set_ylim(-50, max(g['Substitution_burden'].max()*1.25, amp*1.05))
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
    '-- grey = linear regression;  : = extrapolation to $t_B$\n'
    '— purple = VBLS shape (human calibration in the time scale of each species), '
    'amplitude fitted per species to the individual means',
    fontsize=11, fontweight='bold', y=0.985)

plt.savefig('cagan_fig1_trajectories.pdf', bbox_inches='tight')
plt.savefig('cagan_fig1_trajectories.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: cagan_fig1_trajectories")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Mutation burden at the end of observation
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
            label=f'Mean over species = {sigma_univ:.0f} mut.')
ax2.axhspan(sigma_univ - sigma_std, sigma_univ + sigma_std,
            alpha=0.10, color=COL_UNIV, zorder=2,
            label=f'±1σ = [{sigma_univ-sigma_std:.0f}–{sigma_univ+sigma_std:.0f}]')

# Naked mole-rat annotation
nmr_idx = sp_ord.index('Naked mole-rat') if 'Naked mole-rat' in sp_ord else None
if nmr_idx is not None:
    nmr_age = df.loc[df['Species'] == 'Naked mole-rat', 'Age_years'].max()
    ax2.annotate(f'Naked mole-rat: oldest sampled\nanimal {nmr_age:.1f} yr '
                 f'({100 * nmr_age / TMAX["Naked mole-rat"]:.0f}% of $T_{{max}}$)',
                 xy=(nmr_idx, vals[nmr_idx]),
                 xytext=(nmr_idx+0.6, vals[nmr_idx]+1150), va='bottom',
                 fontsize=8, color='#4CAF50',
                 arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=0.9))

ax2.set_xticks(range(len(sp_ord)))
ax2.set_xticklabels(sp_ord, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('Mutations per genome at end of observation', fontsize=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.93)
ax2.set_title(
    'Mutation burden at the end of observation: 15 species (Cagan 2022)\n'
    'A common damage level at death is a transfer assumption (App. D.5.1.5, item 3),\n'
    'not a prediction of $(\\mathcal{S})$; somatic mutations are one proxy of $\\Sigma$',
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
diss_h = BASE_ABS['sigma0'] * sc_h * (BASE_ABS['mu']*E_h**2 + BASE_ABS['lam']*E_h)
dSdt_h = np.maximum(0, diss_h - R_h * BASE_ABS['Phi_cap'] * sc_h)
dSdt_norm_h = dSdt_h / max(dSdt_h.max(), 1e-12)

# Phases of E(t), Section 10: t_peak = argmax E; t_col = point of minimum d2E/dt2 after t_peak
E_norm_h = E_h / max(E_h.max(), 1e-9)
t_peak_h = float(t_h[np.argmax(E_norm_h)])
d2E_h    = np.gradient(np.gradient(E_norm_h, t_h), t_h)
win_h    = np.where((t_h > t_peak_h + 0.01*tB_h) & (t_h < 0.99*tB_h))[0]
t_col_h  = float(t_h[win_h[np.argmin(d2E_h[win_h])]]) if len(win_h) else tB_h

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
ax4.text(tB_h*0.99, end_b_h*1.12,
         f'$t_B={tB_h:.0f}$ yr', ha='right', fontsize=9, color='#444')

# Phase labels
y_top = end_b_h * 1.05
for xc, lbl, fc in [(t_peak_h/2, 'I Growth', '#2980b9'),
                    ((t_peak_h+t_col_h)/2, 'II Plateau', '#b7950b'),
                    ((t_col_h+tB_h)/2, 'III Decline', '#c0392b')]:
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

# ══════════════════════════════════════════════════════════════════════════════
# MODEL AT THE SAMPLED AGES (Sections 17.3.5.1, 17.3.5.2)
# Model burden ~ Sigma(t)/Sigma(t_B) (a common damage level at death, App. D.5.1.5,
# item 3), taken at the individual ages of Table 4. Species with several ages: OLS
# over the individuals, as for the data -> zero crossing L0; species with a single
# age: rate = burden/age. The model exponent phi is the slope of log rate on log T_max.
# ══════════════════════════════════════════════════════════════════════════════
print("Model at the sampled ages...")
model_rate, model_L0, data_L0 = {}, {}, {}
for sp in sorted(TMAX, key=TMAX.get):
    g = df[df['Species'] == sp]
    ind = g.groupby(['Individual', 'Age_years'])['Substitution_burden'].mean().reset_index()
    ages = ind['Age_years'].to_numpy()
    sol = run_vbls(TMAX[sp])
    tB = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    M = sol.sol(ages)[1] / sol.sol(tB)[1]
    if len(np.unique(ages)) >= 2:
        fm = linregress(ages, M)
        fd = linregress(ages, ind['Substitution_burden'])
        model_rate[sp] = fm.slope
        model_L0[sp] = -fm.intercept / fm.slope
        data_L0[sp] = -fd.intercept / fd.slope
    else:
        model_rate[sp] = M[0] / ages[0]
sp_m = sorted(model_rate)
phi_model = linregress(np.log([TMAX[s] for s in sp_m]),
                       np.log([model_rate[s] for s in sp_m])).slope

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICATION SUMMARY — CAGAN 2022")
print("="*60)
print(f"Species with age span > 1yr:  {len(sp_use)}")
print(f"Total crypts (clean):         {len(df)}")
print(f"Mean end burden (proxy):      {sigma_univ:.0f} ± {sigma_std:.0f} mutations")
print(f"Human regression slope:       {slope_h:.1f} mut/yr  R²={r_h**2:.3f}")
print(f"Fig 1 VBLS amplitude:         least squares over individual means; relative RMSE "
      f"median {np.median(list(fit_rel_rmse.values())):.2f} ({len(fit_rel_rmse)} species)")
print(f"Model phi, sampled windows:   {phi_model:+.2f} ({len(sp_m)} species, T_max from AnAge)")
print("Zero crossing of the regressions, yr (data / model):")
for sp in sorted(model_L0, key=TMAX.get):
    print(f"  {sp:24s} T_max={TMAX[sp]:6.1f}  data {data_L0[sp]:5.1f}  model {model_L0[sp]:5.1f}")
print("="*60)
print("\nAll figures saved.")
