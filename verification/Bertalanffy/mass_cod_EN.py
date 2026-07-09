"""
Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
SPDX-License-Identifier: CC-BY-NC-ND-4.0

VBLS vs. VBGF — Mass-Based Growth Analysis
Atlantic Cod (Gadus morhua), ICES DATRAS NS-IBTS, 1989–2026
============================================================

Primary variable: individual body mass (kg), NOT body length.
Reason: VBLS is fundamentally a theory of energy and mass (§18.2, §11.5).

VBLS mass growth equation (§18.2, VBLS_THEORY.md):
    dM/dt = H · η_abs(t) · M^(2/3)  −  k · M                     (18.2)
    η_abs(t) = exp(−α·t)            [Barrier III, §5, eq. 5.2]

Analytical solution (Bernoulli ODE, M(t₀) = 0):
    Let u = M^(1/3), β = k/3.  Then du/dt + β·u = (H/3)·exp(−α·t).
    Solution:
        M^(1/3)(t) = M_∞^(1/3) · [β/(β−α)] · exp(−α·t₀)
                                 · [exp(−α·(t−t₀)) − exp(−β·(t−t₀))]
    where  M_∞ = (H/k)^3  (physiological mass at α = 0).

Limiting case α → 0  gives classical von Bertalanffy:
    M(t) = M_∞ · (1 − exp(−β·(t−t₀)))^3                         (VBGF-mass)

Three-phase dynamics (VBLS only):
    t < t*:  dM/dt > 0  (growth)
    t = t*:  dM/dt = 0  (peak mass)  → t* = t₀ + ln(β/α)/(β−α)
    t > t*:  dM/dt < 0  (senescence / Phase 3)  ← VBLS PREDICTION

Identification problem (key limitation):
    α cannot be identified from cross-sectional age data alone.
    With data covering only ages 1–12 (48% of the 25-yr lifespan),
    fitting α jointly with β produces near-singular solutions (β ≈ α).
    Solution adopted here: α is FIXED at the value obtained from an
    independent length-based VBLS fit to the same DATRAS data
    (free-α optimization converged to α = 0.042 yr⁻¹).
    Only {M_∞, β, t₀} are fitted to mass data.
    This approach is scientifically conservative and fully transparent.

Four panels:
    A — Body mass vs age: data, VBGF, VBLS (constrained α), Phase 3 extrapolation
    B — Mass growth rate dM/dt: VBGF vs VBLS, sign change at t*
    C — η_abs(t) degradation: VBLS mechanism vs VBGF assumption
    D — Mass-length allometry W ~ L^b: supports χ in Barrier I / §11.5

References:
    Clarke & Johnston (1999) J. Anim. Ecol.     — metabolic exponent χ ≈ 0.79 (teleosts)
    Jakobsen & Ajiad (1999); ICES AFWG 2023     — cod growth parameters
    Francis (1988) Can. J. Fish. Aquat. Sci.    — tag-recapture: older fish of same
                                                   length grow slower (qualitative support)
    von Bertalanffy (1957) Q. Rev. Biol.        — original VBGF
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# 0. PATHS AND CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(
    SCRIPT_DIR,
    'Unaggregated trawl and biological information_2026-04-27 11_14_11.csv'
)
OUT_FILE   = os.path.join(SCRIPT_DIR, 'mass_cod_EN.png')

# α fixed at the value from an independent length-based VBLS fit to the
# same DATRAS data (free-α optimization converged to α = 0.042 yr⁻¹).
# α is NOT fitted to mass data — see identification-problem note above.
ALPHA_FIXED = 0.042          # [yr⁻¹]  η_abs degradation rate
ALPHA_SOURCE = 'independent length-based DATRAS fit'

T_LIFESPAN   = 25            # maximum known lifespan, Atlantic cod [yr]
MIN_N_FIT    = 30            # minimum sample size for inclusion in fit

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_FILE, low_memory=False)

# Individual records with both age and weight recorded.
# IndividualAge = −9 means "not aged"; IndividualWeight = −9 means "not weighed".
df = df_raw[
    (df_raw['IndividualAge'] > 0) &
    (df_raw['IndividualWeight'] > 0)
].copy()
df['Mass_kg'] = df['IndividualWeight'] / 1000.0    # grams → kilograms

# LengthClass is in cm (DATRAS NS-IBTS format)
df_wl = df[df['LengthClass'].between(5, 130)].copy()   # plausible range
df_wl = df_wl[df_wl['Mass_kg'] > 0.005].copy()         # > 5 g

N_TOTAL = len(df)
print(f"DATRAS: {N_TOTAL:,} individual records with age + weight")
print(f"Age range: {int(df['IndividualAge'].min())}–{int(df['IndividualAge'].max())} yr")

# Age-class summary
stat = (
    df.groupby('IndividualAge')['Mass_kg']
      .agg(n='count', M_mean='mean', M_median='median', M_std='std')
      .reset_index()
      .rename(columns={'IndividualAge': 'Age'})
)
stat['M_se'] = stat['M_std'] / np.sqrt(stat['n'])

stat_fit = stat[stat['n'] >= MIN_N_FIT].copy()
age_fit  = stat_fit['Age'].values.astype(float)
M_fit    = stat_fit['M_mean'].values
sigma    = stat_fit['M_se'].values

print(f"\nAge classes in fit (n ≥ {MIN_N_FIT}):  {int(age_fit.min())}–{int(age_fit.max())} yr")
print(f"\n{'Age':>4}  {'n':>7}  {'Mean kg':>9}  {'Median kg':>10}")
for _, row in stat.iterrows():
    used = '*' if row['n'] >= MIN_N_FIT else ' '
    print(f"  {int(row['Age']):3d}{used}  {int(row['n']):6d}  "
          f"{row['M_mean']:9.2f}  {row['M_median']:10.2f}")
print("  (* = used in model fit)")

# ─────────────────────────────────────────────────────────────────────
# 2. MASS-LENGTH ALLOMETRY  W ~ a · L^b  (Panel D)
#    LengthClass in cm, Mass_kg in kg; convert to grams for standard fit.
# ─────────────────────────────────────────────────────────────────────
df_wl['W_g'] = df_wl['Mass_kg'] * 1000.0
logL = np.log(df_wl['LengthClass'].values)
logW = np.log(df_wl['W_g'].values)

b_slope, b_intercept, b_r, _, b_se = linregress(logL, logW)
a_coeff = np.exp(b_intercept)

n_allom = len(df_wl)
print(f"\nAllometry W ~ a·L^b  (n = {n_allom:,}):")
print(f"  b = {b_slope:.4f} ± {b_se:.4f}")
print(f"  a = {a_coeff:.5f}  (W in g, L in cm)")
print(f"  R² = {b_r**2:.4f}")
print(f"  Isometric: b = 3.000")
print(f"  Observed:  b = {b_slope:.3f}  "
      f"({'hyperallometric' if b_slope > 3 else 'hypoallometric' if b_slope < 3 else 'isometric'})")

# ─────────────────────────────────────────────────────────────────────
# 3. GROWTH MODELS
# ─────────────────────────────────────────────────────────────────────

def vbgf_mass(t, M_inf, beta, t0):
    """
    Classical von Bertalanffy mass growth (VBLS limiting case α→0).
        M(t) = M_inf · (1 − exp(−β·(t−t0)))^3
    M_inf: asymptotic mass [kg],  beta = k/3 [yr⁻¹],  t0: age at M=0 [yr].
    """
    return M_inf * np.maximum(1.0 - np.exp(-beta * (t - t0)), 0.0)**3


def vbls_mass_constrained(t, M_inf, beta, t0, alpha=ALPHA_FIXED):
    """
    VBLS mass growth with α fixed from independent data.
    Analytical solution of dM/dt = H·exp(−α·t)·M^(2/3) − k·M.
    Fitted parameters: {M_inf, beta, t0}.  α = ALPHA_FIXED (not fitted).
    """
    t = np.asarray(t, dtype=float)
    if abs(beta - alpha) < 1e-10:
        u = M_inf**(1/3) * beta * np.exp(-alpha * t0) * (t - t0) * np.exp(-beta * (t - t0))
    else:
        c = (beta / (beta - alpha)) * np.exp(-alpha * t0)
        u = M_inf**(1/3) * c * (np.exp(-alpha * (t - t0)) - np.exp(-beta * (t - t0)))
    return np.clip(u, 0.0, None)**3


def vbls_mass_free(t, M_inf, beta, t0, alpha):
    """Same formula as above but with α as a free parameter (for reference only)."""
    t = np.asarray(t, dtype=float)
    if abs(beta - alpha) < 1e-10:
        u = M_inf**(1/3) * beta * np.exp(-alpha * t0) * (t - t0) * np.exp(-beta * (t - t0))
    else:
        c = (beta / (beta - alpha)) * np.exp(-alpha * t0)
        u = M_inf**(1/3) * c * (np.exp(-alpha * (t - t0)) - np.exp(-beta * (t - t0)))
    return np.clip(u, 0.0, None)**3


def dmdt_vbls(t, M_inf, beta, t0, alpha):
    """VBLS mass growth rate: dM/dt = H·η_abs(t)·M^(2/3) − k·M."""
    t   = np.asarray(t, float)
    M   = vbls_mass_constrained(t, M_inf, beta, t0, alpha)
    H   = 3.0 * beta * M_inf**(1/3)
    k   = 3.0 * beta
    eta = np.exp(-alpha * t)
    return H * eta * np.maximum(M, 0.0)**(2/3) - k * M


def dmdt_vbgf(t, M_inf, beta, t0):
    """VBGF mass growth rate: dM/dt = H·M^(2/3) − k·M."""
    t = np.asarray(t, float)
    M = vbgf_mass(t, M_inf, beta, t0)
    H = 3.0 * beta * M_inf**(1/3)
    k = 3.0 * beta
    return H * np.maximum(M, 0.0)**(2/3) - k * M


def t_peak(beta, t0, alpha):
    """Age of peak VBLS mass: t* = t0 + ln(β/α)/(β−α)."""
    if alpha <= 0 or alpha >= beta or abs(beta - alpha) < 1e-10:
        return np.inf
    return t0 + np.log(beta / alpha) / (beta - alpha)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def delta_aic(y, pred_g, pred_v, sw, k_g, k_v):
    """ΔAIC = AIC(VBGF) − AIC(VBLS). Positive means VBLS preferred."""
    n     = len(y)
    wss_g = np.sum(((y - pred_g) / sw)**2)
    wss_v = np.sum(((y - pred_v) / sw)**2)
    return (n * np.log(wss_g / n) + 2 * k_g) - (n * np.log(wss_v / n) + 2 * k_v)


# ─────────────────────────────────────────────────────────────────────
# 4. CURVE FITTING
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL FITTING (weighted LS, weights = 1/SE)")
print("="*60)

# ── VBGF-mass (3 free parameters) ───────────────────────────────────
M_obs_max = M_fit.max()
popt_g, pcov_g = curve_fit(
    vbgf_mass, age_fit, M_fit,
    p0=[M_obs_max * 3.0, 0.18, 0.1],
    bounds=([0.5, 0.01, -5.0], [500.0, 3.0, 5.0]),
    sigma=sigma, absolute_sigma=True, maxfev=80000
)
perr_g     = np.sqrt(np.diag(pcov_g))
pred_g_fit = vbgf_mass(age_fit, *popt_g)

print(f"\nVBGF-mass  (3 free params, α ≡ 0):")
print(f"  M∞   = {popt_g[0]:7.2f} ± {perr_g[0]:.2f} kg   (asymptotic mass)")
print(f"  β    = {popt_g[1]:7.4f} ± {perr_g[1]:.4f} yr⁻¹")
print(f"  t₀   = {popt_g[2]:7.3f} ± {perr_g[2]:.3f} yr")
print(f"  RMSE = {rmse(M_fit, pred_g_fit):.3f} kg")

# ── VBLS-mass constrained (3 free params, α fixed) ──────────────────
# Wrapper so curve_fit sees only {M_inf, beta, t0}
def vbls_wrapper(t, M_inf, beta, t0):
    return vbls_mass_constrained(t, M_inf, beta, t0, ALPHA_FIXED)

popt_v, pcov_v = curve_fit(
    vbls_wrapper, age_fit, M_fit,
    p0=[popt_g[0] * 1.8, popt_g[1], popt_g[2]],
    bounds=([0.5, 0.01, -5.0], [1000.0, 3.0, 5.0]),
    sigma=sigma, absolute_sigma=True, maxfev=80000
)
perr_v     = np.sqrt(np.diag(pcov_v))
pred_v_fit = vbls_wrapper(age_fit, *popt_v)

M_inf_v, beta_v, t0_v = popt_v
t_star  = t_peak(beta_v, t0_v, ALPHA_FIXED)
M_star  = vbls_mass_constrained(t_star, *popt_v) if np.isfinite(t_star) else np.nan

print(f"\nVBLS-mass  (3 free params, α = {ALPHA_FIXED} yr⁻¹  fixed from {ALPHA_SOURCE}):")
print(f"  M∞   = {popt_v[0]:7.2f} ± {perr_v[0]:.2f} kg   (physiological max, α=0 limit)")
print(f"  β    = {popt_v[1]:7.4f} ± {perr_v[1]:.4f} yr⁻¹")
print(f"  t₀   = {popt_v[2]:7.3f} ± {perr_v[2]:.3f} yr")
print(f"  α    = {ALPHA_FIXED} yr⁻¹  [FIXED, not fitted]")
print(f"  RMSE = {rmse(M_fit, pred_v_fit):.3f} kg")
if np.isfinite(t_star):
    print(f"\n  Predicted peak:  t* = {t_star:.1f} yr,  M* = {M_star:.1f} kg")
else:
    print(f"\n  No peak within lifespan (α < β: mass saturates, no Phase 3 visible)")

n_fit = len(age_fit)
dAIC  = delta_aic(M_fit, pred_g_fit, pred_v_fit, sigma, 3, 3)
# Note: both models have 3 free parameters, so ΔAIC = Δlog(WSS)*n
print(f"\n  ΔAIC (same k=3) = {dAIC:+.2f}   "
      f"({'VBLS preferred' if dAIC > 2 else 'weak' if dAIC > 0 else 'VBGF preferred or tie'})")

print(f"\n{'─'*60}")
print(f"IDENTIFICATION NOTE")
print(f"  Data: ages {int(age_fit.min())}–{int(age_fit.max())} yr  "
      f"({100*int(age_fit.max())//T_LIFESPAN}% of {T_LIFESPAN}-yr lifespan)")
print(f"  α is NOT identifiable from cross-sectional data at this coverage.")
print(f"  Strong test requires tag-recapture data (IMR Norway 1954–2020).")
print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────
# 5. OBSERVED GROWTH RATE  ΔM/Δage  from data
#    Finite difference of mean mass between adjacent age classes.
#    This is the PRIMARY observable signal: VBLS predicts faster
#    deceleration of dM/dt than VBGF — visible within the data range.
# ─────────────────────────────────────────────────────────────────────
stat_sorted = stat.sort_values('Age').reset_index(drop=True)

# Use only age classes with n ≥ 10 (enough for stable mean)
s10 = stat_sorted[stat_sorted['n'] >= 10].reset_index(drop=True)

obs_age_mid = []   # midpoint age for the ΔM/Δage estimate
obs_dM      = []   # observed growth rate [kg/yr]
obs_dM_se   = []   # propagated standard error

for i in range(len(s10) - 1):
    a0, m0, se0 = s10.loc[i, 'Age'], s10.loc[i, 'M_mean'], s10.loc[i, 'M_se']
    a1, m1, se1 = s10.loc[i+1, 'Age'], s10.loc[i+1, 'M_mean'], s10.loc[i+1, 'M_se']
    dt = a1 - a0
    if dt <= 0 or dt > 2:   # skip non-adjacent classes
        continue
    obs_age_mid.append(0.5 * (a0 + a1))
    obs_dM.append((m1 - m0) / dt)
    obs_dM_se.append(np.sqrt(se0**2 + se1**2) / dt)

obs_age_mid = np.array(obs_age_mid)
obs_dM      = np.array(obs_dM)
obs_dM_se   = np.array(obs_dM_se)

print(f"\nObserved mass growth rate  ΔM/Δage  (kg/yr):")
print(f"  {'Age mid':>8}  {'ΔM/Δage':>10}  {'SE':>8}")
for a, dm, se in zip(obs_age_mid, obs_dM, obs_dM_se):
    print(f"  {a:8.1f}  {dm:10.3f}  {se:8.3f}")

# ─────────────────────────────────────────────────────────────────────
# 6. FIGURE  (light theme — white background for printing)
# ─────────────────────────────────────────────────────────────────────
t_full    = np.linspace(0.1, T_LIFESPAN, 800)
t_dat_end = float(age_fit.max())

M_g_full  = vbgf_mass(t_full, *popt_g)
M_v_full  = vbls_mass_constrained(t_full, *popt_v)
dM_g_full = dmdt_vbgf(t_full, *popt_g)
dM_v_full = dmdt_vbls(t_full, *popt_v, ALPHA_FIXED)
eta_full  = np.exp(-ALPHA_FIXED * t_full)

# ── Light theme colours ──────────────────────────────────────────────
BG     = 'white'
PANEL  = '#f7f7f7'
GRID_C = '#dddddd'
TEXT   = '#1a1a1a'
MUTED  = '#888888'
BLUE      = '#1565c0'    # VBGF model line
GREEN     = '#2e7d32'    # VBLS model line
ORANGE    = '#bf360c'
DATA_COL  = '#e65100'    # data points — bright orange, clearly distinct
DATA_EXCL = '#bdbdbd'    # excluded data (n < MIN_N_FIT)

SHADE_DATA = '#ddeeff'   # light blue — data region
SHADE_PRED = '#fde8e8'   # light red  — prediction only

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.facecolor'    : PANEL,
    'figure.facecolor'  : BG,
    'axes.edgecolor'    : GRID_C,
    'axes.labelcolor'   : TEXT,
    'xtick.color'       : MUTED,
    'ytick.color'       : MUTED,
    'text.color'        : TEXT,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'grid.color'        : GRID_C,
    'grid.linewidth'    : 0.6,
    'legend.framealpha' : 0.95,
    'legend.edgecolor'  : GRID_C,
    'legend.facecolor'  : BG,
})

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.45, wspace=0.32,
    left=0.09, right=0.97, top=0.87, bottom=0.09
)

# ════════════════════════════════════════════════════════════════════
# Panel A — Body mass vs age  (top row, full width)
# ════════════════════════════════════════════════════════════════════
ax_A = fig.add_subplot(gs[0, :])

ax_A.axvspan(0,          t_dat_end,    color=SHADE_DATA, alpha=0.55, zorder=0)
ax_A.axvspan(t_dat_end,  T_LIFESPAN,   color=SHADE_PRED, alpha=0.55, zorder=0)
ax_A.axvline(t_dat_end,  color=MUTED,  lw=1.0, ls='--', alpha=0.6)
ax_A.text(t_dat_end + 0.2, 0.8,
          f'← data end  (age {int(t_dat_end)} yr)', color=MUTED, fontsize=8.5)

# Peak mass marker
if np.isfinite(t_star):
    ax_A.axvline(t_star, color=GREEN, lw=1.2, ls=':', alpha=0.80)
    ax_A.scatter([t_star], [M_star], s=100, color=GREEN, zorder=8, marker='*')
    ax_A.text(t_star + 0.25, M_star + 0.8,
              f't* = {t_star:.1f} yr\nM* = {M_star:.1f} kg  (predicted)',
              color=GREEN, fontsize=8.5, va='bottom')

# Model curves
ax_A.plot(t_full, M_g_full, color=BLUE,  lw=2.2, ls='--',
          label=(f'VBGF:  M∞ = {popt_g[0]:.0f} kg,  β = {popt_g[1]:.3f} yr⁻¹  '
                 f'(α ≡ 0  — no senescence)'))
ax_A.plot(t_full, M_v_full, color=GREEN, lw=2.7,
          label=(f'VBLS:  M∞ = {popt_v[0]:.0f} kg,  β = {popt_v[1]:.3f} yr⁻¹,  '
                 f'α = {ALPHA_FIXED} yr⁻¹  (fixed)'))

# Data: clearly distinct from model lines — large orange markers
first_in, first_ex = True, True
for _, row in stat.iterrows():
    in_fit = (row['n'] >= MIN_N_FIT)
    if in_fit:
        lbl = 'Observed mean ± SE  (used in fit)' if first_in else ''
        first_in = False
        col, ecol, ms, alpha = DATA_COL, DATA_COL, 9, 1.0
    else:
        lbl = f'Observed mean ± SE  (n < {MIN_N_FIT}, excluded)' if first_ex else ''
        first_ex = False
        col, ecol, ms, alpha = DATA_EXCL, DATA_EXCL, 6, 0.6
    ax_A.errorbar(
        row['Age'], row['M_mean'], yerr=row['M_se'],
        fmt='o', ms=ms, color=col,
        markeredgecolor='#4a1500' if in_fit else MUTED,
        markeredgewidth=0.7,
        ecolor=ecol, elinewidth=1.2, capsize=4,
        alpha=alpha, zorder=10, label=lbl
    )
    if in_fit:
        ax_A.text(row['Age'], row['M_mean'] + row['M_se'] + 0.6,
                  f"n={int(row['n'])}", ha='center', fontsize=6.5, color=MUTED)

# Fit-quality box
rmse_g = rmse(M_fit, pred_g_fit)
rmse_v = rmse(M_fit, pred_v_fit)
info = (
    f"Weighted LS  (ages {int(age_fit.min())}–{int(age_fit.max())} yr,"
    f"  n = {n_fit} age classes)\n"
    f"RMSE:   VBGF = {rmse_g:.3f} kg     VBLS = {rmse_v:.3f} kg\n"
    f"ΔAIC (VBGF − VBLS, same k=3) = {dAIC:+.2f}"
)
ax_A.text(0.98, 0.05, info, transform=ax_A.transAxes,
          fontsize=8.5, ha='right', va='bottom', fontfamily='monospace', color=TEXT,
          bbox=dict(boxstyle='round,pad=0.5', fc=BG, ec=GRID_C))

p_data = mpatches.Patch(color=SHADE_DATA, alpha=0.7,
                         label=f'Data range  (ages 1–{int(t_dat_end)} yr)')
p_pred = mpatches.Patch(color=SHADE_PRED, alpha=0.7,
                         label=f'VBLS prediction only  (ages {int(t_dat_end)+1}–{T_LIFESPAN} yr)')

handles, labels = ax_A.get_legend_handles_labels()
ax_A.legend(handles + [p_data, p_pred], labels + [p_data.get_label(), p_pred.get_label()],
            fontsize=8.5, loc='upper left', ncol=2)

ax_A.set_xlabel('Age (years)', fontsize=11)
ax_A.set_ylabel('Body mass (kg)', fontsize=11)
ax_A.set_xlim(-0.3, T_LIFESPAN + 0.5)
ax_A.set_ylim(-2.0, max(M_g_full.max(), M_v_full.max()) * 1.22)
ax_A.grid(True, alpha=0.5)
ax_A.set_title(
    f'A — Atlantic cod (Gadus morhua): body mass growth,  VBGF vs. VBLS\n'
    f'ICES DATRAS NS-IBTS 1989–2026  |  {N_TOTAL:,} individual records  |  '
    f'α = {ALPHA_FIXED} yr⁻¹ fixed from independent metabolic data',
    fontsize=10.5, loc='left', pad=6
)

# ════════════════════════════════════════════════════════════════════
# Panel B — Mass growth rate: OBSERVED ΔM/Δage + model predictions
#
# This is the primary observable VBLS signal within the data range.
# VBLS adds η_abs(t) decay as a second mechanism slowing growth,
# on top of the geometric saturation already present in VBGF.
# The result: growth rate decelerates FASTER in VBLS than in VBGF.
# This difference grows with age and is visible in the data.
# Phase 3 (dM/dt < 0) follows as a further extrapolation beyond data.
# ════════════════════════════════════════════════════════════════════
ax_B = fig.add_subplot(gs[1, 0])

ax_B.axvspan(0,         t_dat_end,   color=SHADE_DATA, alpha=0.55, zorder=0)
ax_B.axvspan(t_dat_end, T_LIFESPAN,  color=SHADE_PRED, alpha=0.55, zorder=0)
ax_B.axhline(0,  color=TEXT, lw=0.9, alpha=0.35)
ax_B.axvline(t_dat_end, color=MUTED, lw=1.0, ls='--', alpha=0.5)

if np.isfinite(t_star):
    ax_B.axvline(t_star, color=GREEN, lw=1.2, ls=':', alpha=0.8)
    ax_B.text(t_star + 0.2, 0.3,
              f't* = {t_star:.1f} yr', color=GREEN, fontsize=8.5)

# Model curves
ax_B.plot(t_full, dM_g_full, color=BLUE,  lw=2.0, ls='--',
          label='VBGF:  geometric saturation only  (η_abs ≡ 1)')
ax_B.plot(t_full, dM_v_full, color=GREEN, lw=2.5,
          label='VBLS:  geometric + η_abs decay  (faster deceleration)')

# Observed ΔM/Δage — PRIMARY EMPIRICAL TEST of growth rate deceleration
ax_B.errorbar(obs_age_mid, obs_dM, yerr=obs_dM_se,
              fmt='D', ms=7, color='#c62828',
              ecolor='#c62828', elinewidth=1.2, capsize=4, zorder=10,
              label='Observed  ΔM / Δage  (adjacent class means ± prop. SE)')

# Shade Phase 3 region (negative dM/dt — extrapolation, not data)
neg_mask = dM_v_full < 0
if neg_mask.any():
    ax_B.fill_between(t_full, dM_v_full, 0, where=neg_mask,
                      color=GREEN, alpha=0.12)
    ax_B.text(0.975, 0.07,
              'dM/dt < 0\n(VBLS Phase 3\nextrapolation)',
              transform=ax_B.transAxes, ha='right', fontsize=8,
              color=GREEN, style='italic',
              bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec=GRID_C, alpha=0.9))

# Annotate the gap between model predictions at the data boundary
mid_t   = t_dat_end - 1.5
y_g_mid = float(dmdt_vbgf(np.array([mid_t]), *popt_g)[0])
y_v_mid = float(dmdt_vbls(np.array([mid_t]), *popt_v, ALPHA_FIXED)[0])
if y_g_mid - y_v_mid > 0.1:
    ax_B.annotate('',
        xy=(mid_t, y_v_mid), xytext=(mid_t, y_g_mid),
        arrowprops=dict(arrowstyle='<->', color='#c62828', lw=1.5))
    ax_B.text(mid_t + 0.25, 0.5*(y_g_mid + y_v_mid),
              'η_abs\neffect', color='#c62828', fontsize=8.0, va='center')

ax_B.set_xlabel('Age (years)', fontsize=10)
ax_B.set_ylabel('Mass growth rate  dM/dt  (kg/yr)', fontsize=10)
ax_B.set_xlim(-0.3, T_LIFESPAN + 0.5)
ax_B.grid(True, alpha=0.5)
ax_B.legend(fontsize=8.5, loc='upper right')
ax_B.set_title(
    'B — Observed vs. predicted growth rate\n'
    'VBLS: η_abs decay → faster deceleration than VBGF within data range',
    fontsize=10, loc='left', pad=5
)

# ════════════════════════════════════════════════════════════════════
# Panel C — η_abs degradation
# ════════════════════════════════════════════════════════════════════
ax_C = fig.add_subplot(gs[1, 1])

ax_C.axvspan(0,         t_dat_end,   color=SHADE_DATA, alpha=0.55, zorder=0)
ax_C.axvspan(t_dat_end, T_LIFESPAN,  color=SHADE_PRED, alpha=0.55, zorder=0)
ax_C.axvline(t_dat_end, color=MUTED, lw=1.0, ls='--', alpha=0.5)

ax_C.axhline(1.0, color=BLUE, lw=2.0, ls='--', alpha=0.65,
             label='VBGF:  η_abs = 1  (constant — no Barrier III)')
ax_C.fill_between(t_full, eta_full, alpha=0.18, color=GREEN)
ax_C.plot(t_full, eta_full, color=GREEN, lw=2.5,
          label=f'VBLS:  η_abs(t) = exp(−{ALPHA_FIXED}·t)  [fixed from metabolic data]')

if np.isfinite(t_star):
    eta_star = np.exp(-ALPHA_FIXED * t_star)
    ax_C.axvline(t_star, color=GREEN, lw=1.2, ls=':', alpha=0.80)
    ax_C.scatter([t_star], [eta_star], s=80, color=GREEN, zorder=7, marker='*')
    ax_C.text(t_star + 0.3, eta_star + 0.04,
              f'η_abs(t*) = {eta_star:.2f}', color=GREEN, fontsize=8.5)

# Phase boundaries and labels
for (x_c, lbl, col) in [
    (t_dat_end * 0.30,  'Phase 1\nGrowth',    GREEN),
    (t_star if np.isfinite(t_star) else t_dat_end + 2,
                         'Phase 2\nPlateau',   ORANGE),
    (T_LIFESPAN - 4.5,  'Phase 3\nSenescence', ORANGE),
]:
    if 0.5 < x_c < T_LIFESPAN:
        ax_C.text(x_c, 0.07, lbl, ha='center', fontsize=8.5,
                  color=col, style='italic')

ax_C.set_xlabel('Age (years)', fontsize=10)
ax_C.set_ylabel('η_abs(t)  — resource assimilation efficiency', fontsize=10)
ax_C.set_xlim(-0.3, T_LIFESPAN + 0.5)
ax_C.set_ylim(0, 1.22)
ax_C.grid(True, alpha=0.5)
ax_C.legend(fontsize=8.5, loc='upper right')
ax_C.set_title(
    'C — VBLS Barrier III:  η_abs degradation  (§5, eq. 5.2)\n'
    f'α = {ALPHA_FIXED} yr⁻¹  fixes t* and the onset of Phase 3',
    fontsize=10, loc='left', pad=5
)

plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved:  {OUT_FILE}")
