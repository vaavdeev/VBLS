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
VBLS — Section 12.5 — Verification of eta_abs(t) against HUNT3/HUNT4 data
========================================================================
Data:
  - HUNT3: Loe et al. (2013) PLoS ONE 8(5):e64319
    N=3816, direct VO2max measurement, ages 20–90 yr
  - HUNT4: Letnes et al. (2020) Prog. Cardiovasc. Dis. 63(6):730–737
    N=1431 repeated measurements over 10 yr
  - FRIEND: Kaminsky et al. (2019) Mayo Clin. Proc. — for ages 70–89

Logic:
  1. HUNT3 cross-sectional data
  2. HUNT4 longitudinal data (decade declines read off Letnes 2023, fig.3)
  3. VBLS simulation (sigma0=0.0162, final calibration April 2026)
  4. Mapping: VO2max/VO2max_0 = 1 - k*(ΔΣ)^beta, k and beta fit by minimize()
  5. Panel D: Sigma(t) and eta_abs(t) vs cumulative VO2max decline
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── VBLS parameters (calibrated, April 2026) ──────────────────────────────────
MU      = 1.50
LAM     = 0.30
SIGMA0  = 0.0162   # human calibration: cross-checked with Cagan (2022)
ALPHA   = 0.030    # constrained by HUNT3/4 VO2max data
NU0     = 0.006
NU      = 0.060
RHO     = 0.002
E_STAR  = 0.04
R_STAR  = 0.15
E0      = 0.10
E_RANGE = E0 - E_STAR   # = 0.06

# ── Colors ────────────────────────────────────────────────────────────────────
C_M    = '#2980b9'   # men
C_W    = '#c0392b'   # women
C_VBLS = '#8e44ad'   # VBLS prediction
C_SIG  = '#e67e22'   # Sigma(t)
C_MN   = '#95a5a6'   # neutral grey

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
    'grid.alpha':        0.15,
    'grid.linestyle':    '--',
    'figure.dpi':        150,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. External functions (human-calibrated)
# ═══════════════════════════════════════════════════════════════════════════════

def Phi_in(t):
    """Negentropic input flux — grows to plateau ~25 yr, gentle decline after 60."""
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline


def Phi_cap(t):
    """Entropy-export capacity — a property of the environment, constant over the life cycle."""
    baseline    = 0.003
    return baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VBLS simulation
# ═══════════════════════════════════════════════════════════════════════════════

def vbls_run(t_end, sigma0=SIGMA0, n=15000):
    """Integrate the 4-ODE VBLS system (S)."""

    def rhs(t, y):
        E  = max(y[0], 0.0)
        S  = max(y[1], 0.0)
        e  = max(min(y[2], 1.0), 0.0)
        R  = max(min(y[3], 1.0), 0.0)
        F  = MU * E**2 + LAM * E
        # Restoration-potential function g(E) = (E - E*) / (E0 - E*)  ∈ [0, 1]  (eq. 4.2)
        g_E = max(0.0, min(1.0, (E - E_STAR) / E_RANGE))
        return [
            e * Phi_in(t) - F,
            max(0.0, sigma0 * F - R * Phi_cap(t)),
            -ALPHA * S * e,
            -NU0 * R - NU * S * R + RHO * g_E * (1.0 - S) * (1.0 - R),
        ]

    return solve_ivp(
        rhs, (0, t_end), [E0, 0.01, 0.96, 0.99],
        t_eval=np.linspace(0, t_end, n),
        method='RK45', rtol=1e-8, atol=1e-10,
    )


sol     = vbls_run(140)
t_      = sol.t
eta_    = sol.y[2]
Sig_    = sol.y[1]
E_      = sol.y[0]

Sig_25  = float(np.interp(25, t_, Sig_))  # Sigma at age 25


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HUNT data
# ═══════════════════════════════════════════════════════════════════════════════

# HUNT3 cross-sectional (Loe et al., 2013, Table 2 — men and women)
AGES_CS   = np.array([25, 35, 45, 55, 65, 75])
VO2_MEN   = np.array([54.4, 49.0, 47.0, 42.0, 39.0, 34.0])
VO2_WOM   = np.array([43.0, 40.0, 38.0, 34.0, 31.0, 27.0])
NM_MEN    = VO2_MEN / VO2_MEN[0]
NM_WOM    = VO2_WOM / VO2_WOM[0]

# Relative SD (from Loe annotation: SD(20–29) = 8.4 men, 7.7 women)
SD_REL_M  = 8.4  / 54.4
SD_REL_W  = 7.7  / 43.0

# HUNT4 longitudinal
# Ten-year decline per age decade, read off figure 3 (right panel) of
# Letnes, Nes & Wisloff, Int.J.Cardiol.Cardiovasc.Risk Prev. 16:200171
# (2023), which redraws the HUNT trajectories of Letnes et al. (2020)
# separately by sex. That figure gives the annual decline in percent;
# the decade value is ten times it, as the review's own text confirms
# (men after 70 approach 25% per decade in both HUNT and BLSA).
AGES_LONG = np.array([25, 35, 45, 55, 65, 75, 85])
d_men_pct = np.array([5.0, 5.6, 10.4, 12.8, 15.5, 24.5])
d_wom_pct = np.array([3.1, 5.9, 10.3, 11.2, 13.6, 12.8])

def cumulative_traj(d_pct):
    """Cumulative normalised trajectory from vector of decade declines."""
    traj = [1.0]
    for d in d_pct:
        traj.append(traj[-1] * (1 - d / 100))
    return np.array(traj)

HUNT4_MEN = cumulative_traj(d_men_pct)
HUNT4_WOM = cumulative_traj(d_wom_pct)

# Approximate CI (widens toward older ages)
CI_MEN    = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
CI_WOM    = CI_MEN.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Mapping VO2max = 1 − k·ΔΣ^beta
# ═══════════════════════════════════════════════════════════════════════════════

def sigma_increment(ages):
    """ΔΣ(t) = Σ(t) − Σ(25) for an array of ages."""
    return np.array([
        max(float(np.interp(a, t_, Sig_)) - Sig_25, 0.0)
        for a in ages
    ])

def vo2_model(k, b, ages):
    dsig = sigma_increment(ages)
    return np.array([max(1.0 - k * (s ** b), 0.0) for s in dsig])

def loss_fn(params):
    return np.sum((vo2_model(*params, AGES_LONG) - HUNT4_MEN) ** 2)

res    = minimize(loss_fn, [0.39, 0.43], method='Nelder-Mead',
                  options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 10000})
K_OPT, B_OPT = res.x

# Continuous VBLS curve
AGES_FINE    = np.linspace(25, 90, 300)
VO2_FINE     = vo2_model(K_OPT, B_OPT, AGES_FINE)
VO2_VBLS_PTS = vo2_model(K_OPT, B_OPT, AGES_LONG)

RMSE = float(np.sqrt(np.mean((VO2_VBLS_PTS - HUNT4_MEN) ** 2)))

print(f"Mapping: k={K_OPT:.4f}, beta={B_OPT:.4f}")
print(f"RMSE(VBLS vs HUNT4-men) = {RMSE:.4f}")

# Decade declines from VBLS
vbls_declines = [
    (1 - VO2_VBLS_PTS[i+1] / VO2_VBLS_PTS[i]) * 100
    for i in range(len(AGES_LONG) - 1)
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Figure
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.08,
                    wspace=0.32, hspace=0.44)
ax = axes.flatten()

# ── Panel A: absolute values ──────────────────────────────────────────────────
a = ax[0]

a.fill_between(AGES_CS,
               VO2_MEN - SD_REL_M * VO2_MEN,
               VO2_MEN + SD_REL_M * VO2_MEN,
               alpha=0.12, color=C_M)
a.fill_between(AGES_CS,
               VO2_WOM - SD_REL_W * VO2_WOM,
               VO2_WOM + SD_REL_W * VO2_WOM,
               alpha=0.12, color=C_W)

a.plot(AGES_CS, VO2_MEN, 'o-', color=C_M, lw=2, ms=7,
       label='HUNT3 men (cross-sectional)')
a.plot(AGES_CS, VO2_WOM, 's-', color=C_W, lw=2, ms=7,
       label='HUNT3 women (cross-sectional)')
a.plot(AGES_LONG[:-1], HUNT4_MEN[:-1] * VO2_MEN[0], 'o--', color=C_M,
       lw=1.8, ms=6, alpha=0.6, label='HUNT4 men (longitudinal)')
a.plot(AGES_LONG[:-1], HUNT4_WOM[:-1] * VO2_WOM[0], 's--', color=C_W,
       lw=1.8, ms=6, alpha=0.6, label='HUNT4 women (longitudinal)')

# Survival bias annotation
a.annotate('',
           xy=(55, HUNT4_MEN[3] * VO2_MEN[0]),
           xytext=(55, NM_MEN[3] * VO2_MEN[0]),
           arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
a.text(57, (HUNT4_MEN[3] * VO2_MEN[0] + NM_MEN[3] * VO2_MEN[0]) / 2,
       'Survival\nbias', fontsize=8, color='#555')

a.set_xlabel('Age (yr)')
a.set_ylabel('VO₂max (ml/kg/min)')
a.set_title('A.  VO₂max by age: HUNT3 vs HUNT4\nAbsolute values',
            fontweight='bold')
a.legend(fontsize=8.5, framealpha=0.93)
a.text(0.97, 0.92, 'N=3816 (HUNT3)\nN=1431 (HUNT4)',
       transform=a.transAxes, ha='right', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

# ── Panel B: normalized + VBLS ────────────────────────────────────────────────
a = ax[1]

# Phase zones
a.axvspan(25, 45, alpha=0.04, color='#2ecc71')
a.axvspan(45, 65, alpha=0.04, color='#f39c12')
a.axvspan(65, 90, alpha=0.06, color='#e74c3c')
a.text(33, 0.62, 'Phase I\n(Σ≈0)',  ha='center', fontsize=8, color='#27ae60')
a.text(54, 0.62, 'Phase II',         ha='center', fontsize=8, color='#e67e22')
a.text(73, 0.62, 'Phase III',        ha='center', fontsize=8, color='#c0392b')

# CI bands for HUNT4
a.fill_between(AGES_LONG, HUNT4_MEN - CI_MEN, HUNT4_MEN + CI_MEN,
               alpha=0.15, color=C_M)
a.fill_between(AGES_LONG, HUNT4_WOM - CI_WOM, HUNT4_WOM + CI_WOM,
               alpha=0.15, color=C_W)

a.plot(AGES_LONG, HUNT4_MEN, 'o-', color=C_M, lw=2.2, ms=7,
       label='HUNT4 men (longitudinal)')
a.plot(AGES_LONG, HUNT4_WOM, 's-', color=C_W, lw=2.2, ms=7,
       label='HUNT4 women (longitudinal)')
a.plot(AGES_CS, NM_MEN, 'o:', color=C_M, lw=1.2, ms=5, alpha=0.45,
       label='HUNT3 men (cross-sectional)')
a.plot(AGES_CS, NM_WOM, 's:', color=C_W, lw=1.2, ms=5, alpha=0.45,
       label='HUNT3 women (cross-sectional)')
a.plot(AGES_FINE, VO2_FINE, '-', color=C_VBLS, lw=2.5, alpha=0.92,
       label=r'VBLS: $1\!-\!k\cdot\Delta\Sigma^\beta$')

r2 = 1 - RMSE**2 / np.var(HUNT4_MEN)
a.text(0.97, 0.52,
       f'VBLS vs HUNT4 (men):\n'
       f'  RMSE = {RMSE:.3f}\n'
       f'  R²   = {r2:.3f}\n\n'
       f'Mapping:\n'
       r'$\frac{\mathrm{VO}_2}{\mathrm{VO}_{2,25}}=1-k\,\Delta\Sigma^\beta$'
       '\n'
       f'k = {K_OPT:.3f},  beta = {B_OPT:.3f}',
       transform=a.transAxes, ha='right', va='center', fontsize=8.5,
       family='monospace',
       bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec=C_VBLS, lw=1.2))

a.set_ylim(0.55, 1.10)
a.set_xlabel('Age (yr)')
a.set_ylabel('VO₂max / VO₂max(25 yr)')
a.set_title('B.  Normalized VO₂max decline\nVBLS vs HUNT3/HUNT4',
            fontweight='bold')
a.legend(fontsize=8.5, framealpha=0.93, loc='upper right')

# ── Panel C: decade-wise decline ──────────────────────────────────────────────
a = ax[2]

x_pos  = np.arange(6)
w      = 0.26
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']

a.bar(x_pos - w, d_men_pct,    w, color=C_M,    alpha=0.82,
      label='HUNT4 men', zorder=4)
a.bar(x_pos,     d_wom_pct,    w, color=C_W,    alpha=0.82,
      label='HUNT4 women', zorder=4)
a.bar(x_pos + w, vbls_declines, w, color=C_VBLS, alpha=0.82,
      label='VBLS',           zorder=4)

slope_m, icept_m, r_m, _, _ = linregress(x_pos, d_men_pct)
a.plot(x_pos, icept_m + slope_m * x_pos, 'k--', lw=1.5, alpha=0.55,
       label=f'Trend HUNT4-M: +{slope_m:.1f}%/dec.')

a.set_xticks(x_pos)
a.set_xticklabels(labels, fontsize=8.5)
a.set_xlabel('Age decade')
a.set_ylabel('VO₂max decline per 10 yr (%)')
a.set_title('C.  Decade-wise acceleration of VO₂max decline\nHUNT4 + VBLS',
            fontweight='bold')
a.legend(fontsize=8.5, framealpha=0.93)

# Explanation of divergence in last decade
a.text(0.97, 0.15,
       'Decade 6 (70–79 yr):\n'
       f'VBLS: {vbls_declines[-1]:.1f}%   HUNT4: 20.0%\n'
       '→ survival bias in HUNT4:\n'
       '  high sigma0 = dropped out earlier',
       transform=a.transAxes, ha='right', fontsize=8.2,
       bbox=dict(boxstyle='round,pad=0.35', fc='#fff8f0', ec=C_VBLS, lw=1))

# ── Panel D: Σ(t) and η(t) + cumulative VO2max decline ───────────────────────
a = ax[3]

t_plot   = t_[t_ <= 90]
Sig_plot = Sig_[t_ <= 90]
eta_plot = np.clip(eta_[t_ <= 90], 0, 1)

l1, = a.plot(t_plot, Sig_plot, '-',  color=C_SIG,  lw=2.5,
             label='Sigma(t) — structural damage')
l2, = a.plot(t_plot, eta_plot, '--', color=C_VBLS, lw=2.5,
             label='eta_abs(t)')

ax_r = a.twinx()
cum_decline = (1 - HUNT4_MEN) * 100
ax_r.plot(AGES_LONG, cum_decline, 'o-', color=C_M, lw=2.2, ms=7)
ax_r.set_ylabel('Cumulative VO₂max decline (%)', color=C_M)
ax_r.tick_params(axis='y', labelcolor=C_M)
ax_r.spines['right'].set_edgecolor(C_M)

# Phase zones
a.axvspan(25, 45, alpha=0.04, color='#2ecc71')
a.axvspan(45, 65, alpha=0.04, color='#f39c12')
a.axvspan(65, 90, alpha=0.07, color='#e74c3c')

# Key event annotations
sig_at_45 = float(np.interp(45, t_, Sig_))
sig_at_55 = float(np.interp(55, t_, eta_))
a.annotate('Sigma begins\nrising (~45 yr)',
           xy=(45, sig_at_45),
           xytext=(33, 0.5),
           fontsize=8.5, color=C_SIG,
           arrowprops=dict(arrowstyle='->', color=C_SIG, lw=0.9))
a.annotate('eta_abs begins\ndeclining (~55 yr)',
           xy=(55, sig_at_55),
           xytext=(40, 0.65),
           fontsize=8.5, color=C_VBLS,
           arrowprops=dict(arrowstyle='->', color=C_VBLS, lw=0.9))

a.set_xlabel('Age (yr)')
a.set_ylabel('Sigma(t) and eta_abs(t) [dimensionless]')
a.set_title('D.  VBLS dynamics and VO₂max decline\n'
            'Delay between "Sigma rises" and "eta_abs falls" = memory effect',
            fontweight='bold')
lines = [
    l1,
    l2,
    Line2D([0], [0], marker='o', color=C_M, ms=7, lw=2,
           label='Cumulative VO₂max decline (HUNT4, men)'),
]
a.legend(handles=lines, fontsize=8.5, framealpha=0.93, loc='upper left')

plt.savefig('hunt_verification.pdf', bbox_inches='tight')
plt.savefig('hunt_verification.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: mde_hunt_verification.pdf / .png")
print(f"Mapping parameters: k={K_OPT:.4f}, beta={B_OPT:.4f}, RMSE={RMSE:.4f}")
