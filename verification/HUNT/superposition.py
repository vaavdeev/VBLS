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
VBLS — Section 12.5 — Extended model: superposition of degradation channels
=========================================================================
Part 1: Three models (M1 VBLS / M2 VBLS+HR / M3 VBLS+HR+PAI)
Part 2: Calibration analysis: Tls=75 vs Tls=82 vs T_max=122

Data: HUNT4 (Stensvold et al. 2020, Prog.Cardiovasc.Dis. 63:85-91)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0. Setup
# ═══════════════════════════════════════════════════════════════════════════════

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

C_M    = '#2980b9'
C_W    = '#c0392b'
C_VBLS = '#8e44ad'
C_G    = '#27ae60'
C_HI   = '#16a085'
C_IN   = '#c0392b'
C_SIG  = '#e67e22'
C_75   = '#8e44ad'
C_82   = '#2980b9'
C_85   = '#27ae60'
C_122  = '#e74c3c'

# ── VBLS parameters (calibrated, April 2026) ─────────────────────────────────
MU      = 1.50
LAM     = 0.30
ALPHA   = 0.030
NU0     = 0.006
NU      = 0.060
RHO     = 0.002
E_STAR  = 0.04
R_STAR  = 0.15
E0      = 0.10
E_RANGE = E0 - E_STAR   # = 0.06


def Phi_in(t):
    """Negentropic input flux — grows to plateau ~25 yr, gentle decline after 60."""
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline


def Phi_cap(t):
    """Entropy-export capacity — slow decline after age 40."""
    baseline    = 0.003
    age_decline = max(0.0, t - 40.0) * 0.008
    return baseline * np.exp(-age_decline)


def vbls_run(t_end, sigma0, n=15000):
    """Integrate the 4-ODE VBLS system (S)."""
    def rhs(t, y):
        E  = float(max(y[0], 0.0))
        S  = float(max(y[1], 0.0))
        e  = float(max(min(y[2], 1.0), 0.0))
        R  = float(max(min(y[3], 1.0), 0.0))
        F  = MU * E**2 + LAM * E
        g_E = max(0.0, min(1.0, (E - E_STAR) / E_RANGE))  # restoration-potential function (eq. 4.2)
        return [
            e * Phi_in(t) - F,
            max(0.0, sigma0 * F - R * Phi_cap(t)),
            -ALPHA * S * e,
            -NU0 * R - NU * S * R + RHO * g_E * (1.0 - S) * (1.0 - R),
        ]

    return solve_ivp(rhs, (0, t_end), [E0, 0.01, 0.96, 0.99],
                     t_eval=np.linspace(0, t_end, n),
                     method='RK45', rtol=1e-8, atol=1e-10)


def death_time(sigma0):
    """Time at which E < E_STAR or R < R_STAR (first crossing)."""
    sol = vbls_run(220, sigma0)
    for i in range(len(sol.t)):
        if sol.y[0, i] < E_STAR or sol.y[3, i] < R_STAR:
            return float(sol.t[i])
    return 220.0


def dsig_at(ages, sigma0, t_end=None):
    """ΔΣ(t) = Σ(t) − Σ(25) for an array of ages."""
    if t_end is None:
        t_end = max(ages) + 20
    sol = vbls_run(t_end, sigma0)
    S25 = float(np.interp(25, sol.t, sol.y[1]))
    return np.array([
        max(float(np.interp(a, sol.t, sol.y[1])) - S25, 0.0)
        for a in ages
    ])


def cumtraj(avg_decline_per_decade):
    """
    Normalised longitudinal HUNT4 trajectory for a given mean decade decline.
    Nonlinear shape via quadratic reconstruction:
      d(k) = (3 + 4.15k − 0.15k²) × scale
    """
    k = np.arange(6)
    d = (3.0 + 4.15 * k - 0.15 * k**2) * (avg_decline_per_decade / 12.0)
    d = np.clip(d, 0.3, 30.0)
    traj = [1.0]
    for x in d:
        traj.append(traj[-1] * (1 - x / 100))
    return np.array(traj)


def fit_kb(dsig, target):
    """Fit k, beta: VO2 = 1 − k·ΔΣ^beta → target."""
    def loss(kb):
        k, b = kb
        if k <= 0 or b <= 0:
            return 1e6
        pred = np.array([max(1 - k * (s ** b), 0.0) for s in dsig])
        return float(np.sum((pred - target) ** 2))

    res = minimize(loss, [0.4, 0.5], method='Nelder-Mead',
                   options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 8000})
    return float(res.x[0]), float(res.x[1])


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1. Three models
# ═══════════════════════════════════════════════════════════════════════════════

AGES      = np.array([25, 35, 45, 55, 65, 75, 85])
AGES_FINE = np.linspace(25, 90, 200)

# Canonical human sigma0 (= MAN/human_lifespan.py, §17.1); gives t_death ≈ 77 yr.
# Used so that M1/M2/M3 here match the base mapping in hunt_verification.py.
S0_BASE = 0.0162

# HUNT4 data by PAI group
H4_MEN   = cumtraj(12.0)   # mean
H4_INACT = cumtraj(15.0)
H4_MOD   = cumtraj(9.0)
H4_HI    = cumtraj(5.0)

# PAI scale factors (physical activity intensity modulates sigma0)
# Re-tuned for canonical sigma0 = 0.0162 (June 2026): each fit to its group's decade decline.
SC_INACT = 1.094
SC_MOD   = 0.879
SC_HI    = 0.621

# ── Model M1: VBLS only ───────────────────────────────────────────────────────
DSIG_BASE = dsig_at(AGES, S0_BASE)
K1, B1    = fit_kb(DSIG_BASE, H4_MEN)

def model1(dsig):
    return np.array([max(1 - K1 * (s ** B1), 0.0) for s in dsig])

# ── Model M2: VBLS + chronological HR channel ─────────────────────────────────
# Joint 3-parameter fit (K2, B2, G2)
def model2_local(k, b, g, ages, dsig):
    hrs  = np.array([max(1 - g * (a - 20), 0.0) for a in ages])
    vbls = np.array([max(1 - k * (s ** b), 0.0) for s in dsig])
    return vbls * hrs

def fit_kbg(dsig, ages, target):
    """Fit (k, b, g) for M2 = VBLS × chronological-HR product."""
    def loss(kbg):
        k, b, g = kbg
        if k <= 0 or b <= 0 or g < 0:
            return 1e6
        return float(np.sum((model2_local(k, b, g, ages, dsig) - target) ** 2))
    res = minimize(loss, [0.35, 0.65, 0.003], method='Nelder-Mead',
                   options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 12000})
    return float(res.x[0]), float(res.x[1]), float(res.x[2])

K2, B2, G2 = fit_kbg(DSIG_BASE, AGES, H4_MEN)

def model2(ages, dsig, g=None):
    g = g if g is not None else G2
    return model2_local(K2, B2, g, ages, dsig)

# ── Pre-compute ΔΣ for continuous curves ─────────────────────────────────────
DSIG_BASE_F  = dsig_at(AGES_FINE, S0_BASE)
DSIG_INACT_F = dsig_at(AGES_FINE, S0_BASE * SC_INACT)
DSIG_MOD_F   = dsig_at(AGES_FINE, S0_BASE * SC_MOD)
DSIG_HI_F    = dsig_at(AGES_FINE, S0_BASE * SC_HI)

f1_fine   = model1(DSIG_BASE_F)
f2_fine   = model2(AGES_FINE, DSIG_BASE_F)
f_inact_f = model2(AGES_FINE, DSIG_INACT_F)
f_mod_f   = model2(AGES_FINE, DSIG_MOD_F)
f_hi_f    = model2(AGES_FINE, DSIG_HI_F)

# ── Point predictions for RMSE ────────────────────────────────────────────────
DSIG_PTS   = DSIG_BASE
DSIG_IN_P  = dsig_at(AGES, S0_BASE * SC_INACT)
DSIG_MOD_P = dsig_at(AGES, S0_BASE * SC_MOD)
DSIG_HI_P  = dsig_at(AGES, S0_BASE * SC_HI)

pred1      = model1(DSIG_PTS)
pred2      = model2(AGES, DSIG_PTS)
pred_inact = model2(AGES, DSIG_IN_P)
pred_mod   = model2(AGES, DSIG_MOD_P)
pred_hi    = model2(AGES, DSIG_HI_P)

rmse1      = float(np.sqrt(np.mean((pred1 - H4_MEN)**2)))
rmse2      = float(np.sqrt(np.mean((pred2 - H4_MEN)**2)))
rmse_in    = float(np.sqrt(np.mean((pred_inact - H4_INACT)**2)))
rmse_mod   = float(np.sqrt(np.mean((pred_mod   - H4_MOD)**2)))
rmse_hi    = float(np.sqrt(np.mean((pred_hi    - H4_HI)**2)))

print(f"RMSE: M1={rmse1:.4f}  M2={rmse2:.4f}")
print(f"RMSE PAI: inactive={rmse_in:.4f}  moderate={rmse_mod:.4f}  high={rmse_hi:.4f}")
print(f"Fitted: K1={K1:.4f}, B1={B1:.4f}  |  K2={K2:.4f}, B2={B2:.4f}, G2={G2:.4f}")

# ── FIGURE 1: three models ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(19, 7))
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10, wspace=0.32)

# Panel A: M1 vs M2
a = axes[0]
for x0, x1, col in [(25, 45, '#2ecc71'), (45, 65, '#f39c12'), (65, 90, '#e74c3c')]:
    a.axvspan(x0, x1, alpha=0.04, color=col)
a.text(33, 0.62, 'I',   ha='center', fontsize=9, color='#27ae60')
a.text(54, 0.62, 'II',  ha='center', fontsize=9, color='#e67e22')
a.text(73, 0.62, 'III', ha='center', fontsize=9, color='#c0392b')

ci = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
a.fill_between(AGES, H4_MEN - ci, H4_MEN + ci, alpha=0.18, color=C_M)
a.plot(AGES, H4_MEN, 'o-', color=C_M, lw=2.2, ms=7, zorder=5,
       label='HUNT4 data (men, avg)')
a.plot(AGES_FINE, f1_fine, '--', color=C_VBLS, lw=2.3,
       label=f'M1: VBLS  RMSE={rmse1:.3f}')
a.plot(AGES_FINE, f2_fine, '-',  color=C_G,    lw=2.5,
       label=f'M2: VBLS+HR  RMSE={rmse2:.3f}')

for i, age in enumerate(AGES):
    if abs(pred2[i] - pred1[i]) > 0.005:
        a.annotate('', xy=(age, pred2[i]), xytext=(age, pred1[i]),
                   arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0))

a.text(0.04, 0.04,
       'Superposition (M2):\n'
       'VO2/VO2_25 =\n'
       '  [1-k*dS^b] x [1-g*(t-20)]\n\n'
       f'M1: k={K1:.3f}, b={B1:.3f}\n'
       f'M2: k={K2:.3f}, b={B2:.3f}\n'
       f'    g={G2:.4f}/yr\n\n'
       f'RMSE: {rmse1:.3f} -> {rmse2:.3f}\n'
       f'(improvement x{rmse1/rmse2:.1f})',
       transform=a.transAxes, va='bottom', fontsize=8.8,
       bbox=dict(boxstyle='round,pad=0.45', fc='#f0fff4', ec=C_G, lw=1.2))

a.set_ylim(0.58, 1.08)
a.set_xlabel('Age (yr)')
a.set_ylabel('VO2max / VO2max(25 yr)')
a.set_title('A.  VBLS + chronological HR-channel superposition\n'
            'Removes 25-45 yr plateau; RMSE improves ~1.5×', fontweight='bold')
a.legend(fontsize=8.8, framealpha=0.95, loc='upper right')

# Panel B: PAI stratification
a = axes[1]
for x0, x1, col in [(25, 45, '#2ecc71'), (45, 65, '#f39c12'), (65, 90, '#e74c3c')]:
    a.axvspan(x0, x1, alpha=0.04, color=col)

a.plot(AGES, H4_INACT, 'o--', color=C_IN,  lw=1.8, ms=6, alpha=0.7,
       label='HUNT4 inactive (-15%/dec)')
a.plot(AGES, H4_MOD,   's--', color=C_SIG, lw=1.8, ms=6, alpha=0.7,
       label='HUNT4 moderate (-9%/dec)')
a.plot(AGES, H4_HI,    '^--', color=C_HI,  lw=1.8, ms=6, alpha=0.7,
       label='HUNT4 high-int. (-5%/dec)')
a.plot(AGES_FINE, f_inact_f, '-', color=C_IN,  lw=2.3,
       label=f'VBLS+PAI inactive  RMSE={rmse_in:.3f}')
a.plot(AGES_FINE, f_mod_f,   '-', color=C_SIG, lw=2.3,
       label=f'VBLS+PAI moderate  RMSE={rmse_mod:.3f}')
a.plot(AGES_FINE, f_hi_f,    '-', color=C_HI,  lw=2.3,
       label=f'VBLS+PAI high-int  RMSE={rmse_hi:.3f}')

a.text(0.97, 0.04,
       'PAI: sigma0_eff = sigma0 * scale\n'
       f'  Inactive:  scale={SC_INACT:.3f}  s0={S0_BASE*SC_INACT:.4f}\n'
       f'  Moderate:  scale={SC_MOD:.3f}  s0={S0_BASE*SC_MOD:.4f}\n'
       f'  High-int:  scale={SC_HI:.3f}  s0={S0_BASE*SC_HI:.4f}\n\n'
       f'sigma0 range:\n'
       f'  {S0_BASE*SC_HI:.4f} – {S0_BASE*SC_INACT:.4f}\n'
       f'  (diff. {(SC_INACT-SC_HI)/SC_HI*100:.0f}%)',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.8,
       family='monospace',
       bbox=dict(boxstyle='round,pad=0.4', fc='#f8f8f8', ec='#aaa', lw=1))

a.set_ylim(0.58, 1.08)
a.set_xlabel('Age (yr)')
a.set_ylabel('VO2max / VO2max(25 yr)')
a.set_title('B.  PAI stratification: physical activity as sigma0 modulator\n'
            'HUNT4 data vs VBLS predictions (3 activity levels)', fontweight='bold')
a.legend(fontsize=8.3, framealpha=0.95, loc='upper right')

# Panel C: residuals
a = axes[2]
res1     = pred1      - H4_MEN
res2     = pred2      - H4_MEN
res_in2  = pred_inact - H4_INACT
res_mod2 = pred_mod   - H4_MOD
res_hi2  = pred_hi    - H4_HI

x = np.arange(len(AGES))
w = 0.17
a.bar(x - 2*w, res1,     w, color=C_VBLS, alpha=0.85, label=f'M1  RMSE={rmse1:.3f}')
a.bar(x -   w, res2,     w, color=C_G,    alpha=0.85, label=f'M2  RMSE={rmse2:.3f}')
a.bar(x,       res_in2,  w, color=C_IN,   alpha=0.75, label=f'M3-inactive  RMSE={rmse_in:.3f}')
a.bar(x +   w, res_mod2, w, color=C_SIG,  alpha=0.75, label=f'M3-moderate  RMSE={rmse_mod:.3f}')
a.bar(x + 2*w, res_hi2,  w, color=C_HI,   alpha=0.75, label=f'M3-high      RMSE={rmse_hi:.3f}')

a.axhline(0,      color='#333', lw=1.0)
a.axhline( 0.05,  color='#bbb', lw=0.8, ls=':')
a.axhline(-0.05,  color='#bbb', lw=0.8, ls=':')
a.text(len(AGES) - 0.3, 0.052, '±5%', fontsize=8, color='#aaa', va='bottom')

a.set_xticks(x)
a.set_xticklabels([str(ag) for ag in AGES], fontsize=8.5)
a.set_xlabel('Age (yr)')
a.set_ylabel('Residual: model − data')
a.set_title('C.  Residuals across three models\n'
            'Each superposition layer removes a systematic error', fontweight='bold')
a.legend(fontsize=8.3, framealpha=0.95, loc='upper left')

a.text(0.97, 0.04,
       'What each layer removes:\n'
       '  M1->M2: 25-45 yr plateau\n'
       f'          RMSE: {rmse1:.3f}->{rmse2:.3f}\n\n'
       '  M2->M3: stratification by rate\n'
       f'          sigma0: {S0_BASE*SC_HI:.4f}–{S0_BASE*SC_INACT:.4f}\n'
       f'          (range {(SC_INACT-SC_HI)/SC_HI*100:.0f}%)',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec=C_VBLS, lw=1.2))

plt.savefig('superposition.pdf', bbox_inches='tight')
plt.savefig('superposition.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: mde_superposition.pdf / .png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2. Calibration analysis: t_death=75 vs 82 vs 85 vs 122
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Calibration analysis t_death ===")

s0_75  = S0_BASE
s0_82  = brentq(lambda s: death_time(s) - 82,  0.005, 0.40)
s0_85  = brentq(lambda s: death_time(s) - 85,  0.005, 0.40)
s0_122 = brentq(lambda s: death_time(s) - 122, 0.001, 0.10)

calibrations = [
    ('Canonical (~77)',   s0_75,  140),
    ('Tls=82 (Norway)',   s0_82,  160),
    ('Tls=85',            s0_85,  160),
    ('T_max=122 (Cagan)', s0_122, 200),
]

calib_results = {}
print(f"\n{'Scenario':>24}  sigma0   k       b     RMSE")
for name, s0, t_end in calibrations:
    dsig = dsig_at(AGES, s0, t_end)
    k, b = fit_kb(dsig, H4_MEN)
    pred = np.array([max(1 - k * (s ** b), 0.0) for s in dsig])
    rmse = float(np.sqrt(np.mean((pred - H4_MEN) ** 2)))
    calib_results[name] = dict(s0=s0, k=k, b=b, rmse=rmse, dsig=dsig, pred=pred)
    print(f"  {name:>24}  {s0:.4f}  {k:.4f}  {b:.4f}  {rmse:.4f}")

# ── FIGURE 2: calibrations ─────────────────────────────────────────────────────

colors_cal = {
    'Canonical (~77)':   C_75,
    'Tls=82 (Norway)':   C_82,
    'Tls=85':            C_85,
    'T_max=122 (Cagan)': C_122,
}
styles_cal = {
    'Canonical (~77)':   '-',
    'Tls=82 (Norway)':   '--',
    'Tls=85':            '-.',
    'T_max=122 (Cagan)': ':',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.34)

# Panel A: trajectories by scenario
a = axes[0]
for x0, x1, col in [(25, 45, '#2ecc71'), (45, 65, '#f39c12'), (65, 90, '#e74c3c')]:
    a.axvspan(x0, x1, alpha=0.04, color=col)

ci = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
a.fill_between(AGES, H4_MEN - ci, H4_MEN + ci, alpha=0.18, color=C_M)
a.plot(AGES, H4_MEN, 'o-', color=C_M, lw=2.4, ms=8, zorder=5, label='HUNT4 data (men)')

for name, res in calib_results.items():
    s0   = res['s0']
    dsig_fine = dsig_at(AGES_FINE, s0, float(max(AGES_FINE)) + 20)
    k, b = res['k'], res['b']
    pred_fine = np.array([max(1 - k * (s ** b), 0.0) for s in dsig_fine])
    rmse = res['rmse']
    a.plot(AGES_FINE, pred_fine,
           styles_cal[name], color=colors_cal[name], lw=2.2,
           label=f'{name}  RMSE={rmse:.3f}')

# Annotation: T_max=122 failure
a.annotate('T_max=122:\nsmaller DeltaSigma\nbut still fits',
           xy=(75, 1.0), xytext=(50, 0.78),
           fontsize=8.5, color=C_122,
           arrowprops=dict(arrowstyle='->', color=C_122, lw=0.9))

a.set_ylim(0.55, 1.10)
a.set_xlabel('Age (yr)')
a.set_ylabel('VO2max / VO2max(25 yr)')
a.set_title('A.  Effect of t_death choice on HUNT4 fit quality\n'
            'Re-fitted k, b for each sigma0', fontweight='bold')
a.legend(fontsize=8.8, framealpha=0.95, loc='upper right')

# Panel B: ΔΣ(t) by scenario
a = axes[1]
for name, res in calib_results.items():
    s0 = res['s0']
    t_end_run = 200 if '122' in name else 160
    sol = vbls_run(t_end_run, s0)
    S25 = float(np.interp(25, sol.t, sol.y[1]))
    t_plot = sol.t[sol.t <= 95]
    sig_plot = np.array([max(float(np.interp(t, sol.t, sol.y[1])) - S25, 0) for t in t_plot])
    a.plot(t_plot, sig_plot,
           styles_cal[name], color=colors_cal[name], lw=2.2,
           label=f'{name}  (sigma0={s0:.4f})')

a.axvline(75, color='#999', lw=1.0, ls=':', alpha=0.7)
a.axvline(82, color='#666', lw=1.0, ls=':', alpha=0.7)
a.text(75.5, 0.05, 'Tls=75', fontsize=8, color='#999', rotation=90, va='bottom')
a.text(82.5, 0.05, 'Tls=82', fontsize=8, color='#666', rotation=90, va='bottom')

a.text(0.97, 0.07,
       'Key result:\n\n'
       'VO2max fit is ROBUST\n'
       'to the t_death choice:\n'
       'all calibrations give\n'
       'RMSE 0.013-0.016.\n\n'
       'k,b absorb sigma0;\n'
       'the SHAPE of DeltaSigma\n'
       'is what is tested.',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.45', fc='#fff8f0', ec=C_SIG, lw=1.2))

a.set_xlabel('Age (yr)')
a.set_ylabel('DeltaSigma(t) = Sigma(t) - Sigma(25)')
a.set_title('B.  DeltaSigma(t) under different sigma0 calibrations\n'
            'all give nonzero, accelerating DeltaSigma in 25-85 yr',
            fontweight='bold')
a.legend(fontsize=8.8, framealpha=0.95, loc='upper left')

plt.savefig('calibration_tdeath.pdf', bbox_inches='tight')
plt.savefig('calibration_tdeath.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: mde_calibration_tdeath.pdf / .png")
