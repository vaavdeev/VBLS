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
VBLS — Appendix D.2.11 — Extended model: superposition of degradation channels
=========================================================================
Part 1: Three models (M1 VBLS / M2 VBLS+HR / M3 VBLS+HR+activity groups)
Part 2: Calibration analysis: Tls=75 vs Tls=82 vs T_max=122

Data: HUNT4 male trajectory (Letnes et al. 2020, Prog.Cardiovasc.Dis. 63(6):730-737;
      decade values from Fig. 3 of Letnes, Nes & Wisloff 2023). The activity-group
      levels -15/-9/-5 %/decade are assumed inputs whose source in the published
      HUNT results has not been established (App. D.2.11).
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
    """Entropy-export capacity — a property of the environment, constant over the life cycle."""
    baseline    = 0.003
    return baseline


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


# Ten-year decline per age decade, read off figure 3 (right panel) of
# Letnes, Nes & Wisloff, Int.J.Cardiol.Cardiovasc.Risk Prev. 16:200171
# (2023), which redraws the HUNT trajectories of Letnes et al. (2020)
# separately by sex. That figure gives the annual decline in percent;
# the decade value is ten times it, as the review's own text confirms
# (men after 70 approach 25% per decade in both HUNT and BLSA).
D_MEN = np.array([5.0, 5.6, 10.4, 12.8, 15.5, 24.5])


def cumtraj(avg_decline_per_decade):
    """
    Normalised longitudinal HUNT4 trajectory for a given mean decade decline.
    The shape is the measured profile D_MEN; only its level is moved, so the
    activity strata share one shape and differ in mean decline.
    """
    d = D_MEN * (avg_decline_per_decade / D_MEN.mean())
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

# HUNT4 male trajectory and activity groups (assumed levels per decade)
H4_MEN   = cumtraj(D_MEN.mean())   # mean
H4_INACT = cumtraj(15.0)
H4_MOD   = cumtraj(9.0)
H4_HI    = cumtraj(5.0)

# Activity scale factors (physical activity modulates the effective sigma0).
# Fitted below, once K2, B2, G2 are known: each group keeps the common
# mapping and differs only through sigma0_eff = sigma0 * scale.

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
# ── Activity scale factors, fitted at fixed K2, B2, G2 ────────────────────────
def fit_scale(target):
    """Fit sigma0 scale so that M2 reproduces the group's trajectory."""
    def loss(s):
        if s[0] <= 0.05 or s[0] > 5.0:
            return 1e6
        return float(np.sum((model2(AGES, dsig_at(AGES, S0_BASE * s[0]))
                             - target) ** 2))
    res = minimize(loss, [1.0], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-14, 'maxiter': 4000})
    return float(res.x[0])


SC_INACT = fit_scale(H4_INACT)
SC_MOD   = fit_scale(H4_MOD)
SC_HI    = fit_scale(H4_HI)
print(f"Activity scales: inactive={SC_INACT:.3f}  moderate={SC_MOD:.3f}  "
      f"high={SC_HI:.3f}")
print(f"  sigma0_eff: {S0_BASE*SC_INACT:.4f}  {S0_BASE*SC_MOD:.4f}  "
      f"{S0_BASE*SC_HI:.4f}   ratio={SC_INACT/SC_HI:.2f}")

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
print(f"RMSE activity groups: inactive={rmse_in:.4f}  moderate={rmse_mod:.4f}  high={rmse_hi:.4f}")
print(f"Fitted: K1={K1:.4f}, B1={B1:.4f}  |  K2={K2:.4f}, B2={B2:.4f}, G2={G2:.4f}")

# ── FIGURE 1: three models ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(19, 7))
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10, wspace=0.32)

# Panel A: M1 vs M2
a = axes[0]

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
            f'Removes 25-45 yr plateau; RMSE improves ~{rmse1/rmse2:.1f}×', fontweight='bold')
a.legend(fontsize=8.8, framealpha=0.95, loc='upper right')

# Panel B: activity groups (assumed levels)
a = axes[1]

a.plot(AGES, H4_INACT, 'o--', color=C_IN,  lw=1.8, ms=6, alpha=0.7,
       label='assumed: inactive (-15%/dec)')
a.plot(AGES, H4_MOD,   's--', color=C_SIG, lw=1.8, ms=6, alpha=0.7,
       label='assumed: moderate (-9%/dec)')
a.plot(AGES, H4_HI,    '^--', color=C_HI,  lw=1.8, ms=6, alpha=0.7,
       label='assumed: high-int. (-5%/dec)')
a.plot(AGES_FINE, f_inact_f, '-', color=C_IN,  lw=2.3,
       label=f'VBLS+act. inactive  RMSE={rmse_in:.3f}')
a.plot(AGES_FINE, f_mod_f,   '-', color=C_SIG, lw=2.3,
       label=f'VBLS+act. moderate  RMSE={rmse_mod:.3f}')
a.plot(AGES_FINE, f_hi_f,    '-', color=C_HI,  lw=2.3,
       label=f'VBLS+act. high-int  RMSE={rmse_hi:.3f}')

a.text(0.97, 0.04,
       'activity: sigma0_eff = sigma0 * scale\n'
       f'  Inactive:  scale={SC_INACT:.3f}  s0={S0_BASE*SC_INACT:.4f}\n'
       f'  Moderate:  scale={SC_MOD:.3f}  s0={S0_BASE*SC_MOD:.4f}\n'
       '  High-int:  not identified\n\n'
       f'ratio inactive/moderate: {SC_INACT/SC_MOD:.2f}',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.8,
       family='monospace',
       bbox=dict(boxstyle='round,pad=0.4', fc='#f8f8f8', ec='#aaa', lw=1))

a.set_ylim(0.58, 1.08)
a.set_xlabel('Age (yr)')
a.set_ylabel('VO2max / VO2max(25 yr)')
a.set_title('B.  Activity groups: physical activity as sigma0 modulator\n'
            'assumed group levels vs VBLS (3 activity levels)', fontweight='bold')
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
a.set_ylabel('Residual: model − reference curve')
a.set_title('C.  Residuals across three models\n'
            'vs HUNT4 men (M1, M2) and assumed group curves (M3)', fontweight='bold')
a.legend(fontsize=8.3, framealpha=0.95, loc='upper left')

a.text(0.97, 0.04,
       'M1->M2: 25-45 yr plateau\n'
       f'removed; RMSE {rmse1:.3f}->{rmse2:.3f}\n\n'
       'M2->M3: assumed levels;\n'
       f'sigma0 inact./mod.: {SC_INACT/SC_MOD:.2f};\n'
       'high-int. not identified',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec=C_VBLS, lw=1.2))

plt.savefig('superposition.pdf', bbox_inches='tight')
plt.savefig('superposition.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: superposition.pdf / .png")


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
    print(f"  {name:>24}  {s0:.4f}  {k:.4f}  {b:.4f}  {rmse:.4f}"
          f"  dSigma(75)={dsig[5]:.4f}")

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

rmse_all = [res['rmse'] for res in calib_results.values()]
a.text(0.97, 0.07,
       'The fit hardly depends\n'
       'on the t_death choice:\n'
       f'RMSE {min(rmse_all):.3f}-{max(rmse_all):.3f}.\n\n'
       'k and b are refitted for\n'
       'each sigma0, so the fit\n'
       'does not discriminate\n'
       'between the calibrations.',
       transform=a.transAxes, ha='right', va='bottom', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.45', fc='#fff8f0', ec=C_SIG, lw=1.2))

a.set_xlabel('Age (yr)')
a.set_ylabel('DeltaSigma(t) = Sigma(t) - Sigma(25)')
a.set_title('B.  DeltaSigma(t) under different sigma0 calibrations\n'
            'all give a nonzero, growing DeltaSigma over 25-85 yr',
            fontweight='bold')
a.legend(fontsize=8.8, framealpha=0.95, loc='upper left')

plt.savefig('calibration_tdeath.pdf', bbox_inches='tight')
plt.savefig('calibration_tdeath.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: calibration_tdeath.pdf / .png")
