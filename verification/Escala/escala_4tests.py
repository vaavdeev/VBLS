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
VBLS — Scaling law verification (Escala 2022)
Four tests of theoretical predictions:

  Test 1 — Kleiber's law:         P ~ M^chi   (input assumption chi = 3/4; dataset check)
  Test 2 — Longevity allometry:   T_ls ~ M^a    (VBLS: a = 0.25)
  Test 3 — T_ls ~ (P/M)^p_scale: p_scale = -1.0 (VBLS: -1.0)
  Test 4 — Rubner invariant:      P/M × T_ls ≈ const

Additional:
  Full dataset (276 species, 20 orders of magnitude in mass) — cross-class check

Data: Escala A. (2022) Sci. Rep. 12:2407
File: escala_data.csv

All results computed from data — nothing hardcoded.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import linregress

# ── Style ─────────────────────────────────────────────────────────────────────
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
    'figure.dpi':        150,
})

C_MM  = '#95a5a6'   # mammals
C_ALL = '#bdc3c7'   # all classes
C_REG = '#555555'   # regression
C_VB  = '#8e44ad'   # VBLS prediction
C_H1  = '#2980b9'   # Human_1
C_H2  = '#c0392b'   # Human_2
C_AV  = '#27ae60'   # birds
C_EC  = '#e67e22'   # ectotherms

# ── Load and prepare data ──────────────────────────────────────────────────────
# Data taken as-is from Escala (2022) CSV
# Columns: Order, Class, Species, M[kg], P[kJ/day], Tls[yr], PTls[kJ], T[C], Tls_calc[yr]

df = pd.read_csv('escala_data.csv', sep=';')
df = df.dropna(subset=['M[kg]', 'P[kJ/day]', 'Tls_calc[yr]'])
df = df[df['Tls_calc[yr]'] > 0].copy()
df['P_per_M'] = df['P[kJ/day]'] / df['M[kg]']

# Subsets
mammals  = df[df['Class'].str.contains('MAMMAL', na=False)].copy()
m_nh     = mammals[~mammals['Species'].str.contains('sapiens', na=False)].copy()
birds    = df[df['Class'].str.contains('AVES',   na=False)].copy()
ecto     = df[df['Class'].str.contains('ECTOTH', na=False)].copy()

# Human parameters (from Escala: P=7560 kJ/day, M=65 kg)
PM_human = 7560 / 65          # 116.3 kJ/day/kg
TLS1     = 75.0               # population standard
TLS2     = 122.0              # biological maximum

# ── Regressions (all computed from data) ───────────────────────────────────────

def regress_powerlaw(x, y):
    """Power-law log-log regression; returns (slope, intercept, R², p, CI_95)."""
    lx, ly = np.log(x), np.log(y)
    slope, ic, r, p, se = linregress(lx, ly)
    ci = 1.96 * se
    return slope, ic, r**2, p, ci

# Test 1: P ~ M^chi (Kleiber)
a1, ic1, r2_1, p1, ci1 = regress_powerlaw(m_nh['M[kg]'],       m_nh['P[kJ/day]'])

# Test 2: T_ls ~ M^a (longevity allometry)
a2, ic2, r2_2, p2, ci2 = regress_powerlaw(m_nh['M[kg]'],       m_nh['Tls_calc[yr]'])

# Test 3: T_ls ~ (P/M)^p_scale
a3, ic3, r2_3, p3, ci3 = regress_powerlaw(m_nh['P_per_M'],     m_nh['Tls_calc[yr]'])

# Test 4: regression prediction for humans (Test 1 basis)
pred_pm_h1 = np.exp(ic1) * 65**a1 / 65   # P/M from Kleiber at M=65 kg
a_pm, ic_pm, r2_pm, p_pm, ci_pm = regress_powerlaw(m_nh['Tls_calc[yr]'], m_nh['P_per_M'])
pred_h1 = np.exp(ic_pm) * TLS1**a_pm
pred_h2 = np.exp(ic_pm) * TLS2**a_pm

# Rubner product
m_nh['rubner'] = m_nh['P_per_M'] * m_nh['Tls_calc[yr]']
rub_med = m_nh['rubner'].median()
rub_cv  = m_nh['rubner'].std() / m_nh['rubner'].mean() * 100
rub1    = PM_human * TLS1
rub2    = PM_human * TLS2

# Full dataset
a_all, ic_all, r2_all, p_all, ci_all = regress_powerlaw(df['P_per_M'], df['Tls_calc[yr]'])

# Print results
print("=" * 65)
print("  VBLS — Scaling law verification (Escala 2022)")
print("=" * 65)

print(f"\nTEST 1 — Kleiber's law P ~ M^chi  (n={len(m_nh)} species)")
print(f"  chi_obs    = {a1:.3f} ± {ci1:.3f}  |  Kleiber (input): 0.750")
print(f"  R²         = {r2_1:.3f}  |  p = {p1:.2e}")
print(f"  Deviation from 0.75 = {abs(a1-0.75)/0.75*100:.1f}%")

print(f"\nTEST 2 — Longevity allometry T_ls ~ M^a  (n={len(m_nh)} species)")
print(f"  a_obs     = {a2:.3f} ± {ci2:.3f}  |  VBLS: 0.250")
print(f"  R²        = {r2_2:.3f}  |  p = {p2:.2e}")
print(f"  Deviation from prediction = {abs(a2-0.25)/0.25*100:.1f}%")

print(f"\nTEST 3 — T_ls ~ (P/M)^p_scale  (n={len(m_nh)} species)")
print(f"  p_obs      = {a3:.3f} ± {ci3:.3f}  |  VBLS: -1.000")
print(f"  R²         = {r2_3:.3f}  |  p = {p3:.2e}")
print(f"  Deviation from prediction = {abs(a3-(-1.0))/1.0*100:.1f}%")

print(f"\nTEST 4 — Rubner invariant P/M × T_ls  (n={len(m_nh)} species)")
print(f"  Median = {rub_med:.0f} kJ/kg·yr")
print(f"  CV     = {rub_cv:.0f}%  |  log10-std = {np.log10(m_nh['rubner']).std():.4f}")
print(f"  Range  = {m_nh['rubner'].max()/m_nh['rubner'].min():.1f}×")
print(f"  Human_1 ({TLS1:.0f} yr): {rub1:.0f} = {rub1/rub_med:.1f}× median")
print(f"  Human_2 ({TLS2:.0f} yr): {rub2:.0f} = {rub2/rub_med:.1f}× median")

print(f"\nADD. — Full dataset  (n={len(df)} species, all classes)")
print(f"  p_all      = {a_all:.3f} ± {ci_all:.3f}  |  R² = {r2_all:.3f}  |  p = {p_all:.2e}")
print(f"  Mass range: {df['M[kg]'].min():.1e} – {df['M[kg]'].max():.0f} kg")

# ── Figure: 4 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(left=0.08, right=0.97, top=.92, bottom=0.08,
                    wspace=0.38, hspace=0.48)
ax = axes.flatten()

M_line   = np.logspace(np.log10(0.01), np.log10(3000), 300)
T_line   = np.logspace(np.log10(0.3),  np.log10(200),  300)
PM_line  = np.logspace(np.log10(0.5),  np.log10(3000), 300)

# ─── PANEL A: Kleiber's law P ~ M^chi ───────────────────────────────────────
a = ax[0]

# All classes (grey background)
a.scatter(ecto['M[kg]'], ecto['P[kJ/day]'],
          s=15, color=C_EC, alpha=0.3, zorder=2, label='Ectotherms')
a.scatter(birds['M[kg]'], birds['P[kJ/day]'],
          s=15, color=C_AV, alpha=0.3, zorder=2, label='Birds')

# Mammals
szs = 15 + 30 * (np.log10(m_nh['M[kg]']+0.001) - np.log10(0.001)) / \
      (np.log10(3001) - np.log10(0.001))
a.scatter(m_nh['M[kg]'], m_nh['P[kJ/day]'],
          s=szs, color=C_MM, alpha=0.6, zorder=3, label=f'Mammals (n={len(m_nh)})')

# Regression (from data)
a.plot(M_line, np.exp(ic1) * M_line**a1,
       color=C_REG, lw=2, ls='--', zorder=4,
       label=f'Regression: chi = {a1:.3f} ± {ci1:.3f}')

# Kleiber 3/4 (input assumption)
a.plot(M_line, np.exp(ic1) * M_line**0.75,
       color=C_VB, lw=1.8, ls=':', zorder=4, alpha=0.85,
       label='Kleiber (input): chi = 0.750')

a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('Body mass M (kg)')
a.set_ylabel('Energy intake P (kJ/day)')
a.set_title(f'A.  Kleiber\'s law  $P \\sim M^{{\\chi}}$\n'
            f'$\\chi_{{obs}}$ = {a1:.3f} ± {ci1:.3f}  |  '
            f'input: 0.750  |  $R^2$ = {r2_1:.3f}',
            fontweight='bold')
a.legend(fontsize=8, framealpha=0.93, loc='upper left')
a.text(0.97, 0.05,
       f'Deviation from 0.75:\n{abs(a1-0.75)/0.75*100:.1f}%',
       transform=a.transAxes, ha='right', va='bottom', fontsize=9,
       bbox=dict(boxstyle='round,pad=0.35', fc='#f0f0ff', ec=C_VB, lw=1))

# ─── PANEL B: Longevity allometry T_ls ~ M^a ───────────────────────────────
a = ax[1]

a.scatter(m_nh['M[kg]'], m_nh['Tls_calc[yr]'],
          s=szs, color=C_MM, alpha=0.6, zorder=3,
          label=f'Mammals (n={len(m_nh)})')

a.plot(M_line, np.exp(ic2) * M_line**a2,
       color=C_REG, lw=2, ls='--', zorder=4,
       label=f'Regression: a = {a2:.3f} ± {ci2:.3f}')

a.plot(M_line, np.exp(ic2) * M_line**0.25,
       color=C_VB, lw=1.8, ls=':', zorder=4, alpha=0.85,
       label='VBLS: a = 0.250')

# Human
a.scatter(65, TLS1, s=200, color=C_H1, marker='*',
          edgecolors='white', lw=0.9, zorder=6, label=f'Human₁ ({TLS1:.0f} yr)')
a.scatter(65, TLS2, s=200, color=C_H2, marker='*',
          edgecolors='white', lw=0.9, zorder=6, label=f'Human₂ ({TLS2:.0f} yr)')

a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('Body mass M (kg)')
a.set_ylabel('Lifespan $T_{ls}$ (yr)')
a.set_title(f'B.  Longevity allometry  $T_{{ls}} \\sim M^{{a}}$\n'
            f'$a_{{obs}}$ = {a2:.3f} ± {ci2:.3f}  |  '
            f'VBLS: 0.250  |  $R^2$ = {r2_2:.3f}',
            fontweight='bold')
a.legend(fontsize=8, framealpha=0.93, loc='upper left')
a.text(0.97, 0.05,
       f'Deviation from VBLS:\n{abs(a2-0.25)/0.25*100:.1f}%',
       transform=a.transAxes, ha='right', va='bottom', fontsize=9,
       bbox=dict(boxstyle='round,pad=0.35', fc='#f0f0ff', ec=C_VB, lw=1))

# ─── PANEL C: T_ls ~ (P/M)^p_scale ──────────────────────────────────────────────
a = ax[2]

a.scatter(m_nh['P_per_M'], m_nh['Tls_calc[yr]'],
          s=szs, color=C_MM, alpha=0.6, zorder=3,
          label=f'Mammals (n={len(m_nh)})')

a.plot(PM_line, np.exp(ic3) * PM_line**a3,
       color=C_REG, lw=2, ls='--', zorder=4,
       label=f'Regression: p_scale = {a3:.3f} ± {ci3:.3f}')

a.plot(PM_line, np.exp(ic3) * PM_line**(-1.0),
       color=C_VB, lw=1.8, ls=':', zorder=4, alpha=0.85,
       label='VBLS: p_scale = −1.000')

# Human
a.scatter(PM_human, TLS1, s=200, color=C_H1, marker='*',
          edgecolors='white', lw=0.9, zorder=6, label=f'Human₁ ({TLS1:.0f} yr)')
a.scatter(PM_human, TLS2, s=200, color=C_H2, marker='*',
          edgecolors='white', lw=0.9, zorder=6, label=f'Human₂ ({TLS2:.0f} yr)')

a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('Specific metabolic rate P/M (kJ/day/kg)')
a.set_ylabel('Lifespan $T_{ls}$ (yr)')
a.set_title(f'C.  $T_{{ls}} \\sim (P/M)^{{p_{{\\mathrm{{scale}}}}}}$\n'
            f'$p_{{obs}}$ = {a3:.3f} ± {ci3:.3f}  |  '
            f'VBLS: −1.000  |  $R^2$ = {r2_3:.3f}',
            fontweight='bold')
a.legend(fontsize=8, framealpha=0.93, loc='upper right')
a.text(0.97, 0.65,
       f'Deviation from VBLS:\n{abs(a3-(-1.0))/1.0*100:.1f}%',
       transform=a.transAxes, ha='right', va='bottom', fontsize=9,
       bbox=dict(boxstyle='round,pad=0.35', fc='#f0f0ff', ec=C_VB, lw=1))

# ─── PANEL D: Rubner invariant ─────────────────────────────────────────────────
a = ax[3]

a.scatter(m_nh['Tls_calc[yr]'], m_nh['rubner'],
          s=szs, color=C_MM, alpha=0.6, zorder=3,
          label=f'Mammals (n={len(m_nh)})')
a.axhline(rub_med, color=C_REG, lw=1.8, ls='--', zorder=4,
          label=f'Median = {rub_med:.0f} kJ/kg·yr')
a.axhspan(rub_med * 0.5, rub_med * 2.0,
          alpha=0.07, color=C_REG, label='×2 from median')

a.scatter(TLS1, rub1, s=200, color=C_H1, marker='*',
          edgecolors='white', lw=0.9, zorder=6)
a.scatter(TLS2, rub2, s=200, color=C_H2, marker='*',
          edgecolors='white', lw=0.9, zorder=6)
a.annotate(f'H₁: {rub1:.0f}\n({rub1/rub_med:.1f}×)',
           (TLS1, rub1), (TLS1 * 0.55, rub1 * 1.3),
           fontsize=8.5, color=C_H1, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=C_H1, lw=0.8))
a.annotate(f'H₂: {rub2:.0f}\n({rub2/rub_med:.1f}×)',
           (TLS2, rub2), (TLS2 * 1.08, rub2 * 1.2),
           fontsize=8.5, color=C_H2, fontweight='bold',
           arrowprops=dict(arrowstyle='->', color=C_H2, lw=0.8))

a.set_xscale('log')
a.set_xlabel('Lifespan $T_{ls}$ (yr)')
a.set_ylabel('P/M × $T_{ls}$ (kJ/kg·yr)')
a.set_title(f'D.  Rubner invariant  $P/M \\times T_{{ls}}$\n'
            f'CV = {rub_cv:.0f}%  |  range = '
            f'{m_nh["rubner"].max()/m_nh["rubner"].min():.1f}×',
            fontweight='bold')
a.legend(fontsize=8, framealpha=0.93, loc='upper right')
a.text(0.03, 0.05,
       f'log10-std = {np.log10(m_nh["rubner"]).std():.3f}',
       transform=a.transAxes, ha='left', va='bottom', fontsize=8.5,
       bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

plt.savefig('escala_4tests.pdf', bbox_inches='tight')
plt.savefig('escala_4tests.png', bbox_inches='tight', dpi=180)
plt.close()
print("\nSaved: escala_4tests.pdf / .png")
