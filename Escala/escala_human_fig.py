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
VBLS — Verification of eta_abs via metabolic data from Escala (2022)
Comparison: Human_1 (Tls=75 yr, population) vs Human_2 (Tls=122 yr, biological maximum)

Three panels:
A — P/M vs Tls: power law + positions of Human_1 / Human_2
B — Rubner product P/M × Tls: distribution across 95 species + both humans
C — Comparison of Tls from Escala vs T_max from Cagan: cat anomaly

Run: python3 escala_human_fig.py
Out: escala_fig_human.pdf / .png
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
    'font.family':      'DejaVu Serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.15,
    'grid.linestyle':   '--',
    'figure.dpi':       150,
})

C_MAMMALS = '#95a5a6'
C_H1      = '#2980b9'   # Human_1 — blue
C_H2      = '#c0392b'   # Human_2 — red
C_CAT     = '#e67e22'   # cat — orange
C_VBLS    = '#8e44ad'   # VBLS — purple
C_REG     = '#555'      # regression — dark grey

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('escala_data.csv', sep=';')
m  = df[df['Class'].str.contains('MAMMAL', na=False)].copy()
m  = m.dropna(subset=['M[kg]', 'P[kJ/day]', 'Tls_calc[yr]'])
m  = m[m['Tls_calc[yr]'] > 0]
m['P_per_M'] = m['P[kJ/day]'] / m['M[kg]']

# Exclude humans (plotted separately twice)
m_no_human = m[~m['Species'].str.contains('sapiens', na=False)].copy()

# Human parameters
PM_human  = 7560 / 65           # 116.3 kJ/day/kg
TLS1      = 75.0                 # population standard
TLS2      = 122.0                # biological maximum
M_human   = 65.0

# Regression P/M ~ Tls^alpha (non-human mammals)
log_T  = np.log(m_no_human['Tls_calc[yr]'])
log_PM = np.log(m_no_human['P_per_M'])
alpha, intercept, r, _, _ = linregress(log_T, log_PM)

# Rubner product
m_no_human = m_no_human.copy()
m_no_human['rubner'] = m_no_human['P_per_M'] * m_no_human['Tls_calc[yr]']
rub_median = m_no_human['rubner'].median()
rub1 = PM_human * TLS1
rub2 = PM_human * TLS2

# Regression predictions for both humans
pred1 = np.exp(intercept) * TLS1**alpha
pred2 = np.exp(intercept) * TLS2**alpha

# Escala–Cagan overlap species (Panel C)
overlap = {
    'Mouse':   (1.5,  4.0),
    'Rat':     (3.5,  3.8),
    'Rabbit':  (8.0, 13.0),
    'Dog':     (18.0, 20.0),
    'Cat':     (11.0, 30.0),
    'Horse':   (40.0, 57.0),
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.11, wspace=0.38)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL A — power law P/M ~ Tls^alpha
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[0]

# All non-human mammals
sizes = 30 + 25 * (np.log10(m_no_human['M[kg]'] + 0.001) - np.log10(0.001)) / \
        (np.log10(3001) - np.log10(0.001))
ax.scatter(m_no_human['Tls_calc[yr]'], m_no_human['P_per_M'],
           s=sizes, color=C_MAMMALS, alpha=0.55, zorder=3, label='95 mammalian species')

# Cat — highlighted to show Tls_Escala=11 anomaly
cat = m_no_human[m_no_human['Species'].str.contains('silvestris|Felis', na=False)]
if len(cat):
    ax.scatter(cat['Tls_calc[yr]'], cat['P_per_M'],
               s=80, color=C_CAT, zorder=5, marker='D', edgecolors='white', lw=0.8)
    ax.annotate('Cat\n(Tls=11 yr)', (cat['Tls_calc[yr]'].values[0], cat['P_per_M'].values[0]),
                xytext=(7, 350), fontsize=8, color=C_CAT,
                arrowprops=dict(arrowstyle='->', color=C_CAT, lw=0.8))

# Regression line
T_line = np.logspace(np.log10(0.5), np.log10(150), 200)
ax.plot(T_line, np.exp(intercept) * T_line**alpha,
        color=C_REG, lw=1.8, ls='--', zorder=4,
        label=f'Regression: alpha={alpha:.2f}, R²={r**2:.2f}')

# VBLS line (alpha = −1) through Human_1
A_VBLS = PM_human * TLS1
ax.plot(T_line, A_VBLS / T_line,
        color=C_VBLS, lw=1.5, ls=':', zorder=4, alpha=0.75,
        label='VBLS: alpha = −1')

# Human_1 and Human_2 at the same horizontal P/M=116.3
for Tls, color, label, mk in [(TLS1, C_H1, f'Human_1\n(75 yr, population)', '*'),
                               (TLS2, C_H2, f'Human_2\n(122 yr, maximum)', '*')]:
    ax.scatter(Tls, PM_human, s=220, color=color, marker=mk,
               edgecolors='white', lw=0.8, zorder=6)
    pred = np.exp(intercept) * Tls**alpha
    ax.vlines(Tls, pred, PM_human, color=color, lw=1.2, ls='-', alpha=0.6, zorder=5)

# Horizontal bracket between two humans
ax.annotate('', xy=(TLS2, PM_human), xytext=(TLS1, PM_human),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
ax.text((TLS1+TLS2)/2, PM_human*1.25, 'Δ = 47 yr\n(Phase III?)',
        ha='center', fontsize=8.5, color='#333',
        bbox=dict(boxstyle='round,pad=0.25', fc='#f8f8f8', ec='#ccc'))

# Human labels
ax.text(TLS1*0.82, PM_human*0.68, f'Human_1\n75 yr', ha='center', fontsize=8.5, color=C_H1)
ax.text(TLS2*1.05, PM_human*1.08, f'Human_2\n122 yr', ha='left', fontsize=8.5, color=C_H2)

# Deviation annotation
ax.text(0.97, 0.27,
        f'Deviation from regression:\n'
        f'Human_1: ×{PM_human/pred1:.1f} (+{(PM_human/pred1-1)*100:.0f}%)\n'
        f'Human_2: ×{PM_human/pred2:.1f} (+{(PM_human/pred2-1)*100:.0f}%)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', fc='#fff8f0', ec=C_H2, lw=1))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Lifespan Tls (yr)', fontsize=10)
ax.set_ylabel('Specific metabolic rate P/M (kJ/day/kg)', fontsize=10)
ax.set_title('A.  Power law P/M ~ Tls^alpha\n95 mammalian species (Escala 2022)',
             fontsize=10.5, fontweight='bold')

handles = [
    mpatches.Patch(color=C_MAMMALS, alpha=0.6, label='95 mammalian species'),
    Line2D([0],[0], color=C_REG, lw=1.8, ls='--', label=f'Regression alpha={alpha:.2f}, R²={r**2:.2f}'),
    Line2D([0],[0], color=C_VBLS, lw=1.5, ls=':', alpha=0.8, label='VBLS: alpha = −1'),
    Line2D([0],[0], marker='*', color=C_H1, ms=10, ls='none', label='Human_1 (75 yr)'),
    Line2D([0],[0], marker='*', color=C_H2, ms=10, ls='none', label='Human_2 (122 yr)'),
    Line2D([0],[0], marker='D', color=C_CAT, ms=7, ls='none', label='Cat (11 yr in Escala)'),
]
ax.legend(handles=handles, fontsize=8, framealpha=0.92, loc='upper right')

# ═══════════════════════════════════════════════════════════════════════════
# PANEL B — Rubner product P/M × Tls
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[1]

rub_vals = m_no_human['rubner'].values
Tls_vals  = m_no_human['Tls_calc[yr]'].values
order     = np.argsort(Tls_vals)

ax.scatter(Tls_vals[order], rub_vals[order],
           s=35, color=C_MAMMALS, alpha=0.55, zorder=3)

# Median line
ax.axhline(rub_median, color=C_REG, lw=1.5, ls='--', zorder=4,
           label=f'Median = {rub_median:.0f}')
ax.axhspan(rub_median*0.5, rub_median*2.0, alpha=0.07, color=C_REG)

# Human_1 and Human_2
ax.scatter(TLS1, rub1, s=220, color=C_H1, marker='*', edgecolors='white', lw=0.8, zorder=6)
ax.scatter(TLS2, rub2, s=220, color=C_H2, marker='*', edgecolors='white', lw=0.8, zorder=6)
ax.text(TLS1*0.82, rub1*1.15, f'{rub1:.0f}\n({rub1/rub_median:.1f}×)', ha='center',
        fontsize=8.5, color=C_H1, fontweight='bold')
ax.text(TLS2*0.80, rub2*1.15, f'{rub2:.0f}\n({rub2/rub_median:.1f}×)', ha='center',
        fontsize=8.5, color=C_H2, fontweight='bold')
ax.set_ylim(top=rub2*1.35)

# Arrow between two humans
ax.annotate('', xy=(TLS2, rub2), xytext=(TLS1, rub1),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
ax.text(96, (rub1+rub2)/2 * 1.05, 'Δ = 5467\n(Phase III)', ha='center',
        fontsize=8, color='#333',
        bbox=dict(boxstyle='round,pad=0.25', fc='#f8f8f8', ec='#ccc'))

ax.set_xscale('log')
ax.set_xlabel('Tls (yr)', fontsize=10)
ax.set_ylabel('Lifetime specific energy\nP/M × Tls (kJ/kg·yr)', fontsize=10)
ax.set_title('B.  Rubner invariant P/M × Tls\nHumans exceptional: 5–8× above median',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.92)

# Range without humans
ax.text(0.97, 0.05,
        f'95 species: {m_no_human["rubner"].min():.0f}–{m_no_human["rubner"].max():.0f}\n'
        f'(range {m_no_human["rubner"].max()/m_no_human["rubner"].min():.1f}×)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

# ═══════════════════════════════════════════════════════════════════════════
# PANEL C — Tls Escala vs T_max Cagan for overlapping species
# ═══════════════════════════════════════════════════════════════════════════
ax = axes[2]

sp_labels = list(overlap.keys())
tls_E     = [overlap[s][0] for s in sp_labels]
tls_C     = [overlap[s][1] for s in sp_labels]

for i, (sp, te, tc) in enumerate(zip(sp_labels, tls_E, tls_C)):
    is_cat = sp == 'Cat'
    col = C_CAT if is_cat else C_MAMMALS
    ms  = 120 if is_cat else 60
    ax.scatter(te, tc, s=ms, color=col, zorder=4,
               edgecolors='white' if is_cat else 'none', lw=0.8,
               marker='D' if is_cat else 'o')
    offset_x = te * 1.08
    offset_y = tc * 1.05
    ax.annotate(sp, (te, tc), (offset_x, offset_y), fontsize=8.5,
                color=col if is_cat else '#333')

# Humans (two variants)
ax.scatter(TLS1, TLS1, s=160, color=C_H1, marker='*', zorder=5, edgecolors='white', lw=0.8)
ax.scatter(TLS1, TLS2, s=160, color=C_H2, marker='*', zorder=5, edgecolors='white', lw=0.8)
ax.text(TLS1*1.05, TLS1*0.94, 'H₁ (75,75)', fontsize=8.5, color=C_H1)
ax.text(TLS1*1.05, TLS2*1.03, 'H₂ (75→122)', fontsize=8.5, color=C_H2)
ax.annotate('', xy=(TLS1, TLS2), xytext=(TLS1, TLS1),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

# Equality diagonal
lim_max = 130
ax.plot([0, lim_max], [0, lim_max], color=C_REG, lw=1.2, ls='--', alpha=0.6,
        label='Tls_Escala = T_max_Cagan')
ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)

# Cat annotation
ax.annotate('Cat: Escala 11 yr\n   Cagan 30 yr (!)',
            xy=(11, 30), xytext=(30, 22),
            fontsize=8.5, color=C_CAT,
            arrowprops=dict(arrowstyle='->', color=C_CAT, lw=0.9))

ax.set_xlabel('Tls from Escala (2022), yr', fontsize=10)
ax.set_ylabel('T_max from Cagan (2022), yr', fontsize=10)
ax.set_title('C.  Comparison of lifespan standards\nBetween Escala and Cagan datasets',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.92, loc='upper left')

ax.text(0.97, 0.05,
        'Above diagonal: Cagan overestimates T_max\n(cat is an extreme case)',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f0', ec=C_CAT, lw=1))

plt.savefig('escala_fig_human.pdf', bbox_inches='tight')
plt.savefig('escala_fig_human.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved.")
