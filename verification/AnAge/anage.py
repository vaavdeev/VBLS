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
VBLS — Cross-species verification against the AnAge database
=================================================================
Version: final (with improved filtering + Rubner by order)

Database: AnAge Build 14 (Tacutu et al. 2018, Nucleic Acids Res.)
File:     anage_data.csv (semicolon-delimited)

Two samples:
  Base     N=318: Data quality != 'low', without Chiroptera
  Improved N=111: Data quality = 'acceptable', sample_size >= medium

Four tests (0 free parameters):
  Test 1: T_max ~ M^a             VBLS: a = +0.250
  Test 2: P/M  ~ M^beta           VBLS: beta  = -0.250  (Kleiber)
  Test 3: T_max ~ (P/M)^p_scale  VBLS: p_scale = -1.000  (prediction 1)
  Test 4: T_max x P/M = const     Rubner invariant; primary metric: log-std

Figure 1 (anage_verification.pdf/png):     base sample 2x2
Figure 2 (anage_improved.pdf/png):         comparison + by order 2x3
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════

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
})

ORDER_COLORS = {
    'Rodentia':        '#e74c3c',
    'Primates':        '#2980b9',
    'Carnivora':       '#27ae60',
    'Artiodactyla':    '#e67e22',
    'Chiroptera':      '#8e44ad',
    'Soricomorpha':    '#16a085',
    'Dasyuromorphia':  '#c0392b',
    'Diprotodontia':   '#f39c12',
}
C_BASE = '#2980b9'
C_FILT = '#e74c3c'


def gcol(order):
    return ORDER_COLORS.get(order, '#95a5a6')


KNOWN_SPECIES = {
    ('Mus',       'musculus'):   'Mouse',
    ('Rattus',    'norvegicus'): 'Rat',
    ('Homo',      'sapiens'):    'Human',
    ('Elephas',   'maximus'):    'Elephant',
    ('Loxodonta', 'africana'):   'Savanna elephant',
    ('Bos',       'taurus'):     'Cow',
    ('Sus',       'scrofa'):     'Pig',
    ('Felis',     'catus'):      'Cat',
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND FILTER
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv('anage_data.csv', sep=';', encoding='utf-8-sig', on_bad_lines='skip')

mam = df[df['Class'] == 'Mammalia'].copy()

for col in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    mam[col] = pd.to_numeric(mam[col], errors='coerce')

mam = mam.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)'])
mam = mam[
    (mam['Body mass (g)'] > 0) &
    (mam['Maximum longevity (yrs)'] > 0) &
    (mam['Metabolic rate (W)'] > 0)
]

mam['PM']     = mam['Metabolic rate (W)'] / mam['Body mass (g)'] * 1000  # W/kg
mam['Rubner'] = mam['Maximum longevity (yrs)'] * mam['PM']
mam['label']  = mam.apply(
    lambda r: KNOWN_SPECIES.get((r['Genus'], r['Species']), ''), axis=1)

# Base sample: without 'low', without Chiroptera
base = mam[(mam['Data quality'] != 'low') & (mam['Order'] != 'Chiroptera')].copy()

# Improved sample: acceptable + medium+, without Chiroptera
filt = mam[
    (mam['Data quality'] == 'acceptable') &
    (mam['Sample size'].isin(['medium', 'large', 'huge'])) &
    (mam['Order'] != 'Chiroptera')
].copy()

# Bats — separately for display
bats = mam[mam['Order'] == 'Chiroptera'].copy()

print(f"Total Mammalia:       {len(mam)}")
print(f"Base sample:          {len(base)} (excl. Chiroptera={len(mam[mam['Order']=='Chiroptera'])})")
print(f"Improved sample:      {len(filt)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSIONS
# ══════════════════════════════════════════════════════════════════════════════

def log_reg(x, y, label=''):
    sl, ic, r, p, se = stats.linregress(np.log10(x.values), np.log10(y.values))
    if label:
        print(f"  {label:>26s}: a={sl:+.4f}±{se:.4f}  R²={r**2:.4f}  p={p:.2e}  n={len(x)}")
    return sl, ic, r**2, p, se


print('\n=== TEST 1: T_max ~ M^a  (VBLS a=+0.250) ===')
a1b, b1b, r1b, _, se1b = log_reg(base['Body mass (g)'], base['Maximum longevity (yrs)'], 'Base')
a1f, b1f, r1f, _, se1f = log_reg(filt['Body mass (g)'], filt['Maximum longevity (yrs)'], 'Improved')

print('\n=== TEST 2: P/M ~ M^beta  (Kleiber input, beta=-0.250) ===')
a2b, b2b, r2b, _, se2b = log_reg(base['Body mass (g)'], base['PM'], 'Base')
a2f, b2f, r2f, _, se2f = log_reg(filt['Body mass (g)'], filt['PM'], 'Improved')

print('\n=== TEST 3: T_max ~ (P/M)^p_scale  (VBLS p_scale=-1.000) ===')
a3b, b3b, r3b, _, se3b = log_reg(base['PM'], base['Maximum longevity (yrs)'], 'Base')
a3f, b3f, r3f, _, se3f = log_reg(filt['PM'], filt['Maximum longevity (yrs)'], 'Improved')

print('\n=== TEST 4: Rubner T_max x P/M (primary metric: log-std) ===')
lrb_b = np.log10(base['Rubner'])
lrb_f = np.log10(filt['Rubner'])
print(f"  Base:     median={base['Rubner'].median():.2f}  "
      f"log-std={lrb_b.std():.4f}  2sigma=x{10**(2*lrb_b.std()):.2f}")
print(f"  Improved: median={filt['Rubner'].median():.2f}  "
      f"log-std={lrb_f.std():.4f}  2sigma=x{10**(2*lrb_f.std()):.2f}")
print(f"  Improvement log-std: {(1-lrb_f.std()/lrb_b.std())*100:.1f}%")

print('\n=== SUMMARY ===')
print(f"  Test 1: a_VBLS=+0.250  base={a1b:.3f}±{se1b:.3f}  filt={a1f:.3f}±{se1f:.3f}")
print(f"  Test 2: beta_Kleiber=-0.250  base={a2b:.3f}±{se2b:.3f}  filt={a2f:.3f}±{se2f:.3f}")
print(f"  Test 3: p_VBLS=-1.000  base={a3b:.3f}±{se3b:.3f}  filt={a3f:.3f}±{se3f:.3f}")
print(f"  a+beta: base={a1b+a2b:.4f}  filt={a1f+a2f:.4f}  (VBLS: 0.000)")
print(f"  Indirect p_scale = a/|beta|: base={a1b/abs(a2b):.4f}  filt={a1f/abs(a2f):.4f}")

print('\n=== Rubner by order (improved sample) ===')
for order, grp in sorted(filt.groupby('Order'), key=lambda x: -len(x[1])):
    if len(grp) >= 3:
        rb_o = grp['Rubner']
        lrb_o = np.log10(rb_o)
        print(f"  {order:22s}: n={len(grp):3d}  median={rb_o.median():.1f}  "
              f"log-std={lrb_o.std():.3f}")

print('\n=== Familiar species ===')
for (g, s), name, en in [(('Mus','musculus'), 'Mouse', 'Mouse'),
                          (('Rattus','norvegicus'), 'Rat', 'Rat'),
                          (('Homo','sapiens'), 'Human', 'Human'),
                          (('Elephas','maximus'), 'Elephant', 'Elephant')]:
    row = base[(base['Genus'] == g) & (base['Species'] == s)]
    if len(row):
        r = row.iloc[0]
        print(f"  {en:10s}: T_max={r['Maximum longevity (yrs)']:.1f}  "
              f"P/M={r['PM']:.3f}  Rubner={r['Rubner']:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def scatter_orders(ax, x_col, y_col, df_main, df_bats=None, annotate=True):
    for order, grp in df_main.groupby('Order'):
        ax.scatter(grp[x_col], grp[y_col], s=16, c=gcol(order), alpha=0.55, zorder=3)
    if df_bats is not None and len(df_bats):
        ax.scatter(df_bats[x_col], df_bats[y_col],
                   s=24, c='#8e44ad', alpha=0.65, marker='^', zorder=4, label='Chiroptera')
    if annotate:
        for _, row in df_main[df_main['label'] != ''].iterrows():
            ax.annotate(row['label'],
                        xy=(row[x_col], row[y_col]),
                        xytext=(5, 3), textcoords='offset points',
                        fontsize=7.5, color='#333', zorder=6)


def order_legend(ax):
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, ms=8, label=o)
        for o, c in list(ORDER_COLORS.items())[:6]
    ]
    handles += [Line2D([0], [0], marker='^', color='w',
                        markerfacecolor='#8e44ad', ms=8, label='Chiroptera')]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.93, ncol=2, loc='lower right')


m_range  = np.logspace(np.log10(base['Body mass (g)'].min()),
                        np.log10(base['Body mass (g)'].max()), 200)
pm_range = np.logspace(np.log10(base['PM'].min()),
                        np.log10(base['PM'].max()), 200)

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — Base sample 2x2
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.07, wspace=0.30, hspace=0.40)
ax = axes.flatten()

# Panel A: T_max ~ M
scatter_orders(ax[0], 'Body mass (g)', 'Maximum longevity (yrs)', base, bats)
ax[0].plot(m_range, 10**b1b * m_range**a1b, 'k-', lw=2.2, zorder=5,
           label=f'N={len(base)}: a={a1b:.3f}±{se1b:.3f}  R²={r1b:.3f}')
ax[0].plot(m_range, 10**b1b * m_range**0.25, 'k--', lw=1.5, alpha=0.45, label='VBLS a=0.250')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('Body mass (g)'); ax[0].set_ylabel('T_max (yr)')
ax[0].set_title(f'A.  T_max ~ M^a\na_emp={a1b:.3f}  a_VBLS=0.250  Delta={a1b-0.25:+.3f}',
                fontweight='bold')
order_legend(ax[0])

# Panel B: P/M ~ M
scatter_orders(ax[1], 'Body mass (g)', 'PM', base, bats, annotate=False)
for _, row in base[base['label'] != ''].iterrows():
    ax[1].annotate(row['label'], xy=(row['Body mass (g)'], row['PM']),
                   xytext=(5, 3), textcoords='offset points', fontsize=7.5, color='#333')
ax[1].plot(m_range, 10**b2b * m_range**a2b, 'k-', lw=2.2, zorder=5,
           label=f'N={len(base)}: beta={a2b:.3f}±{se2b:.3f}  R²={r2b:.3f}')
ax[1].plot(m_range, 10**b2b * m_range**(-0.25), 'k--', lw=1.5, alpha=0.45,
           label='Kleiber (input): beta=-0.250')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel('Body mass (g)'); ax[1].set_ylabel('P/M (W/kg)')
ax[1].set_title(f'B.  P/M ~ M^beta  (Kleiber law)\nbeta_emp={a2b:.3f}  beta_Kleiber=-0.250  Delta={a2b+0.25:+.3f}',
                fontweight='bold')
ax[1].legend(fontsize=8.5, framealpha=0.93, loc='upper right')

# Panel C: T_max ~ (P/M)^p_scale  — prediction 1
scatter_orders(ax[2], 'PM', 'Maximum longevity (yrs)', base, bats)
ax[2].plot(pm_range, 10**b3b * pm_range**a3b, 'k-', lw=2.5, zorder=5,
           label=f'N={len(base)}: p_scale={a3b:.3f}±{se3b:.3f}  R²={r3b:.3f}')
ax[2].plot(pm_range, 10**b3b * pm_range**(-1.0), 'k--', lw=1.8, alpha=0.45,
           label='VBLS p_scale=-1.000')
ax[2].text(0.04, 0.06,
           f'Indirect p_scale = a/|beta| = {a1b/abs(a2b):.3f}\n'
           f'Direct: p_scale = {a3b:.3f}\n\n'
           f'P/M explains {r3b*100:.1f}% var(T_max)\n'
           f'{(1-r3b)*100:.1f}% — other factors',
           transform=ax[2].transAxes, va='bottom', fontsize=8.5,
           bbox=dict(boxstyle='round,pad=0.4', fc='#fff8f0', ec='#e67e22', lw=1.2))
ax[2].set_xscale('log'); ax[2].set_yscale('log')
ax[2].set_xlabel('P/M (W/kg)'); ax[2].set_ylabel('T_max (yr)')
ax[2].set_title(f'C.  T_max ~ (P/M)^p_scale  — prediction 1 (with transfer assumptions)\np_emp={a3b:.3f}  p_VBLS=-1.000  Delta={a3b+1.0:+.3f}',
                fontweight='bold')
ax[2].legend(fontsize=8.5, framealpha=0.93, loc='upper right')

# Panel D: Rubner histogram (base sample)
bins = np.linspace(-0.5, 2.8, 36)
lrb_bats = np.log10(bats['Rubner']) if len(bats) else pd.Series([], dtype=float)
ax[3].hist(lrb_b, bins=bins, color=C_BASE, alpha=0.75, label=f'Excl. Chiroptera  n={len(base)}')
if len(lrb_bats):
    ax[3].hist(lrb_bats, bins=bins, color='#8e44ad', alpha=0.70,
               label=f'Chiroptera  n={len(bats)}')
ax[3].axvline(lrb_b.median(), color='#1a252f', lw=2.0, ls='-',
              label=f'Median = {base["Rubner"].median():.1f}  W·yr/kg')
# Familiar species
for (g, s), name, col in zip(
        [('Mus','musculus'), ('Rattus','norvegicus'), ('Homo','sapiens'), ('Elephas','maximus')],
        ['Mouse', 'Rat', 'Human', 'Elephant'],
        ['#e74c3c', '#c0392b', '#2980b9', '#e67e22']):
    row = base[(base['Genus'] == g) & (base['Species'] == s)]
    if len(row):
        rv = float(np.log10(row['Rubner'].values[0]))
        ax[3].axvline(rv, color=col, lw=1.8, ls='--', alpha=0.85)
        ax[3].text(rv + 0.04, 28 - ['Mouse','Rat','Human','Elephant'].index(name) * 5.5,
                   name, fontsize=8, color=col)
ax[3].text(0.97, 0.97,
           f'VBLS: T_max x P/M = const\n\n'
           f'Base N={len(base)}:\n'
           f'  Median = {base["Rubner"].median():.1f}\n'
           f'  log-std = {lrb_b.std():.4f}\n'
           f'  2sigma = x{10**(2*lrb_b.std()):.2f}\n\n'
           f'Primates: median = 62.9 (+1.65x)\n'
           f'Chiroptera: above median',
           transform=ax[3].transAxes, ha='right', va='top', fontsize=8.5,
           bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec=C_BASE, lw=1.2))
ax[3].set_xlabel('log10(T_max x P/M)  [W·yr/kg]')
ax[3].set_ylabel('Number of species')
ax[3].set_title(f'D.  Rubner invariant: T_max x P/M = const\n'
                f'Median={base["Rubner"].median():.1f}  '
                f'log-std={lrb_b.std():.3f}  '
                f'2sigma=x{10**(2*lrb_b.std()):.2f}',
                fontweight='bold')
ax[3].legend(fontsize=8.5, framealpha=0.93, loc='upper left')

plt.savefig('anage_verification.pdf', bbox_inches='tight')
plt.savefig('anage_verification.png', bbox_inches='tight', dpi=180)
plt.close()
print('\nSaved: anage_verification.pdf / .png')

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — Base vs improved comparison 2x3
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97, top=0.91, bottom=0.07,
                       wspace=0.30, hspace=0.40)
axes2 = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# Panel A: T_max ~ M base sample
scatter_orders(axes2[0], 'Body mass (g)', 'Maximum longevity (yrs)', base, bats)
axes2[0].plot(m_range, 10**b1b * m_range**a1b, '-', color=C_BASE, lw=2.2, zorder=5,
              label=f'N={len(base)}: a={a1b:.3f}±{se1b:.3f}  R²={r1b:.3f}')
axes2[0].plot(m_range, 10**b1b * m_range**0.25, 'k--', lw=1.5, alpha=0.45, label='VBLS a=0.250')
axes2[0].set_xscale('log'); axes2[0].set_yscale('log')
axes2[0].set_xlabel('Body mass (g)'); axes2[0].set_ylabel('T_max (yr)')
axes2[0].set_title(f'A.  T_max ~ M^a  (base N={len(base)})\n'
                   f'a_emp={a1b:.3f}  Delta={a1b-0.25:+.3f}', fontweight='bold')
axes2[0].legend(fontsize=8.5, framealpha=0.93)

# Panel B: T_max ~ M improved sample
scatter_orders(axes2[1], 'Body mass (g)', 'Maximum longevity (yrs)', filt)
axes2[1].plot(m_range, 10**b1b * m_range**a1b, '--', color=C_BASE, lw=1.5, alpha=0.5,
              label=f'Base N={len(base)}: a={a1b:.3f}')
axes2[1].plot(m_range, 10**b1f * m_range**a1f, '-', color=C_FILT, lw=2.3, zorder=5,
              label=f'Filter N={len(filt)}: a={a1f:.3f}±{se1f:.3f}  R²={r1f:.3f}')
axes2[1].plot(m_range, 10**b1f * m_range**0.25, 'k--', lw=1.5, alpha=0.45, label='VBLS a=0.250')
axes2[1].set_xscale('log'); axes2[1].set_yscale('log')
axes2[1].set_xlabel('Body mass (g)'); axes2[1].set_ylabel('T_max (yr)')
axes2[1].set_title(f'B.  T_max ~ M^a  (acceptable + medium+)\n'
                   f'a_emp={a1f:.3f}  Delta={a1f-0.25:+.3f}', fontweight='bold')
axes2[1].legend(fontsize=8.5, framealpha=0.93)

# Panel C: T_max ~ (P/M)^p_scale both samples
scatter_orders(axes2[2], 'PM', 'Maximum longevity (yrs)', base, bats)
axes2[2].plot(pm_range, 10**b3b * pm_range**a3b, '--', color=C_BASE, lw=2.0, alpha=0.7,
              label=f'Base: p_scale={a3b:.3f}±{se3b:.3f}  R²={r3b:.3f}')
axes2[2].plot(pm_range, 10**b3f * pm_range**a3f, '-', color=C_FILT, lw=2.5, zorder=5,
              label=f'Filter: p_scale={a3f:.3f}±{se3f:.3f}  R²={r3f:.3f}')
axes2[2].plot(pm_range, 10**b3b * pm_range**(-1.0), 'k--', lw=1.5, alpha=0.45,
              label='VBLS p_scale=-1.000')
axes2[2].set_xscale('log'); axes2[2].set_yscale('log')
axes2[2].set_xlabel('P/M (W/kg)'); axes2[2].set_ylabel('T_max (yr)')
axes2[2].set_title(f'C.  T_max ~ (P/M)^p_scale  — prediction 1\n'
                   f'Base p={a3b:.3f}  ->  Filter p={a3f:.3f}  VBLS=-1.000',
                   fontweight='bold')
axes2[2].legend(fontsize=8.5, framealpha=0.93)

# Panel D: Rubner histogram both samples
axes2[3].hist(lrb_b, bins=bins, color=C_BASE, alpha=0.55,
              label=f'Base N={len(base)}  log-std={lrb_b.std():.3f}')
axes2[3].hist(lrb_f, bins=bins, color=C_FILT, alpha=0.70,
              label=f'Filter N={len(filt)}  log-std={lrb_f.std():.3f}')
axes2[3].axvline(lrb_b.median(), color=C_BASE, lw=2.0, ls='--')
axes2[3].axvline(lrb_f.median(), color=C_FILT, lw=2.0, ls='-')
axes2[3].set_xlabel('log10(T_max x P/M)  [W·yr/kg]')
axes2[3].set_ylabel('Number of species')
axes2[3].set_title(f'D.  Rubner: primary metric log-std\n'
                   f'log-std: {lrb_b.std():.3f} -> {lrb_f.std():.3f}  '
                   f'(-{(1-lrb_f.std()/lrb_b.std())*100:.1f}%)',
                   fontweight='bold')
axes2[3].legend(fontsize=8.5, framealpha=0.93)

# Panel E: Rubner by order (improved, IQR boxes)
orders_show = ['Rodentia', 'Carnivora', 'Primates', 'Artiodactyla',
               'Dasyuromorphia', 'Diprotodontia', 'Soricomorpha']
order_data = [(o, filt[filt['Order'] == o]['Rubner'])
              for o in orders_show if len(filt[filt['Order'] == o]) >= 3]

for i, (order, rb) in enumerate(order_data):
    lrb_o = np.log10(rb)
    med = lrb_o.median()
    q1, q3 = lrb_o.quantile(0.25), lrb_o.quantile(0.75)
    lo, hi = lrb_o.min(), lrb_o.max()
    c = gcol(order)
    axes2[4].barh(i, q3 - q1, left=q1, height=0.6, color=c, alpha=0.75, zorder=3)
    axes2[4].plot([lo, hi], [i, i], '-', color=c, lw=1.2, alpha=0.6)
    axes2[4].scatter([med], [i], color=c, s=40, zorder=5)
    axes2[4].text(hi + 0.03, i, f'n={len(rb)} sigma={lrb_o.std():.2f}',
                  va='center', fontsize=8, color='#444')

axes2[4].axvline(lrb_f.median(), color='#333', lw=1.5, ls='--', alpha=0.7,
                 label=f'Median={filt["Rubner"].median():.1f} W·yr/kg')
axes2[4].set_yticks(range(len(order_data)))
axes2[4].set_yticklabels([o for o, _ in order_data], fontsize=9)
axes2[4].set_xlabel('log10(T_max x P/M)')
axes2[4].set_title('E.  Rubner by order (filtered sample)\n'
                   'Boxes = IQR, dot = median, whiskers = min-max', fontweight='bold')
axes2[4].legend(fontsize=8.5, framealpha=0.93)

# Panel F: summary table
axes2[5].axis('off')
rows = [
    ['', 'VBLS', f'Base\n(N={len(base)})', f'Filter\n(N={len(filt)})', 'Delta'],
    ['a (T_max~M)',        '+0.250',
     f'{a1b:.3f}\n±{se1b:.3f}', f'{a1f:.3f}\n±{se1f:.3f}',
     f'{abs(a1f-0.25):.3f}\nvs {abs(a1b-0.25):.3f}'],
    ['beta (P/M~M)',       '-0.250',
     f'{a2b:.3f}\n±{se2b:.3f}', f'{a2f:.3f}\n±{se2f:.3f}',
     f'{abs(a2f+0.25):.3f}\nvs {abs(a2b+0.25):.3f}'],
    ['p (T_max~P/M)',      '-1.000',
     f'{a3b:.3f}\n±{se3b:.3f}', f'{a3f:.3f}\n±{se3f:.3f}',
     f'{abs(a3f+1.0):.3f}\nvs {abs(a3b+1.0):.3f}'],
    ['log-std\n(Rubner)', '0',
     f'{lrb_b.std():.4f}', f'{lrb_f.std():.4f}',
     f'-{lrb_b.std()-lrb_f.std():.4f}\n({(1-lrb_f.std()/lrb_b.std())*100:.1f}%)'],
    ['2sigma range', 'x1',
     f'x{10**(2*lrb_b.std()):.2f}', f'x{10**(2*lrb_f.std()):.2f}',
     f'improved\n-{10**(2*lrb_b.std())-10**(2*lrb_f.std()):.2f}'],
]
table = axes2[5].table(cellText=rows[1:], colLabels=rows[0],
                       loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 2.5)
for j in range(5):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
for j in range(5):
    table[4, j].set_facecolor('#eaf4fb')
axes2[5].set_title('F.  Summary: base vs improved filtering\n'
                   'log-std — primary metric (not CV)', fontweight='bold')

plt.savefig('anage_improved.pdf', bbox_inches='tight')
plt.savefig('anage_improved.png', bbox_inches='tight', dpi=180)
plt.close()
print('Saved: anage_improved.pdf / .png')
