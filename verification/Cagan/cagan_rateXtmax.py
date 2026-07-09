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
VBLS — Analysis of the rate x T_max product
Supplementary section: testing the Cagan hypothesis of a universal constant
=================================================================

Out: cagan_fig3_rateXtmax.pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

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

COL_VBLS  = '#8e44ad'
COL_UNIV  = '#c0392b'
COL_ANOM  = '#e67e22'   # anomalies
COL_REG   = '#7f8c8d'

SP_COLORS = {
    'Human':                  '#1565C0',
    'Mouse':                  '#F44336',
    'Rat':                    '#E91E63',
    'Naked mole-rat':         '#4CAF50',
    'Dog':                    '#FF9800',
    'Cat':                    '#9C27B0',
    'Horse':                  '#00BCD4',
    'Tiger':                  '#FF5722',
    'Cow':                    '#607D8B',
    'Rabbit':                 '#8BC34A',
    'Giraffe':                '#FFC107',
    'Lion':                   '#795548',
    'Ferret':                 '#03A9F4',
    'Ring-tailed lemur':      '#673AB7',
    'Black-and-white colobus':'#009688',
}

# ── Species max lifespan (identical to mde_cagan_verification.py) ────────────
# Cat: biological max ~17 yr; AnAge entry of 30 yr is unreliable (see §17.3.5.2)
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
}

# Anomalous species: rate × T_max deviates due to identified causes (see §17.3.5.2)
# Cat excluded: with T_max=17 yr its product ~9300 falls within the normal range
ANOMALY = {'Horse', 'Tiger', 'Rabbit', 'Naked mole-rat'}

# ── Per-species statistics (15 species; regression slope × T_max) ─────────────
# Cat: biological T_max = 17 yr (AnAge entry of 30 yr is unreliable for the
# domestic cat — see §17.3.5.2); with T_max = 17 its product ~9300 is normal,
# so the cat is NOT an anomaly. The 5 species with a single age point keep their
# externally estimated slopes.
# (sp, rate/yr, T_max, product, burden_end, frac_sampled%, flag)
records = [
    ('Rat',                     607,   3.8,  2307, 1337, 48, 'N'),
    ('Mouse',                   761,   4.0,  3046, 1558, 50, 'N'),
    ('Ferret',                  485,  10.0,  4851, 3516, 71, 'N'),
    ('Rabbit',                   98,  13.0,  1268,  875, 39, 'A'),
    ('Dog',                     222,  20.0,  4435, 2837, 60, 'N'),
    ('Tiger',                   367,  26.0,  9536, 3892, 61, 'A'),
    ('Ring-tailed lemur',       102,  27.0,  2753, 2638, 99, 'N'),
    ('Lion',                    160,  29.0,  4645, 2950, 64, 'N'),
    ('Cat',                     547,  17.0,  9299, 3247, 57, 'N'),
    ('Cow',                     220,  30.0,  6612, 1870, 33, 'N'),
    ('Black-and-white colobus',  79,  30.0,  2369, 1125, 47, 'N'),
    ('Naked mole-rat',          103,  37.0,  3798,  727, 20, 'A'),
    ('Giraffe',                 100,  39.5,  3960, 1462, 37, 'N'),
    ('Horse',                   201,  57.0, 11452, 2752, 33, 'A'),
    ('Human',                    53, 122.5,  6522, 3282, 58, 'N'),
]
records.sort(key=lambda x: x[2])   # sort by T_max ascending

sp_names = [r[0] for r in records]
rates    = np.array([r[1] for r in records])
tmaxs    = np.array([r[2] for r in records])
prods    = np.array([r[3] for r in records])
burds    = np.array([r[4] for r in records])
fracs    = np.array([r[5] for r in records])
flags    = [r[6] for r in records]

prod_mean   = prods.mean()
prod_median = np.median(prods)
prod_std    = prods.std()
burd_mean   = burds.mean()
burd_std    = burds.std()

# VBLS predicted: rate x T_max = const from scaling
# median of "normal" species (N flag)
normal_idx  = [i for i,f in enumerate(flags) if f == 'N']
prod_normal = prods[normal_idx]
prod_vlbs   = np.median(prod_normal)

# ── Figure: 3 panels ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.20,
                    wspace=0.38)

# ── Panel 1: rate x T_max bar chart ──────────────────────────────────────────
ax1 = axes[0]

# Sort by T_max
order = np.argsort(tmaxs)
sp_s  = [sp_names[i] for i in order]
pr_s  = prods[order]
fl_s  = [flags[i] for i in order]
fr_s  = fracs[order]
cols1 = []
alphs = []
for i in order:
    c = SP_COLORS.get(sp_names[i], '#888')
    cols1.append(c)
    alphs.append(0.60 if flags[i]=='A' else 0.82)

bars1 = ax1.bar(range(len(sp_s)), pr_s, color=cols1, alpha=0.82,
                edgecolor='white', lw=0.8, zorder=3)

# Hatch anomalies
for bi, (bar, fl) in enumerate(zip(bars1, fl_s)):
    if fl == 'A':
        bar.set_hatch('///')
        bar.set_edgecolor(COL_ANOM)
        bar.set_linewidth(1.2)

# VBLS predicted constant (median of normal species)
ax1.axhline(prod_vlbs, color=COL_VBLS, lw=2.0, ls='--', zorder=4,
            label=f'VBLS: rate × $T_{{max}}$ = {prod_vlbs:.0f}\n(median of normal species)')
ax1.axhspan(prod_vlbs * 0.5, prod_vlbs * 1.5,
            alpha=0.08, color=COL_VBLS, zorder=2, label='±50% band')

# Annotate anomalies (Cat excluded: with T_max=17yr product ~9300, within normal range)
annots = {
    'Horse':         ('Sample 33%\nT_max extrapolated', 0.2, 0.88),
    'Tiger':         ('Sample 61%\nnonlinearity?', 0.2, 0.78),
    'Rabbit':        ('Sample 39%\nnoisy regression', 0.2, 0.25),
    'Naked mole-rat':('E(t) channel\n(anomalously low Sigma*)', 0.2, 0.18),
}
for bi, sp in enumerate(sp_s):
    if sp in annots:
        txt, dx, yf = annots[sp]
        ymax = ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else pr_s.max()*1.3
        ax1.annotate(txt,
                     xy=(bi, pr_s[bi]),
                     xytext=(bi + dx*3, pr_s.max() * yf),
                     fontsize=7.2, color=COL_ANOM,
                     arrowprops=dict(arrowstyle='->', color=COL_ANOM,
                                     lw=0.8, alpha=0.7))

ax1.set_xticks(range(len(sp_s)))
ax1.set_xticklabels(sp_s, rotation=42, ha='right', fontsize=8.5)
ax1.set_ylabel('rate × $T_{max}$ (mut. per lifetime)', fontsize=10)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.92)
ax1.set_title(f'Product rate × $T_{{max}}$\nacross {len(sp_s)} species', fontsize=10.5,
              fontweight='bold')

# CV annotation
cv_all = prod_std / prod_mean * 100
cv_norm = prod_normal.std() / prod_normal.mean() * 100
ax1.text(0.98, 0.97,
         f'CV all: {cv_all:.0f}%\nCV without anomalies: {cv_norm:.0f}%',
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=8.5, color='#333',
         bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

# ── Panel 2: power law rate ~ T_max^alpha  ────────────────────────────────────
ax2 = axes[1]

for i, (sp, r, tm, fl) in enumerate(zip(sp_names, rates, tmaxs, flags)):
    col = SP_COLORS.get(sp, '#888')
    mk  = 'D' if fl == 'A' else 'o'
    ax2.scatter(tm, r, color=col, s=65, zorder=4,
                marker=mk, edgecolors='white', lw=0.8,
                alpha=0.70 if fl=='A' else 0.90)
    offset_x = tm * 1.04
    offset_y = r * 1.05
    if sp in ['Human','Mouse','Naked mole-rat','Cat','Rabbit']:
        ax2.annotate(sp, (tm, r), (offset_x, offset_y),
                     fontsize=7.5, color=col)

# Regression on ALL
phi_all, loga_all, r_all, _, _ = linregress(np.log(tmaxs), np.log(rates))
tm_line = np.logspace(np.log10(tmaxs.min()*0.8), np.log10(tmaxs.max()*1.2), 200)
ax2.plot(tm_line, np.exp(loga_all)*tm_line**phi_all,
         color=COL_REG, lw=1.5, ls='--', zorder=3,
         label=f'Regression (all): φ={phi_all:.2f}, $R^2={r_all**2:.2f}$')

# Regression on NORMAL only
tmaxs_n = tmaxs[normal_idx]; rates_n = rates[normal_idx]
phi_n, loga_n, r_n, _, _ = linregress(np.log(tmaxs_n), np.log(rates_n))
ax2.plot(tm_line, np.exp(loga_n)*tm_line**phi_n,
         color=COL_VBLS, lw=2.0, ls='-', zorder=4, alpha=0.80,
         label=f'Regression (no anomalies): φ={phi_n:.2f}, $R^2={r_n**2:.2f}$')

# VBLS prediction: exact -1 line through median
A_vlbs = prod_vlbs
ax2.plot(tm_line, A_vlbs / tm_line,
         color=COL_VBLS, lw=1.5, ls=':', zorder=5, alpha=0.55,
         label=f'VBLS: φ=−1  (rate × $T_{{max}}$={A_vlbs:.0f})')

ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('$T_{max}$ (years)', fontsize=10)
ax2.set_ylabel('Mutation rate (mut./genome/yr)', fontsize=10)
ax2.legend(fontsize=8, framealpha=0.92, loc='upper right')
ax2.set_title('Power law: rate ~ $T_{max}^{\\varphi}$\nVBLS predicts φ = −1',
              fontsize=10.5, fontweight='bold')

# ── Panel 3: burden at end vs corrected burden ────────────────────────────────
ax3 = axes[2]

x = np.arange(len(sp_names))
order3 = np.argsort(tmaxs)
sp_s3  = [sp_names[i] for i in order3]
burd_s = burds[order3]
frac_s = fracs[order3] / 100
cols3  = [SP_COLORS.get(sp_names[i],'#888') for i in order3]
fl_s3  = [flags[i] for i in order3]

# Corrected (extrapolated to T_max assuming linear)
burd_corr = burd_s / frac_s

w = 0.38
bars_obs  = ax3.bar(x - w/2, burd_s,    width=w, color=cols3,
                    alpha=0.82, edgecolor='white', lw=0.8, zorder=3,
                    label='Observed burden at end of sample')
bars_corr = ax3.bar(x + w/2, burd_corr, width=w, color=cols3,
                    alpha=0.40, edgecolor='white', lw=0.8, zorder=3,
                    hatch='...', label='Extrapolated to $T_{max}$ (linear)')

# Universal Sigma* line
ax3.axhline(burd_mean, color=COL_UNIV, lw=1.8, ls='--', zorder=4,
            label=f'$\\Sigma^*_{{univ}}$ obs. = {burd_mean:.0f} ± {burd_std:.0f}')

ax3.set_xticks(x)
ax3.set_xticklabels(sp_s3, rotation=42, ha='right', fontsize=8.5)
ax3.set_ylabel('Mutations per genome', fontsize=10)
ax3.legend(fontsize=8, framealpha=0.92, loc='upper left')
ax3.set_title('Observed vs extrapolated\nmutation burden at end of life',
              fontsize=10.5, fontweight='bold')

cv_obs  = burd_std / burd_mean * 100
cv_corr = burd_corr.std() / burd_corr.mean() * 100
ax3.text(0.98, 0.97,
         f'CV obs.: {cv_obs:.0f}%\nCV extrap.: {cv_corr:.0f}%',
         transform=ax3.transAxes, ha='right', va='top',
         fontsize=8.5, color='#333',
         bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    f'Cagan hypothesis: rate × $T_{{max}}$ ≈ const  |  {len(sp_names)} mammalian species\n'
    f'VBLS predicts φ = −1; data yield φ = {phi_all:.2f} due to anomalies and incomplete age coverage\n'
    f'Hatched diamonds = {len(ANOMALY)} anomalous species (explanations in text)',
    fontsize=10.5, fontweight='bold', y=0.99)

fig.text(0.99, 0.005,
         'Data: Cagan et al. (2022) Nature 604:517-524',
         ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('cagan_fig3_rateXtmax.pdf', bbox_inches='tight')
plt.savefig('cagan_fig3_rateXtmax.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: cagan_fig3_rateXtmax")

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY: rate x T_max analysis")
print(f"{'='*65}")
print(f"  All species:      mean={prod_mean:.0f}  median={prod_median:.0f}  CV={cv_all:.0f}%")
print(f"  Normal species:   mean={prod_normal.mean():.0f}  median={prod_vlbs:.0f}  CV={cv_norm:.0f}%")
print(f"  Power law phi:    all={phi_all:.3f}  normal={phi_n:.3f}  (vbls=-1.000)")
print(f"  R2 power law:     all={r_all**2:.3f}  normal={r_n**2:.3f}")
print(f"  burden_end CV:    {cv_obs:.0f}% (Cagan: ~3x range)")
print(f"  burden_corr CV:   {cv_corr:.0f}%")
