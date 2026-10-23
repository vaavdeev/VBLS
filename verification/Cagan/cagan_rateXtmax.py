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

T_max is the maximum longevity from AnAge (verification/AnAge/anage_data.csv) for
all 15 species, without corrections (Section 17.3). No species is excluded from
the regression; the species whose product lies outside the +-50% band around the
median are marked and discussed in Section 17.3.5.3.

Mutation rate per species (mutations/genome/yr): OLS slope of the individual-mean
burden on age, Cagan et al. (2022) Supplementary Table 4 (Table_4.csv); the five
species represented by a single individual keep their externally estimated slopes.

Out: cagan_fig3_rateXtmax.pdf / .png
"""

import csv
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
COL_ANOM  = '#e67e22'   # species outside the +-50% band
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

HERE = os.path.dirname(os.path.abspath(__file__))
ANAGE_FILE = os.path.join(HERE, '..', 'AnAge', 'anage_data.csv')

# Cagan species -> AnAge genus and species (the domestic ferret is listed under Mustela putorius)
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

# Species represented by a single individual: externally estimated slopes (mut./genome/yr)
SINGLE_AGE_RATE = {
    'Ferret':                  485.10,
    'Ring-tailed lemur':       101.96,
    'Lion':                    160.17,
    'Black-and-white colobus':  78.97,
    'Giraffe':                 100.25,
}

# ── Data: rate, burden at the end of observation, oldest sampled age ─────────
df = pd.read_csv(os.path.join(HERE, 'Table_4.csv'), sep=';', na_values=['NA', 'na', ''])
df['Age_years'] = pd.to_numeric(df['Age_years'], errors='coerce')
df['Substitution_burden'] = pd.to_numeric(df['Substitution_burden'], errors='coerce')
df = df.dropna(subset=['Age_years', 'Substitution_burden'])
df = df[df['Species'].isin(list(TMAX))]

records = []
for sp, g in df.groupby('Species'):
    ind = g.groupby(['Individual', 'Age_years'])['Substitution_burden'].mean().reset_index()
    oldest = ind['Age_years'].max()
    # crypts at the oldest sampled age, as N_mut,end in cagan_verification.py (Section 17.3.6)
    burden_end = g.loc[g['Age_years'] == oldest, 'Substitution_burden'].mean()
    if ind['Age_years'].nunique() >= 2:
        rate = linregress(ind['Age_years'], ind['Substitution_burden']).slope
    else:
        rate = SINGLE_AGE_RATE[sp]
    records.append((sp, rate, TMAX[sp], rate * TMAX[sp], burden_end, oldest / TMAX[sp] * 100))
records.sort(key=lambda x: x[2])   # sort by T_max ascending

sp_names = [r[0] for r in records]
rates    = np.array([r[1] for r in records])
tmaxs    = np.array([r[2] for r in records])
prods    = np.array([r[3] for r in records])
burds    = np.array([r[4] for r in records])
fracs    = np.array([r[5] for r in records])

prod_mean   = prods.mean()
prod_median = np.median(prods)
prod_std    = prods.std()
burd_mean   = burds.mean()
burd_std    = burds.std()

# Species outside the +-50% band around the median product (descriptive, none excluded)
OUT = [sp for sp, p in zip(sp_names, prods) if abs(p / prod_median - 1.0) > 0.5]

# ── Figure: 3 panels ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.20,
                    wspace=0.38)

# ── Panel 1: rate x T_max bar chart ──────────────────────────────────────────
ax1 = axes[0]

order = np.argsort(tmaxs)
sp_s  = [sp_names[i] for i in order]
pr_s  = prods[order]
cols1 = [SP_COLORS.get(sp_names[i], '#888') for i in order]

bars1 = ax1.bar(range(len(sp_s)), pr_s, color=cols1, alpha=0.82,
                edgecolor='white', lw=0.8, zorder=3)

# Hatch the species outside the +-50% band
for bar, sp in zip(bars1, sp_s):
    if sp in OUT:
        bar.set_hatch('///')
        bar.set_edgecolor(COL_ANOM)
        bar.set_linewidth(1.2)

# VBLS: constant product (line at the median of all species)
ax1.axhline(prod_median, color=COL_VBLS, lw=2.0, ls='--', zorder=4,
            label=f'VBLS: rate × $T_{{max}}$ = const\n(line at the median\nof all species, {prod_median:.0f})')
ax1.axhspan(prod_median * 0.5, prod_median * 1.5,
            alpha=0.08, color=COL_VBLS, zorder=2, label='±50% band')

# Annotate the species outside the band (reasons: Section 17.3.5.3)
annots = {
    'Horse':  ('Sample 33%\nT_max extrapolated', -0.2, 0.92),
    'Tiger':  ('Sample 61%\n2 individuals', -0.25, 0.64),
    'Cat':    ('Sample 32%\nnarrow window', -1.3, 0.80),
    'Rabbit': ('Sample 39%\nnoisy regression', -0.93, 0.45),
}
for bi, sp in enumerate(sp_s):
    if sp in OUT and sp in annots:
        txt, dx, yf = annots[sp]
        ax1.annotate(txt,
                     xy=(bi, pr_s[bi]),
                     xytext=(bi + dx * 3, pr_s.max() * yf),
                     fontsize=7.2, color=COL_ANOM,
                     arrowprops=dict(arrowstyle='->', color=COL_ANOM,
                                     lw=0.8, alpha=0.7))

ax1.set_xticks(range(len(sp_s)))
ax1.set_xticklabels(sp_s, rotation=42, ha='right', fontsize=8.5)
ax1.set_ylabel('rate × $T_{max}$ (mut. per lifetime)', fontsize=10)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.92)
ax1.set_title(f'Product rate × $T_{{max}}$\nacross {len(sp_s)} species', fontsize=10.5,
              fontweight='bold')

cv_all = prod_std / prod_mean * 100
ax1.text(0.98, 0.97, f'CV: {cv_all:.0f}%',
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=8.5, color='#333',
         bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8', ec='#ccc'))

# ── Panel 2: power law rate ~ T_max^phi ──────────────────────────────────────
ax2 = axes[1]

for sp, r, tm in zip(sp_names, rates, tmaxs):
    col = SP_COLORS.get(sp, '#888')
    out = sp in OUT
    ax2.scatter(tm, r, color=col, s=65, zorder=4,
                marker='D' if out else 'o', edgecolors='white', lw=0.8,
                alpha=0.70 if out else 0.90)
    if sp in ['Human', 'Mouse', 'Naked mole-rat', 'Cat', 'Rabbit']:
        ax2.annotate(sp, (tm, r), (tm * 1.04, r * 1.05), fontsize=7.5, color=col)

phi_all, loga_all, r_all, _, _ = linregress(np.log(tmaxs), np.log(rates))
tm_line = np.logspace(np.log10(tmaxs.min() * 0.8), np.log10(tmaxs.max() * 1.2), 200)
ax2.plot(tm_line, np.exp(loga_all) * tm_line ** phi_all,
         color=COL_REG, lw=1.5, ls='--', zorder=3,
         label=f'Regression (all 15): φ={phi_all:.2f}, $R^2={r_all**2:.2f}$')

# VBLS: exact -1 line through the median product
ax2.plot(tm_line, prod_median / tm_line,
         color=COL_VBLS, lw=1.5, ls=':', zorder=5, alpha=0.70,
         label=f'VBLS: φ=−1  (rate × $T_{{max}}$={prod_median:.0f})')

ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('$T_{max}$ (years)', fontsize=10)
ax2.set_ylabel('Mutation rate (mut./genome/yr)', fontsize=10)
ax2.legend(fontsize=8, framealpha=0.92, loc='upper right')
ax2.set_title('Power law: rate ~ $T_{max}^{\\varphi}$\nVBLS, uniform scaling: φ = −1',
              fontsize=10.5, fontweight='bold')

# ── Panel 3: burden at the end of observation vs extrapolated ────────────────
ax3 = axes[2]

x = np.arange(len(sp_names))
order3 = np.argsort(tmaxs)
sp_s3  = [sp_names[i] for i in order3]
burd_s = burds[order3]
frac_s = fracs[order3] / 100
cols3  = [SP_COLORS.get(sp_names[i], '#888') for i in order3]

# Extrapolated to T_max assuming linear accumulation
burd_corr = burd_s / frac_s

w = 0.38
ax3.bar(x - w/2, burd_s, width=w, color=cols3,
        alpha=0.82, edgecolor='white', lw=0.8, zorder=3,
        label='Observed burden at end of sample')
ax3.bar(x + w/2, burd_corr, width=w, color=cols3,
        alpha=0.40, edgecolor='white', lw=0.8, zorder=3,
        hatch='...', label='Extrapolated to $T_{max}$ (linear)')

# Mean end-of-sample burden (one component of the Sigma index, Section 3.4)
ax3.axhline(burd_mean, color=COL_UNIV, lw=1.8, ls='--', zorder=4,
            label=f'Mean end-of-sample burden = {burd_mean:.0f}')

ax3.set_xticks(x)
ax3.set_xticklabels(sp_s3, rotation=42, ha='right', fontsize=8.5)
ax3.set_ylabel('Mutations per genome', fontsize=10)
ax3.legend(fontsize=8, framealpha=0.92, loc='upper left')
ax3.set_title('Burden at end of sample vs\nlinear extrapolation to $T_{max}$',
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
    f'Cagan hypothesis: rate × $T_{{max}}$ ≈ const  |  {len(sp_names)} mammalian species, $T_{{max}}$ from AnAge\n'
    f'VBLS with uniform scaling gives φ = −1, also in the sampled age windows; '
    f'data give φ = {phi_all:.2f}, which the model does not explain\n'
    f'Hatched bars (left) and diamonds (centre): {len(OUT)} species outside the ±50% band, see text',
    fontsize=10.5, fontweight='bold', y=0.99)

fig.text(0.99, 0.005,
         'Data: Cagan et al. (2022) Nature 604:517-524; $T_{max}$: AnAge',
         ha='right', fontsize=8, color='#666', style='italic')

plt.savefig(os.path.join(HERE, 'cagan_fig3_rateXtmax.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(HERE, 'cagan_fig3_rateXtmax.png'), bbox_inches='tight', dpi=180)
plt.close()
print("Saved: cagan_fig3_rateXtmax")

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY: rate x T_max analysis (T_max from AnAge, all 15 species)")
print(f"{'='*65}")
print(f"  {'species':24s} {'rate':>8s} {'T_max':>6s} {'product':>8s} {'end burden':>10s} {'window%':>8s}")
for sp, r, t, p, b, f in records:
    print(f"  {sp:24s} {r:8.2f} {t:6.1f} {p:8.0f} {b:10.0f} {f:8.0f}")
print(f"\n  Products:         mean={prod_mean:.0f}  median={prod_median:.0f}  CV={cv_all:.0f}%  "
      f"range {prods.min():.0f}-{prods.max():.0f} ({prods.max()/prods.min():.1f}x)")
print(f"  Outside +-50%:    {', '.join(OUT)}")
rest = np.array([p for sp, p in zip(sp_names, prods) if sp not in OUT])
print(f"  Without them:     range {rest.min():.0f}-{rest.max():.0f} ({rest.max()/rest.min():.1f}x)")
print(f"  Power law phi:    {phi_all:.3f}  R2={r_all**2:.3f}  (vbls=-1.000)")
print(f"  burden_end CV:    {cv_obs:.0f}% (Cagan: ~3x range)")
print(f"  burden_corr CV:   {cv_corr:.0f}%")
