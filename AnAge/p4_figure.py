# -*- coding: utf-8 -*-
"""Figure for P4 (РЕПАРАЦИЯ.md §12): sigma0_eff vs body mass, and the
trend estimate under three phylogeny treatments (OLS / taxonomic PGLS /
molecular PGLS). sigma0_eff ∝ 1/Rubner = 1/(T_max * P/M).  P4 predicts slope 0.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER_COLORS = {'Rodentia': '#e74c3c', 'Primates': '#2980b9', 'Carnivora': '#27ae60',
                'Artiodactyla': '#e67e22', 'Soricomorpha': '#16a085',
                'Dasyuromorphia': '#c0392b', 'Diprotodontia': '#f39c12'}

df = pd.read_csv(os.path.join(HERE, 'anage_data.csv'), sep=';',
                 encoding='utf-8-sig', on_bad_lines='skip')
m = df[df['Class'] == 'Mammalia'].copy()
for c in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m = m.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)'])
m = m[(m['Body mass (g)'] > 0) & (m['Maximum longevity (yrs)'] > 0) & (m['Metabolic rate (W)'] > 0)]
m['PM'] = m['Metabolic rate (W)'] / m['Body mass (g)'] * 1000
m['Rubner'] = m['Maximum longevity (yrs)'] * m['PM']
base = m[(m['Data quality'] != 'low') & (m['Order'] != 'Chiroptera')].copy()
base['sigma0'] = base['Rubner'].median() / base['Rubner']      # ∝1/Rubner, median=1

LM = np.log10(base['Body mass (g)'].values)
LS = np.log10(base['sigma0'].values)
sl, ic, r, p, se = stats.linregress(LM, LS)                     # OLS slope on sigma0 = -(a+b)

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                               gridspec_kw={'width_ratios': [1.55, 1]})

# Panel A: sigma0_eff vs M, by order
for order, g in base.groupby('Order'):
    axA.scatter(g['Body mass (g)'], g['sigma0'], s=18,
                c=ORDER_COLORS.get(order, '#95a5a6'), alpha=0.6, zorder=3,
                label=order if order in ORDER_COLORS else None)
xr = np.logspace(LM.min(), LM.max(), 100)
axA.plot(xr, 10**ic * xr**sl, 'k-', lw=2.2, zorder=5,
         label=f'OLS: slope={sl:+.3f}±{se:.3f}')
axA.plot(xr, np.full_like(xr, np.median(base['sigma0'])), 'k--', lw=1.4, alpha=0.5,
         zorder=4, label='P4 prediction: slope = 0')
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlabel('Body mass  M (g)')
axA.set_ylabel(r'$\sigma_0^{\rm eff} \propto 1/(T_{\max}\cdot P/M)$  (median=1)')
axA.set_title(r'A.  $\sigma_0^{\rm eff}$ vs body mass  (Mammalia, $n=%d$)' % len(base),
              fontweight='bold')
hands = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, ms=8, label=o)
         for o, c in ORDER_COLORS.items()]
hands += [Line2D([0], [0], color='k', lw=2.2, label=f'OLS slope {sl:+.3f}'),
          Line2D([0], [0], color='k', ls='--', lw=1.4, alpha=0.6, label='P4: slope 0')]
axA.legend(handles=hands, fontsize=7.5, ncol=2, loc='upper right', framealpha=0.93)

# Panel B: trend estimate under three phylogeny treatments (sigma0 slope = +exponent)
methods = ['OLS\n(naive)', 'Taxonomic\nPGLS', 'Molecular\nPGLS (OTL)']
slopes  = [0.059, 0.113, 0.124]          # sigma0 ∝ M^(+slope)  (= -(a+b))
errs    = [0.012, 0.017, 0.015]
pvals   = ['2e-6', '6e-11', '2e-15']
cols    = ['#95a5a6', '#e67e22', '#c0392b']
xpos = np.arange(3)
axB.axhline(0, color='k', ls='--', lw=1.4, alpha=0.6)
axB.errorbar(xpos, slopes, yerr=np.array(errs) * 1.96, fmt='o', ms=9,
             capsize=5, lw=1.8, ecolor='#444',
             markerfacecolor='none', markeredgewidth=1.8)
for i, (s, pv, c) in enumerate(zip(slopes, pvals, cols)):
    axB.scatter([xpos[i]], [s], s=80, color=c, zorder=5)
    axB.annotate(f'{s:+.3f}\np={pv}', (xpos[i], s), xytext=(0, 12),
                 textcoords='offset points', ha='center', fontsize=8.5, color='#222')
axB.text(0.03, 0.95, 'P4 predicts 0\n(N-independent $\\sigma_0$)', transform=axB.transAxes,
         va='top', fontsize=9, color='#555',
         bbox=dict(boxstyle='round,pad=0.35', fc='#f7f7f7', ec='#bbb'))
axB.set_xticks(xpos); axB.set_xticklabels(methods, fontsize=9)
axB.set_ylabel(r'exponent of  $\sigma_0^{\rm eff}\propto M^{\,k}$')
axB.set_ylim(-0.03, 0.17)
axB.set_title('B.  Residual mass-trend strengthens\nwith phylogenetic correction',
              fontweight='bold')

fig.suptitle('P4 test: is $\\sigma_0^{\\rm eff}$ independent of body mass?  '
             '— rejected (weak but robust $\\sigma_0\\propto M^{+0.12}$)',
             fontsize=11, fontweight='bold', y=1.01)
fig.tight_layout()
out = os.path.join(HERE, 'p4_sigma0_mass.png')
fig.savefig(out, dpi=170, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('saved:', out)
print(f'OLS sigma0 slope = {sl:+.4f} ± {se:.4f}  p={p:.2e}  (= -(alpha+beta))')
