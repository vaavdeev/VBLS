"""
Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
SPDX-License-Identifier: CC-BY-NC-ND-4.0

VBLS -- Appendix C.9.2: empirical calibration of the class invariant B*
========================================================================

Data: anage_data.csv (AnAge database, semicolon-delimited)
Run:  python bstar_calibration.py
Output: Fig1_allometry.png, Fig2_Bstar_normalized.png, Fig3_taxa_boxplots.png

Class B (Bauer action invariant): B = (E0 - E*) * tau_energy, formula (14.6).
Empirically, B ~ M^(5/4) (formula C.8); the normalized class invariant
B* = B / M^1.25 (formula C.9) is checked here on 306 species of class
Mammalia (AnAge Build 14, quality "acceptable" or "good", with data on
M, T_max and Phi_in).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# --- Style settings ----------------------------------------------------------
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'axes.titleweight': 'bold',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'legend.frameon':   True,
    'legend.framealpha':0.9,
    'figure.dpi':       150,
    'savefig.dpi':      200,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

# --- Physical constants -------------------------------------------------------
k_B    = 1.38e-23      # J/K
T_phys = 310.0         # K  (37 deg C)
kBT    = k_B * T_phys  # 4.28e-21 J

# --- Mammalia class parameters -------------------------------------------------
beta_cell = 60_000        # cellular structural complexity (calibrated from JCVI-syn3.0)
m_cell    = 1e-12         # kg, typical mammalian cell mass
c_tissue  = 3500          # J/(kg*K) -- specific heat of living tissue

# --- Loading and filtering AnAge data -------------------------------------------
print("Loading AnAge...")
df_raw = pd.read_csv('anage_data.csv', sep=';', encoding='utf-8-sig')

df = df_raw[
    (df_raw['Class'] == 'Mammalia') &
    df_raw['Maximum longevity (yrs)'].notna() &
    df_raw['Body mass (g)'].notna() &
    df_raw['Metabolic rate (W)'].notna() &
    df_raw['Data quality'].isin(['acceptable', 'good']) &
    (df_raw['Maximum longevity (yrs)'] > 1) &
    (df_raw['Body mass (g)'] > 0) &
    (df_raw['Metabolic rate (W)'] > 0)
].copy().reset_index(drop=True)

print(f"Species after filtering: {len(df)}")

# --- Computing variables -------------------------------------------------------
M_kg    = df['Body mass (g)'].values   / 1000.0
Phi_in  = df['Metabolic rate (W)'].values
T_max   = df['Maximum longevity (yrs)'].values
orders  = df['Order'].values

N_cells    = M_kg / m_cell
E_struct   = beta_cell * kBT * N_cells            # structural energy of the organism
tau_star   = (M_kg * c_tissue) / Phi_in           # characteristic time tau* (s)
B_org      = E_struct * tau_star                  # Bauer action invariant B (J*s)
B_star     = B_org / (M_kg ** 1.25)               # normalized B* (J*s*kg^-5/4)

# Allometric fit: B ~ M^alpha
slope_M, inter_M, r_M, p_M, se_M = stats.linregress(np.log10(M_kg), np.log10(B_org))

print(f"B ~ M^{slope_M:.4f}  r^2={r_M**2:.4f}")
print(f"B* median = {np.median(B_star):.4f},  sigma(log10) = {np.std(np.log10(B_star)):.3f}")

# --- Color palette by order ----------------------------------------------------
unique_orders = sorted(set(orders))
n_ord = len(unique_orders)
palette = plt.cm.tab20(np.linspace(0, 1, n_ord))
ord_color  = {o: palette[i] for i, o in enumerate(unique_orders)}
ord_marker = {o: m for o, m in zip(unique_orders,
    ['o','s','^','v','D','P','*','X','h','<','>','p','H','8','+','x']*(n_ord//16+1))}

# --- Orders with N>=5 for boxplots ---------------------------------------------
ord_n5 = [o for o in unique_orders if np.sum(orders == o) >= 5]


# ==============================================================================
# FIGURE 1.  Allometric law  B ~ M^(5/4)
# ==============================================================================
fig1, ax = plt.subplots(figsize=(9, 7))

added_labels = set()
for i in range(len(df)):
    o = orders[i]
    lbl = o if o not in added_labels else None
    ax.scatter(M_kg[i], B_org[i],
               color=ord_color[o], marker=ord_marker[o],
               s=28, alpha=0.65, linewidths=0, label=lbl)
    added_labels.add(o)

M_fit = np.logspace(np.log10(M_kg.min())*1.05, np.log10(M_kg.max())*1.05, 300)

ax.plot(M_fit, 10**inter_M * M_fit**slope_M,
        'k-', lw=2.5, zorder=5,
        label=fr'Data: $\mathcal{{B}} \sim M^{{{slope_M:.3f}}}$  '
              fr'($r^2={r_M**2:.3f}$,  theory exponent $=5/4=1.250$)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass $M$ (kg)', fontsize=12)
ax.set_ylabel(r'$\mathcal{B} = (E_0-E^*)\cdot\tau^*$  (J*s)', fontsize=12)
ax.set_title('Fig. C.9.1.  Allometric law of the Bauer action invariant\n'
             '306 species of class Mammalia (AnAge database)', fontsize=12)
ax.legend(fontsize=8, ncol=3, loc='upper left',
          bbox_to_anchor=(0, 1.0), columnspacing=0.8)
ax.grid(True, which='both', alpha=0.2, linewidth=0.5)
ax.tick_params(which='both', direction='in')

plt.tight_layout()
fig1.savefig('Fig1_allometry.png')
print("Fig1 saved.")
plt.close(fig1)


# ==============================================================================
# FIGURE 2.  Normalized invariant B* = B/M^(5/4)
# ==============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# --- 2a: B* vs M (scatter) ------------------------------------------------------
ax2a = axes2[0]
sc = ax2a.scatter(M_kg, B_star,
                  c=np.log10(T_max), cmap='plasma',
                  s=30, alpha=0.7, linewidths=0.2, edgecolors='grey')
cb = plt.colorbar(sc, ax=ax2a, pad=0.02)
cb.set_label(r'$\log_{10}(T_{max},\ \mathrm{yr})$', fontsize=10)

med  = np.median(B_star)
p16  = np.percentile(B_star, 16)
p84  = np.percentile(B_star, 84)
p2   = np.percentile(B_star, 2.5)
p975 = np.percentile(B_star, 97.5)

ax2a.axhline(med,  color='crimson', lw=2.5, zorder=6,
             label=fr'Median: $\mathcal{{B}}^*={med:.3f}$ J*s*kg$^{{-5/4}}$')
ax2a.axhspan(p16, p84, alpha=0.15, color='crimson',
             label=fr'$\pm1\sigma$  [{p16:.3f} - {p84:.3f}]')
ax2a.axhspan(p2, p975, alpha=0.07, color='crimson',
             label='95% interval')

ax2a.set_xscale('log'); ax2a.set_yscale('log')
ax2a.set_xlabel('Body mass $M$ (kg)', fontsize=12)
ax2a.set_ylabel(r'$\mathcal{B}^* = \mathcal{B}\,/\,M^{5/4}$  (J*s*kg$^{-5/4}$)', fontsize=12)
ax2a.set_title('a)  Normalized $\\mathcal{B}^*$ by mass\n(color -- lifespan)')
ax2a.legend(fontsize=9)
ax2a.grid(True, which='both', alpha=0.2, linewidth=0.5)

# --- 2b: Histogram of log10(B*) -------------------------------------------------
ax2b = axes2[1]
log_bstar = np.log10(B_star)
ax2b.hist(log_bstar, bins=30, color='steelblue', alpha=0.75,
          edgecolor='white', linewidth=0.5, density=True, label='Data')

x_kde = np.linspace(log_bstar.min()-0.3, log_bstar.max()+0.3, 300)
kde   = gaussian_kde(log_bstar, bw_method='silverman')
ax2b.plot(x_kde, kde(x_kde), 'k-', lw=2.2, label='KDE')
ax2b.axvline(np.log10(med), color='crimson', lw=2, ls='-',
             label=fr'Median = {np.log10(med):.2f}')
ax2b.axvline(np.log10(med)-np.std(log_bstar), color='crimson',
             lw=1.5, ls='--')
ax2b.axvline(np.log10(med)+np.std(log_bstar), color='crimson',
             lw=1.5, ls='--', label=fr'$\pm\sigma$ ($\sigma$={np.std(log_bstar):.3f})')
ax2b.set_xlabel(r'$\log_{10}(\mathcal{B}^*)$', fontsize=12)
ax2b.set_ylabel('Density', fontsize=12)
ax2b.set_title('b)  Distribution of $\\log_{10}(\\mathcal{B}^*)$\n'
               fr'N={len(df)},  median$\pm\sigma$ = {np.log10(med):.2f}$\pm${np.std(log_bstar):.2f}')
ax2b.legend(fontsize=9)
ax2b.grid(True, alpha=0.2, linewidth=0.5)

fig2.suptitle('Fig. C.9.2.  Normalized Bauer action invariant '
              r'$\mathcal{B}^* = \mathcal{B}/M^{5/4}$  --  class invariant of Mammalia',
              fontsize=12, y=1.01)
plt.tight_layout()
fig2.savefig('Fig2_Bstar_normalized.png')
print("Fig2 saved.")
plt.close(fig2)


# ==============================================================================
# FIGURE 3.  B* by taxonomic order (boxplots)
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(13, 7))

data_bp   = [np.log10(B_star[orders == o]) for o in ord_n5]
colors_bp = [ord_color[o] for o in ord_n5]
counts_bp = [np.sum(orders == o) for o in ord_n5]

bp = ax3.boxplot(data_bp, patch_artist=True, notch=False, vert=True,
                 medianprops=dict(color='black', lw=2.2),
                 whiskerprops=dict(lw=1.4),
                 capprops=dict(lw=1.4),
                 flierprops=dict(marker='o', markersize=4,
                                 markerfacecolor='grey', alpha=0.5))

for patch, col in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(col)
    patch.set_alpha(0.78)

ax3.axhline(np.log10(np.median(B_star)), color='crimson',
            lw=2.2, ls='--', zorder=5,
            label=fr'Class median: $\log_{{10}}\mathcal{{B}}^*={np.log10(np.median(B_star)):.2f}$')
# Theory line omitted here: its absolute value depends on the definition of
# tau* and is not on the same scale as the empirical data. The theoretical
# exponent alpha=5/4 is verified in Fig1.

all_vals = np.concatenate(data_bp)
pad = 0.25
ax3.set_ylim(all_vals.min() - pad, all_vals.max() + pad)

tick_labels = [f'{o}\n(N={c})' for o, c in zip(ord_n5, counts_bp)]
ax3.set_xticks(range(1, len(ord_n5)+1))
ax3.set_xticklabels(tick_labels, rotation=40, ha='right', fontsize=9)
ax3.set_ylabel(r'$\log_{10}(\mathcal{B}^*)$  [J*s*kg$^{-5/4}$]', fontsize=12)
ax3.set_title('Fig. C.9.3.  Bauer action invariant $\\mathcal{B}^*$ by taxonomic order',
              fontsize=12)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.25, axis='y', linewidth=0.5)
ax3.set_xlim(0.5, len(ord_n5)+0.5)

plt.tight_layout()
fig3.savefig('Fig3_taxa_boxplots.png')
print("Fig3 saved.")
plt.close(fig3)


# --- Text summary --------------------------------------------------------------
print("\n" + "="*62)
print("SUMMARY OF RESULTS")
print("="*62)
print(f"Number of species:        N  = {len(df)}")
print(f"Allometry:          B ~ M^{slope_M:.4f}  (theory 5/4=1.250)")
print(f"Allometry r^2:             {r_M**2:.4f}  (p < 10^{int(np.log10(p_M))-1})")
print(f"B* median:           {np.median(B_star):.4f} J*s*kg^(-5/4)")
print(f"sigma(log10 B*):           {np.std(np.log10(B_star)):.4f}  (spread x{10**np.std(np.log10(B_star)):.2f})")
print(f"95% interval of B*:  [{np.percentile(B_star,2.5):.3f} - {np.percentile(B_star,97.5):.3f}]")
print("\nFiles saved:")
print("  Fig1_allometry.png          -- B ~ M^(5/4)")
print("  Fig2_Bstar_normalized.png   -- B* and its distribution")
print("  Fig3_taxa_boxplots.png      -- B* by order (boxplots)")
