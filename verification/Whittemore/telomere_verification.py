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
VBLS — Consistency with Whittemore et al. (2019) PNAS
Telomere shortening data: 9 species (birds & mammals)

PRIMARY RESULT (Variant A): all 9 species, single universal R*_univ,
formula t_B = (1 - R*_univ) / nu*. R²_log = 0.909.

SUPPLEMENTARY (Variant B): two-channel separation (n=7 R-channel + n=2 E-channel).
R²_log = 0.976 for cluster 1 — shows structural refinement, not primary claim.

4 Figures:
  Fig. 1 — Universal R*: residual telomere fraction at death (bar chart)
  Fig. 2 — Predicted vs Observed T_max: Variant A primary (all 9), channel labels
  Fig. 3 — Power law: T_max ~ ν*^(-1) from VBLS vs Whittemore regression
  Fig. 4 — Sensitivity analysis: MAPE vs R*_univ perturbation (all 9 species)

Run: python telomere_verification.py
Output: telomere_fig1..4  (.pdf + .png)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── Style (same as mde_fig1_fig2) ──────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.2,
    'figure.dpi':        150,
})

COL_R   = '#1a6faf'   # R(t) channel — blue
COL_E   = '#c0392b'   # E(t) channel — red
COL_REF = '#27ae60'   # Whittemore regression — green
COL_MDE = '#8e44ad'   # MDE prediction — purple
PH_COL  = ['#ddeeff', '#fff9e6', '#ffe6e6']  # phase background colours

# ── Data (Table S1 + S2, Whittemore 2019) ──────────────────────────────────
#  Species, L0_kb, rate_bp_yr, T_max, T_avg, residual_%_at_T_max
raw = [
    ("Mouse",              50.0,   7000, 4.0,    2.0,  44.0),
    ("Goat",               10.4,    363, 20.8,  16.5,  27.4),
    ("Audouin's gull",     35.0,    771, 25.0,  21.0,  44.9),
    ("Reindeer",           19.8,    531, 21.8,  15.0,  41.5),
    ("Griffon vulture",    19.8,    209, 41.4,  37.0,  56.3),
    ("Bottlenose dolphin", 90.7,    766, 51.6,  17.0,  56.4),
    ("American flamingo",  21.0,    105, 60.0,  40.0,  70.0),
    ("Sumatran elephant",  36.3,    109, 65.5,  60.0,  80.3),
    ("Human",              15.0,     71, 122.5, 79.0,  42.0),
]

species  = [d[0] for d in raw]
L0       = np.array([d[1] for d in raw])
rate     = np.array([d[2] for d in raw])
T_max    = np.array([d[3] for d in raw])
T_avg    = np.array([d[4] for d in raw])
res_pct  = np.array([d[5] for d in raw])

nu_star  = rate / (L0 * 1000)        # fractional loss rate per year [1/yr]
R_star   = res_pct / 100.0           # observed residual fraction at death

# ── Cluster split at R*_obs = 0.60 (used only for Variant B / structural analysis)
THRESHOLD  = 0.60
mask_R     = R_star < THRESHOLD       # Cluster 1 — R(t) channel
mask_E     = R_star >= THRESHOLD      # Cluster 2 — E(t) channel

# ══════════════════════════════════════════════════════════════════════════════
# VARIANT A (PRIMARY): all 9 species, single universal R*_univ
# ══════════════════════════════════════════════════════════════════════════════
R_univ_A    = np.mean(R_star)          # mean over all 9 species
R_univ_A_std = np.std(R_star)
t_pred_A    = (1.0 - R_univ_A) / nu_star
MAPE_A      = np.mean(np.abs(t_pred_A - T_max) / T_max) * 100
r_A, _      = pearsonr(np.log(t_pred_A), np.log(T_max))
R2_A        = r_A**2

print(f"=== VARIANT A (primary, all 9) ===")
print(f"Universal R*_univ = {R_univ_A:.3f} ± {R_univ_A_std:.3f}")
print(f"MAPE = {MAPE_A:.1f}%,  R²_log = {R2_A:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# VARIANT B (supplementary): two-channel separation
# ══════════════════════════════════════════════════════════════════════════════
R_univ_B    = np.mean(R_star[mask_R])  # mean over cluster 1 only
R_univ_B_std = np.std(R_star[mask_R])
t_pred_B_c1 = (1.0 - R_univ_B) / nu_star[mask_R]
T_obs_c1    = T_max[mask_R]
MAPE_B      = np.mean(np.abs(t_pred_B_c1 - T_obs_c1) / T_obs_c1) * 100
r_B, _      = pearsonr(np.log(t_pred_B_c1), np.log(T_obs_c1))
R2_B        = r_B**2

print(f"\n=== VARIANT B (supplementary, cluster 1 n=7) ===")
print(f"R*_univ = {R_univ_B:.3f} ± {R_univ_B_std:.3f}")
print(f"MAPE = {MAPE_B:.1f}%,  R²_log = {R2_B:.3f}")

# ── Whittemore power law fit for comparison (all 9) ─────────────────────────
log_nu_all = np.log(nu_star)
log_T_all  = np.log(T_max)
psi_w, logA_w = np.polyfit(log_nu_all, log_T_all, 1)
A_w   = np.exp(logA_w)
T_wfit_all = A_w * nu_star**psi_w
r_w, _ = pearsonr(np.log(T_wfit_all), log_T_all)
R2_w   = r_w**2
# R² as reported by Whittemore et al. (2019) — empirical benchmark cited in the
# monograph (§17.2). NB: the script-recomputed R2_w equals VBLS R2_A by
# construction (both = corr(log ν*, log T_max)²), so it must NOT be labelled as
# Whittemore's published value; the figure box and console cite 0.829 separately.
R2_w_reported = 0.829
print(f"\nLog-log power-law fit on these 9 species: ψ = {-psi_w:.3f},  R² = {R2_w:.3f}")
print(f"  (this R² equals VBLS R²_log by construction; "
      f"Whittemore 2019 reported R² = {R2_w_reported:.3f})")

# Power-law fit on cluster 1 only (R-channel, n=7) — slope cited as ψ_emp in §17.2.6
psi_w7, logA_w7 = np.polyfit(np.log(nu_star[mask_R]), np.log(T_max[mask_R]), 1)
A_w7 = np.exp(logA_w7)
print(f"Cluster-1 (R-channel, n=7) log-log fit: ψ = {-psi_w7:.3f}")

# ── Sensitivity grid (Variant A: all 9) ─────────────────────────────────────
perturb = np.linspace(-0.25, 0.25, 201)
mape_grid = []
for dp in perturb:
    R_test = R_univ_A * (1 + dp)
    t_test = (1.0 - R_test) / nu_star
    mape_grid.append(np.mean(np.abs(t_test - T_max) / T_max) * 100)
mape_grid = np.array(mape_grid)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Universal R*: residual telomere fraction at death
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 1...")
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.subplots_adjust(left=0.12, right=0.96, top=0.87, bottom=0.20)

x_pos = np.arange(len(species))
colours = [COL_R if not mask_E[i] else COL_E for i in range(len(species))]

bars = ax.bar(x_pos, res_pct, color=colours, alpha=0.82, edgecolor='white',
              linewidth=0.8, zorder=3, width=0.65)

# R* zone — cluster 1
ax.axhspan(0, THRESHOLD * 100, alpha=0.07, color=COL_R, zorder=0)
ax.axhline(R_univ_B * 100, color=COL_R, lw=1.8, ls='--', alpha=0.80, zorder=4)
ax.text(len(species) - 0.4, R_univ_B * 100 + 1.5,
        f'$R^*_{{univ}}$ = {R_univ_B*100:.0f}%  (Cluster 1)',
        color=COL_R, fontsize=9.5, ha='right', fontweight='bold')

# E(t) channel zone
ax.axhspan(THRESHOLD * 100, 105, alpha=0.07, color=COL_E, zorder=0)
ax.axhline(THRESHOLD * 100, color=COL_E, lw=1.4, ls=':', alpha=0.65, zorder=4)
ax.text(0.01, THRESHOLD * 100 + 1.5,
        'Cluster split threshold = 60%',
        color=COL_E, fontsize=8.5, style='italic')

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, res_pct)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=8.5,
            color='#333', fontweight='bold')

# Legend patches
import matplotlib.patches as mpatches
p1 = mpatches.Patch(color=COL_R, alpha=0.82, label='Cluster 1 — death by R(t) channel')
p2 = mpatches.Patch(color=COL_E, alpha=0.82, label='Cluster 2 — death by E(t) channel (telomere not limiting)')
ax.legend(handles=[p1, p2], fontsize=9, loc='upper left', framealpha=0.92)

ax.set_xticks(x_pos)
ax.set_xticklabels(species, rotation=30, ha='right', fontsize=9.5)
ax.set_ylabel('Residual telomere fraction at death (%)', fontsize=10.5)
ax.set_ylim(0, 100)
ax.set_xlim(-0.55, len(species) - 0.45)

ax.set_title(
    'Universal Threshold $R^*$: Residual Telomere Fraction at Death (9 Species)\n'
    'Species die not at zero telomeres, but at a conserved non-zero threshold  '
    '— consistent with VBLS bifurcation at $R(t) = R^*$',
    fontsize=11, fontweight='bold'
)
ax.text(0.99, -0.22,
        'Data: Whittemore et al. (2019) PNAS, Table S2',
        transform=ax.transAxes, ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('telomere_fig1_Rstar.pdf', bbox_inches='tight')
plt.savefig('telomere_fig1_Rstar.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: telomere_fig1_Rstar")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Predicted vs Observed T_max: Variant A primary (all 9 species)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2...")
fig, ax = plt.subplots(figsize=(8, 7.5))
fig.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.11)

# Perfect prediction line + band
ax.plot([0, 140], [0, 140], color='#aaa', lw=1.3, ls='--', zorder=1, label='Perfect prediction')
ax.fill_between([0, 140], [0, 112], [0, 168], alpha=0.07, color='#888', zorder=0, label='±20% band')

# All 9 — Variant A primary result
offsets_all = {
    'Mouse':              ( 2, -9),
    'Goat':               ( 2,  2),
    "Audouin's gull":     ( 2,  2),
    'Reindeer':           ( 2, -9),
    'Griffon vulture':    ( 2,  3),
    'Bottlenose dolphin': (-52, 3),
    'American flamingo':  ( 2,  3),
    'Sumatran elephant':  ( 2,  3),
    'Human':              (-48, 3),
}
for i, name in enumerate(species):
    col   = COL_E if mask_E[i] else COL_R
    mark  = 's' if mask_E[i] else 'o'
    ax.scatter(T_max[i], t_pred_A[i], color=col, s=100, zorder=5,
               edgecolors='white', linewidths=0.8, marker=mark)
    dx, dy = offsets_all.get(name, (3, 3))
    ax.annotate(name, xy=(T_max[i], t_pred_A[i]),
                xytext=(T_max[i]+dx, t_pred_A[i]+dy), fontsize=8.5, color=col)

# Legend entries
import matplotlib.patches as mpatches
p_r = mpatches.Patch(color=COL_R, label=f'R(t)-channel species  (n=7)')
p_e = mpatches.Patch(color=COL_E, label=f'E(t)-channel species  (n=2, telomere not limiting)')
p_all = mpatches.Patch(color='none', label=f'All 9: MAPE={MAPE_A:.1f}%,  R²_log={R2_A:.3f}')
ax.legend(handles=[p_r, p_e, p_all], fontsize=9, loc='upper left', framealpha=0.93)

ax.set_xlim(-3, 145)
ax.set_ylim(-3, 230)
ax.set_xlabel('Observed $T_{max}$ (years)', fontsize=11)
ax.set_ylabel('VBLS-predicted $t_B$ (years)', fontsize=11)

ax.set_title(
    'VBLS vs Observed Lifespan — Variant A: all 9 species, single $R^*_{univ}$\n'
    r'$t_B = (1 - R^*_{univ})\;/\;\nu^*$  |  $R^*_{univ} = $'
    f'{R_univ_A:.3f}  (mean of all 9)',
    fontsize=10.5, fontweight='bold'
)
ax.text(0.99, -0.09,
        'Data: Whittemore et al. (2019) PNAS',
        transform=ax.transAxes, ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('telomere_fig2_pred_vs_obs.pdf', bbox_inches='tight')
plt.savefig('telomere_fig2_pred_vs_obs.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: telomere_fig2_pred_vs_obs")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Power law: T_max vs ν*  (all 9 species)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3...")
fig, ax = plt.subplots(figsize=(8.5, 6.5))
fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.12)

nu_range = np.logspace(np.log10(nu_star.min() * 0.5),
                        np.log10(nu_star.max() * 2.0), 200)

# VBLS Variant A analytical line (all 9)
t_vbls_line = (1.0 - R_univ_A) / nu_range
ax.plot(nu_range, t_vbls_line, color=COL_MDE, lw=2.4, zorder=3,
        label=r'VBLS (derived): $\psi=1$')

# Power-law regression — all 9 species (biased by the 2 E-channel outliers)
t_wh_line = A_w * nu_range**psi_w
ax.plot(nu_range, t_wh_line, color=COL_REF, lw=2.0, ls='--', zorder=3,
        label=f'Regression, all 9 species: $\\psi={-psi_w:.3f}$')

# Power-law regression — cluster 1 only (R-channel, n=7); slope cited as ψ_emp
t_wh7_line = A_w7 * nu_range**psi_w7
ax.plot(nu_range, t_wh7_line, color='#e67e22', lw=2.0, ls='-.', zorder=4,
        label=f'Regression, R-channel (n=7): $\\psi={-psi_w7:.3f}$')

# All 9 scatter
for i, name in enumerate(species):
    col  = COL_E if mask_E[i] else COL_R
    mark = 's' if mask_E[i] else 'o'
    ax.scatter(nu_star[i], T_max[i], color=col, s=80, zorder=6,
               edgecolors='white', linewidths=0.8, marker=mark)
    ax.annotate(name, xy=(nu_star[i], T_max[i]),
                xytext=(nu_star[i] * 1.15, T_max[i] * 1.12),
                fontsize=7.5, color=col,
                arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.6))

import matplotlib.patches as mpatches
p_r  = mpatches.Patch(color=COL_R, label='R(t)-channel species (n=7)')
p_e  = mpatches.Patch(color=COL_E, label='E(t)-channel species (n=2)')
line_handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=line_handles + [p_r, p_e],
          fontsize=7.5, loc='upper right', framealpha=0.93)

ax.text(0.03, 0.17,
        f'VBLS: $R^2_{{log}} = {R2_A:.3f}$,  MAPE $= {MAPE_A:.1f}\\%$\n'
        f'Whittemore (2019): $R^2 = {R2_w_reported:.3f}$',
        transform=ax.transAxes, fontsize=9.0,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#bbb', alpha=0.93),
        verticalalignment='bottom')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Telomere shortening rate  $\nu^*$ (fractional/year)', fontsize=11)
ax.set_ylabel('Maximum lifespan  $T_{max}$ (years)', fontsize=11)
ax.set_title(
    r'Power Law $T_{max} \sim \nu^{*\,-1}$: VBLS Derivation vs Empirical Regression' + '\n'
    r'VBLS derives $\psi=1$ analytically (Variant A, all 9 species);  regression fits post hoc',
    fontsize=10.5, fontweight='bold'
)
ax.text(0.99, -0.10, 'Data: Whittemore et al. (2019) PNAS',
        transform=ax.transAxes, ha='right', fontsize=8, color='#666', style='italic')

plt.savefig('telomere_fig3_powerlaw.pdf', bbox_inches='tight')
plt.savefig('telomere_fig3_powerlaw.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: telomere_fig3_powerlaw")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 4...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.14, wspace=0.30)

# ── LEFT: MAPE vs R*_univ perturbation (Variant A, all 9) ───────────────────
ax = axes[0]
pct_axis = perturb * 100
ax.plot(pct_axis, mape_grid, color=COL_MDE, lw=2.5, zorder=3)
ax.axvline(0, color='#555', lw=1.2, ls=':', zorder=2)
ax.axhline(MAPE_A, color='#555', lw=1.0, ls='--', alpha=0.60, zorder=2)

ax.axvspan(-10, 10, alpha=0.10, color=COL_MDE, zorder=0)
ax.text(0, mape_grid[len(mape_grid)//2] + 2, '±10% zone',
        ha='center', fontsize=8.5, color=COL_MDE, style='italic')

idx_m10 = np.argmin(np.abs(perturb + 0.10))
idx_p10 = np.argmin(np.abs(perturb - 0.10))
ax.scatter([-10, 10],
           [mape_grid[idx_m10], mape_grid[idx_p10]],
           color=COL_MDE, s=60, zorder=5)
ax.annotate(f'MAPE={mape_grid[idx_m10]:.1f}%',
            xy=(-10, mape_grid[idx_m10]),
            xytext=(-22, mape_grid[idx_m10] + 3),
            fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.9))
ax.annotate(f'MAPE={mape_grid[idx_p10]:.1f}%',
            xy=(10, mape_grid[idx_p10]),
            xytext=(12, mape_grid[idx_p10] + 3),
            fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.9))

ax.set_xlabel(r'Perturbation of $R^*_{univ}$ (%)', fontsize=10.5)
ax.set_ylabel('MAPE of $t_B$ prediction (%)', fontsize=10.5)
ax.set_title(r'Sensitivity of $t_B$ to $R^*_{univ}$ perturbation' + '\n(Variant A, all 9 species)',
             fontsize=10.5, fontweight='bold')

# ── RIGHT: per-species residuals all 9 ──────────────────────────────────────
ax = axes[1]
ratios = t_pred_A / T_max
colours_all = [COL_E if mask_E[i] else COL_R for i in range(len(species))]
x_all = np.arange(len(species))

bars = ax.bar(x_all, ratios, color=colours_all, alpha=0.80,
              edgecolor='white', linewidth=0.8, width=0.6, zorder=3)
ax.axhline(1.0, color='#555', lw=1.5, ls='--', zorder=4, label='Perfect prediction')
ax.axhspan(0.80, 1.20, alpha=0.10, color=COL_R, zorder=0, label='±20% band')

for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.015 if val >= 1 else val - 0.04,
            f'{val:.2f}', ha='center', va='bottom' if val >= 1 else 'top',
            fontsize=8.0, color='#333', fontweight='bold')

ax.set_xticks(x_all)
ax.set_xticklabels(species, rotation=30, ha='right', fontsize=8.5)
ax.set_ylabel(r'$t_{B,pred} \;/\; T_{max,obs}$', fontsize=10.5)
ax.set_ylim(0, max(ratios) * 1.25)
ax.legend(fontsize=9, loc='upper left')
ax.set_title(r'Per-species prediction ratio $t_{B,pred} / T_{max,obs}$' + '\n'
             f'Variant A (all 9)  |  MAPE = {MAPE_A:.1f}%  |  R²_log = {R2_A:.3f}',
             fontsize=10.5, fontweight='bold')

fig.suptitle(
    'VBLS Consistency — Sensitivity & Robustness (Whittemore et al., 2019)',
    fontsize=12, fontweight='bold', y=0.99
)

plt.savefig('telomere_fig4_sensitivity.pdf', bbox_inches='tight')
plt.savefig('telomere_fig4_sensitivity.png', bbox_inches='tight', dpi=180)
plt.close()
print("  Saved: telomere_fig4_sensitivity")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONSISTENCY SUMMARY")
print("=" * 60)
print(f"VARIANT A (primary, all 9 species)")
print(f"  R*_univ = {R_univ_A:.3f} ± {R_univ_A_std:.3f}")
print(f"  MAPE    = {MAPE_A:.1f}%")
print(f"  R²_log  = {R2_A:.3f}")
print(f"  Sensitivity ±10% → MAPE range: "
      f"{mape_grid[idx_m10]:.1f}% – {mape_grid[idx_p10]:.1f}%")
print(f"\nVARIANT B (supplementary, cluster 1 n=7)")
print(f"  R*_univ = {R_univ_B:.3f} ± {R_univ_B_std:.3f}")
print(f"  MAPE    = {MAPE_B:.1f}%")
print(f"  R²_log  = {R2_B:.3f}")
print(f"\nLog-log fit on 9 species: R² = {R2_w:.3f} (= VBLS R²_log); "
      f"Whittemore 2019 reported R² = {R2_w_reported:.3f}")
print(f"E-channel species: Flamingo + Elephant (telomere NOT limiting)")
print("=" * 60)
print("\nAll figures saved.")
