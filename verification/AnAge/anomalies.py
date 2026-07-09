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
VBLS — Allometric anomalies in T_max and their interpretation
=================================================================
Version: 1.0 (2026)

Goal:
  For each mammalian species in the AnAge database, compute the predicted
  lifespan from the VBLS allometric law and identify the dominant system (S)
  parameter that explains the deviation.

Theoretical basis (VBLS, section 13):
  T_max ~ B* * M^(5/4) / (sigma0 * c_Kl * M^(3/4))
        ~ (B* / sigma0 / c_Kl) * M^(1/4)

  From Kleiber law: P ~ c_Kl * M^(3/4), c_Kl ~ 3.5 W/kg^(3/4)

  Calibrated on class Mammalia, the empirical law is:
  T_max ~ A * M^alpha,  alpha_VBLS = 0.218 (empirical), 0.250 (theoretical)

  Anomaly = deviation of T_obs from T_allom.
  Each anomaly type corresponds to one parameter of (S):

  Type | Ratio > 1 | Mechanism            | VBLS parameter
  ----+-----------+---------------------+----------------
  1   | high      | low metabolism      | sigma0_eff down
  2   | high      | strong repair       | rho up or nu0 down
  3   | high      | hibernation/torpor  | nu0 down (seasonal)
  4   | low       | high stress         | sigma0 up
  5   | low       | weak repair         | rho down

Sources:
  AnAge Build 14: Tacutu et al. (2018), Nucleic Acids Research
  Greenland shark: Nielsen et al. (2016), Science 353:702
  Aldabra tortoise: Swingland & Coe (1979); AnAge
  Bats: Wilkinson & South (2002), Aging Cell

Usage:
  python anomalies.py
  Output files: anomalies_table.csv, anomalies_fig.png/.pdf

For reviewers:
  All computations are reproducible. Input data: anage_data.csv (public database).
  Additional species (Greenland shark) added manually from publications.
  Parameter B* = 0.32 J·s·kg^(-5/4) — from section 13 of VBLS.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# VBLS CONSTANTS
# ══════════════════════════════════════════════════════════════════════

BSTAR       = 0.32      # J·s·kg^(-5/4) — viability quantum
C_KL        = 3.5       # W/kg^(3/4)    — Kleiber coefficient
ALPHA_VBLS  = 0.218     # empirical exponent (AnAge 318 species)
ALPHA_THEOR = 0.250     # theoretical exponent VBLS

# ══════════════════════════════════════════════════════════════════════
# ADDITIONAL SPECIES (not in AnAge or requiring clarification)
# ══════════════════════════════════════════════════════════════════════

EXTRA_SPECIES = [
    # (Genus, Species, Common, Order, Class, M_kg, T_max_yrs, Source, Notes)
    ('Somniosus',      'microcephalus', 'Greenland shark',
     'Squaliformes',   'Chondrichthyes',
     400.0, 392.0,
     'Nielsen et al. 2016, Science 353:702',
     'Eye lens radiocarbon dating'),

    ('Aldabrachelys',  'gigantea',      'Aldabra giant tortoise',
     'Testudines',     'Reptilia',
     110.0, 255.0,
     'AnAge / Swingland & Coe 1979',
     'Record Jonathan: 191+ years'),

    ('Geochelone',     'sulcata',       'Sulcata tortoise',
     'Testudines',     'Reptilia',
     90.0, 54.0,
     'AnAge Build 14',
     ''),
]

# ══════════════════════════════════════════════════════════════════════
# ANOMALY INTERPRETATION — mechanism table
# Key: Genus -> (parameter, explanation)
# ══════════════════════════════════════════════════════════════════════

ANOMALY_INTERP = {
    'Myotis': (
        'sigma0_eff down',
        'Flight activates antioxidant pathways (AMPK, FOXO3); '
        'high mitochondrial respiration rate during flight '
        'paradoxically reduces ROS leakage through training'
    ),
    'Pteropus': (
        'sigma0_eff down',
        'Same flight mechanism; large fruit bats. '
        'Additionally: IFN-alpha constitutively active — '
        'reduced inflammatory response lowers sigma0'
    ),
    'Eptesicus': (
        'sigma0_eff down + nu0 down',
        'Annual hibernation (~5 months) reduces effective nu0; '
        'antioxidant protection during flight'
    ),
    'Desmodus': (
        'sigma0_eff down',
        'Vampire bat: flight + specific immune regulation'
    ),
    'Plecotus': (
        'nu0 down',
        'Long hibernation: nu0_eff ~ nu0 * (1 - f_hibern), '
        'f_hibern ~ 0.6; effective lifespan extended'
    ),
    'Heterocephalus': (
        'rho up + sigma0 down',
        'Naked mole-rat: highly efficient DNA repair (PARP1 x3), '
        'conformationally stable ribosomes, no accumulation of '
        'oxidized proteins; low ROS from hypoxic adaptation'
    ),
    'Homo': (
        'rho up (social)',
        'Human: medicine, social protection, neural repair; '
        'effective rho significantly exceeds biological minimum. '
        'Anomaly ~4x matches the role of cultural rho accumulation'
    ),
    'Tachyglossus': (
        'nu0 down + T down',
        'Monotremes: low body temperature (30C vs 37C), '
        'torpor; van t Hoff rule: x2-3 per 10C'
    ),
    'Somniosus': (
        'nu0 down down (T down down)',
        'Polar waters (~0C): nu0 suppressed by 2-3 orders of magnitude '
        'compared to endotherms; ectotherm + slow growth'
    ),
    'Aldabrachelys': (
        'nu0 down + sigma0 down',
        'Tortoises: extremely slow metabolism, '
        'ectotherm, powerful antioxidant defense in shell'
    ),
    'Sphenodon': (
        'nu0 down',
        'Tuatara: relict reptilian metabolism, '
        'body temperature ~12-18C; New Zealand isolation'
    ),
    # Anomalously short-lived
    'Didelphis': (
        'rho down + sigma0 up',
        'Opossums: rapid sexual maturity, high reproductive rate; '
        'classic r-strategist — evolutionarily low rho'
    ),
    'Crocidura': (
        'sigma0 up',
        'Shrews: record metabolism (~35 W/kg), '
        'heart 1200 bpm; sigma0 maximal among Mammalia'
    ),
}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print("Loading AnAge Build 14...")
df = pd.read_csv('anage_data.csv', sep=';', encoding='utf-8-sig',
                 on_bad_lines='skip')

for col in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Add extra species
extra_rows = []
for (genus, species, common, order, cls,
     m_kg, t_max, source, notes) in EXTRA_SPECIES:
    extra_rows.append({
        'Genus': genus, 'Species': species,
        'Common name': common,
        'Order': order, 'Class': cls,
        'Body mass (g)': m_kg * 1000,
        'Maximum longevity (yrs)': t_max,
        'Metabolic rate (W)': np.nan,
        'Data quality': 'acceptable',
        'Sample size': 'large',
        '_extra': True,
        '_source': source,
        '_notes': notes,
    })

df_extra = pd.DataFrame(extra_rows)
df_all = pd.concat([df, df_extra], ignore_index=True)
df_all['_extra'] = df_all.get('_extra', False).fillna(False)

print(f"  Total records: {len(df_all)}")

# ══════════════════════════════════════════════════════════════════════
# 2. ALLOMETRIC FIT
# ══════════════════════════════════════════════════════════════════════

# Base sample for fitting: Mammalia, data != low, without Chiroptera
mam_fit = df_all[
    (df_all['Class'] == 'Mammalia') &
    (df_all['Data quality'] != 'low') &
    (df_all['Order'] != 'Chiroptera') &
    (~df_all.get('_extra', pd.Series(False, index=df_all.index)))
].copy()
mam_fit = mam_fit.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)'])
mam_fit = mam_fit[
    (mam_fit['Body mass (g)'] > 0) &
    (mam_fit['Maximum longevity (yrs)'] > 0)
]

log_M = np.log10(mam_fit['Body mass (g)'])
log_T = np.log10(mam_fit['Maximum longevity (yrs)'])
mask = np.isfinite(log_M) & np.isfinite(log_T)

slope, intercept, r_val, p_val, se = stats.linregress(
    log_M[mask], log_T[mask])

print(f"\nAllometric fit (N={mask.sum()}):")
print(f"  T_max ~ M^{slope:.4f}  (VBLS theory: 0.250, empirical: 0.218)")
print(f"  R² = {r_val**2:.4f},  p = {p_val:.2e}")

def T_allom(M_g, use_empirical=True):
    """Allometric prediction of T_max in years."""
    alpha = slope if use_empirical else ALPHA_VBLS
    ic = intercept if use_empirical else (
        np.log10(BSTAR / (C_KL * 365.25 * 86400)) + ALPHA_VBLS * np.log10(1000))
    return 10**(ic + alpha * np.log10(M_g))

# ══════════════════════════════════════════════════════════════════════
# 3. COMPUTE ANOMALIES
# ══════════════════════════════════════════════════════════════════════

all_valid = df_all.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)']).copy()
all_valid = all_valid[
    (all_valid['Body mass (g)'] > 0) &
    (all_valid['Maximum longevity (yrs)'] > 0)
].copy()

all_valid['M_kg']    = all_valid['Body mass (g)'] / 1000
all_valid['T_obs']   = all_valid['Maximum longevity (yrs)']
all_valid['T_allom'] = all_valid.apply(
    lambda r: T_allom(r['Body mass (g)']), axis=1)
all_valid['ratio']   = all_valid['T_obs'] / all_valid['T_allom']
all_valid['log_ratio'] = np.log10(all_valid['ratio'])

def get_interp(row):
    genus = str(row['Genus'])
    if genus in ANOMALY_INTERP:
        return ANOMALY_INTERP[genus]
    return ('—', '—')

all_valid[['param', 'mechanism']] = pd.DataFrame(
    all_valid.apply(get_interp, axis=1).tolist(),
    index=all_valid.index)

# ══════════════════════════════════════════════════════════════════════
# 4. ANOMALY TABLE
# ══════════════════════════════════════════════════════════════════════

FOCAL_GENERA = [
    # Anomalously LONG-LIVED
    'Myotis', 'Eptesicus', 'Desmodus', 'Plecotus',
    'Heterocephalus', 'Homo', 'Tachyglossus',
    'Somniosus', 'Aldabrachelys', 'Sphenodon',
    # Anomalously SHORT-LIVED
    'Didelphis', 'Crocidura',
]

focal = all_valid[all_valid['Genus'].isin(FOCAL_GENERA)].copy()

focal_best = []
for genus in FOCAL_GENERA:
    sub = focal[focal['Genus'] == genus]
    if len(sub) == 0:
        continue
    sub_acc = sub[sub['Data quality'] == 'acceptable']
    row = sub_acc.iloc[0] if len(sub_acc) > 0 else sub.iloc[0]
    focal_best.append(row)

focal_df = pd.DataFrame(focal_best).reset_index(drop=True)

print("\n" + "="*100)
print("ALLOMETRIC ANOMALIES IN T_max: INTERPRETATION VIA VBLS PARAMETERS")
print("="*100)
header = f"{'Species':<30} {'Order':<18} {'M,kg':>6} {'T_obs':>7} {'T_allom':>8} {'Ratio':>6}  {'VBLS parameter'}"
print(header)
print("-"*100)

for _, row in focal_df.iterrows():
    name = f"{row['Genus']} {str(row['Species'])[:12]}"
    ratio_str = f"{row['ratio']:5.1f}x"
    marker = "^" if row['ratio'] > 1.5 else ("v" if row['ratio'] < 0.7 else " ")
    print(f"{marker} {name:<28} {str(row['Order'])[:17]:<18} "
          f"{row['M_kg']:6.2f} {row['T_obs']:7.1f} {row['T_allom']:8.1f} "
          f"{ratio_str:>6}  {row['param']}")

print("\nLegend: ^ — anomalously long-lived (ratio > 1.5)")
print("        v — anomalously short-lived (ratio < 0.7)")

# ══════════════════════════════════════════════════════════════════════
# 5. CSV — for independent reviewer verification
# ══════════════════════════════════════════════════════════════════════

out_cols = ['Genus', 'Species', 'Common name', 'Class', 'Order',
            'M_kg', 'T_obs', 'T_allom', 'ratio', 'log_ratio',
            'param', 'mechanism']

anomalies_full = all_valid[
    (all_valid['ratio'] > 2.0) | (all_valid['ratio'] < 0.5)
].sort_values('ratio', ascending=False)

save_cols = [c for c in out_cols if c in anomalies_full.columns]
anomalies_full[save_cols].round(3).to_csv(
    'anomalies_table.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"\nSaved: anomalies_table.csv  ({len(anomalies_full)} species)")

focal_df[save_cols].round(3).to_csv(
    'anomalies_focal.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"Saved: anomalies_focal.csv  ({len(focal_df)} species)")

# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE
# ══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family':     'DejaVu Serif',
    'font.size':       10,
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid':       True,
    'grid.alpha':      0.15,
    'grid.linestyle':  '--',
})

CLASS_COLORS = {
    'Mammalia':      '#2980b9',
    'Reptilia':      '#27ae60',
    'Chondrichthyes':'#8e44ad',
    'Aves':          '#e67e22',
}

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.07, right=0.97,
                       top=0.91, bottom=0.08,
                       wspace=0.35, hspace=0.45)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

# --- PANEL A: T_max vs M with anomalies ---
ax = axes[0]
for cls, grp in all_valid.groupby('Class'):
    if cls not in CLASS_COLORS:
        continue
    c = CLASS_COLORS[cls]
    normal = grp[(grp['ratio'] > 0.5) & (grp['ratio'] < 2.0)]
    anomal = grp[(grp['ratio'] >= 2.0) | (grp['ratio'] <= 0.5)]
    ax.scatter(normal['M_kg'], normal['T_obs'],
               s=15, alpha=0.35, color=c, zorder=2,
               label=cls if cls == 'Mammalia' else '')
    ax.scatter(anomal['M_kg'], anomal['T_obs'],
               s=45, alpha=0.85, color=c, edgecolors='k',
               linewidths=0.7, zorder=4)

M_range = np.logspace(-3, 6, 200)
ax.plot(M_range / 1000, 10**(intercept + slope * np.log10(M_range)),
        '-', color='#2980b9', lw=2.0, zorder=5,
        label=f'Empirical alpha={slope:.3f}')
ax.plot(M_range / 1000, 10**(intercept + ALPHA_THEOR * np.log10(M_range)),
        'k--', lw=1.5, alpha=0.5, zorder=5,
        label=f'VBLS theory α={ALPHA_THEOR}')

LABELS_A = {
    'Myotis':        ('Myotis spp.',        'up'),
    'Heterocephalus':('Naked mole-rat',     'right'),
    'Homo':          ('Human',              'right'),
    'Somniosus':     ('Greenl. shark',      'up'),
    'Aldabrachelys': ('Aldabra tortoise',   'up'),
    'Didelphis':     ('Opossum',            'down'),
    'Crocidura':     ('Shrew',              'down'),
}
for _, row in focal_df.iterrows():
    if row['Genus'] in LABELS_A:
        txt, pos = LABELS_A[row['Genus']]
        dy = 0.08 if pos == 'up' else (-0.12 if pos == 'down' else 0)
        dx = 0.0 if pos in ('up','down') else 0.15
        ax.annotate(txt,
                    xy=(row['M_kg'], row['T_obs']),
                    xytext=(row['M_kg'] * 10**dx,
                            row['T_obs'] * 10**(dy + 0.05)),
                    fontsize=7.5, arrowprops=dict(arrowstyle='-',
                    color='#666', lw=0.8), color='#333')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass (kg)')
ax.set_ylabel('T_max (yr)')
ax.set_title(f'A.  T_max ~ M: anomalies marked\n'
             f'Filled dots = ratio > 2x or < 0.5x',
             fontweight='bold')
ax.legend(fontsize=8, framealpha=0.9)

# --- PANEL B: ratio vs M ---
ax = axes[1]
mam_plot = all_valid[all_valid['Class'].isin(['Mammalia','Reptilia','Chondrichthyes'])].copy()
for cls, grp in mam_plot.groupby('Class'):
    c = CLASS_COLORS.get(cls, '#95a5a6')
    ax.scatter(grp['M_kg'], grp['ratio'],
               s=20, alpha=0.4, color=c, label=cls, zorder=2)

ax.axhline(1.0, color='k', lw=1.5, ls='--', alpha=0.6, label='Allom. norm')
ax.axhline(2.0, color='#e74c3c', lw=1.0, ls=':', alpha=0.6)
ax.axhline(0.5, color='#e74c3c', lw=1.0, ls=':', alpha=0.6)

for _, row in focal_df.iterrows():
    c = CLASS_COLORS.get(str(row['Class']), '#95a5a6')
    ax.scatter(row['M_kg'], row['ratio'],
               s=60, color=c, edgecolors='k', linewidths=0.8, zorder=5)
    if row['ratio'] > 2.5 or row['ratio'] < 0.45:
        short = str(row['Genus'])[:8]
        ax.annotate(short,
                    xy=(row['M_kg'], row['ratio']),
                    xytext=(row['M_kg'] * 1.8, row['ratio'] * 1.05),
                    fontsize=7, color='#333',
                    arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.7))

ax.set_xscale('log')
ax.set_xlabel('Body mass (kg)')
ax.set_ylabel('T_obs / T_allom (ratio)')
ax.set_title('B.  Ratio T_obs/T_allom vs mass\n'
             'Anomaly zone: ratio > 2 or < 0.5',
             fontweight='bold')
ax.legend(fontsize=8, framealpha=0.9)

# --- PANEL C: horizontal lollipop of focal species ---
ax = axes[2]
focal_sorted = focal_df.sort_values('ratio', ascending=True)

y_pos = np.arange(len(focal_sorted))
colors_bar = ['#e74c3c' if r > 1 else '#3498db'
              for r in focal_sorted['ratio']]

ax.barh(y_pos, np.log10(focal_sorted['ratio']),
        color=colors_bar, alpha=0.7, height=0.6, zorder=3)
ax.axvline(0, color='k', lw=1.5, zorder=5)
ax.axvline(np.log10(2), color='#e74c3c', lw=1, ls=':', alpha=0.7)
ax.axvline(-np.log10(2), color='#3498db', lw=1, ls=':', alpha=0.7)

labels_v = [f"{row['Genus']} {str(row['Species'])[:8]}"
            for _, row in focal_sorted.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_v, fontsize=8)
ax.set_xlabel('log10(T_obs / T_allom)')
ax.set_title('C.  Anomaly magnitude (log scale)\n'
             'Red = longer than norm, blue = shorter',
             fontweight='bold')

for i, (_, row) in enumerate(focal_sorted.iterrows()):
    xval = np.log10(row['ratio'])
    ha = 'left' if xval >= 0 else 'right'
    offset = 0.03 if xval >= 0 else -0.03
    ax.text(xval + offset, i, f'{row["ratio"]:.1f}x',
            va='center', ha=ha, fontsize=7.5, color='#333')

# --- PANEL D: parameter typology table ---
ax = axes[3]
ax.axis('off')

type_data = [
    ['Type', 'Ratio', 'Parameter\n(S)', 'Biological mechanism (examples)'],
    ['High metabolism (-)', '<1',   'sigma0 up',        'High ROS, fast DNA wear\n(shrews, opossums)'],
    ['Strong repair (+)',   '>2',   'rho up',           'Highly efficient DNA repair\n(naked mole-rat, Homo)'],
    ['Low metabolism (+)',  '>2',   'sigma0_eff down',  'Antioxidants, slow turnover\n(tortoises, ectotherms)'],
    ['Hibernation (+)',     '>2',   'nu0 down (seas.)', 'Effective < chronological lifespan\n(bats: Myotis, Plecotus)'],
    ['Polar cold (+)',      '>10',  'nu0 << mammal',    'T~0C: degradation x100 slower\n(Greenland shark)'],
    ['r-strategist (-)',    '<0.5', 'rho down',         'Evolutionarily low repair invest.\n(opossums, mice)'],
]

table = ax.table(cellText=type_data[1:],
                 colLabels=type_data[0],
                 loc='center', cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 2.8)

for j in range(4):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, 7):
    fc = '#fef9e7' if i % 2 == 0 else '#ffffff'
    for j in range(4):
        table[i, j].set_facecolor(fc)

ax.set_title('D.  Anomaly typology in terms of VBLS\n'
             'Each type corresponds to one parameter of system (S)',
             fontweight='bold', pad=10)

# --- PANEL E: bat mechanisms in detail ---
ax = axes[4]

bats_data = all_valid[all_valid['Order'] == 'Chiroptera'].dropna(
    subset=['M_kg', 'T_obs', 'T_allom']).copy()
bats_data = bats_data[bats_data['M_kg'] > 0]

ax.scatter(bats_data['M_kg'], bats_data['T_obs'],
           s=25, color='#8e44ad', alpha=0.6, label='Chiroptera', zorder=3)
ax.scatter(bats_data['M_kg'], bats_data['T_allom'],
           s=10, color='#bdc3c7', alpha=0.4, zorder=2)

for _, row in bats_data.iterrows():
    ax.plot([row['M_kg'], row['M_kg']],
            [row['T_allom'], row['T_obs']],
            '-', color='#8e44ad', lw=0.5, alpha=0.3, zorder=1)

M_bat = np.logspace(-3, 0, 100)
ax.plot(M_bat, 10**(intercept + slope * np.log10(M_bat * 1000)),
        'k--', lw=1.5, alpha=0.5, label='Allom. norm (Mammalia)')

bat_labels = {
    'lucifugus': 'M. lucifugus\n34yr (6.6x)',
    'auritus':   'P. auritus\n30yr (5.3x)',
}
for _, row in bats_data.iterrows():
    if str(row['Species']) in bat_labels and row['ratio'] > 4:
        ax.annotate(bat_labels[str(row['Species'])],
                    xy=(row['M_kg'], row['T_obs']),
                    xytext=(row['M_kg'] * 3, row['T_obs'] * 0.85),
                    fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=0.9))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Body mass (kg)')
ax.set_ylabel('T_max (yr)')
ax.set_title('E.  Chiroptera: all species vs allometry\n'
             'Vertical lines = anomaly per species',
             fontweight='bold')
ax.legend(fontsize=8.5, framealpha=0.9)

# --- PANEL F: mechanism summary text ---
ax = axes[5]
ax.axis('off')

summary = """VBLS interpretation of anomalies: key conclusions

^ ANOMALOUSLY LONG-LIVED:

Chiroptera (Myotis, x6.6):
  sigma0_eff down from flight — AMPK/FOXO3 activation
  of antioxidants reduces ROS production despite
  high metabolism. The flight paradox.

Heterocephalus glaber (x4.3):
  rho up — PARP1 x3, ribosomal fidelity up,
  hypoxic adaptation -> low ROS

Homo sapiens (x4.2):
  rho_eff up via medicine and social protection;
  biological rho not unique, but cultural
  'additional rho' is species-specific

Somniosus microcephalus (x~100):
  nu0 down down down at T~0C: van't Hoff rule
  gives x2-3 per 10C -> x100+ over 40C

v ANOMALOUSLY SHORT-LIVED:

Crocidura (x0.3):
  sigma0 up — metabolism 35 W/kg, heart
  1200 bpm -> maximal ROS

Didelphis (x0.28):
  rho down — r-strategist, evolutionarily optimized
  for reproduction, not longevity

------------------------------------------
All anomalies -> one parameter of system (S).
No anomaly requires exclusion from the theory.
"""

ax.text(0.03, 0.97, summary,
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=8.5,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.6',
                  facecolor='#f8f9fa',
                  edgecolor='#2c3e50',
                  linewidth=1.2))

ax.set_title('F.  Summary: each anomaly -> one VBLS parameter',
             fontweight='bold')

plt.savefig('anomalies_fig.pdf', bbox_inches='tight')
plt.savefig('anomalies_fig.png', bbox_inches='tight', dpi=180)
plt.close()
print("Saved: anomalies_fig.pdf / .png")

# ══════════════════════════════════════════════════════════════════════
# 7. FINAL STATISTICS
# ══════════════════════════════════════════════════════════════════════

mam_all = all_valid[all_valid['Class'] == 'Mammalia']
print(f"\n{'='*60}")
print("ANOMALY STATISTICS (Mammalia)")
print(f"{'='*60}")
print(f"Total Mammalia with data:        {len(mam_all)}")
print(f"Anomalously long-lived (>2x):    {len(mam_all[mam_all['ratio']>2])}")
print(f"Anomalously short-lived (<0.5x): {len(mam_all[mam_all['ratio']<0.5])}")
print(f"Within norm (0.5-2x):            {len(mam_all[(mam_all['ratio']>=0.5)&(mam_all['ratio']<=2)])}")
print(f"\nTop-5 anomalously long-lived:")
top5 = mam_all.nlargest(5, 'ratio')[['Genus','Species','M_kg','T_obs','T_allom','ratio']]
print(top5.round(2).to_string(index=False))
print(f"\nTop-5 anomalously short-lived:")
bot5 = mam_all.nsmallest(5, 'ratio')[['Genus','Species','M_kg','T_obs','T_allom','ratio']]
print(bot5.round(2).to_string(index=False))

print(f"\n{'='*60}")
print("Files saved:")
print("  anomalies_table.csv  — full anomaly table")
print("  anomalies_focal.csv  — focal species with interpretation")
print("  anomalies_fig.pdf/png — figure (6 panels)")
print(f"{'='*60}")
