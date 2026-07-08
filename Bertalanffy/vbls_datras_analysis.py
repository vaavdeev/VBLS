"""
VBLS vs VBGF — анализ реальных данных ICES DATRAS
Gadus morhua, NS-IBTS, 1989–2026, n=21794

Три теста:
  T1. VBGF vs VBLS: подгонка на реальных данных (средние + разброс)
  T2. Аллометрия W~L: получить показатель b → морфологический χ
  T3. Рост до/после зрелости: эффект зрелости на замедление роста
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('/home/claude/cod_datras_clean.csv')
print(f"Загружено: {len(df)} записей, {df['Year'].min()}–{df['Year'].max()}")

# ══════════════════════════════════════════════════════════════
# МОДЕЛИ
# ══════════════════════════════════════════════════════════════
def vbgf(t, Linf, K, t0):
    return Linf * (1 - np.exp(-K*(t-t0)))

def vbls_growth(t, Linf, K, t0, alpha):
    t = np.asarray(t, float)
    if abs(K-alpha) < 1e-10:
        return Linf*K*(t-t0)*np.exp(-K*(t-t0)+alpha*t0)
    c = Linf*K/(K-alpha)*np.exp(-alpha*t0)
    return np.clip(c*(np.exp(-alpha*(t-t0)) - np.exp(-K*(t-t0))), 0, None)

def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
def aic_f(n,r,k): return n*np.log(r**2) + 2*k

# ══════════════════════════════════════════════════════════════
# ТЕСТ 1: VBGF vs VBLS по средним длинам при возрасте
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ТЕСТ 1: VBGF vs VBLS — подгонка на реальных данных DATRAS")
print("="*60)

# Средние и SD по возрасту
stat = df.groupby('Age').agg(
    n=('L_cm','count'),
    L_mean=('L_cm','mean'),
    L_std=('L_cm','std')
).reset_index()
stat = stat[stat['n'] >= 10]  # минимум 10 рыб

age_arr = stat['Age'].values.astype(float)
L_arr   = stat['L_mean'].values
sigma   = stat['L_std'].values / np.sqrt(stat['n'].values)  # SE

# VBGF
popt_g, pcov_g = curve_fit(vbgf, age_arr, L_arr,
    p0=[120, 0.2, -0.5], bounds=([50,0.01,-5],[250,2,5]),
    sigma=sigma, absolute_sigma=True, maxfev=20000)
perr_g = np.sqrt(np.diag(pcov_g))
pred_g = vbgf(age_arr, *popt_g)

# VBLS
popt_v, pcov_v = curve_fit(vbls_growth, age_arr, L_arr,
    p0=[130, 0.18, -0.5, 0.015],
    bounds=([50,0.01,-5,1e-5],[300,2,5,0.5]),
    sigma=sigma, absolute_sigma=True, maxfev=50000)
perr_v = np.sqrt(np.diag(pcov_v))
pred_v = vbls_growth(age_arr, *popt_v)

r_g = rmse(L_arr, pred_g)
r_v = rmse(L_arr, pred_v)
n   = len(age_arr)

print(f"\nVBGF: L∞={popt_g[0]:.1f}±{perr_g[0]:.1f}  "
      f"K={popt_g[1]:.3f}±{perr_g[1]:.3f}  t₀={popt_g[2]:.2f}")
print(f"VBLS: L∞={popt_v[0]:.1f}±{perr_v[0]:.1f}  "
      f"K={popt_v[1]:.3f}±{perr_v[1]:.3f}  t₀={popt_v[2]:.2f}  "
      f"α={popt_v[3]:.5f}±{perr_v[3]:.5f}")
print(f"\nRMSE: VBGF={r_g:.3f} cm,  VBLS={r_v:.3f} cm")
print(f"ΔAIC(VBGF−VBLS) = {aic_f(n,r_g,3)-aic_f(n,r_v,4):+.2f}")

# ══════════════════════════════════════════════════════════════
# ТЕСТ 2: Аллометрия W ~ L^b (морфологический показатель)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ТЕСТ 2: Аллометрия W ~ L^b  (морфологический χ)")
print("="*60)

df_w = df[df['Weight'].notna() & df['L_cm'].notna() &
          (df['Weight'] > 5) & (df['L_cm'] > 5)].copy()

logL = np.log(df_w['L_cm'].values)
logW = np.log(df_w['Weight'].values)
slope, intercept, r_val, p_val, se = linregress(logL, logW)

print(f"\nW = exp({intercept:.3f}) · L^{slope:.4f}")
print(f"b (показатель) = {slope:.4f} ± {se:.4f}")
print(f"R² = {r_val**2:.4f},  n = {len(df_w)}")
print(f"\nИзометрический рост: b=3.000")
print(f"Наблюдаемый:         b={slope:.3f}  ({'гипераллометрия' if slope>3 else 'гипоаллометрия' if slope<3 else 'изометрия'})")
print(f"\nДля VBLS §11.5 (M* формула):")
print(f"  Морфологическое b={slope:.3f} → χ_metabolic ≈ 0.79 (Clarke & Johnston 1999)")
print(f"  M* ∝ ν₀^(-1/(1-χ)) = ν₀^(-{1/(1-0.79):.2f})")

# По возрастным группам (проверка онтогенетического сдвига b)
print(f"\nАллометрия b по возрастным группам:")
for age_g, label in [([1,2,3],'Ювенильные 1-3'), ([4,5,6],'Молодые 4-6'),
                      ([7,8,9,10,11,12],'Взрослые 7+')]:
    sub = df_w[df_w['Age'].isin(age_g)]
    if len(sub) < 10: continue
    sl, ic, rv, pv, se_s = linregress(np.log(sub['L_cm']), np.log(sub['Weight']))
    print(f"  {label:25s}: b={sl:.3f}±{se_s:.3f}  R²={rv**2:.3f}  n={len(sub)}")

# ══════════════════════════════════════════════════════════════
# ТЕСТ 3: Рост до/после зрелости
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ТЕСТ 3: Эффект зрелости на траекторию роста")
print("="*60)

# Коды зрелости DATRAS: 1=незрелые, 2-4=зрелеющие/зрелые
# Коды с буквами: I=immature, M=mature
# Числовые 61-66: шкала ICES (61=незрелые, 62-66=зрелые)
def is_immature(x):
    if pd.isna(x): return None
    x = str(x).strip()
    if x in ['1','61','I','A']: return True
    if x in ['2','3','4','62','63','64','65','66','M','Bb','Ba','Cb','Ca','Da','Db']: return False
    return None

df['Mature'] = df['IndividualMaturity'].apply(lambda x: not is_immature(x)
              if is_immature(x) is not None else None)

imm = df[df['Mature']==False].groupby('Age')['L_cm'].agg(['mean','std','count'])
mat = df[df['Mature']==True ].groupby('Age')['L_cm'].agg(['mean','std','count'])
imm.columns = ['L_imm','std_imm','n_imm']
mat.columns = ['L_mat','std_mat','n_mat']
both = imm.join(mat, how='inner')
both = both[both['n_imm']>=5]

print("\nВозраст  L_незрелые(±sd)    L_зрелые(±sd)      ΔL(зрел−незрел)")
for age, row in both.iterrows():
    dL = row['L_mat'] - row['L_imm']
    print(f"  {int(age):2d}    {row['L_imm']:5.1f}±{row['std_imm']:4.1f}cm   "
          f"{row['L_mat']:5.1f}±{row['std_mat']:4.1f}cm   {dL:+.1f}cm")

# VBLS-интерпретация: при одном возрасте зрелые особи КРУПНЕЕ
# (они достигли зрелости раньше → быстрее росли).
# Но темп роста у них НИЖЕ (больший η_abs ущерб от репродукции).
# Тест: у зрелых рыб одного возраста длина должна быть выше,
# но стандартное отклонение — тоже выше (из-за расхождения траекторий).
print("\nVBLS-предсказание: у зрелых рыб одного возраста —")
print("  > средняя длина (зрелость наступает у более крупных особей)")
print("  > разброс длин (η_abs варьирует → траектории расходятся)")
for age, row in both.iterrows():
    ratio_sd = row['std_mat']/row['std_imm'] if row['std_imm']>0 else np.nan
    print(f"  Возраст {int(age):2d}: SD_зрелые/SD_незрелые = {ratio_sd:.2f}")

# ══════════════════════════════════════════════════════════════
# РИСУНОК
# ══════════════════════════════════════════════════════════════
BG='#0d1117'; PANEL='#161b22'; GRID_C='#21262d'
TEXT='#e6edf3'; MUTED='#7d8590'
BLUE='#58a6ff'; TEAL='#3fb950'; GOLD='#d29922'; RED='#f85149'; WHITE='#ffffff'

plt.rcParams.update({
    'font.family':'monospace','axes.facecolor':PANEL,'figure.facecolor':BG,
    'axes.edgecolor':GRID_C,'axes.labelcolor':TEXT,'xtick.color':MUTED,
    'ytick.color':MUTED,'text.color':TEXT,
    'axes.spines.top':False,'axes.spines.right':False,
    'grid.color':GRID_C,'grid.linewidth':0.55,
})

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 3, figure=fig,
    hspace=0.42, wspace=0.32, left=0.07, right=0.97, top=0.89, bottom=0.08)

t_plot = np.linspace(0.3, 14, 400)
L_g_p = vbgf(t_plot, *popt_g)
L_v_p = vbls_growth(t_plot, *popt_v)

# ── A: кривые роста + данные ─────────────────────────────────
ax = fig.add_subplot(gs[0, :2])

# Violin-style: индивидуальные точки с джиттером
np.random.seed(42)
jitter = np.random.uniform(-0.18, 0.18, len(df))
ax.scatter(df['Age']+jitter, df['L_cm'], s=1.5, color=MUTED, alpha=0.12,
           zorder=2, rasterized=True)

# Средние ± SD
ax.errorbar(age_arr, L_arr, yerr=stat['L_std'].values,
            fmt='o', color=WHITE, ms=6, ecolor=MUTED,
            elinewidth=0.8, capsize=3, zorder=8, label='Средние ± SD (DATRAS)')

# Кривые
ax.plot(t_plot, L_g_p, color=BLUE, lw=2.2, ls='--',
        label=f"VBGF: L∞={popt_g[0]:.0f}cm, K={popt_g[1]:.3f}")
ax.plot(t_plot, L_v_p, color=TEAL, lw=2.8,
        label=f"VBLS: L∞={popt_v[0]:.0f}cm, α={popt_v[3]:.4f}")

# RMSE бокс
btext = (f"RMSE (cm)\nVBGF  {r_g:.3f}\nVBLS  {r_v:.3f}\n"
         f"ΔAIC  {aic_f(n,r_g,3)-aic_f(n,r_v,4):+.2f}")
ax.text(0.02, 0.97, btext, transform=ax.transAxes,
        fontsize=9, color=TEXT, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.45', fc=GRID_C, ec=MUTED, alpha=0.9))

ax.set_xlabel('Возраст (лет)', fontsize=11); ax.set_ylabel('Длина (cm)', fontsize=11)
ax.set_xlim(0, 13.5); ax.set_ylim(0, 150)
ax.grid(True, alpha=0.35)
ax.legend(fontsize=9, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED, loc='lower right')
ax.set_title('A — DATRAS реальные данные: VBGF vs VBLS (NS-IBTS, Gadus morhua, n=21794)',
             fontsize=11, color=TEXT, loc='left', pad=6)

# ── B: аллометрия W~L ────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
logL_plot = np.log(df_w['L_cm'])
logW_plot = np.log(df_w['Weight'])
ax2.scatter(logL_plot, logW_plot, s=1.5, color=TEAL, alpha=0.12,
            rasterized=True, zorder=2)
x_line = np.linspace(logL.min(), logL.max(), 200)
ax2.plot(x_line, intercept + slope*x_line, color=GOLD, lw=2.2,
         label=f'W = a·L^{slope:.3f}')
ax2.plot(x_line, intercept + 3.0*x_line, color=MUTED, lw=1.2, ls='--',
         alpha=0.6, label='Изометрия b=3.000')
ax2.set_xlabel('ln(L, cm)', fontsize=9); ax2.set_ylabel('ln(W, г)', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED)
ax2.set_title(f'B — Аллометрия W~L\nb={slope:.3f}±{se:.3f}, R²={r_val**2:.3f}',
              fontsize=10, color=TEXT, loc='left', pad=5)
# VBLS M* оценка
chi_m = 0.79  # Clarke & Johnston 1999
ax2.text(0.03, 0.10,
         f'χ_metabolic≈{chi_m}\n(Clarke & Johnston 1999)\nM*∝ν₀^(-{1/(1-chi_m):.1f})',
         transform=ax2.transAxes, fontsize=8, color=GOLD,
         bbox=dict(boxstyle='round,pad=0.35', fc=GRID_C, ec=MUTED, alpha=0.85))

# ── C: незрелые vs зрелые ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
ages_common = both.index.values

L_i = both['L_imm'].values; s_i = both['std_imm'].values
L_m = both['L_mat'].values; s_m = both['std_mat'].values

ax3.fill_between(ages_common, L_i-s_i, L_i+s_i, color=BLUE, alpha=0.15)
ax3.fill_between(ages_common, L_m-s_m, L_m+s_m, color=GOLD, alpha=0.15)
ax3.plot(ages_common, L_i, color=BLUE, lw=2.2, marker='o', ms=6,
         label='Незрелые (imm.)')
ax3.plot(ages_common, L_m, color=GOLD, lw=2.2, marker='D', ms=6,
         label='Зрелые (mat.)')

# dL/dt приближение: разница соседних возрастов
if len(ages_common) >= 3:
    dL_i = np.gradient(L_i, ages_common)
    dL_m = np.gradient(L_m, ages_common)
    ax3r = ax3.twinx()
    ax3r.plot(ages_common, dL_i, color=BLUE, lw=1.5, ls=':', alpha=0.7,
              label='dL/dt незрелые')
    ax3r.plot(ages_common, dL_m, color=GOLD, lw=1.5, ls=':', alpha=0.7,
              label='dL/dt зрелые')
    ax3r.set_ylabel('dL/dt (cm/yr)', fontsize=8, color=MUTED)
    ax3r.tick_params(axis='y', colors=MUTED, labelsize=7)
    ax3r.axhline(0, color=MUTED, lw=0.7)

ax3.set_xlabel('Возраст (лет)', fontsize=11)
ax3.set_ylabel('Длина (cm)', fontsize=11)
ax3.grid(True, alpha=0.35)
ax3.legend(fontsize=9, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED)
ax3.set_title('C — Траектории роста: незрелые vs зрелые особи одного возраста\n'
              'VBLS: зрелые крупнее (быстрее росли), но расходятся сильнее (η_abs вариабельнее)',
              fontsize=10, color=TEXT, loc='left', pad=5)

# ── D: разброс длин по возрасту (ключевой тест VBLS) ─────────
ax4 = fig.add_subplot(gs[1, 2])
cv = (stat['L_std'] / stat['L_mean'] * 100).values  # коэффициент вариации
ax4.bar(age_arr, cv, color=[TEAL if a<=6 else GOLD for a in age_arr],
        alpha=0.8, width=0.6)

# Тренд
z = np.polyfit(age_arr, cv, 1)
p = np.poly1d(z)
x_cv = np.linspace(age_arr.min(), age_arr.max(), 100)
ax4.plot(x_cv, p(x_cv), color=RED, lw=1.8, ls='--', alpha=0.85,
         label=f'Тренд: {z[0]:+.2f}%/yr')

ax4.set_xlabel('Возраст (лет)', fontsize=9)
ax4.set_ylabel('CV = σ/μ × 100%', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=8.5, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED)
ax4.set_title('D — Коэффициент вариации длины\nVBLS: CV↑ с возрастом (расхождение η_abs)',
              fontsize=10, color=TEXT, loc='left', pad=5)

ax4.text(0.05, 0.92,
         'VBGF: CV=const\n(единственная траектория)\nVBLS: CV↑\n(η_abs варьирует между особями)',
         transform=ax4.transAxes, fontsize=7.5, color=TEXT,
         bbox=dict(boxstyle='round,pad=0.35', fc=GRID_C, ec=MUTED, alpha=0.88))

fig.text(0.5, 0.955,
         'VBLS vs VBGF — реальные данные ICES DATRAS: Gadus morhua, NS-IBTS 1989–2026',
         ha='center', fontsize=13, color=WHITE, fontweight='bold')
fig.text(0.5, 0.932,
         f'n=21794 особей  |  возраст из отолитов  |  b={slope:.3f} (W~L)  |  '
         f'α={popt_v[3]:.5f} yr⁻¹ (VBLS)',
         ha='center', fontsize=9, color=MUTED)

plt.savefig('/mnt/user-data/outputs/vbls_datras_real.png',
            dpi=155, bbox_inches='tight', facecolor=BG)
print("\nРисунок сохранён.")
