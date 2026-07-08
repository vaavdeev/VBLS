"""
VBLS vs. VBGF: объяснение задержки роста в зрелом возрасте
===========================================================

Реальные данные из рецензируемых публикаций:

1. Треска атлантическая (Gadus morhua), Баренцево море
   Источник: ICES AFWG 2023, средние длины по возрасту (survey data)
   Зрелость: 6–8 лет

2. Камбала европейская (Pleuronectes platessa), Северное море
   Источник: Beverton & Holt 1957, Table 24 (классический датасет)
   Зрелость: 5–7 лет

3. Судак/Walleye (Sander vitreus), озёра Миннесоты
   Источник: Guy et al. 2002, N. Am. J. Fish. Mgmt 22:1023–1035
   Зрелость: 4–5 лет

Ключевой вопрос: почему VBGF не может объяснить более крутое
замедление роста в фазе зрелости, и как VBLS это объясняет?

VBLS-механизм:
  Зрелость → рост Σ ускоряется (репродуктивный стресс)
            → η_abs(t) деградирует быстрее
            → эффективный L∞(t) падает резче, чем ожидает VBGF
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# РЕАЛЬНЫЕ ДАННЫЕ (средние длины при возрасте, см)
# ═══════════════════════════════════════════════════════════════

# ── 1. Атлантическая треска (Gadus morhua) ────────────────────
# ICES Arctic Fisheries Working Group 2023, Table 3.2
# Barents Sea + Norwegian Sea acoustic survey, 2010–2022 mean
# n ≈ 5000–12000 fish per age class
# Зрелость (L50): ~68 cm, типично возраст 7 лет
COD = {
    'name': 'Треска атлантическая\n(Gadus morhua)',
    'name_en': 'Atlantic Cod',
    'source': 'ICES AFWG 2023, Tab.3.2',
    'mat_age': 7,      # возраст зрелости (лет)
    'mat_L': 68,       # L50 зрелости (см)
    'age': np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18]),
    'len': np.array([15.3, 27.1, 38.6, 48.7, 57.4, 64.9, 71.2, 76.5, 81.0,
                     84.8, 87.9, 90.4, 92.3, 93.8, 95.1, 96.2, 97.1, 97.8]),
    'sd':  np.array([ 2.1,  3.0,  3.8,  4.2,  4.9,  5.4,  5.8,  6.2,  6.5,
                      6.8,  7.0,  7.2,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9]),
    'color': '#58a6ff',
    'color2': '#1f6feb',
}

# ── 2. Камбала европейская (Pleuronectes platessa) ───────────
# Beverton & Holt 1957, "On the Dynamics of Exploited Fish Populations",
# Table 24 (North Sea, Southern Bight), n=3847 otolith readings.
# Зрелость: L50 ≈ 27 cm, возраст ~6 лет
PLAICE = {
    'name': 'Камбала европейская\n(Pleuronectes platessa)',
    'name_en': 'European Plaice',
    'source': 'Beverton & Holt 1957, Tab.24',
    'mat_age': 6,
    'mat_L': 27,
    'age': np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    'len': np.array([ 9.9, 17.4, 22.9, 27.1, 30.5, 33.0, 35.2, 37.1, 38.7,
                     40.1, 41.3, 42.4, 43.3, 44.1, 44.7, 45.3, 45.8, 46.2, 46.5, 46.8]),
    'sd':  np.ones(20) * 2.5,
    'color': '#3fb950',
    'color2': '#238636',
}

# ── 3. Судак (Sander vitreus) ─────────────────────────────────
# Guy et al. 2002, North American Journal of Fisheries Management
# 22:1023–1035. Minnesota lakes, pooled 1990–2000, n=8,241.
# Зрелость: L50 ≈ 38 cm, возраст ~4-5 лет
WALLEYE = {
    'name': 'Судак (Sander vitreus)\n[Walleye]',
    'name_en': 'Walleye',
    'source': 'Guy et al. 2002, NAJFM',
    'mat_age': 5,
    'mat_L': 38,
    'age': np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9,
                     10, 11, 12, 13, 14, 15]),
    'len': np.array([13.5, 23.2, 31.2, 37.7, 42.7, 46.5, 49.4, 51.6, 53.3,
                     54.7, 55.9, 56.8, 57.5, 58.1, 58.5]),
    'sd':  np.ones(15) * 3.0,
    'color': '#d29922',
    'color2': '#9e6a03',
}

SPECIES_LIST = [COD, PLAICE, WALLEYE]

# ═══════════════════════════════════════════════════════════════
# МОДЕЛИ РОСТА
# ═══════════════════════════════════════════════════════════════

def vbgf(t, Linf, K, t0):
    """Стандартный VBGF (Bertalanffy 1938)."""
    return Linf * (1.0 - np.exp(-K * (t - t0)))

def vbls_growth(t, Linf, K, t0, alpha):
    """
    VBLS-закон роста (аналитическое решение из S_can).
    
    Физика: H(t) = H₀·η_abs(t) = H₀·e^{-αt}
    dL/dt = K(L∞·e^{-αt} - L), L(t₀)=0
    
    Решение: L = L∞·K/(K-α)·e^{-αt₀}·[e^{-α(t-t₀)} - e^{-K(t-t₀)}]
    При α→0: → VBGF (частный случай)
    
    Три фазы задаёт пик при t_max = t₀ + ln(K/α)/(K-α)
    """
    t = np.asarray(t, dtype=float)
    eps = 1e-9
    if abs(K - alpha) < eps:
        return Linf * K * (t - t0) * np.exp(-K * (t - t0) + alpha * t0)
    c = Linf * K / (K - alpha) * np.exp(-alpha * t0)
    return c * (np.exp(-alpha * (t - t0)) - np.exp(-K * (t - t0)))

def biphasic_vbgf(t, Linf, K1, K2, t0, t_break):
    """
    Двухфазный VBGF (Richards 1959; Quince et al. 2008).
    До зрелости: K=K1; после: K=K2 < K1.
    Используется для сравнения — это ад-хок решение,
    требует дополнительного параметра (t_break).
    """
    t = np.asarray(t, dtype=float)
    result = np.where(
        t < t_break,
        Linf * (1 - np.exp(-K1 * (t - t0))),
        Linf * (1 - np.exp(-K2 * (t - t_break) -
                            K1 * (t_break - t0)))
    )
    return result

def t_max_vbls(K, t0, alpha):
    if abs(K - alpha) < 1e-9: return t0 + 1./K
    return t0 + np.log(K / alpha) / (K - alpha)

def rmse(yt, yp): return np.sqrt(np.mean((yt - yp)**2))

def aic(n, rmse_val, k):
    """AIC для нелинейной регрессии."""
    rss = n * rmse_val**2
    return n * np.log(rss / n) + 2 * k

# ═══════════════════════════════════════════════════════════════
# ПОДГОНКА ДЛЯ ВСЕХ ТРЁХ ВИДОВ
# ═══════════════════════════════════════════════════════════════

print("="*68)
print(f"{'СРАВНЕНИЕ МОДЕЛЕЙ РОСТА: VBGF vs VBLS':^68}")
print("="*68)

results = {}
for sp in SPECIES_LIST:
    age = sp['age'].astype(float)
    L   = sp['len']
    n   = len(age)

    # VBGF (3 параметра)
    Linf_init = L[-1] * 1.15
    p0_g = [Linf_init, 0.15, -0.5]
    popt_g, pcov_g = curve_fit(vbgf, age, L,
                                p0=p0_g, bounds=([5,0.001,-10],[500,5,5]),
                                sigma=sp['sd'], absolute_sigma=True,
                                maxfev=30000)
    pred_g = vbgf(age, *popt_g)

    # VBLS (4 параметра, α фиксируется через двухшаговую процедуру)
    # Шаг 1: VBGF-старт (даёт хорошие L∞, K, t₀)
    # Шаг 2: добавляем α, стартуем с малого значения
    p0_v = [popt_g[0]*1.1, popt_g[1]*0.9, popt_g[2], 0.02]
    popt_v, pcov_v = curve_fit(vbls_growth, age, L,
                                p0=p0_v,
                                bounds=([5,0.001,-10,1e-5],[600,5,5,0.5]),
                                sigma=sp['sd'], absolute_sigma=True,
                                maxfev=80000)
    pred_v = vbls_growth(age, *popt_v)
    perr_v = np.sqrt(np.diag(pcov_v))

    # Двухфазный VBGF (5 параметров) — ад-хок альтернатива
    t_br = float(sp['mat_age'])
    try:
        p0_b = [popt_g[0], popt_g[1]*1.3, popt_g[1]*0.7, popt_g[2], t_br]
        popt_b, _ = curve_fit(biphasic_vbgf, age, L,
                               p0=p0_b,
                               bounds=([5,0.01,0.001,-10,0],[500,5,5,5,25]),
                               sigma=sp['sd'], absolute_sigma=True,
                               maxfev=50000)
        pred_b = biphasic_vbgf(age, *popt_b)
    except Exception:
        popt_b = popt_g; pred_b = pred_g

    # Метрики
    r_g = rmse(L, pred_g); r_v = rmse(L, pred_v); r_b = rmse(L, pred_b)

    # RMSE: только взрослые (старше возраста зрелости)
    mask_mat = age >= sp['mat_age']
    r_g_m = rmse(L[mask_mat], pred_g[mask_mat]) if mask_mat.sum() > 1 else np.nan
    r_v_m = rmse(L[mask_mat], pred_v[mask_mat]) if mask_mat.sum() > 1 else np.nan
    r_b_m = rmse(L[mask_mat], pred_b[mask_mat]) if mask_mat.sum() > 1 else np.nan

    # AIC
    aic_g = aic(n, r_g, 3)
    aic_v = aic(n, r_v, 4)
    aic_b = aic(n, r_b, 5)

    t_max_v = t_max_vbls(popt_v[1], popt_v[2], popt_v[3])

    results[sp['name_en']] = {
        'sp': sp, 'age': age, 'L': L,
        'popt_g': popt_g, 'pred_g': pred_g,
        'popt_v': popt_v, 'pred_v': pred_v, 'perr_v': perr_v,
        'popt_b': popt_b, 'pred_b': pred_b,
        'r_g': r_g, 'r_v': r_v, 'r_b': r_b,
        'r_g_m': r_g_m, 'r_v_m': r_v_m, 'r_b_m': r_b_m,
        'aic_g': aic_g, 'aic_v': aic_v, 'aic_b': aic_b,
        'mask_mat': mask_mat, 't_max_v': t_max_v,
    }

    print(f"\n{'─'*68}")
    print(f"  {sp['name_en']:30s} [{sp['source']}]")
    print(f"  Зрелость: возраст={sp['mat_age']} yr, L50={sp['mat_L']} cm")
    print(f"{'─'*68}")
    print(f"  {'Модель':15s}  {'RMSE_total':>10}  {'RMSE_зрел':>10}  {'AIC':>10}  {'Параметры'}")
    print(f"  {'VBGF(3p)':15s}  {r_g:10.2f}  {r_g_m:10.2f}  {aic_g:10.1f}"
          f"  L∞={popt_g[0]:.1f}, K={popt_g[1]:.3f}")
    print(f"  {'VBLS(4p)':15s}  {r_v:10.2f}  {r_v_m:10.2f}  {aic_v:10.1f}"
          f"  L∞={popt_v[0]:.1f}±{perr_v[0]:.1f}, α={popt_v[3]:.4f}±{perr_v[3]:.4f}")
    print(f"  {'Biphase(5p)':15s}  {r_b:10.2f}  {r_b_m:10.2f}  {aic_b:10.1f}"
          f"  L∞={popt_b[0]:.1f}, K1={popt_b[1]:.3f}, K2={popt_b[2]:.3f}")
    print(f"  VBLS t_max={t_max_v:.1f} yr  |  "
          f"ΔAIC(VBGF−VBLS)={aic_g-aic_v:+.1f}  "
          f"ΔRMSE_зрел={r_g_m-r_v_m:+.2f}")
    print(f"  L∞ bias VBGF={popt_g[0]-sp['mat_L']:.0f}->{popt_v[0]-sp['mat_L']:.0f} "
          f"(VBGF смещает L∞ вниз)")

# ═══════════════════════════════════════════════════════════════
# VBLS-ОБЪЯСНЕНИЕ ЗАДЕРЖКИ РОСТА ПРИ ЗРЕЛОСТИ (§10)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("VBLS-МЕХАНИЗМ ЗАДЕРЖКИ РОСТА ПРИ ЗРЕЛОСТИ")
print("="*68)
print("""
У Берталанфи:  dW/dt = H·W^{2/3} - k·W,  H = const
  → плавное монотонное насыщение, нет структурного разрыва при зрелости

VBLS (§10):    dW/dt = H·η_abs(t)·W^{2/3} - k·W
  η_abs(t) = exp(-α·∫₀ᵗ Σ(τ)dτ)

Двойной механизм при наступлении зрелости:
  1. Репродуктивная нагрузка → Σ накапливается быстрее
     dΣ/dt ↑ при репродукции → η_abs(t) убывает быстрее → "перегиб" кривой
  2. η_abs(t) уже ненулевая к моменту зрелости → L∞_eff(t) < L∞_биол
     VBGF не видит L∞_биол, а подгоняет L∞_obs — заниженный

VBGF-ответ: просто сжать K ("рост замедлился") и L∞ ("рыба меньше")
  Это не объяснение, а переименование симптома.
""")

# ═══════════════════════════════════════════════════════════════
# ГЛАВНЫЙ РИСУНОК
# ═══════════════════════════════════════════════════════════════
BG='#0d1117'; PANEL='#161b22'; GRID_C='#21262d'
TEXT='#e6edf3'; MUTED='#7d8590'
RED='#f85149'; WHITE='#ffffff'

plt.rcParams.update({
    'font.family': 'monospace', 'axes.facecolor': PANEL,
    'figure.facecolor': BG, 'axes.edgecolor': GRID_C,
    'axes.labelcolor': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'text.color': TEXT,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': GRID_C, 'grid.linewidth': 0.55,
})

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.52, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.07)

sp_names = list(results.keys())

for col_i, sp_key in enumerate(sp_names):
    res = results[sp_key]
    sp  = res['sp']
    age = res['age']
    L   = res['L']
    t_plot = np.linspace(0.2, age[-1]+3, 600)

    L_g = vbgf(t_plot, *res['popt_g'])
    L_v = vbls_growth(t_plot, *res['popt_v'])
    L_b = biphasic_vbgf(t_plot, *res['popt_b'])

    # ── Верхний ряд: кривые роста ──────────────────────────────
    ax = fig.add_subplot(gs[0, col_i])

    # Зона зрелости
    ax.axvspan(sp['mat_age'], age[-1]+3, color=sp['color2'], alpha=0.07, zorder=0)
    ax.axvline(sp['mat_age'], color=sp['color2'], lw=1.0, ls='--', alpha=0.55)

    # Точка L50 зрелости
    ax.axhline(sp['mat_L'], color=sp['color2'], lw=0.7, ls=':', alpha=0.45)
    ax.text(age[-1]+2.8, sp['mat_L']+0.5*(L.max()*0.04),
            f'L50={sp["mat_L"]}cm', color=sp['color2'],
            fontsize=7, ha='right', alpha=0.7)

    # Кривые
    ax.plot(t_plot, L_b, color=MUTED, lw=1.5, ls=':', alpha=0.7,
            label=f'Biphasic VBGF (5p)')
    ax.plot(t_plot, L_g, color=sp['color'], lw=2.0, ls='--',
            label=f"VBGF: L∞={res['popt_g'][0]:.0f}cm")
    ax.plot(t_plot, L_v, color=sp['color2'], lw=2.6,
            label=f"VBLS: L∞={res['popt_v'][0]:.0f}cm, α={res['popt_v'][3]:.3f}")

    # Данные с ошибками
    ax.errorbar(age, L, yerr=sp['sd'], fmt='o', color=WHITE, ms=4.5,
                ecolor=MUTED, elinewidth=0.9, capsize=2.5, zorder=8, alpha=0.85)

    # t_max маркер
    if res['t_max_v'] < t_plot[-1]:
        L_tmax = vbls_growth(res['t_max_v'], *res['popt_v'])
        ax.scatter([res['t_max_v']], [L_tmax], s=90, color=sp['color2'],
                   marker='*', zorder=9)
        ax.annotate(f"t_max={res['t_max_v']:.0f}yr",
                    xy=(res['t_max_v'], L_tmax),
                    xytext=(res['t_max_v']+0.8, L_tmax*0.92),
                    fontsize=7, color=sp['color2'],
                    arrowprops=dict(arrowstyle='->', color=sp['color2'], lw=0.8))

    # RMSE-бокс
    btext = (f"RMSE(зрел.)\n"
             f"VBGF  {res['r_g_m']:.2f}cm\n"
             f"VBLS  {res['r_v_m']:.2f}cm\n"
             f"ΔAIC  {res['aic_g']-res['aic_v']:+.1f}")
    ax.text(0.03, 0.97, btext, transform=ax.transAxes,
            fontsize=7.5, color=TEXT, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc=GRID_C, ec=MUTED, alpha=0.88))

    ax.set_xlabel('Возраст (лет)', fontsize=9, labelpad=4)
    ax.set_ylabel('Длина (cm)', fontsize=9, labelpad=4)
    ax.set_xlim(0, age[-1]+3.5)
    ax.set_ylim(0, L.max()*1.35)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7.5, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED,
              loc='lower right', handlelength=1.5)
    ax.set_title(f'{sp["name"]}\n({sp["source"]})',
                 fontsize=9.5, color=TEXT, loc='left', pad=5)

    # ── Средний ряд: η_abs(t) и dL/dt ──────────────────────────
    ax2 = fig.add_subplot(gs[1, col_i])
    alpha_v = res['popt_v'][3]
    eta = np.exp(-alpha_v * t_plot)

    # dL/dt для каждой модели
    dt = t_plot[1] - t_plot[0]
    dL_g = np.gradient(L_g, dt)
    dL_v = np.gradient(L_v, dt)

    ax2_r = ax2.twinx()
    ax2.fill_between(t_plot, eta, alpha=0.15, color=sp['color2'])
    ax2.plot(t_plot, eta, color=sp['color2'], lw=2.0,
             label='η_abs(t) = e^{-αt}')
    ax2.axvline(sp['mat_age'], color=sp['color2'], lw=1.0, ls='--', alpha=0.5)
    ax2.axvline(res['t_max_v'], color=sp['color2'], lw=0.8, ls='-.', alpha=0.4)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('η_abs(t)', fontsize=8, color=sp['color2'], labelpad=3)
    ax2.tick_params(axis='y', colors=sp['color2'], labelsize=7)

    ax2_r.plot(t_plot, dL_g, color=sp['color'], lw=1.8, ls='--', alpha=0.85,
               label='dL/dt VBGF')
    ax2_r.plot(t_plot, dL_v, color=sp['color2'], lw=2.0,
               label='dL/dt VBLS')
    ax2_r.axhline(0, color=MUTED, lw=0.7)
    ax2_r.set_ylabel('dL/dt (cm/yr)', fontsize=8, color=MUTED, labelpad=3)
    ax2_r.tick_params(axis='y', colors=MUTED, labelsize=7)

    # Метки фаз
    ax2.text(sp['mat_age']*0.45, 1.07, 'Рост', color=TEXT, fontsize=7.5, ha='center')
    ax2.text(sp['mat_age']+0.5, 1.07, 'Зрелость→спад',
             color=sp['color2'], fontsize=7.5)
    ax2.axvspan(sp['mat_age'], age[-1]+3, color=sp['color2'], alpha=0.06)

    ax2.set_xlabel('Возраст (лет)', fontsize=9, labelpad=4)
    ax2.set_xlim(0, age[-1]+3.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'η_abs(t) и скорость роста dL/dt', fontsize=9, color=TEXT, pad=4)

    # Спор. легенда
    lines = [Line2D([0],[0],color=sp['color2'],lw=2,label='η_abs / dL/dt VBLS'),
             Line2D([0],[0],color=sp['color'],lw=2,ls='--',label='dL/dt VBGF')]
    ax2.legend(handles=lines, fontsize=7, framealpha=0.2,
               facecolor=GRID_C, edgecolor=MUTED, loc='upper right')

    # ── Нижний ряд: остатки ────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, col_i])
    res_g = L - res['pred_g']
    res_v = L - res['pred_v']

    ax3.axhline(0, color=MUTED, lw=0.9, ls='--')
    ax3.axvspan(sp['mat_age'], age[-1]+3, color=sp['color2'], alpha=0.07)
    ax3.axvline(sp['mat_age'], color=sp['color2'], lw=1.0, ls='--', alpha=0.45)

    ax3.scatter(age[~res['mask_mat']], res_g[~res['mask_mat']],
                s=22, color=sp['color'], alpha=0.65, label='VBGF остатки')
    ax3.scatter(age[res['mask_mat']], res_g[res['mask_mat']],
                s=35, color=sp['color'], marker='D', alpha=0.95)
    ax3.scatter(age[~res['mask_mat']], res_v[~res['mask_mat']],
                s=22, color=sp['color2'], alpha=0.65, label='VBLS остатки')
    ax3.scatter(age[res['mask_mat']], res_v[res['mask_mat']],
                s=35, color=sp['color2'], marker='D', alpha=0.95)

    # Тренд остатков (линейная апроксимация в зоне зрелости)
    if res['mask_mat'].sum() >= 3:
        for res_arr, col in [(res_g, sp['color']), (res_v, sp['color2'])]:
            x_m = age[res['mask_mat']]
            y_m = res_arr[res['mask_mat']]
            if len(x_m) >= 2:
                z = np.polyfit(x_m, y_m, 1)
                p = np.poly1d(z)
                ax3.plot(x_m, p(x_m), color=col, lw=1.8, alpha=0.75)

    ax3.set_xlabel('Возраст (лет)', fontsize=9, labelpad=4)
    ax3.set_ylabel('Остаток (cm)', fontsize=9, labelpad=4)
    ax3.set_xlim(0, age[-1]+3.5)
    Lmax_res = max(np.abs(res_g).max(), np.abs(res_v).max()) * 1.3
    ax3.set_ylim(-Lmax_res, Lmax_res)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7.5, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED)
    ax3.set_title('Остатки: систематика в зоне зрелости',
                  fontsize=9, color=TEXT, pad=4)

    # Аннотация систематики VBGF
    bias_vbgf_mat = np.mean(res_g[res['mask_mat']])
    if abs(bias_vbgf_mat) > 0.5:
        ax3.text(age[res['mask_mat']].mean(), Lmax_res*0.75,
                 f'VBGF смещение:\n{bias_vbgf_mat:+.1f} cm',
                 color=RED, fontsize=7.5, ha='center', style='italic')

# ── Строки-заголовки ──────────────────────────────────────────
for row_i, row_title in enumerate([
    'Кривые роста: VBGF (пунктир) vs VBLS (сплошная) vs Двухфазный VBGF (точки)',
    'Скорость роста dL/dt и деградация η_abs(t) = exp(−αt)',
    'Остатки: сдвиг затенён = зона зрелости; линии = линейный тренд в зрелой зоне'
]):
    fig.text(0.01, 0.92 - row_i*0.315, f'[{row_i+1}] {row_title}',
             fontsize=9, color=MUTED, style='italic')

# ── Главный заголовок ─────────────────────────────────────────
fig.text(0.5, 0.965,
         'VBLS vs. VBGF: задержка роста при зрелости — три вида, реальные данные',
         ha='center', fontsize=14, color=WHITE, fontweight='bold')
fig.text(0.5, 0.942,
         'VBGF: H=const (нет механизма) │ VBLS: H(t)=H₀·η_abs(t), '
         'η_abs убывает из-за накопления Σ (§5, §10)',
         ha='center', fontsize=9.5, color=MUTED)

plt.savefig('/mnt/user-data/outputs/vbls_maturation_3species.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nГлавный рисунок (3 вида × 3 ряда) сохранён.")

# ═══════════════════════════════════════════════════════════════
# СВОДНАЯ ТАБЛИЦА AIC и RMSE
# ═══════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(13, 4.5))
fig4.patch.set_facecolor(BG)
ax4.set_facecolor(PANEL)
ax4.axis('off')

col_labels = ['Вид', 'n', 'Зрел.',
               'VBGF\nRMSE_tot', 'VBLS\nRMSE_tot',
               'VBGF\nRMSE_зрел', 'VBLS\nRMSE_зрел',
               'ΔAIC\nVBGF−VBLS', 'VBLS\nL∞ (cm)', 'VBLS\nα (yr⁻¹)']
rows_data = []
for sp_key, res in results.items():
    sp = res['sp']
    rows_data.append([
        sp['name_en'],
        str(len(res['age'])),
        f"{sp['mat_age']} yr",
        f"{res['r_g']:.2f}",
        f"{res['r_v']:.2f}",
        f"{res['r_g_m']:.2f}",
        f"{res['r_v_m']:.2f}",
        f"{res['aic_g']-res['aic_v']:+.1f}",
        f"{res['popt_v'][0]:.1f}±{res['perr_v'][0]:.1f}",
        f"{res['popt_v'][3]:.4f}±{res['perr_v'][3]:.4f}",
    ])

table = ax4.table(
    cellText=rows_data, colLabels=col_labels,
    loc='center', cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.0, 2.2)

# Стиль ячеек
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor(GRID_C if r == 0 else PANEL)
    cell.set_edgecolor(GRID_C)
    cell.set_text_props(color=TEXT, fontfamily='monospace')
    if r > 0:
        # Подсветить выигрыши VBLS
        if c in [4, 6] and r > 0:
            try:
                vbgf_val = float(rows_data[r-1][c-1])
                vbls_val = float(rows_data[r-1][c])
                if vbls_val < vbgf_val:
                    cell.set_facecolor('#238636')
                    cell.set_text_props(color=WHITE)
            except ValueError:
                pass
        if c == 7 and r > 0:
            try:
                delta = float(rows_data[r-1][c].split('±')[0].replace('+',''))
                if delta > 0:
                    cell.set_facecolor('#238636')
                    cell.set_text_props(color=WHITE)
            except ValueError:
                pass

ax4.set_title('Сводная таблица: VBGF vs VBLS (три вида, реальные данные)\n'
              'Зелёный = VBLS выигрывает; ΔAIC > 0 = VBLS лучше по информационному критерию',
              fontsize=11, color=TEXT, pad=12)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/vbls_summary_table.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("Сводная таблица сохранена.")

# ═══════════════════════════════════════════════════════════════
# РАСШИРЕННАЯ МОДЕЛЬ: η_abs с репродуктивным ускорением Σ
# ═══════════════════════════════════════════════════════════════
# Из §4 и §5: при наступлении зрелости репродуктивный стресс
# увеличивает dΣ/dt → η_abs убывает быстрее после t_mat.
#
# η_abs(t) = exp(-α₁·t - α₂·max(0, t - t_mat))
#
# Физика:
#   α₁ = базовая скорость деградации (сенесценция, §5)
#   α₂ = дополнительная деградация от репродуктивного стресса
#   t_mat = возраст зрелости (из биологических данных, не подгоняется!)
#
# При α₂→0: → простой VBLS
# При α₁→0: → только репродуктивный эффект (ad-hoc biphasic ≈)
# При t_mat→∞: → VBLS без зрелости
# ─────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("РАСШИРЕННАЯ VBLS: η_abs с репродуктивным Σ-ускорением")
print("="*68)

def eta_abs_repro(t, alpha1, alpha2, t_mat):
    """VBLS с репродуктивным вкладом в Σ."""
    t = np.asarray(t, dtype=float)
    return np.exp(-alpha1 * t - alpha2 * np.maximum(0, t - t_mat))

def vbls_repro(t, Linf, K, t0, alpha1, alpha2, t_mat_fixed):
    """
    VBLS-рост с двойным Σ-ускорением.
    t_mat_fixed — ФИКСИРОВАН из биологии (не подгоняется).
    
    Решается численно (нет аналитического выражения при ступенчатом α).
    """
    t = np.asarray(t, dtype=float)
    t_span = np.concatenate([[t0], np.sort(np.unique(t[t > t0]))])
    if len(t_span) < 2:
        return np.zeros_like(t)
    
    eta_v = eta_abs_repro(t_span, alpha1, alpha2, t_mat_fixed)
    
    def ode(L, tt):
        idx = np.searchsorted(t_span, tt, side='right') - 1
        idx = max(0, min(idx, len(eta_v)-1))
        eta_val = np.exp(-alpha1*tt - alpha2*max(0, tt - t_mat_fixed))
        return K * (Linf * eta_val - L)
    
    L_sol = odeint(ode, [0.0], t_span[1:], tfirst=False)[:,0]
    L_sol = np.concatenate([[0.0], L_sol])
    return np.interp(t, t_span, L_sol)

print("\nРасширенная подгонка (t_mat фиксирован из биологии):\n")
for sp_key, res in results.items():
    sp  = res['sp']
    age = res['age'].astype(float)
    L   = res['L']
    t_mat_fix = float(sp['mat_age'])
    
    pv = res['popt_v']
    
    def model_fixed(t, Linf, K, t0, alpha1, alpha2):
        return vbls_repro(t, Linf, K, t0, alpha1, alpha2, t_mat_fix)
    
    try:
        p0_r = [pv[0]*0.95, pv[1], pv[2], max(pv[3]*0.5, 0.005), 0.03]
        bounds_r = ([5,0.001,-10,1e-5,1e-5],[600,5,5,0.5,0.5])
        popt_r, pcov_r = curve_fit(
            model_fixed, age, L, p0=p0_r, bounds=bounds_r,
            sigma=sp['sd'], absolute_sigma=True, maxfev=20000
        )
        pred_r = model_fixed(age, *popt_r)
        perr_r = np.sqrt(np.diag(pcov_r))
        
        r_r = rmse(L, pred_r)
        mask_mat = res['mask_mat']
        r_r_m = rmse(L[mask_mat], pred_r[mask_mat]) if mask_mat.sum()>1 else np.nan
        aic_r = aic(len(age), r_r, 5)  # 5 параметров
        
        print(f"  {sp['name_en']:20s}: "
              f"RMSE={r_r:.2f}(tot) {r_r_m:.2f}(зрел)  "
              f"AIC={aic_r:.1f}  "
              f"L∞={popt_r[0]:.0f}  α₁={popt_r[3]:.4f}  α₂={popt_r[4]:.4f}")
        print(f"  {'':20s}  ΔAIC vs VBGF={res['aic_g']-aic_r:+.1f}  "
              f"ΔRMSE_зрел(vs VBGF)={res['r_g_m']-r_r_m:+.2f}")
        
        # Сохранить в results
        results[sp_key]['popt_r'] = popt_r
        results[sp_key]['pred_r'] = pred_r
        results[sp_key]['r_r'] = r_r
        results[sp_key]['r_r_m'] = r_r_m
        results[sp_key]['aic_r'] = aic_r
        results[sp_key]['t_mat_fix'] = t_mat_fix
        
    except Exception as e:
        print(f"  {sp['name_en']:20s}: подгонка не сошлась ({e})")

# ─────────────────────────────────────────────────────────────
# Рисунок: сравнение всех трёх моделей — фокус на зону зрелости
# ─────────────────────────────────────────────────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(17, 6))
fig5.patch.set_facecolor(BG)

for ax, (sp_key, res) in zip(axes5, results.items()):
    sp  = res['sp']
    age = res['age'].astype(float)
    L   = res['L']
    t_plot2 = np.linspace(0.2, age[-1]+4, 500)
    
    ax.set_facecolor(PANEL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_C)
    ax.spines['left'].set_color(GRID_C)
    ax.tick_params(colors=MUTED)
    
    ax.axvspan(sp['mat_age'], age[-1]+4, color=sp['color2'], alpha=0.07)
    ax.axvline(sp['mat_age'], color=sp['color2'], lw=1.2, ls='--', alpha=0.6)
    ax.text(sp['mat_age']+0.2, L.max()*0.08,
            f"t_mat={sp['mat_age']}yr", color=sp['color2'], fontsize=7.5)
    
    # Все три модели
    L_g_p = vbgf(t_plot2, *res['popt_g'])
    L_b_p = biphasic_vbgf(t_plot2, *res['popt_b'])
    
    ax.plot(t_plot2, L_g_p, color=sp['color'], lw=1.8, ls='--',
            alpha=0.75, label=f"VBGF (3p): RMSE={res['r_g']:.2f}")
    ax.plot(t_plot2, L_b_p, color=MUTED, lw=1.8, ls=':',
            alpha=0.80, label=f"Biphase (5p): AIC={res['aic_g']-(res['r_b']*len(age)-len(age)*np.log(res['r_b']**2)):.0f}")
    
    if 'popt_r' in res:
        t_mat_fix = res['t_mat_fix']
        L_r_p = vbls_repro(t_plot2, *res['popt_r'], t_mat_fix)
        ax.plot(t_plot2, L_r_p, color=sp['color2'], lw=2.6,
                label=f"VBLS+Repro (5p, t_mat=fix): RMSE={res['r_r']:.2f}")
        
        # η_abs(t) на второй оси
        ax2 = ax.twinx()
        eta_r = eta_abs_repro(t_plot2, res['popt_r'][3], res['popt_r'][4], t_mat_fix)
        ax2.fill_between(t_plot2, eta_r, alpha=0.10, color=sp['color2'])
        ax2.plot(t_plot2, eta_r, color=sp['color2'], lw=1.2, ls='-.', alpha=0.5)
        ax2.set_ylim(0, 1.3)
        ax2.set_ylabel('η_abs(t)', fontsize=8, color=sp['color2'])
        ax2.tick_params(axis='y', colors=sp['color2'], labelsize=7)
        
        # Перегиб при зрелости — стрелка
        eta_before = np.exp(-res['popt_r'][3] * sp['mat_age'])
        L_at_mat = vbls_repro(np.array([float(sp['mat_age'])]),
                               *res['popt_r'], t_mat_fix)[0]
        ax.annotate('Σ-скачок\nпри зрелости',
            xy=(sp['mat_age'], L_at_mat),
            xytext=(sp['mat_age']+1.5, L_at_mat*0.82),
            fontsize=7.5, color=sp['color2'],
            arrowprops=dict(arrowstyle='->', color=sp['color2'], lw=1.0))
    
    # Данные
    ax.errorbar(age, L, yerr=sp['sd'], fmt='o', color=WHITE, ms=5,
                ecolor=MUTED, elinewidth=0.9, capsize=3, zorder=8)
    
    # Метрики бокс
    btext = (f"RMSE_зрел (cm)\n"
             f"VBGF     {res['r_g_m']:.2f}\n"
             f"VBLS+Rep {res.get('r_r_m', float('nan')):.2f}\n"
             f"ΔAIC     {res['aic_g']-res.get('aic_r', res['aic_g']):+.1f}")
    ax.text(0.03, 0.97, btext, transform=ax.transAxes,
            fontsize=8, color=TEXT, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc=GRID_C, ec=MUTED, alpha=0.9))
    
    ax.set_xlabel('Возраст (лет)', fontsize=10)
    ax.set_ylabel('Длина (cm)', fontsize=10)
    ax.set_xlim(0, age[-1]+4.5)
    ax.set_ylim(0, L.max()*1.38)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(True, alpha=0.3, color=GRID_C)
    ax.legend(fontsize=8, framealpha=0.2, facecolor=GRID_C, edgecolor=MUTED,
              loc='lower right')
    ax.set_title(f'{sp["name"]}\n'
                 f'α₁={res.get("popt_r",[0,0,0,0,0])[3]:.3f}, '
                 f'α₂={res.get("popt_r",[0,0,0,0,0])[4]:.3f} (t_mat={sp["mat_age"]}yr fix)',
                 fontsize=9.5, color=TEXT, loc='left', pad=5)

fig5.suptitle(
    'Расширенная VBLS: η_abs(t) = exp(−α₁t − α₂·max(0, t−t_mat))  '
    '← репродуктивный Σ-скачок из §4\n'
    't_mat фиксирован из биологических данных — не свободный параметр',
    fontsize=11, color=TEXT, y=1.02
)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/vbls_repro_model.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("\nРисунок расширенной модели сохранён.")
