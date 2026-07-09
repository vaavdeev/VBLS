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
VBLS — Human lifespan calibration: four coupled ODEs (system S)
Variables: E(t), Sigma(t), eta_abs(t), R(t)

Calibration target : t_B = 75–83 yr  (WHO 2023, developed countries)
Death event        : R(t) = R*  OR  E(t) = E*  (whichever comes first)
Key addition       : restoration-potential function g(E) in the R-restoration term (eq. 4.2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False


# ============================================================================
# 1. EXTERNAL FUNCTIONS (independent channels)
# ============================================================================

def Phi_in(t):
    """
    Phi_in(t) — negentropic input flux (nutrition, oxygen, information).
    Grows to plateau by ~25 yr; gentle decline after age 60.
    """
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline


def Phi_cap(t):
    """
    Phi_cap(t) — environmental entropy-export capacity.
    Kidney/lung/lymph clearance; ~0.8 %/yr decline after age 40 (GFR data).
    Independent of Phi_in by construction.
    """
    baseline    = 0.003
    age_decline = max(0.0, t - 40.0) * 0.008
    return baseline * np.exp(-age_decline)


# ============================================================================
# 2. PARAMETERS  (calibrated on human lifespan)
# ============================================================================

# --- Dissipation (canonical form F_diss = mu*E^2 + lam*E) ---
mu  = 1.50   # nonlinear dissipation coefficient
lam = 0.30   # linear (basal) dissipation coefficient

# --- Structural entropy ---
sigma_0 = 0.0162  # damage production rate (somatic mutations, oxidation)
                   # cross-checked against Cagan et al. (2022) mutation rates

# --- Absorption efficiency (eta_abs channel) ---
alpha = 0.030     # eta_abs degradation rate; constrained by HUNT3/4 VO2max data

# --- Repair reserve (R channel) ---
nu_0 = 0.006      # intrinsic R decay (Hayflick limit analog)
                   # consistent with Whittemore et al. (2019): nu* ~ 0.0047–0.006 /yr
nu   = 0.060      # damage-accelerated R decay
rho  = 0.002      # R restoration rate; physically constrained: rho << nu_0 + nu

# --- Death thresholds ---
R_star = 0.15     # bifurcation threshold: repair reserve exhaustion
E_star = 0.04     # integrity potential threshold

# --- Initial conditions (birth) ---
E0_init = 0.10
Sigma0  = 0.01
eta0    = 0.96
R0      = 0.99

# --- Restoration-potential function normalization (formula 4.2) ---
E_range = E0_init - E_star    # = 0.06  (denominator of g(E))


# ============================================================================
# 3. SYSTEM OF 4 ODEs  — system (S)
# ============================================================================

def system(t, y):
    E, Sigma, eta_abs, R = y

    # Physical bounds
    E       = max(E,   0.0)
    Sigma   = np.clip(Sigma,   0.0, 1.0)
    eta_abs = np.clip(eta_abs, 0.0, 1.0)
    R       = np.clip(R,       0.0, 1.0)

    F_diss = mu * E**2 + lam * E                               # eq. (6.2)

    # Restoration-potential function  g(E) = (E - E*) / (E0 - E*)  in [0, 1]  (eq. 4.2)
    # Degradation terms in dR/dt are independent of E (time-scale separation).
    # Restoration requires active synthesis — gated by g(E).
    g_E = max(0.0, min(1.0, (E - E_star) / E_range))

    # ── Eq. 1: integrity potential E(t)  (eq. 6.1) ──────────────────────────
    dE = eta_abs * Phi_in(t) - F_diss

    # ── Eq. 2: structural entropy Sigma(t)  (eq. 7.1) ───────────────────────
    # max(0,·) enforces monotonicity (2nd law)
    dSigma = max(0.0, sigma_0 * F_diss - R * Phi_cap(t))

    # ── Eq. 3: absorption efficiency eta_abs(t)  (eq. 5.1) ──────────────────
    deta = -alpha * Sigma * eta_abs

    # ── Eq. 4: repair reserve R(t)  (eq. 4.1) ───────────────────────────────
    # Decay terms: spontaneous (-nu_0*R) + damage-driven (-nu*Sigma*R)
    # Restoration: gated by the restoration-potential function g(E) — zero when E -> E*
    dR = -nu_0 * R - nu * Sigma * R + rho * g_E * (1.0 - Sigma) * (1.0 - R)

    return [dE, dSigma, deta, dR]


def event_death(t, y):
    """Integration stops when R <= R* or E <= E* (first crossing)."""
    return min(y[3] - R_star, y[0] - E_star)

event_death.terminal  = True
event_death.direction = -1


# ============================================================================
# 4. INTEGRATION
# ============================================================================

t_max  = 110.0
t_eval = np.linspace(0, t_max, 6000)

sol = solve_ivp(
    system, [0, t_max], [E0_init, Sigma0, eta0, R0],
    method='RK45', t_eval=t_eval, events=event_death,
    rtol=1e-8, atol=1e-10, max_step=0.05
)

t       = sol.t
E       = sol.y[0]
Sigma   = sol.y[1]
eta_abs = sol.y[2]
R       = sol.y[3]

t_death = sol.t_events[0][0] if sol.t_events[0].size > 0 else None

# Derived quantities
Phi_in_v  = np.array([Phi_in(ti)  for ti in t])
Phi_cap_v = np.array([Phi_cap(ti) for ti in t])
F_diss_v      = mu * E**2 + lam * E
g_E_v         = np.clip((E - E_star) / E_range, 0.0, 1.0)
production_v  = sigma_0 * F_diss_v
export_v      = R * Phi_cap_v
int_Sigma     = np.cumsum(Sigma) * (t[1] - t[0])   # numerical integral of Sigma

# Life-cycle phase boundaries
peak_idx = np.argmax(E);   t_peak = t[peak_idx]
col_idx  = np.where(R < 0.50)[0]
t_col    = t[col_idx[0]] if len(col_idx) > 0 else (t_death or t_max)

in_target = (75.0 <= (t_death or 0) <= 83.0)


# ============================================================================
# 5. CONSOLE OUTPUT
# ============================================================================

SEP = "=" * 72
print(SEP)
print("  VBLS — Human lifespan calibration: system (S) with g(E)")
print(SEP)

if t_death is not None:
    R_fin = float(np.interp(t_death, t, R))
    E_fin = float(np.interp(t_death, t, E))
    if R_fin <= R_star + 0.01:
        cause = "R-channel  (repair reserve exhausted, R -> R*)"
    else:
        cause = "E-channel  (integrity potential lost, E -> E*)"
    status = "PASS" if in_target else "RECALIBRATE"
    print(f"\n  t_B (lifespan)  : {t_death:.1f} yr   [target 75–83 yr]  [{status}]")
    print(f"  Death channel   : {cause}")
    print(f"  g(E) at t_B     : {float(np.interp(t_death, t, g_E_v)):.4f}")
    print(f"\n  Growth phase    : 0 — {t_peak:.0f} yr")
    print(f"  Plateau phase   : {t_peak:.0f} — {t_col:.0f} yr")
    print(f"  Collapse phase  : {t_col:.0f} — {t_death:.0f} yr")
    i70 = np.argmin(np.abs(t - 70))
    print(f"\n  Sigma(70 yr)    = {Sigma[i70]:.3f}   (accumulated structural damage)")
    print(f"  R(70 yr)        = {R[i70]:.3f}   (R* = {R_star})")
    print(f"  g(E)(70 yr)     = {g_E_v[i70]:.3f}   (restoration-potential function)")
else:
    print("\n  No death event within t_max — system survives to end of integration.")

print()
header = (f"{'Age':>5} | {'E(t)':>6} | {'Sigma':>6} | {'eta':>6} | "
          f"{'R(t)':>6} | {'g(E)':>6} | {'intSig':>7} | Stage")
print(header)
print("-" * len(header))

STAGES = {
     0: "Newborn",        5: "Early childhood", 10: "Adolescent",
    15: "Youth",         20: "Young adult",     30: "Adult",
    40: "Peak",          50: "Middle age",      60: "Elderly",
    70: "Old age",       80: "Advanced age",    90: "Centenarian"
}

for age, label in STAGES.items():
    if age > t[-1] + 0.5:
        break
    i = np.argmin(np.abs(t - age))
    print(f"{age:>5} | {E[i]:>6.3f} | {Sigma[i]:>6.3f} | {eta_abs[i]:>6.3f} | "
          f"{R[i]:>6.3f} | {g_E_v[i]:>6.3f} | {int_Sigma[i]:>7.2f} | {label}")

print(SEP)


# ============================================================================
# 6. FIGURES
# ============================================================================

BG  = '#f7f9fc'
c_g = '#27ae60'    # growth phase color
c_p = '#2980b9'    # plateau phase color
c_c = '#e74c3c'    # collapse phase color
c_d = '#8e44ad'    # death-event / threshold color

t_xlim = min(t_max, (t_death or t_max) + 2)

fig = plt.figure(figsize=(19, 14), facecolor=BG)
title_str = (f'VBLS — Human lifespan: system (S) with restoration-potential function g(E)\n'
             f't_B = {t_death:.0f} yr  (R-channel: R ≤ R* = {R_star})'
             if t_death else 'VBLS — Human lifespan: no death event')
fig.suptitle(title_str, fontsize=13, fontweight='bold', color='#1a252f', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)


def shading(ax):
    ax.axvspan(0,      t_peak, alpha=0.07, color=c_g, zorder=0)
    ax.axvspan(t_peak, t_col,  alpha=0.07, color=c_p, zorder=0)
    if t_death:
        ax.axvspan(t_col, t_death, alpha=0.09, color=c_c, zorder=0)
        ax.axvline(t_death, color=c_d, ls='--', lw=1.8, alpha=0.85, zorder=5)
    ax.set_xlim(0, t_xlim)
    ax.set_facecolor(BG)


def phase_labels(ax):
    ylo, yhi = ax.get_ylim()
    yy = yhi - (yhi - ylo) * 0.09
    ax.text(t_peak / 2,            yy, 'GROWTH',  ha='center', fontsize=8, color=c_g, fontweight='bold')
    ax.text((t_peak + t_col) / 2,  yy, 'PLATEAU', ha='center', fontsize=8, color=c_p, fontweight='bold')
    if t_death:
        ax.text((t_col + t_death) / 2, yy, 'COLLAPSE', ha='center', fontsize=8, color=c_c, fontweight='bold')


# ── Panel 1: E(t) ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(t, E, color='#1a252f', lw=2.8, label='E(t)')
ax1.axhline(E_star, color=c_d, ls=':', lw=1.2, alpha=0.5)
ax1.text(t_xlim * 0.98, E_star + 0.003, f'E* = {E_star}',
         ha='right', fontsize=8, color=c_d)
shading(ax1); phase_labels(ax1)
ax1.set_title('Integrity potential E(t)', fontsize=10)
ax1.set_ylabel('E(t)')

# ── Panel 2: Sigma(t) ─────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.fill_between(t, 0, Sigma, alpha=0.2, color='#c0392b')
ax2.plot(t, Sigma, color='#c0392b', lw=2.8)
shading(ax2); phase_labels(ax2)
ax2.set_title(r'Structural entropy $\Sigma(t)$ — monotone (Lemma A.1)', fontsize=10)
ax2.set_ylabel(r'$\Sigma(t)$')
ax2.set_ylim(0)

# ── Panel 3: R(t) and eta_abs(t) ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :2])
ax3.plot(t, R,       color='#27ae60', lw=2.8,
         label='R(t) — repair reserve (Hayflick analog)')
ax3.plot(t, eta_abs, color='#e67e22', lw=2.2, ls='--',
         label=r'$\eta_{abs}(t)$ — absorption efficiency')
ax3.plot(t, g_E_v,   color='#9b59b6', lw=1.6, ls=':',
         label='g(E) — restoration-potential function (eq. 4.2)')
ax3.axhline(R_star, color=c_d, ls=':', lw=1.8,
            label=f'R* = {R_star}  (bifurcation threshold)')
ax3.fill_between(t, 0, R_star, alpha=0.06, color=c_d)
shading(ax3); phase_labels(ax3)
ax3.set_title(r'Repair reserve R(t), $\eta_{abs}(t)$, and restoration potential g(E)', fontsize=10)
ax3.set_ylabel('Value')
ax3.set_xlabel('Age (yr)')
ax3.legend(fontsize=7.5, loc='upper right')

# ── Panel 4: Damage production vs export ─────────────────────────────────────
ax4 = fig.add_subplot(gs[0, 2])
ax4.plot(t, production_v, color='#c0392b', lw=2,
         label=r'Production $\sigma_0 F_{diss}$')
ax4.plot(t, export_v,     color='#27ae60', lw=2,
         label=r'Export $R\cdot\Phi_{cap}$')
ax4.fill_between(t, export_v, production_v,
                 where=(production_v >= export_v),
                 alpha=0.30, color='#c0392b', label=r'$\Sigma$ growing')
ax4.fill_between(t, export_v, production_v,
                 where=(production_v < export_v),
                 alpha=0.15, color='#27ae60')
if t_death:
    ax4.axvline(t_death, color=c_d, ls='--', lw=1.5)
ax4.set_xlim(0, t_xlim)
ax4.set_facecolor(BG)
ax4.set_title('Damage production\nvs export rate', fontsize=9)
ax4.set_xlabel('Age (yr)')
ax4.set_ylabel(r'$d\Sigma/dt$ components')
ax4.legend(fontsize=7)

# ── Panel 5: External functions ───────────────────────────────────────────────
ax5   = fig.add_subplot(gs[1, 2])
ax5_r = ax5.twinx()
ax5.plot(t, Phi_in_v,   color='#3498db', lw=2, label=r'$\Phi_{in}$ (left)')
ax5_r.plot(t, Phi_cap_v, color='#9b59b6', lw=2, ls='--', label=r'$\Phi_{cap}$ (right)')
if t_death:
    ax5.axvline(t_death,   color=c_d, ls='--', lw=1.5)
    ax5_r.axvline(t_death, color=c_d, ls='--', lw=1.5)
ax5.set_xlim(0, t_xlim)
ax5.set_facecolor(BG)
ax5.set_title(r'$\Phi_{in}$ and $\Phi_{cap}$ —' + '\nindependent channels', fontsize=9)
ax5.set_xlabel('Age (yr)')
ax5.set_ylabel(r'$\Phi_{in}$', color='#3498db')
ax5_r.set_ylabel(r'$\Phi_{cap}$', color='#9b59b6')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_r.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

# ── Panel 6: Phase portrait R(Sigma) ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
sc = ax6.scatter(Sigma, R, c=t, cmap='plasma', s=8, zorder=3)
ax6.axhline(R_star, color=c_d, ls='--', lw=1.8, label=f'R* = {R_star}')
ax6.fill_between([0.0, 1.0], 0, R_star, alpha=0.08, color=c_d)

mid = len(t) // 2
if mid + 50 < len(t):
    ax6.annotate(
        '', xy=(Sigma[mid + 50], R[mid + 50]),
        xytext=(Sigma[mid], R[mid]),
        arrowprops=dict(arrowstyle='->', color='#1a252f', lw=2.0)
    )
    ax6.text(Sigma[mid], R[mid] + 0.02, 'time →', fontsize=7, color='#1a252f')

if t_death is not None:
    death_idx = np.argmin(np.abs(t - t_death))
    ax6.scatter([Sigma[death_idx]], [R[death_idx]],
                c='red', s=100, zorder=10, marker='X',
                label=f'Death ({t_death:.0f} yr)')

ax6.set_facecolor(BG)
ax6.set_xlabel(r'$\Sigma(t)$ — accumulated damage')
ax6.set_ylabel('R(t) — repair reserve')
ax6.set_title('Phase portrait R(Σ)\n(state-space trajectory)', fontsize=9)
ax6.legend(fontsize=7)
plt.colorbar(sc, ax=ax6, label='Age (yr)', shrink=0.8)

plt.savefig('human_lifespan_EN.png', dpi=150, bbox_inches='tight', facecolor=BG)
print("\nFigure saved: human_lifespan_EN.png")
