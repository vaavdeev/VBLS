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
VBLS — Alpha sensitivity: non-monotonic dependence of t_B on α with optimum α* ≈ 0.10 yr⁻¹.

Three distinct regimes (Homo sapiens, calibrated parameters):
  α ≤ 0.020 yr⁻¹  — α-passive regime: memory decay too slow to affect R-channel;
                     t_B = 73–75 yr, change < 2.1% vs baseline.
  0.020 < α ≤ 0.10 — α-active regime: POSITIVE sensitivity (counter-intuitive);
                     mechanism: α↑ → η_abs↓ → E↓ → F_diss↓ → σ_0·F_diss↓ → Σ slower
                              → ν·Σ·R weaker → R depletes slower → t_B increases.
                     Peak at α* ≈ 0.10 yr⁻¹ → t_B ≈ 88.8 yr (maximum).
  α > 0.10 yr⁻¹   — collapse regime: NEGATIVE sensitivity;
                     η_abs decays so fast that E collapses before R-channel fires;
                     t_B falls sharply (α=0.18 → 87.1 yr; α=1.9 → 45 yr; α=10000 → 2.3 yr).

Calibrated baseline: α = 0.030 yr⁻¹ → t_B = 76.9 yr (within WHO 75–83 yr target).
Theoretical optimum: α* ≈ 0.10 yr⁻¹ → t_B = 88.8 yr (~12 yr above calibrated baseline).
Biological interpretation: humans operate below α*, consistent with an evolutionary
trade-off between lifespan maximisation and metabolic intensity for reproduction.

Four panels:
  (A) t_B vs α — non-monotonic curve, optimum at α* ≈ 0.10 yr⁻¹
  (B) η_abs(t) trajectories for multiple α values
  (C) Σ(t) trajectories showing slower accumulation at higher α (α-active regime)
  (D) R(t) trajectories showing slower depletion at higher α (→ later R-channel death)

Output: mde_alpha_sensitivity.png / .pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    10.5,
    'axes.labelsize':    9.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})

# ============================================================================
# CANONICAL PARAMETERS (April 2026)
# ============================================================================

BASE = dict(
    mu      = 1.50,
    lam     = 0.30,
    sigma_0 = 0.0162,
    alpha   = 0.030,
    nu_0    = 0.006,
    nu      = 0.060,
    rho     = 0.002,
    R_star  = 0.15,
    E_star  = 0.04,
)
IC = [0.10, 0.01, 0.96, 0.99]   # E0, Σ0, η0, R0
E0    = IC[0]
E_RANGE = E0 - BASE['E_star']   # 0.06
T_MAX = 130.0


def Phi_in(t):
    return (1.0 - np.exp(-t / 10.0)) * np.exp(-max(0.0, t - 60.0) / 30.0)


def Phi_cap(t):
    return 0.003 * np.exp(-max(0.0, t - 40.0) * 0.008)


# ============================================================================
# SIMULATION KERNEL
# ============================================================================

def run(alpha_val):
    """
    Integrate system (S) with a specified α value.
    Returns (t_B, sol) where sol.y has shape (4, n_steps).
    """
    p = {**BASE, 'alpha': alpha_val}

    def rhs(t, y):
        E, S, eta, R = y
        E   = max(E,   0.0)
        S   = np.clip(S,   0.0, 1.0)
        eta = np.clip(eta, 0.0, 1.0)
        R   = np.clip(R,   0.0, 1.0)
        F   = p['mu'] * E**2 + p['lam'] * E
        g_E = max(0.0, min(1.0, (E - p['E_star']) / E_RANGE))
        return [
            eta * Phi_in(t) - F,
            max(0.0, p['sigma_0'] * F - R * Phi_cap(t)),
            -p['alpha'] * S * eta,
            -p['nu_0'] * R - p['nu'] * S * R + p['rho'] * g_E * (1.0 - S) * (1.0 - R),
        ]

    def event(t, y):
        return min(y[3] - p['R_star'], y[0] - p['E_star'])
    event.terminal  = True
    event.direction = -1

    sol = solve_ivp(rhs, [0, T_MAX], IC,
                    method='RK45', events=event,
                    rtol=1e-7, atol=1e-9, max_step=0.2)
    t_B = float(sol.t_events[0][0]) if sol.t_events[0].size > 0 else T_MAX
    return t_B, sol


# ============================================================================
# SCAN OVER ALPHA VALUES
# ============================================================================

# Wide range: 0.01 to 0.20 (baseline = 0.030)
ALPHA_SCAN  = np.linspace(0.005, 0.20, 60)
t_B_scan    = np.array([run(a)[0] for a in ALPHA_SCAN])

# Trajectory set: 7 representative α values for panels B/C/D
ALPHA_TRAJ  = [0.0005, 0.005, 0.010, 0.020, 0.030, 0.060, 0.100, 0.180, 0.280, 0.380, 0.98, 1.9, 4.5, 20.0, 30.0, 70.0, 120.0, 300.0, 800.0, 10000.0]
TRAJ_CMAP   = cm.get_cmap('plasma', len(ALPHA_TRAJ))
TRAJ_COLORS = [TRAJ_CMAP(i) for i in range(len(ALPHA_TRAJ))]

print("Running trajectory set...")
trajs = {}
for a in ALPHA_TRAJ:
    tB, sol = run(a)
    trajs[a] = (tB, sol)
    print(f"  α = {a:.3f}  →  t_B = {tB:.1f} yr")

print(f"\nBaseline α = {BASE['alpha']:.3f}  →  t_B = {trajs[BASE['alpha']][0]:.1f} yr")


# ============================================================================
# FIGURE
# ============================================================================

BG = '#f7f9fc'
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle(
    r'VBLS — $\alpha$ sensitivity: non-monotonic $t_B(\alpha)$ with optimum $\alpha^* \approx 0.10$ yr$^{-1}$'
    '\n'
    r'Active regime ($\alpha < \alpha^*$): $\alpha\uparrow$ $\Rightarrow$ $\eta_{abs}\downarrow$ $\Rightarrow$ $F_{diss}\downarrow$'
    r' $\Rightarrow$ $\Sigma\downarrow$ $\Rightarrow$ $t_B\uparrow$  |  '
    r'Collapse regime ($\alpha > \alpha^*$): $\eta_{abs}$ collapses, $E\to E^*$ early $\Rightarrow$ $t_B\downarrow$',
    fontsize=11, fontweight='bold', y=1.01
)
fig.subplots_adjust(hspace=0.42, wspace=0.34, left=0.09, right=0.97,
                    top=0.92, bottom=0.09)

ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]
for ax in axes.flat:
    ax.set_facecolor(BG)

# ── Panel A: t_B vs α ──────────────────────────────────────────────────────────
ax_a.plot(ALPHA_SCAN, t_B_scan, color='#2c3e50', lw=2.0, zorder=4)
ax_a.axvline(BASE['alpha'], color='#c0392b', lw=1.5, ls='--',
             label=f"Baseline α = {BASE['alpha']:.3f}")
ax_a.axhspan(75, 83, alpha=0.12, color='#27ae60', label='WHO target 75–83 yr')

# Mark trajectory points
for a, col in zip(ALPHA_TRAJ, TRAJ_COLORS):
    tB = trajs[a][0]
    ax_a.scatter([a], [tB], color=col, s=55, zorder=5)

ax_a.set_xlabel(r'$\alpha$ (yr$^{-1}$)', fontsize=10)
ax_a.set_ylabel(r'$t_B$ (yr)', fontsize=10)
ax_a.set_title(r'(A)  $t_B$ vs $\alpha$  — non-monotonic: optimum $\alpha^* \approx 0.10$ yr$^{-1}$',
               fontsize=10.5)
ax_a.legend(fontsize=8.5, framealpha=0.8)
ax_a.text(0.98, 0.06,
          r'$\alpha < \alpha^*$: $t_B\uparrow$ (feedback via $F_{diss}{\to}\Sigma{\to}R$)'
          '\n'
          r'$\alpha > \alpha^*$: $t_B\downarrow$ (collapse of $\eta_{abs}$)'
          '\n'
          r'$\alpha^* \approx 0.10$ yr$^{-1}$  →  $t_B^{max} \approx 88.8$ yr',
          transform=ax_a.transAxes, ha='right', va='bottom',
          fontsize=8.5, color='#2c3e50',
          bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#aaa', alpha=0.85))

# ── Panel B: η_abs(t) trajectories ────────────────────────────────────────────
for a, col in zip(ALPHA_TRAJ, TRAJ_COLORS):
    tB, sol = trajs[a]
    t_plot = sol.t[sol.t <= tB + 1]
    eta_plot = sol.y[2, :len(t_plot)]
    lw = 2.2 if abs(a - BASE['alpha']) < 1e-4 else 1.3
    ls = '-' if abs(a - BASE['alpha']) < 1e-4 else '--'
    ax_b.plot(t_plot, eta_plot, color=col, lw=lw, ls=ls,
              label=f'α = {a:.3f}  (t_B = {tB:.0f} yr)')

ax_b.axvline(BASE['alpha'] * 1000, color='none')   # dummy, keeps legend tidy
ax_b.set_xlabel('Age (yr)', fontsize=10)
ax_b.set_ylabel(r'$\eta_{abs}(t)$', fontsize=10)
ax_b.set_title(r'(B)  $\eta_{abs}(t)$ — faster decay at higher $\alpha$', fontsize=10.5)
ax_b.legend(fontsize=7.5, loc='lower left', framealpha=0.8)

# ── Panel C: Σ(t) trajectories ────────────────────────────────────────────────
for a, col in zip(ALPHA_TRAJ, TRAJ_COLORS):
    tB, sol = trajs[a]
    t_plot = sol.t[sol.t <= tB + 1]
    sig_plot = sol.y[1, :len(t_plot)]
    lw = 2.2 if abs(a - BASE['alpha']) < 1e-4 else 1.3
    ls = '-' if abs(a - BASE['alpha']) < 1e-4 else '--'
    ax_c.plot(t_plot, sig_plot, color=col, lw=lw, ls=ls,
              label=f'α = {a:.3f}')

ax_c.set_xlabel('Age (yr)', fontsize=10)
ax_c.set_ylabel(r'$\Sigma(t)$', fontsize=10)
ax_c.set_title(r'(C)  $\Sigma(t)$ — slower damage accumulation at higher $\alpha$',
               fontsize=10.5)
ax_c.legend(fontsize=7.5, loc='upper left', framealpha=0.8)
ax_c.text(0.98, 0.04,
          r'Higher $\alpha$ $\Rightarrow$ lower $E$ $\Rightarrow$'
          '\n' r'lower $F_{diss}$ $\Rightarrow$ slower $\Sigma\uparrow$',
          transform=ax_c.transAxes, ha='right', va='bottom',
          fontsize=8.0, color='#2c3e50',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa', alpha=0.85))

# ── Panel D: R(t) trajectories ────────────────────────────────────────────────
for a, col in zip(ALPHA_TRAJ, TRAJ_COLORS):
    tB, sol = trajs[a]
    t_plot = sol.t[sol.t <= tB + 1]
    R_plot = sol.y[3, :len(t_plot)]
    lw = 2.2 if abs(a - BASE['alpha']) < 1e-4 else 1.3
    ls = '-' if abs(a - BASE['alpha']) < 1e-4 else '--'
    ax_d.plot(t_plot, R_plot, color=col, lw=lw, ls=ls,
              label=f'α = {a:.3f}')

ax_d.axhline(BASE['R_star'], color='#c0392b', lw=1.2, ls=':',
             label=f"R* = {BASE['R_star']}")
ax_d.set_xlabel('Age (yr)', fontsize=10)
ax_d.set_ylabel(r'$R(t)$', fontsize=10)
ax_d.set_title(r'(D)  $R(t)$ — slower depletion at higher $\alpha$ (→ later death)',
               fontsize=10.5)
ax_d.legend(fontsize=7.5, loc='lower left', framealpha=0.8)
ax_d.text(0.98, 0.30,
          'Death occurs when R = R*\n'
          r'Slower $\Sigma$ $\Rightarrow$ R decays more slowly',
          transform=ax_d.transAxes, ha='right', va='bottom',
          fontsize=8.0, color='#2c3e50',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa', alpha=0.85))

fig.text(0.99, 0.004,
         'VBLS system (S)  |  σ₀=0.0162, ν₀=0.006, ν=0.060, ρ=0.002  |  '
         'alpha_sensitivity.py',
         ha='right', fontsize=7.5, color='#666', style='italic')

plt.savefig('alpha_sensitivity.pdf', bbox_inches='tight')
plt.savefig('alpha_sensitivity.png', bbox_inches='tight', dpi=180)
plt.close()
print("\nSaved: alpha_sensitivity.pdf / .png")

# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

print(f"\n{'='*62}")
print(f"  α vs t_B summary (baseline α = {BASE['alpha']:.3f})")
print(f"{'='*62}")
print(f"  {'α':>8}  {'t_B (yr)':>10}  {'Δt_B (yr)':>12}  {'Δt_B (%)':>10}")
print(f"  {'-'*50}")
tB_base = trajs[BASE['alpha']][0]
for a in ALPHA_TRAJ:
    tB = trajs[a][0]
    delta = tB - tB_base
    pct   = delta / tB_base * 100
    marker = ' ← baseline' if abs(a - BASE['alpha']) < 1e-4 else ''
    print(f"  {a:>8.3f}  {tB:>10.1f}  {delta:>+12.1f}  {pct:>+9.1f}%{marker}")
print(f"{'='*62}")
print(f"\nConclusion: non-monotonic t_B(α) with optimum α* ≈ 0.10 yr⁻¹ (t_B = 88.8 yr).")
print(f"Calibrated α = 0.030 yr⁻¹ lies in the α-active regime, ~3× below α*.")
print(f"Biological interpretation: humans trade lifespan maximisation for metabolic")
print(f"intensity (reproduction, activity), operating below the theoretical optimum.")
print(f"Qualitatively consistent with caloric restriction / rate-of-living observations.")
