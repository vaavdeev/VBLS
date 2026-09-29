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
VBLS — Sensitivity analysis of the human lifespan calibration.
Addresses reviewer question: "How were parameters chosen? Can anything be fitted?"

Three complementary tests:
  (A) Tornado chart     — each parameter varied individually (±10 %, ±20 %, ±50 %)
  (B) Sensitivity bars  — |Δt_B| / t_B  at ±10 % perturbation
  (C) 2D iso-lifespan   — heatmap for the two most sensitive parameters
  (D) Monte Carlo       — all parameters perturbed simultaneously ±20 %

Key conclusion: the model is structurally stable; nu and sigma_0 produce the
largest shifts (≈10 % each at ±10 % perturbation); mu, lam, E_star are inert
because g(E)=1 throughout the normal human aging trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import time

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False


# ============================================================================
# 1. BASELINE PARAMETERS AND INITIAL CONDITIONS
# ============================================================================

BASE = dict(
    mu      = 1.50,    # nonlinear dissipation
    lam     = 0.30,    # linear (basal) dissipation
    sigma_0 = 0.0162,   # damage production rate
    alpha   = 0.030,   # eta_abs degradation rate
    nu_0    = 0.006,   # intrinsic R decay (Hayflick limit)
    nu      = 0.060,   # damage-accelerated R decay
    rho     = 0.002,   # R restoration rate
    R_star  = 0.15,    # R death threshold
    E_star  = 0.04,    # E death threshold
)
IC = dict(E0=0.10, Sigma0=0.01, eta0=0.96, R0=0.99)

T_MAX    = 130.0   # integration ceiling (yr)
T_TARGET = None    # filled after baseline run

# ============================================================================
# 2. EXTERNAL FUNCTIONS (unchanged across all sensitivity runs)
# ============================================================================

def Phi_in(t):
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline

def Phi_cap(t):
    return 0.003


# ============================================================================
# 3. SIMULATION KERNEL
# ============================================================================

def run_sim(params=None, ic=None):
    """
    Integrate system (S) and return t_B (yr).
    Returns T_MAX if no death event is triggered within T_MAX.
    """
    p = {**BASE,  **(params or {})}
    c = {**IC,    **(ic     or {})}

    E_range = max(c['E0'] - p['E_star'], 1e-9)   # avoid division by zero

    def system(t, y):
        E, Sig, eta, R = y
        E   = max(E,   0.0)
        Sig = np.clip(Sig, 0.0, 1.0)
        eta = np.clip(eta, 0.0, 1.0)
        R   = np.clip(R,   0.0, 1.0)
        F   = p['mu'] * E**2 + p['lam'] * E
        g_E = max(0.0, min(1.0, (E - p['E_star']) / E_range))
        dE   = eta * Phi_in(t) - F
        dSig = max(0.0, p['sigma_0'] * F - R * Phi_cap(t))
        deta = -p['alpha'] * Sig * eta
        dR   = (-p['nu_0'] * R - p['nu'] * Sig * R
                + p['rho'] * g_E * (1.0 - Sig) * (1.0 - R))
        return [dE, dSig, deta, dR]

    def event(t, y):
        return min(y[3] - p['R_star'], y[0] - p['E_star'])
    event.terminal  = True
    event.direction = -1

    sol = solve_ivp(
        system, [0, T_MAX], [c['E0'], c['Sigma0'], c['eta0'], c['R0']],
        method='RK45', events=event,
        rtol=1e-7, atol=1e-9, max_step=0.1
    )
    if sol.t_events[0].size > 0:
        return float(sol.t_events[0][0])
    return float(T_MAX)


# ============================================================================
# 4. BASELINE RUN
# ============================================================================

print("Running baseline...")
T_TARGET = run_sim()
print(f"  Baseline t_B = {T_TARGET:.2f} yr")


# ============================================================================
# 5. 1-D SENSITIVITY: tornado data
# ============================================================================

PARAM_LABELS = {
    'nu_0':    r'$\nu_0$ — intrinsic R decay',
    'nu':      r'$\nu$ — damage-accel. R decay',
    'sigma_0': r'$\sigma_0$ — damage production',
    'rho':     r'$\rho$ — R restoration rate',
    'alpha':   r'$\alpha$ — $\eta_{abs}$ degradation',
    'R_star':  r'$R^*$ — R death threshold',
    'E_star':  r'$E^*$ — E death threshold',
    'mu':      r'$\mu$ — nonlinear dissipation',
    'lam':     r'$\lambda$ — linear dissipation',
}

DELTAS = [-0.50, -0.20, -0.10, +0.10, +0.20, +0.50]

print("\nComputing 1-D sensitivity (tornado)...")
sensitivity = {}
for pname in PARAM_LABELS:
    base_val = BASE[pname]
    row = {}
    for delta in DELTAS:
        row[delta] = run_sim(params={pname: base_val * (1.0 + delta)})
    sensitivity[pname] = row
    lo = row[-0.10]; hi = row[+0.10]
    pct = abs(hi - lo) / T_TARGET * 100
    print(f"  {pname:8s}: t_B in [{row[-0.50]:.1f}, {row[+0.50]:.1f}] yr  "
          f"| ±10% range = {pct:.1f}%")

# Sort by ±10% sensitivity (descending)
range10 = {p: abs(sensitivity[p][+0.10] - sensitivity[p][-0.10])
           for p in PARAM_LABELS}
sorted_params = sorted(range10, key=range10.get, reverse=True)
pct10 = {p: range10[p] / T_TARGET * 100 for p in sorted_params}

max_pct = max(pct10.values())
print(f"\n  Max ±10% sensitivity: {max_pct:.1f}%  "
      f"(parameter: {sorted_params[0]})")


# ============================================================================
# 6. 2-D HEATMAP: two most sensitive parameters
# ============================================================================

p1 = sorted_params[0]
p2 = sorted_params[1]
print(f"\nComputing 2-D heatmap: {p1} × {p2}  (20×20 grid)...")

N_GRID = 20
p1_vals = BASE[p1] * np.linspace(0.50, 1.50, N_GRID)
p2_vals = BASE[p2] * np.linspace(0.50, 1.50, N_GRID)
heatmap = np.zeros((N_GRID, N_GRID))
t0 = time.time()
for i, v1 in enumerate(p1_vals):
    for j, v2 in enumerate(p2_vals):
        heatmap[i, j] = run_sim(params={p1: v1, p2: v2})
print(f"  Done in {time.time()-t0:.1f} s")


# ============================================================================
# 7. MONTE CARLO: all parameters ±20 % simultaneously
# ============================================================================

N_MC = 600
print(f"\nMonte Carlo: {N_MC} samples, all params ±20%...")
rng = np.random.default_rng(seed=42)
mc_t_B = np.empty(N_MC)
t0 = time.time()
for k in range(N_MC):
    perturbed = {p: BASE[p] * rng.uniform(0.80, 1.20) for p in BASE}
    mc_t_B[k] = run_sim(params=perturbed)
print(f"  Done in {time.time()-t0:.1f} s")
print(f"  t_B: mean={mc_t_B.mean():.1f} yr,  "
      f"std={mc_t_B.std():.1f} yr,  "
      f"range=[{mc_t_B.min():.1f}, {mc_t_B.max():.1f}] yr")
frac_in_target = np.mean((mc_t_B >= 76) & (mc_t_B <= 85)) * 100
print(f"  Fraction in WHO target [76–85 yr]: {frac_in_target:.0f}%")


# ============================================================================
# 8. FIGURE
# ============================================================================

BG = '#f7f9fc'

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle(
    'VBLS — Sensitivity analysis: human lifespan calibration\n'
    r'Response of $t_B$ to parameter perturbations  |  system (S) with restoration-potential function g(E)',
    fontsize=13, fontweight='bold', color='#1a252f', y=0.99
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.36)

y_pos = np.arange(len(sorted_params))

# ── Panel A: Tornado chart ────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(BG)
ax_a.set_title(r'(A) Tornado chart — $\Delta t_B$ from baseline (yr)', fontsize=10)

color_map = {
    -0.50: '#922b21', -0.20: '#e74c3c', -0.10: '#f39c12',
    +0.10: '#27ae60', +0.20: '#1e8449', +0.50: '#0b5345'
}
BAR_H = 0.13
for k, delta in enumerate(DELTAS):
    offsets = [sensitivity[p][delta] - T_TARGET for p in sorted_params]
    ax_a.barh(
        y_pos + (k - 2.5) * BAR_H, offsets,
        height=BAR_H, color=color_map[delta],
        label=f'{delta:+.0%}', alpha=0.88
    )

ax_a.axvline(0, color='black', lw=1.3, ls='--')
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels([PARAM_LABELS[p] for p in sorted_params], fontsize=8.5)
ax_a.set_xlabel(r'$\Delta t_B$ from baseline (yr)', fontsize=9)
ax_a.legend(fontsize=7.5, title='Perturbation', loc='lower right',
            framealpha=0.7)

# ── Panel B: Sensitivity magnitude ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(BG)
ax_b.set_title(r'(B) Sensitivity magnitude at $\pm10\%$ perturbation', fontsize=10)

bar_colors = ['#c0392b' if pct10[p] > 5 else '#2980b9' for p in sorted_params]
ax_b.barh(y_pos, [pct10[p] for p in sorted_params],
          color=bar_colors, alpha=0.85, height=0.6)
ax_b.axvline(5.0, color='#7d3c98', lw=1.6, ls=':', label='5 % line')
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([PARAM_LABELS[p] for p in sorted_params], fontsize=8.5)
ax_b.set_xlabel(r'$|\Delta t_B| / t_B$  (%)', fontsize=9)
ax_b.legend(fontsize=8.5)

for i, p in enumerate(sorted_params):
    ax_b.text(pct10[p] + 0.15, i, f'{pct10[p]:.1f}%',
              va='center', fontsize=7.5)

# ── Panel C: 2-D iso-lifespan heatmap ─────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor(BG)

im = ax_c.imshow(
    heatmap, origin='lower', aspect='auto', cmap='RdYlGn',
    extent=[p2_vals[0], p2_vals[-1], p1_vals[0], p1_vals[-1]],
    vmin=50, vmax=120
)
ct = ax_c.contour(
    p2_vals, p1_vals, heatmap,
    levels=[60, 70, 76, 80, 85, 90, 100, 110],
    colors='black', linewidths=0.7, alpha=0.55
)
ax_c.clabel(ct, fmt='%d yr', fontsize=7, inline=True)

# WHO target band: 76–85 yr region bounded by contour lines (already shown)
ax_c.axvline(BASE[p2], color='white', ls='--', lw=2.0, alpha=0.9, label='Baseline')
ax_c.axhline(BASE[p1], color='white', ls='--', lw=2.0, alpha=0.9)
ax_c.scatter([BASE[p2]], [BASE[p1]], color='white', s=80, zorder=10,
             marker='*', label=f'Baseline ({T_TARGET:.0f} yr)')

ax_c.set_xlabel(PARAM_LABELS[p2], fontsize=9)
ax_c.set_ylabel(PARAM_LABELS[p1], fontsize=9)
ax_c.set_title(
    f'(C) 2-D sensitivity: {p1} × {p2}\n'
    '(iso-lifespan contours, yr; green = longer life)',
    fontsize=9
)
ax_c.legend(fontsize=8, loc='upper right', framealpha=0.7)
plt.colorbar(im, ax=ax_c, label=r'$t_B$ (yr)', shrink=0.85)

# ── Panel D: Monte Carlo histogram ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor(BG)

counts, edges, patches = ax_d.hist(
    mc_t_B, bins=40, color='#2980b9', alpha=0.75,
    edgecolor='white', lw=0.5
)
# Colour bars inside WHO target
for patch, left in zip(patches, edges[:-1]):
    if 76.0 <= left <= 85.0:
        patch.set_facecolor('#27ae60')
        patch.set_alpha(0.85)

ax_d.axvline(T_TARGET, color='#e74c3c', lw=2.2, ls='--',
             label=f'Baseline $t_B$ = {T_TARGET:.1f} yr')
ax_d.axvspan(76, 85, alpha=0.10, color='#27ae60',
             label='WHO target 76–85 yr')
ax_d.set_xlabel(r'$t_B$ (yr)', fontsize=10)
ax_d.set_ylabel('Count', fontsize=10)
ax_d.set_title(
    f'(D) Monte Carlo: all params ±20 % simultaneously\n'
    rf'N = {N_MC}  |  mean = {mc_t_B.mean():.1f} yr  |  '
    rf'$\sigma$ = {mc_t_B.std():.1f} yr  |  {frac_in_target:.0f}% in target',
    fontsize=9
)
ax_d.legend(fontsize=8.5, framealpha=0.7)

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.set_facecolor(BG)

plt.savefig('human_sensitivity_EN.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\nFigure saved: human_sensitivity_EN.png")

# ============================================================================
# 9. SUMMARY TABLE
# ============================================================================
print(f"\n{'='*72}")
print(f"  VBLS — Sensitivity summary  |  baseline t_B = {T_TARGET:.1f} yr")
print(f"{'='*72}")
print(f"  {'Parameter':<12}  {'−50%':>6}  {'−20%':>6}  {'−10%':>6}  "
      f"{'Baseline':>8}  {'+10%':>6}  {'+20%':>6}  {'+50%':>6}  {'±10% sens':>10}")
print(f"  {'-'*88}")
for p in sorted_params:
    row = sensitivity[p]
    print(f"  {p:<12}  "
          f"{row[-0.50]:>6.1f}  {row[-0.20]:>6.1f}  {row[-0.10]:>6.1f}  "
          f"{T_TARGET:>8.1f}  "
          f"{row[+0.10]:>6.1f}  {row[+0.20]:>6.1f}  {row[+0.50]:>6.1f}  "
          f"{pct10[p]:>9.1f}%")
print(f"\n  Max ±10% sensitivity = {max_pct:.1f}%  ({sorted_params[0]})")
print(f"  Monte Carlo ±20%: {frac_in_target:.0f}% of samples within WHO target [76–85 yr]")
print(f"{'='*72}")
