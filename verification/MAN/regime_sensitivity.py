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
VBLS — Regime sensitivity to nu_0 (extension of mde_human_sensitivity.py).

Goal: numerical demonstration that the zero sensitivity of t_B to (mu, lam, E*)
documented in Table 8.3.1 is a REGIME property of normal aging
(R-channel domination), not an architectural defect.

For decreasing nu_0:
  (i)  the death channel switches from R-channel to E-channel,
  (ii) sensitivity to (mu, lam, E*) becomes non-zero.

This is the falsifiable prediction requested by Reviewer 5
(Devil's advocate).

Outputs:
  - Console table with (nu_0, t_B, dominant channel, sens_mu, sens_lam, sens_E*).
  - Figure mde_regime_sensitivity.png with two panels:
      (A) t_B vs nu_0 with channel color-coding,
      (B) sensitivities to mu/lam/E* vs nu_0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# ============================================================================
# 1. BASELINE (Homo sapiens calibration) — identical to mde_human_sensitivity.py
# ============================================================================

BASE = dict(
    mu=1.50,
    lam=0.30,
    sigma_0=0.0162,
    alpha=0.030,
    nu_0=0.006,
    nu=0.060,
    rho=0.002,
    R_star=0.15,
    E_star=0.04,
)
IC = dict(E0=0.10, Sigma0=0.01, eta0=0.96, R0=0.99)

# Increased horizon because at low nu_0 lifespans can exceed 500 yr
T_MAX = 2000.0


def Phi_in(t):
    growth = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline


def Phi_cap(t):
    return 0.003


# ============================================================================
# 2. SIMULATION KERNEL with per-channel event tracking
# ============================================================================

def run_sim(params=None, ic=None, return_channel=False):
    """
    Integrate system (S). Two terminal events:
      - event_R: R(t) - R_star = 0
      - event_E: E(t) - E_star = 0
    The earlier event determines t_B and the dominant death channel.

    Returns:
        t_B (float)
        if return_channel: also (channel in {'R','E','NONE'}, E_end, R_end)
    """
    p = {**BASE, **(params or {})}
    c = {**IC, **(ic or {})}

    E_range = max(c['E0'] - p['E_star'], 1e-9)

    def system(t, y):
        E, Sig, eta, R = y
        E = max(E, 0.0)
        Sig = np.clip(Sig, 0.0, 1.0)
        eta = np.clip(eta, 0.0, 1.0)
        R = np.clip(R, 0.0, 1.0)
        F = p['mu'] * E**2 + p['lam'] * E
        g_E = max(0.0, min(1.0, (E - p['E_star']) / E_range))
        dE = eta * Phi_in(t) - F
        dSig = max(0.0, p['sigma_0'] * F - R * Phi_cap(t))
        deta = -p['alpha'] * Sig * eta
        dR = (-p['nu_0'] * R - p['nu'] * Sig * R
              + p['rho'] * g_E * (1.0 - Sig) * (1.0 - R))
        return [dE, dSig, deta, dR]

    def event_R(t, y):
        return y[3] - p['R_star']
    event_R.terminal = True
    event_R.direction = -1

    def event_E(t, y):
        return y[0] - p['E_star']
    event_E.terminal = True
    event_E.direction = -1

    sol = solve_ivp(
        system, [0, T_MAX], [c['E0'], c['Sigma0'], c['eta0'], c['R0']],
        method='RK45', events=(event_R, event_E),
        rtol=1e-7, atol=1e-9, max_step=0.5,
    )

    tR = sol.t_events[0][0] if sol.t_events[0].size > 0 else np.inf
    tE = sol.t_events[1][0] if sol.t_events[1].size > 0 else np.inf

    if tR == np.inf and tE == np.inf:
        if return_channel:
            return float(T_MAX), 'NONE', float(sol.y[0][-1]), float(sol.y[3][-1])
        return float(T_MAX)

    if tR < tE:
        t_B, ch = tR, 'R'
    else:
        t_B, ch = tE, 'E'

    if return_channel:
        return float(t_B), ch, float(sol.y[0][-1]), float(sol.y[3][-1])
    return float(t_B)


# ============================================================================
# 3. nu_0 SCAN with channel attribution
#    (R-channel parameters scaled together: nu_0 and nu by the same factor,
#     because biologically R-channel suppression in long-lived species affects
#     both spontaneous and damage-driven decay of the repair reserve.)
# ============================================================================

print("=" * 78)
print("  VBLS — REGIME SENSITIVITY:  scan over R-channel (nu_0, nu)")
print("=" * 78)
print(f"  Baseline (Homo sapiens): nu_0 = {BASE['nu_0']:.4f}, nu = {BASE['nu']:.4f} yr^-1")
print(f"  T_MAX integration horizon: {T_MAX:.0f} yr")
print(f"  Scan mode: both nu_0 and nu scaled together by factor k\n")

# Multiplicative factors applied to both nu_0 and nu simultaneously.
scale_factors = np.array([
    1.0,        # baseline (human)
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.01,
    0.003,
    0.001,
    0.0003,
])
nu0_grid = BASE['nu_0'] * scale_factors
nu_grid  = BASE['nu']  * scale_factors

print(f"  {'factor k':>8}  {'nu_0':>10}  {'nu':>10}  {'t_B (yr)':>9}  "
      f"{'channel':>7}  {'E_end':>7}  {'R_end':>7}")
print("  " + "-" * 74)

scan_results = []
t0 = time.time()
for k, nu0, nu_v in zip(scale_factors, nu0_grid, nu_grid):
    tB, ch, Eend, Rend = run_sim(
        params={'nu_0': nu0, 'nu': nu_v}, return_channel=True)
    scan_results.append({'k': k, 'nu_0': nu0, 'nu': nu_v, 't_B': tB,
                          'channel': ch, 'E_end': Eend, 'R_end': Rend})
    print(f"  {k:>8.4f}  {nu0:>10.2e}  {nu_v:>10.2e}  {tB:>9.1f}  "
          f"{ch:>7s}  {Eend:>7.4f}  {Rend:>7.4f}")
print(f"\n  Scan completed in {time.time()-t0:.1f} s")


# ============================================================================
# 4. SENSITIVITY OF t_B TO (mu, lam, E*) AT EACH nu_0
# ============================================================================

PARAMS_E = ['mu', 'lam', 'E_star']
PARAM_LABELS_E = {
    'mu': r'$\mu$',
    'lam': r'$\lambda$',
    'E_star': r'$E^*$',
}

print(f"\n{'=' * 78}")
print("  Tornado for (mu, lam, E*) at each nu_0  (perturbation +/-50%)")
print(f"{'=' * 78}")
print(f"  {'k':>6}  {'nu_0':>10}  {'t_B':>7}  {'ch':>3}  "
      f"{'sens_mu':>8}  {'sens_lam':>8}  {'sens_E*':>8}")
print("  " + "-" * 64)

sensitivity_grid = {p: [] for p in PARAMS_E}
tB_baseline = []
channels = []

t0 = time.time()
for k, nu0, nu_v in zip(scale_factors, nu0_grid, nu_grid):
    base_tB, base_ch, _, _ = run_sim(
        params={'nu_0': nu0, 'nu': nu_v}, return_channel=True)
    tB_baseline.append(base_tB)
    channels.append(base_ch)
    row_print = f"  {k:>6.4f}  {nu0:>10.2e}  {base_tB:>7.1f}  {base_ch:>3s}"
    for p in PARAMS_E:
        bv = BASE[p]
        t_lo = run_sim(params={'nu_0': nu0, 'nu': nu_v, p: bv * 0.5})
        t_hi = run_sim(params={'nu_0': nu0, 'nu': nu_v, p: bv * 1.5})
        sens = abs(t_hi - t_lo) / max(base_tB, 1e-9) * 100.0
        sensitivity_grid[p].append(sens)
        row_print += f"  {sens:>7.1f}%"
    print(row_print)
print(f"\n  Sensitivity grid completed in {time.time()-t0:.1f} s")


# ============================================================================
# 4b. SECOND SCAN: damage production rate (sigma_0)
#     Tests whether low damage production (the Cagan signature of naked mole-rat)
#     also drives the system into the E-channel regime — or whether different
#     "long-lived" species have different mechanistic profiles.
# ============================================================================

print(f"\n{'=' * 78}")
print("  SECOND SCAN: damage production (sigma_0)")
print(f"{'=' * 78}")
print(f"  Baseline (Homo sapiens): sigma_0 = {BASE['sigma_0']:.4f}")
print(f"  Scan mode: sigma_0 scaled by factor k (nu_0, nu kept at baseline)\n")

sigma_factors = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01, 0.003])

print(f"  {'k':>8}  {'sigma_0':>10}  {'t_B':>7}  {'ch':>3}  "
      f"{'sens_mu':>8}  {'sens_lam':>8}  {'sens_E*':>8}  {'E_end':>7}  {'R_end':>7}")
print("  " + "-" * 84)

sigma_results = []
sens_sigma_grid = {p: [] for p in PARAMS_E}
t0 = time.time()
for k in sigma_factors:
    s0 = BASE['sigma_0'] * k
    base_tB, base_ch, Eend, Rend = run_sim(
        params={'sigma_0': s0}, return_channel=True)
    sigma_results.append({'k': k, 'sigma_0': s0, 't_B': base_tB,
                          'channel': base_ch, 'E_end': Eend, 'R_end': Rend})
    row_print = f"  {k:>8.4f}  {s0:>10.2e}  {base_tB:>7.1f}  {base_ch:>3s}"
    for p in PARAMS_E:
        bv = BASE[p]
        t_lo = run_sim(params={'sigma_0': s0, p: bv * 0.5})
        t_hi = run_sim(params={'sigma_0': s0, p: bv * 1.5})
        sens = abs(t_hi - t_lo) / max(base_tB, 1e-9) * 100.0
        sens_sigma_grid[p].append(sens)
        row_print += f"  {sens:>7.1f}%"
    row_print += f"  {Eend:>7.4f}  {Rend:>7.4f}"
    print(row_print)
print(f"\n  sigma_0 scan completed in {time.time()-t0:.1f} s")


# ============================================================================
# 5. FIGURE
# ============================================================================

BG = '#f7f9fc'
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig.suptitle(
    'VBLS — Regime sensitivity: two independent paths into the E-channel\n'
    r'(Top: R-channel suppression $\nu_0, \nu \to k\,\nu_0, k\,\nu$. '
    r'Bottom: damage suppression $\sigma_0 \to k\,\sigma_0$.)',
    fontsize=12, fontweight='bold', color='#1a252f', y=0.995
)

colors_map = {'R': '#c0392b', 'E': '#27ae60', 'NONE': '#7f8c8d'}
colors_p = {'mu': '#2980b9', 'lam': '#8e44ad', 'E_star': '#d35400'}

# ----------------------------------------------------------------------
# Row 1: R-channel scan
# ----------------------------------------------------------------------
ax = axes[0, 0]
ax.set_facecolor(BG)
channel_arr = np.array(channels)
tB_arr = np.array(tB_baseline)
for ch_name in ['R', 'E', 'NONE']:
    mask = channel_arr == ch_name
    if mask.any():
        ax.scatter(scale_factors[mask], tB_arr[mask],
                   c=colors_map[ch_name], s=110, edgecolor='black',
                   linewidth=0.6, zorder=3,
                   label=f'{ch_name}-channel' if ch_name != 'NONE'
                         else 'No death within T_MAX')
ax.plot(scale_factors, tB_arr, 'k-', alpha=0.25, lw=1.0, zorder=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k$  (joint scaling of $\nu_0, \nu$)', fontsize=10)
ax.set_ylabel(r'$t_B$  (yr)', fontsize=10)
ax.set_title('(A)  R-channel scan: lifespan and dominant channel', fontsize=10)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9, loc='best', framealpha=0.85)
ax.axvline(1.0, color='#2c3e50', ls=':', lw=1.2, alpha=0.7)

ax = axes[0, 1]
ax.set_facecolor(BG)
for p in PARAMS_E:
    ax.plot(scale_factors, sensitivity_grid[p], 'o-',
            color=colors_p[p], lw=2.0, markersize=7,
            label=PARAM_LABELS_E[p])
ax.axhline(5.0, color='gray', ls='--', lw=1.2, label='5% threshold')
ax.axvline(1.0, color='#2c3e50', ls=':', lw=1.2, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel(r'$k$  (joint scaling of $\nu_0, \nu$)', fontsize=10)
ax.set_ylabel(r'Sensitivity (%)', fontsize=10)
ax.set_title(r'(B)  Identifiability of $\mu, \lambda, E^*$ in R-channel scan',
             fontsize=10)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9, loc='best', framealpha=0.85)

# ----------------------------------------------------------------------
# Row 2: sigma_0 scan
# ----------------------------------------------------------------------
sigma_tB = np.array([r['t_B'] for r in sigma_results])
sigma_ch = np.array([r['channel'] for r in sigma_results])

ax = axes[1, 0]
ax.set_facecolor(BG)
for ch_name in ['R', 'E', 'NONE']:
    mask = sigma_ch == ch_name
    if mask.any():
        ax.scatter(sigma_factors[mask], sigma_tB[mask],
                   c=colors_map[ch_name], s=110, edgecolor='black',
                   linewidth=0.6, zorder=3,
                   label=f'{ch_name}-channel' if ch_name != 'NONE'
                         else 'No death within T_MAX')
ax.plot(sigma_factors, sigma_tB, 'k-', alpha=0.25, lw=1.0, zorder=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$k$  (scaling of $\sigma_0$)', fontsize=10)
ax.set_ylabel(r'$t_B$  (yr)', fontsize=10)
ax.set_title(r'(C)  $\sigma_0$ scan: lifespan and dominant channel', fontsize=10)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9, loc='best', framealpha=0.85)
ax.axvline(1.0, color='#2c3e50', ls=':', lw=1.2, alpha=0.7)

ax = axes[1, 1]
ax.set_facecolor(BG)
for p in PARAMS_E:
    ax.plot(sigma_factors, sens_sigma_grid[p], 'o-',
            color=colors_p[p], lw=2.0, markersize=7,
            label=PARAM_LABELS_E[p])
ax.axhline(5.0, color='gray', ls='--', lw=1.2, label='5% threshold')
ax.axvline(1.0, color='#2c3e50', ls=':', lw=1.2, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel(r'$k$  (scaling of $\sigma_0$)', fontsize=10)
ax.set_ylabel(r'Sensitivity (%)', fontsize=10)
ax.set_title(r'(D)  Identifiability of $\mu, \lambda, E^*$ in $\sigma_0$ scan',
             fontsize=10)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=9, loc='best', framealpha=0.85)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = 'regime_sensitivity.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nFigure saved: {out}")


# ============================================================================
# 6. SUMMARY
# ============================================================================

def _find_switch(channel_list):
    for i in range(1, len(channel_list)):
        if channel_list[i - 1] == 'R' and channel_list[i] == 'E':
            return i
    return None


print(f"\n{'=' * 78}")
print("  SUMMARY — two regime-switch pathways")
print(f"{'=' * 78}")

# Pathway 1: R-channel suppression
sw1 = _find_switch(channels)
print("\n  Pathway 1: joint R-channel suppression (nu_0, nu) -> k * (nu_0, nu)")
if sw1 is not None:
    print(f"    Channel switch R -> E between k = {scale_factors[sw1-1]:.4g} "
          f"and k = {scale_factors[sw1]:.4g}")
    print(f"    Lifespan plateau in E-regime: "
          f"~{np.array(tB_baseline)[channel_arr=='E'].mean():.0f} yr")
else:
    print("    No clear R -> E switch in this scan.")
for p in PARAMS_E:
    arr = np.array(sensitivity_grid[p])
    above = np.where(arr > 5.0)[0]
    if above.size > 0:
        print(f"    Sensitivity to {p:7s} > 5% first at k = {scale_factors[above[0]]:.4g}; "
              f"max = {arr.max():.1f}%")
    else:
        print(f"    Sensitivity to {p:7s} stays below 5% across this scan.")

# Pathway 2: damage suppression
sigma_channels = [r['channel'] for r in sigma_results]
sw2 = _find_switch(sigma_channels)
print("\n  Pathway 2: damage suppression sigma_0 -> k * sigma_0")
if sw2 is not None:
    print(f"    Channel switch R -> E between k = {sigma_factors[sw2-1]:.4g} "
          f"and k = {sigma_factors[sw2]:.4g}")
    e_lives = [r['t_B'] for r in sigma_results if r['channel'] == 'E']
    if e_lives:
        print(f"    Lifespan plateau in E-regime: ~{np.mean(e_lives):.0f} yr")
else:
    print("    No clear R -> E switch in this scan.")
for p in PARAMS_E:
    arr = np.array(sens_sigma_grid[p])
    above = np.where(arr > 5.0)[0]
    if above.size > 0:
        print(f"    Sensitivity to {p:7s} > 5% first at k = {sigma_factors[above[0]]:.4g}; "
              f"max = {arr.max():.1f}%")
    else:
        print(f"    Sensitivity to {p:7s} stays below 5% across this scan.")

print("\n  Interpretation:")
print("    The zero sensitivity of (mu, lam, E*) at human baseline (Table 8.3.1)")
print("    is a REGIME property of R-channel domination, not a structural defect.")
print("    Two biologically distinct mechanisms drive the system into the")
print("    E-channel and restore identifiability of E-channel parameters:")
print("      (1) suppression of R-channel decay (Bats, Greenland shark profile)")
print("      (2) suppression of damage production (Naked mole-rat profile, Cagan)")
print(f"{'=' * 78}")
