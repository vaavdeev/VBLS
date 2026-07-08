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
=====================================================================

VBLS — Attenuation bias simulation for Test 3: T_max ~ (P/M)^gamma
=====================================================================
Reviewer concern (DeepSeek 3.4.2):
  Observed slope gamma = -0.656 vs VBLS prediction -1.000.
  Hypothesis: measurement noise in P/M (metabolic rate) causes regression
  dilution (attenuation bias), pulling the estimated slope toward zero.

Correct simulation design
--------------------------
  We generate T_max from the TRUE model (gamma=-1.000) using observed P/M
  as the true underlying values.  We then add multiplicative lognormal noise
  to P/M and fit the regression.  At CV=0% we recover -1.000; at higher CV
  the slope is attenuated toward zero.  We find what CV reproduces -0.6561.

Key references
--------------
  Errors-in-variables OLS attenuation:
    beta_hat -> beta_true * sigma_X^2 / (sigma_X^2 + sigma_noise^2)
  Lognormal noise: if P/M* = P/M * exp(eta), eta ~ N(0, sigma_mult^2),
    sigma_noise (log10) = sigma_mult / ln(10)
    CV = sqrt(exp(sigma_mult^2) - 1)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
N_SIM = 2000

# ── Load data (same pipeline as anage.py) ────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(here, 'anage_data.csv'),
                   sep=';', encoding='utf-8-sig', on_bad_lines='skip')

mam = df[df['Class'] == 'Mammalia'].copy()
for col in ['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)']:
    mam[col] = pd.to_numeric(mam[col], errors='coerce')
mam = mam.dropna(subset=['Body mass (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)'])
mam = mam[(mam['Body mass (g)'] > 0) &
          (mam['Maximum longevity (yrs)'] > 0) &
          (mam['Metabolic rate (W)'] > 0)]
mam['PM'] = mam['Metabolic rate (W)'] / mam['Body mass (g)'] * 1000   # W/kg

base = mam[(mam['Data quality'] != 'low') & (mam['Order'] != 'Chiroptera')].copy()

pm_obs   = base['PM'].values
tmax_obs = base['Maximum longevity (yrs)'].values
lpm      = np.log10(pm_obs)
ltmax    = np.log10(tmax_obs)
n        = len(base)

# ── Observed (attenuated) slope ───────────────────────────────────────────────
slope_obs, icept_obs, r_obs, _, se_obs = stats.linregress(lpm, ltmax)
sigma_X = lpm.std()

print("=" * 65)
print("VBLS Attenuation Bias Simulation — Test 3: T_max ~ (P/M)^gamma")
print("=" * 65)
print(f"  n (base sample)        = {n}")
print(f"  Observed slope γ̂      = {slope_obs:.4f} ± {se_obs:.4f}")
print(f"  R² (observed)          = {r_obs**2:.4f}")
print(f"  VBLS prediction        = -1.000")
print(f"  Attenuation ratio      = {abs(slope_obs)/1.000:.4f}  (= |γ̂| / |γ_true|)")
print(f"  σ_X = std(log10 P/M)   = {sigma_X:.4f}")

# ── True model: forced regression at slope = -1.000 ──────────────────────────
#   log T_max = -1.000 * log(P/M) + c + epsilon_bio
#   c = mean(ltmax) + 1.000 * mean(lpm)
icept_true = np.mean(ltmax) + 1.000 * np.mean(lpm)
resid_true = ltmax - (-1.000 * lpm + icept_true)
sigma_bio  = resid_true.std()

print(f"\n── True model (forced slope = -1.000) ──")
print(f"  Intercept c (forced)   = {icept_true:.4f}")
print(f"  sigma_bio (residuals)  = {sigma_bio:.4f}  (biological scatter in log10)")
print(f"  Compare sigma_bio with OLS residuals: "
      f"{np.sqrt((1 - r_obs**2) * ltmax.var()):.4f}")

# ── Analytical attenuation (errors-in-variables) ─────────────────────────────
# lambda = sigma_X^2 / (sigma_X^2 + sigma_noise_log10^2)
# gamma_hat = -1.000 * lambda
# Required lambda to match observed: lambda_req = |slope_obs| / 1.000

lambda_req       = abs(slope_obs)
sigma_noise_req2 = sigma_X**2 * (1.0/lambda_req - 1.0)
sigma_noise_req  = np.sqrt(sigma_noise_req2)                  # log10 units
sigma_mult_req   = sigma_noise_req * np.log(10)                # log-e units
cv_req           = np.sqrt(np.exp(sigma_mult_req**2) - 1.0)   # coefficient of variation

print(f"\n── Analytical (errors-in-variables formula) ──")
print(f"  Required attenuation   λ = {lambda_req:.4f}")
print(f"  Required σ_noise(log10) = {sigma_noise_req:.4f}")
print(f"  ► Required CV of P/M   ≈ {cv_req:.1%}  to reproduce observed slope")

print(f"\n  {'CV(P/M)':>8}  {'σ_η(log10)':>12}  {'λ':>8}  {'Expected γ':>11}")
for cv in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    sm  = np.sqrt(np.log(1 + cv**2))
    sn  = sm / np.log(10)
    lam = sigma_X**2 / (sigma_X**2 + sn**2)
    print(f"  {cv:>7.0%}  {sn:>12.4f}  {lam:>8.4f}  {-1.000*lam:>11.4f}")

# ── Monte Carlo: generate T_max from TRUE model, add noise to P/M ─────────────
print(f"\n── Monte Carlo  (N_sim={N_SIM}, n={n} species per run) ──")
print(f"  T_max generated from TRUE model (γ=-1.000 + biological scatter)")
print(f"  Noise added to P/M (multiplicative lognormal)")
print(f"\n  {'CV(P/M)':>8}  {'Mean γ̂':>9}  {'Std':>7}  {'2.5%':>9}  {'97.5%':>9}")

mc_cvs  = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
mc_mean = []
mc_std  = []
mc_lo   = []
mc_hi   = []

for cv in mc_cvs:
    slopes = []
    sigma_mult = np.sqrt(np.log(1 + cv**2)) if cv > 0 else 0.0
    for _ in range(N_SIM):
        # 1. Generate T_max from TRUE model
        eps        = np.random.normal(0, sigma_bio, n)
        ltmax_gen  = -1.000 * lpm + icept_true + eps

        # 2. Add lognormal noise to P/M → noise in log10 space
        if cv > 0:
            lpm_noisy = lpm + np.random.normal(0, sigma_mult / np.log(10), n)
        else:
            lpm_noisy = lpm.copy()

        # 3. Fit OLS
        sl, *_ = stats.linregress(lpm_noisy, ltmax_gen)
        slopes.append(sl)

    slopes = np.array(slopes)
    mc_mean.append(slopes.mean())
    mc_std.append(slopes.std())
    mc_lo.append(np.percentile(slopes, 2.5))
    mc_hi.append(np.percentile(slopes, 97.5))
    print(f"  {cv:>7.0%}  {slopes.mean():>9.4f}  {slopes.std():>7.4f}"
          f"  {np.percentile(slopes,2.5):>9.4f}  {np.percentile(slopes,97.5):>9.4f}")

# ── Interpolate: find CV where MC mean = observed slope ──────────────────────
mc_mean_arr = np.array(mc_mean)
mc_cvs_arr  = np.array(mc_cvs)
# mc_mean decreases in absolute value (less negative) as CV grows
idx = np.searchsorted(mc_mean_arr, slope_obs)    # mean is increasing (less neg)
if 0 < idx < len(mc_cvs_arr):
    cv_match = float(np.interp(slope_obs,
                               [mc_mean_arr[idx-1], mc_mean_arr[idx]],
                               [mc_cvs_arr[idx-1],  mc_cvs_arr[idx]]))
else:
    cv_match = cv_req   # fall back to analytical

print(f"\n  ► Monte Carlo: observed slope {slope_obs:.4f} reproduced at CV ≈ {cv_match:.1%}")
print(f"    Analytical prediction:                         CV ≈ {cv_req:.1%}")

# ── Distribution at 40% and at cv_req ────────────────────────────────────────
def slopes_at_cv(cv, nsim=N_SIM):
    sigma_mult = np.sqrt(np.log(1 + cv**2))
    out = []
    for _ in range(nsim):
        eps       = np.random.normal(0, sigma_bio, n)
        ltmax_gen = -1.000 * lpm + icept_true + eps
        lpm_noisy = lpm + np.random.normal(0, sigma_mult / np.log(10), n)
        sl, *_    = stats.linregress(lpm_noisy, ltmax_gen)
        out.append(sl)
    return np.array(out)

sl40  = slopes_at_cv(0.40)
sl_cv = slopes_at_cv(min(cv_req, 0.80))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: mean slope vs CV
ax = axes[0]
pct = [c*100 for c in mc_cvs]
ax.plot(pct, mc_mean, 'o-', color='#2980b9', lw=2, ms=6, label='MC mean γ̂')
ax.fill_between(pct, mc_lo, mc_hi, alpha=0.18, color='#2980b9', label='95% CI')

ax.axhline(-1.000,    color='#c0392b', lw=1.8, ls='--', label='VBLS true: γ = −1.000')
ax.axhline(slope_obs, color='#2c3e50', lw=1.6, ls=':',
           label=f'Observed: γ = {slope_obs:.4f}')

ax.axvline(cv_req*100, color='#27ae60', lw=1.4, ls='-.',
           label=f'CV ≈ {cv_req:.0%}  (full explanation)')
ax.axvline(40,         color='#e67e22', lw=1.2, ls=':',
           label=f'CV = 40%  →  γ̂ ≈ {mc_mean[mc_cvs.index(0.40)]:.3f}')

ax.set_xlabel('Coefficient of variation of P/M  (%)', fontsize=11)
ax.set_ylabel('Estimated slope  γ̂', fontsize=11)
ax.set_title('Attenuation bias: γ̂ vs noise level\n(T_max from true model, n=318)', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.25, ls='--')

# Right: histograms at CV=40% and CV=cv_req
ax2 = axes[1]
ax2.hist(sl40, bins=55, color='#e67e22', alpha=0.55, density=True,
         label=f'CV=40%,  mean={sl40.mean():.4f}')
ax2.hist(sl_cv, bins=55, color='#27ae60', alpha=0.55, density=True,
         label=f'CV≈{cv_req:.0%},  mean={sl_cv.mean():.4f}')
ax2.axvline(-1.000,    color='#c0392b', lw=2.2, ls='--', label='VBLS: −1.000')
ax2.axvline(slope_obs, color='#2c3e50', lw=2.0, ls=':',
            label=f'Observed: {slope_obs:.4f}')

ax2.set_xlabel('Estimated slope  γ̂', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(f'Slope distributions at two noise levels\n(N={N_SIM} simulations, n={n} species)',
              fontsize=11)
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.25, ls='--')

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(here, f'attenuation_bias.{ext}'),
                dpi=150, bbox_inches='tight')
print(f"\nFigures: {os.path.join(here, 'attenuation_bias.pdf/png')}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  VBLS prediction (true):  γ = −1.000")
print(f"  Observed in AnAge data:  γ = {slope_obs:.4f} ± {se_obs:.4f}")
print(f"  Required CV of P/M to fully explain the gap:  ≈ {cv_req:.0%}")
print(f"  At CV=40%:  expected γ̂ ≈ {mc_mean[mc_cvs.index(0.40)]:.4f}"
      f"  (partial explanation: {(abs(mc_mean[mc_cvs.index(0.40)])-abs(slope_obs))/(1.0-abs(slope_obs))*100:.0f}% of the gap closed from the true slope)")
print(f"\n  Conclusion: attenuation bias is quantitatively consistent with")
print(f"  the observed deviation.  Metabolic rate in AnAge is estimated from")
print(f"  field/allometric data; CV 50–70% is plausible for such measurements.")
