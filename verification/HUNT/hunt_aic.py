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
hunt_aic.py
VBLS — Appendix D.2.9 — AIC comparison
The mapping 1 - k*DeltaSigma(t)^beta through Sigma(t) vs a simple exponential and alternatives.
The memory mechanism eta_abs ~ exp(-alpha*int(Sigma)) itself is not tested by this comparison:
a direct identification of eta_abs with VO2max (M0, no fitted parameters) gives RMSE 0.058, as the
exponential does, yet by AICc it is on a par with M1, so these data do not separate the shares of
the memory and damage-level components of the VO2max decline (App. D.2.4, D.2.9).

Models:
  M0  Direct        eta_abs(t) / eta_abs(25)           0 params (eta_abs from the same ODE and calibration)
  M1  VBLS          1 - k * DeltaSigma(t)^beta        2 params (k, beta)
                    DeltaSigma(t) from ODE, pre-calibrated — NOT fit to HUNT
  M2  Exp-time      exp(-lambda*(t-25))                1 param  (lambda)
  M3  Exp-mapped    1 - k * (exp(lambda*(t-25))-1)^beta 3 params (k, beta, lambda)
  M4  Power-law     (25/t)^s                           1 param  (s)

Selection criterion: AICc (Akaike corrected for small samples)
Data: HUNT4 men — 7 age groups (25, 35, 45, 55, 65, 75, 85 yr)
      men's decade declines from Letnes, Nes & Wisloff (2023) fig.3,
      redrawing Letnes et al. (2020) Prog.Cardiovasc.Dis. 63(6):730-737
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── VBLS parameters (human calibration, April 2026) ───────────────────────────
MU, LAM = 1.50, 0.30
SIGMA0  = 0.0162
ALPHA   = 0.030
NU0, NU = 0.006, 0.060
RHO     = 0.002
E_STAR  = 0.04
E0      = 0.10
E_RANGE = E0 - E_STAR

# ── External functions ─────────────────────────────────────────────────────────
def Phi_in(t):
    growth  = 1.0 - np.exp(-t / 10.0)
    decline = np.exp(-max(0.0, t - 60.0) / 30.0)
    return growth * decline

def Phi_cap(t):
    baseline    = 0.003
    return baseline

# ── VBLS ODE simulation ────────────────────────────────────────────────────────
def vbls_run(t_end=140, n=15000):
    def rhs(t, y):
        E = max(y[0], 0.0)
        S = max(y[1], 0.0)
        e = max(min(y[2], 1.0), 0.0)
        R = max(min(y[3], 1.0), 0.0)
        F = MU * E**2 + LAM * E
        g_E = max(0.0, min(1.0, (E - E_STAR) / E_RANGE))
        return [
            e * Phi_in(t) - F,
            max(0.0, SIGMA0 * F - R * Phi_cap(t)),
            -ALPHA * S * e,
            -NU0 * R - NU * S * R + RHO * g_E * (1.0 - S) * (1.0 - R),
        ]
    return solve_ivp(rhs, (0, t_end), [E0, 0.01, 0.96, 0.99],
                     t_eval=np.linspace(0, t_end, n),
                     method='RK45', rtol=1e-8, atol=1e-10)

print("Running VBLS simulation...")
sol  = vbls_run()
t_   = sol.t
Sig_ = sol.y[1]
eta_ = sol.y[2]
print("  Done.")

# ── HUNT4 data ────────────────────────────────────────────────────────────
AGES    = np.array([25, 35, 45, 55, 65, 75, 85], dtype=float)
N_DATA  = len(AGES)

# Ten-year decline per age decade, read off figure 3 (right panel) of
# Letnes, Nes & Wisloff, Int.J.Cardiol.Cardiovasc.Risk Prev. 16:200171
# (2023), which redraws the HUNT trajectories of Letnes et al. (2020)
# separately by sex. That figure gives the annual decline in percent;
# the decade value is ten times it, as the review's own text confirms
# (men after 70 approach 25% per decade in both HUNT and BLSA).
d_men_pct = np.array([5.0, 5.6, 10.4, 12.8, 15.5, 24.5])

def cumulative(d_pct):
    traj = [1.0]
    for d in d_pct:
        traj.append(traj[-1] * (1 - d / 100))
    return np.array(traj)

VO2_OBS = cumulative(d_men_pct)
print(f"\nHUNT4 men:")
for a, v in zip(AGES, VO2_OBS):
    print(f"  age {int(a):3d}: {v:.4f}")

# ── AICc helper ───────────────────────────────────────────────────────────────
def aicc(rss, n, p):
    """
    AICc = n*ln(RSS/n) + 2K + 2K(K+1)/(n-K-1),  K = p + 1
    Gaussian MLE; RSS = sum of squared residuals. In the least-squares form the
    residual variance is itself an estimated parameter and is counted in K
    (Burnham & Anderson 2002, p.63), hence K = p + 1 for p structural
    parameters. At n = 7 the correction term is nonlinear in K, so omitting the
    variance does not cancel between models of unequal p.
    """
    K = p + 1
    aic = n * np.log(rss / n) + 2 * K
    corr = 2 * K * (K + 1) / max(n - K - 1, 1)
    return aic + corr

# ── M1: VBLS ──────────────────────────────────────────────────────────────────
Sig25 = float(np.interp(25, t_, Sig_))
eta25 = float(np.interp(25, t_, eta_))

def dsig(ages):
    return np.array([max(float(np.interp(a, t_, Sig_)) - Sig25, 0.0) for a in ages])

def m1_pred(k, b, ages=AGES):
    ds = dsig(ages)
    return np.array([max(1.0 - k * s**b, 0.0) for s in ds])

def m1_loss(p):
    return np.sum((m1_pred(*p) - VO2_OBS)**2)

r1   = minimize(m1_loss, [0.39, 0.43], method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
k1, b1 = r1.x
pred1  = m1_pred(k1, b1)
rss1   = float(np.sum((pred1 - VO2_OBS)**2))
rmse1  = float(np.sqrt(rss1 / N_DATA))
aic1   = aicc(rss1, N_DATA, 2)

# ── M2: pure exponential in time ──────────────────────────────────────────────
def m2_pred(lam, ages=AGES):
    return np.exp(-lam * (ages - 25))

def m2_loss(p):
    return np.sum((m2_pred(p[0]) - VO2_OBS)**2)

r2    = minimize(m2_loss, [0.007], method='Nelder-Mead',
                 options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
lam2  = float(r2.x[0])
pred2 = m2_pred(lam2)
rss2  = float(np.sum((pred2 - VO2_OBS)**2))
rmse2 = float(np.sqrt(rss2 / N_DATA))
aic2  = aicc(rss2, N_DATA, 1)

# ── M3: mapped exponential (same form as M1, free lambda) ─────────────────────
def m3_pred(k, b, lam, ages=AGES):
    x = np.array([max(np.exp(lam * (a - 25)) - 1.0, 0.0) for a in ages])
    return np.array([max(1.0 - k * xi**b, 0.0) for xi in x])

def m3_loss(p):
    return np.sum((m3_pred(*p) - VO2_OBS)**2)

r3       = minimize(m3_loss, [0.20, 0.40, 0.01], method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
k3, b3, l3 = r3.x
pred3    = m3_pred(k3, b3, l3)
rss3     = float(np.sum((pred3 - VO2_OBS)**2))
rmse3    = float(np.sqrt(rss3 / N_DATA))
aic3     = aicc(rss3, N_DATA, 3)

# ── M4: power law ─────────────────────────────────────────────────────────────
def m4_pred(gamma, ages=AGES):
    return np.array([(25.0 / a)**gamma for a in ages])

def m4_loss(p):
    return np.sum((m4_pred(p[0]) - VO2_OBS)**2)

r4    = minimize(m4_loss, [0.15], method='Nelder-Mead',
                 options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
gam4  = float(r4.x[0])
pred4 = m4_pred(gam4)
rss4  = float(np.sum((pred4 - VO2_OBS)**2))
rmse4 = float(np.sqrt(rss4 / N_DATA))
aic4  = aicc(rss4, N_DATA, 1)

# ── M0: direct identification VO2max/VO2max(25) = eta_abs(t)/eta_abs(25) ──────
# No fitted parameters: eta_abs(t) comes from the same ODE and calibration as M1.
pred0 = np.array([float(np.interp(a, t_, eta_)) / eta25 for a in AGES])
rss0  = float(np.sum((pred0 - VO2_OBS)**2))
rmse0 = float(np.sqrt(rss0 / N_DATA))
aic0  = aicc(rss0, N_DATA, 0)

# ── Summary table ─────────────────────────────────────────────────────────────
models = [
    ("M0 Direct eta_abs (no params)", 0, rss0, rmse0, aic0, pred0),
    ("M1 VBLS (k, beta)",         2, rss1, rmse1, aic1, pred1),
    ("M2 Exp-time (lambda)",       1, rss2, rmse2, aic2, pred2),
    ("M3 Exp-mapped (k, beta, l)", 3, rss3, rmse3, aic3, pred3),
    ("M4 Power-law (s)",       1, rss4, rmse4, aic4, pred4),
]
best_aic = min(m[4] for m in models)

print("\n" + "="*72)
print(f"  AIC comparison  |  HUNT4 men (n={N_DATA})  |  baseline t_B = 76.9 yr")
print("="*72)
print(f"  {'Model':<35} {'p':>3} {'RMSE':>8} {'AICc':>9} {'ΔAICc':>8}")
print("-"*72)
for name, p, rss, rmse, aic, _ in models:
    print(f"  {name:<35} {p:>3} {rmse:>8.5f} {aic:>9.2f} {aic-best_aic:>8.2f}")
print("="*72)
print(f"\n  Best model by AICc: {models[np.argmin([m[4] for m in models])][0]}")
print(f"\n  Key comparisons (relative to M1 VBLS):")
print(f"  M2 vs M1: ΔAICc = {aic2-aic1:.1f}  {'→ no support for M2' if aic2-aic1 > 10 else '→ weak evidence' if aic2-aic1 > 4 else ''}")
print(f"  M3 vs M1: ΔAICc = {aic3-aic1:.1f}  (M3 has 1 extra param vs M1; check for degeneracy)")
print(f"  M0 vs M1: ΔAICc = {aic0-aic1:.1f}  (M0 has no fitted parameters)")
print(f"\n  M1 params: k={k1:.4f}, beta={b1:.4f}")
print(f"  M2 params: lambda={lam2:.8f} yr-1")
print(f"  M3 params: k={k3:.4f}, beta={b3:.4f}, lambda={l3:.10f} yr-1  [A=k*lam^b={k3*l3**b3:.6f}]")
print(f"  M4 params: s={gam4:.4f}")

# ── M3b: explicit power-law in (t-25) — what M3 degenerates to ────────────────
# M3b: 1 - A*(t-25)^beta  (2 params)
def m3b_pred(A, b, ages=AGES):
    return np.array([max(1.0 - A * (a - 25)**b, 0.0) for a in ages])

def m3b_loss(p):
    return np.sum((m3b_pred(*p) - VO2_OBS)**2)

r3b       = minimize(m3b_loss, [0.001, 1.5], method='Nelder-Mead',
                     options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
A3b, b3b  = r3b.x
pred3b    = m3b_pred(A3b, b3b)
rss3b     = float(np.sum((pred3b - VO2_OBS)**2))
rmse3b    = float(np.sqrt(rss3b / N_DATA))
aic3b     = aicc(rss3b, N_DATA, 2)

print(f"\nM3b Time-power-law (A, beta): A={A3b:.6f}, beta={b3b:.4f}")
print(f"  RSS={rss3b:.6f}, RMSE={rmse3b:.4f}, AICc={aic3b:.2f}")

# Confirm M3 degeneracy: compare M3 prediction vs M3b prediction
print(f"\n  M3 vs M3b degeneracy check (should be near-identical):")
for a, p3, p3b_ in zip(AGES, pred3, pred3b):
    print(f"  age {int(a):3d}: M3={p3:.4f}  M3b={p3b_:.4f}  diff={abs(p3-p3b_):.6f}")

# Full comparison including M3b
all_models = models + [("M3b Time-power (A,beta)", 2, rss3b, rmse3b, aic3b, pred3b)]
best_aic2  = min(m[4] for m in all_models)
print("\n" + "="*72)
print(f"  AIC comparison (including M3b)  |  HUNT4 men (n={N_DATA})")
print("="*72)
print(f"  {'Model':<35} {'p':>3} {'RMSE':>8} {'AICc':>9} {'ΔAICc':>8}")
print("-"*72)
for name, p, rss, rmse, aic, _ in all_models:
    print(f"  {name:<35} {p:>3} {rmse:>8.5f} {aic:>9.2f} {aic-best_aic2:>8.2f}")
print("="*72)

# ── eta_abs shape analysis ─────────────────────────────────────────────────────
# Compare cumulative integral of Sigma vs linear time
print(f"\n  Shape of int(Sigma) vs linear t:")
for a in AGES:
    s = float(np.interp(a, t_, Sig_))
    # integral of Sigma from 25 to a using trapz
    mask = (t_ >= 25) & (t_ <= a)
    if mask.sum() > 1:
        int_s = float(np.trapz(Sig_[mask], t_[mask]))
    else:
        int_s = 0.0
    eta_vbls_val = float(np.interp(a, t_, eta_)) / eta25
    eta_simp_val = np.exp(-lam2 * (a - 25))
    print(f"  age {int(a):3d}: Sigma={s:.4f}  int(Sigma)={int_s:.4f}  "
          f"eta_VBLS={eta_vbls_val:.4f}  eta_simp={eta_simp_val:.4f}")

# ── Figure ────────────────────────────────────────────────────────────────────
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
    'figure.dpi':      150,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.11, wspace=0.36)

AGES_FINE = np.linspace(25, 90, 400)

# Panel A — model fits
ax = axes[0]
dsig_fine  = dsig(AGES_FINE)
vo2_m1     = np.array([max(1.0 - k1 * s**b1, 0.0) for s in dsig_fine])
vo2_m2     = m2_pred(lam2, AGES_FINE)
vo2_m3b    = m3b_pred(A3b, b3b, AGES_FINE)
vo2_m4     = m4_pred(gam4, AGES_FINE)

ax.plot(AGES,      VO2_OBS,  'ko',  ms=8, zorder=6, label='HUNT4 men (data)')
ax.plot(AGES_FINE, vo2_m1,   '-',   color='#8e44ad', lw=2.5,
        label=f'M1 VBLS        AICc={aic1:.1f}  ΔAICc={aic1-aic3b:.1f}')
ax.plot(AGES_FINE, vo2_m2,   '--',  color='#e74c3c', lw=1.8,
        label=f'M2 Exp-time    AICc={aic2:.1f}  ΔAICc={aic2-aic3b:.1f}')
ax.plot(AGES_FINE, vo2_m3b,  '-.',  color='#27ae60', lw=1.8,
        label=f'M3b Time-power AICc={aic3b:.1f}  ΔAICc={aic3b-aic3b:.1f}')
ax.plot(AGES_FINE, vo2_m4,   ':',   color='#e67e22', lw=1.8,
        label=f'M4 Power-law   AICc={aic4:.1f}  ΔAICc={aic4-aic3b:.1f}')
ax.set_xlabel('Age (yr)')
ax.set_ylabel('VO₂max / VO₂max(25 yr)')
ax.set_title('A.  Model fits to HUNT4 (men)\nAICc comparison', fontweight='bold')
ax.legend(fontsize=8.5, framealpha=0.95, loc='lower left')

# Panel B — eta_abs shape
ax = axes[1]
mask_plot = (t_ >= 25) & (t_ <= 90)
t_plot    = t_[mask_plot]
eta_norm  = eta_[mask_plot] / eta25

# Simple exponential best-fit to VO2max
eta_simp  = np.exp(-lam2 * (t_plot - 25))

# Pure memory-mechanism term exp(-alpha * int Sigma)
int_sig   = np.array([
    float(np.trapz(Sig_[(t_ >= 25) & (t_ <= a)],
                   t_[(t_ >= 25) & (t_ <= a)])) if np.sum((t_ >= 25) & (t_ <= a)) > 1 else 0.0
    for a in t_plot
])
eta_mem   = np.exp(-ALPHA * int_sig)

ax.plot(t_plot, eta_norm,  '-',  color='#8e44ad', lw=2.5,
        label=r'VBLS $\eta_{abs}(t)/\eta_0$ (from ODE)')
ax.plot(t_plot, eta_mem,   '--', color='#2980b9', lw=2.0,
        label=r'$\exp(-\alpha\int_{25}^t\Sigma\,d\tau)$  [memory term]')
ax.plot(t_plot, eta_simp,  ':',  color='#e74c3c', lw=2.0,
        label=f'Simple exp: $e^{{-\\lambda t}}$, $\\lambda$={lam2:.4f} yr⁻¹')

ax.set_xlabel('Age (yr)')
ax.set_ylabel('Normalized amplitude')
ax.set_title(r'B.  Shape of the $\eta_{abs}$ profile vs simple exponential (form only)'
             '\nSlower decline while Sigma is still small; not tested by the AICc comparison in A',
             fontweight='bold')
ax.legend(fontsize=8.5, framealpha=0.95)
ax.set_ylim(0.50, 1.05)

plt.savefig('hunt_aic.png', dpi=180, bbox_inches='tight')
plt.close()
print("\nSaved: hunt_aic.png")
