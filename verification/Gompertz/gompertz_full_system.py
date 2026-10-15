# SPDX-FileCopyrightText: 2026 Vasiliy Avdeev <vaavdeev64@gmail.com>
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================

  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
  SPDX-License-Identifier: CC-BY-NC-ND-4.0

================================================================================
VBLS — Fig. 16.2: the Gompertz law on the full system (S)
================================================================================

PURPOSE
    Check, on the full field of system (S) with the canonical human calibration
    (Section 17.1.1), the statement of Section 16.2.4: in the late approximation
    for the R-channel, with a rate nu_eff = nu_0 + nu*Sigma_late common to the
    population, the mortality rate is Gompertzian if and only if the reserve
    R_late is Weibull-distributed across the population, with slope
    b = k_W * nu_eff. System (S) is deterministic and does not generate the spread
    of the reserve: the spread is an assumption, and here it is set explicitly.

DESIGN
    1. One canonical trajectory from birth (parameters and fluxes of
       Section 17.1.1, as in verification/MAN/human_lifespan.py), continued past
       its death time so that Sigma(t) is defined at all ages.
    2. Three populations share E, Sigma and eta_abs of the canonical trajectory at
       the age t_late and differ in the reserve R(t_late):
         (W)   Weibull spread of R_late, shape k_W = 2, conditioned on R_late > R*,
               scale chosen so that the mean equals the canonical R(t_late);
         (L)   log-normal spread with the same mean and standard deviation of
               ln R_late as (W), on the same interval (R*, 1];
         (W+z) as (W), plus an individual factor z on the wear rates nu_0 and nu
               (gamma-distributed, mean 1, CV 0.3), i.e. a spread of nu_eff
               (Section 16.2.7).
    3. Every individual is integrated on the full field: both cut-offs of g(E)
       (eq. 4.2), max(0, 1 - Sigma) in the restoration term (eq. 4.1), no clipping
       of Sigma. Death is the first crossing R <= R* or E <= E* (Section 9.1).
    4. The mortality rate is estimated from the simulated death times by the
       life-table method; no analytic hazard is plotted as data. The straight
       line in panel C is a least-squares fit over the central 90% of the deaths.

WHAT IS AN ASSUMPTION
    The three spreads, the shape k_W = 2, the age t_late = 50 yr and the CV of z
    are choices of this illustration, not results of the theory.

Dependencies: numpy, scipy, matplotlib
Run:          python gompertz_full_system.py
Output:       gompertz_full_system.png in the working directory
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
    'lines.linewidth':   2.0,
    'figure.dpi':        150,
})

# ── Canonical calibration (Section 17.1.1; verification/MAN/human_lifespan.py) ──
mu, lam = 1.50, 0.30          # F_diss = mu*E^2 + lam*E
sigma_0 = 0.0162
alpha = 0.030
nu_0, nu, rho = 0.006, 0.060, 0.002
R_star, E_star = 0.15, 0.04
E0, Sigma0, eta0, R0 = 0.10, 0.01, 0.96, 0.99   # E0 is also the class parameter of g(E)
PHI_CAP = 0.003


def Phi_in(t):
    """Inflow of Section 17.1.1: growth to a plateau, gentle decline after age 60."""
    return (1.0 - np.exp(-t / 10.0)) * np.exp(-max(0.0, t - 60.0) / 30.0)


def F_diss(E):
    E = np.maximum(E, 0.0)                         # extended by zero for E < 0 (axiom A5)
    return mu * E**2 + lam * E


def g(E):
    return np.clip((E - E_star) / (E0 - E_star), 0.0, 1.0)   # eq. (4.2), both cut-offs


def field(t, E, S, eta, R, z):
    """Full field of system (S); z multiplies the wear rates nu_0 and nu."""
    Fd = F_diss(E)
    dE = eta * Phi_in(t) - Fd                                           # eq. (6.1)
    dS = np.maximum(0.0, sigma_0 * Fd - R * PHI_CAP)                    # eq. (7.1)
    deta = -alpha * S * eta                                             # eq. (5.1)
    dR = (-(z * nu_0) * R - (z * nu) * S * R
          + rho * g(E) * np.maximum(0.0, 1.0 - S) * (1.0 - R))          # eq. (4.1)
    return dE, dS, deta, dR


# ── 1. Canonical trajectory from birth, continued past its death time ─────────
T_LATE = 50.0


def rhs_scalar(t, y):
    E, S, eta, R = (np.array(v) for v in y)
    return [float(v) for v in field(t, E, S, eta, R, 1.0)]


def death_event(t, y):
    return min(y[3] - R_star, y[0] - E_star)


death_event.direction = -1

can = solve_ivp(rhs_scalar, [0.0, 125.0], [E0, Sigma0, eta0, R0], method='RK45',
                rtol=1e-9, atol=1e-11, max_step=0.02, events=death_event, dense_output=True)
t_B_can = can.t_events[0][0]
E_l, S_l, eta_l, R_l = can.sol(T_LATE)
print('Canonical trajectory: t_B = %.2f yr; at t_late = %.0f yr: E = %.3f, Sigma = %.3f, '
      'eta = %.3f, R = %.3f' % (t_B_can, T_LATE, E_l, S_l, eta_l, R_l))
print('Canonical Sigma: %.3f at 60, %.3f at 70, %.3f at t_B' % (
    can.sol(60.0)[1], can.sol(70.0)[1], can.sol(t_B_can)[1]))


# ── 2. Populations ────────────────────────────────────────────────────────────
rng = np.random.default_rng(20261015)
N = 100_000
K_W = 2.0


def weibull_from_uniform(u, k, scale):
    """Inverse transform for P(R > r | R > R*) = exp(-(r/scale)^k + (R*/scale)^k)."""
    return scale * ((R_star / scale) ** k - np.log(u)) ** (1.0 / k)


# scale r_W such that the mean of R_late (restricted to R <= 1) equals R(t_late); common random numbers
u_fix = rng.random(400_000)
lo, hi = 0.05, 2.0
for _ in range(50):
    mid = 0.5 * (lo + hi)
    r = weibull_from_uniform(u_fix, K_W, mid)
    if r[r <= 1.0].mean() < R_l:
        lo = mid
    else:
        hi = mid
R_W = 0.5 * (lo + hi)


def sample_weibull(n):
    out = np.empty(0)
    while out.size < n:
        r = weibull_from_uniform(rng.random(2 * n), K_W, R_W)
        out = np.concatenate([out, r[r <= 1.0]])
    return out[:n]


Rw = sample_weibull(N)
m_ln, s_ln = np.log(Rw).mean(), np.log(Rw).std()
Rl = np.empty(0)
while Rl.size < N:
    x = np.exp(rng.normal(m_ln, s_ln, 2 * N))
    Rl = np.concatenate([Rl, x[(x > R_star) & (x <= 1.0)]])
Rl = Rl[:N]

CV_Z = 0.3
z = rng.gamma(1.0 / CV_Z**2, CV_Z**2, N)

print('Weibull spread: k_W = %.1f, r_W = %.4f; R_late mean %.4f (canonical %.4f), sd %.4f; '
      'share above 1 before restriction %.4f'
      % (K_W, R_W, Rw.mean(), R_l, Rw.std(), np.mean(weibull_from_uniform(u_fix, K_W, R_W) > 1.0)))
print('Log-normal spread: ln R mean %.4f, sd %.4f; R_late mean %.4f, sd %.4f; ln R after truncation '
      'mean %.4f, sd %.4f' % (m_ln, s_ln, Rl.mean(), Rl.std(), np.log(Rl).mean(), np.log(Rl).std()))


# ── 3. Integration on the full field (vectorized RK4) ─────────────────────────
DT, T_END = 0.01, 125.0


def rk4_step(t, E, S, eta, R, zf):
    k1 = field(t, E, S, eta, R, zf)
    k2 = field(t + DT / 2, E + DT / 2 * k1[0], S + DT / 2 * k1[1], eta + DT / 2 * k1[2], R + DT / 2 * k1[3], zf)
    k3 = field(t + DT / 2, E + DT / 2 * k2[0], S + DT / 2 * k2[1], eta + DT / 2 * k2[2], R + DT / 2 * k2[3], zf)
    k4 = field(t + DT, E + DT * k3[0], S + DT * k3[1], eta + DT * k3[2], R + DT * k3[3], zf)
    return (E + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            S + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            eta + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            R + DT / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]))


def simulate(R_init, zfac):
    """Death time of every individual (first crossing R <= R* or E <= E*) and the channel."""
    n = R_init.size
    E = np.full(n, E_l); S = np.full(n, S_l); eta = np.full(n, eta_l); R = R_init.copy()
    zf = zfac.copy(); idx = np.arange(n)
    t_death = np.full(n, np.nan)
    channel = np.zeros(n, dtype=int)            # 1 = R-channel, 2 = E-channel
    t = T_LATE
    while t < T_END - 1e-9 and idx.size:
        En, Sn, etan, Rn = rk4_step(t, E, S, eta, R, zf)
        deadR = Rn <= R_star
        deadE = En <= E_star
        dead = deadR | deadE
        if dead.any():
            fr = np.where(deadR[dead], (R[dead] - R_star) / np.maximum(R[dead] - Rn[dead], 1e-15), 2.0)
            fe = np.where(deadE[dead], (E[dead] - E_star) / np.maximum(E[dead] - En[dead], 1e-15), 2.0)
            frac = np.clip(np.minimum(fr, fe), 0.0, 1.0)
            t_death[idx[dead]] = t + frac * DT
            channel[idx[dead]] = np.where(fr <= fe, 1, 2)
        keep = ~dead
        E, S, eta, R, zf, idx = En[keep], Sn[keep], etan[keep], Rn[keep], zf[keep], idx[keep]
        t += DT
    return t_death, channel


def paths(R_init):
    """Full-field reserve of a few individuals, continued until death."""
    n = R_init.size
    E = np.full(n, E_l); S = np.full(n, S_l); eta = np.full(n, eta_l); R = R_init.copy()
    t = T_LATE; ts_, Rs_ = [t], [R.copy()]
    alive = np.ones(n, dtype=bool)
    while t < T_END - 1e-9 and alive.any():
        E, S, eta, R = rk4_step(t, E, S, eta, R, np.ones(n))
        t += DT
        alive &= (R > R_star) & (E > E_star)
        ts_.append(t); Rs_.append(np.where(alive, R, np.nan))
    return np.array(ts_), np.array(Rs_)


ones = np.ones(N)
tW, chW = simulate(Rw, ones)
tL, chL = simulate(Rl, ones)
tZ, chZ = simulate(Rw, z)
for name, tt, ch in (('W', tW, chW), ('L', tL, chL), ('W+z', tZ, chZ)):
    print('Population %-4s: died %d of %d; R-channel %d, E-channel %d; median t_B = %.2f yr, '
          '5-95%%: %.1f-%.1f yr' % (name, np.isfinite(tt).sum(), tt.size, (ch == 1).sum(), (ch == 2).sum(),
                                   np.nanmedian(tt), np.nanpercentile(tt, 5), np.nanpercentile(tt, 95)))


# ── 4. Life tables ────────────────────────────────────────────────────────────
def life_table(tdeath, width=1.0, min_risk=200, min_d=10):
    tdeath = tdeath[np.isfinite(tdeath)]
    edges = np.arange(T_LATE, tdeath.max() + width, width)
    d, _ = np.histogram(tdeath, bins=edges)
    at_risk = tdeath.size - np.concatenate(([0], np.cumsum(d)[:-1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_hat = -np.log(1.0 - d / at_risk) / width
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = (at_risk >= min_risk) & (d >= min_d)
    return mid[ok], mu_hat[ok]


aW, mW = life_table(tW)
aL, mL = life_table(tL)
aZ, mZ = life_table(tZ)

# Gompertz fit over the central 90% of deaths of (W)
lo_a, hi_a = np.nanpercentile(tW, [5, 95])
win = (aW >= lo_a) & (aW <= hi_a)
b_fit, ln_a = np.polyfit(aW[win], np.log(mW[win]), 1)
resid = np.log(mW[win]) - (ln_a + b_fit * aW[win])

# nu_eff along the canonical trajectory (common Sigma): k_W * nu_eff(t)
ts = np.linspace(T_LATE, 120.0, 701)
nu_eff_t = nu_0 + nu * can.sol(ts)[1]


def local_slope(a, m, at, h=5.0):
    sel = (a >= at - h) & (a <= at + h)
    if sel.sum() < 4:
        return np.nan
    return np.polyfit(a[sel], np.log(m[sel]), 1)[0]


print('\nGompertz fit of (W) over ages %.1f-%.1f yr: b = %.4f per yr; rms residual of ln(mu) %.3f'
      % (lo_a, hi_a, b_fit, np.sqrt(np.mean(resid**2))))
print('k_W * nu_eff(t) along the canonical trajectory: %.4f at %.0f yr, %.4f at %.0f yr, %.4f at %.0f yr'
      % (K_W * nu_eff_t[0], ts[0], K_W * np.interp(t_B_can, ts, nu_eff_t), t_B_can,
         K_W * np.interp(95.0, ts, nu_eff_t), 95.0))
print('\nLocal slope d ln(mu)/dt (+-5 yr):')
print('  age |   (W)   |   (L)   |  (W+z)  | k_W*nu_eff')
for at in (55, 60, 65, 70, 75, 80, 85, 90, 95, 100):
    print('  %3d | %7.4f | %7.4f | %7.4f | %7.4f' % (
        at, local_slope(aW, mW, at), local_slope(aL, mL, at), local_slope(aZ, mZ, at),
        K_W * np.interp(at, ts, nu_eff_t)))

# late approximation for (W): R = R_late * exp(-N(t)), N = integral of nu_eff along the canonical Sigma,
# so mu(t) = k_W * nu_eff(t) * (R*/r_W)^k_W * exp(k_W * N(t)) while R* exp(N(t)) <= 1 (Section 16.2.4)
N_t = np.concatenate(([0.0], np.cumsum(0.5 * (nu_eff_t[1:] + nu_eff_t[:-1]) * np.diff(ts))))
ok_late = R_star * np.exp(N_t) < 1.0
mu_late = K_W * nu_eff_t * (R_star / R_W) ** K_W * np.exp(K_W * N_t)
print('Late approximation for (W): mu(%.0f) = %.4f, mu(77) = %.4f; full system: mu(%.1f) = %.4f, mu(%.1f) = %.4f'
      % (ts[0], mu_late[0], np.interp(77.0, ts, mu_late), aW[0], mW[0],
         aW[np.argmin(np.abs(aW - 77.0))], mW[np.argmin(np.abs(aW - 77.0))]))

# share of restoration in the reserve balance along the canonical trajectory
for at in (50, 60, 70, t_B_can):
    E_, S_, eta_, R_ = can.sol(at)
    wear = (nu_0 + nu * S_) * R_
    rest = rho * g(E_) * max(0.0, 1.0 - S_) * (1.0 - R_)
    print('  canonical %5.1f yr: restoration / wear = %.3f' % (at, rest / wear))


# ── 5. Figure ─────────────────────────────────────────────────────────────────
cW, cL, cZ = '#1f5fa8', '#c0392b', '#2e8b57'
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.subplots_adjust(hspace=0.40, wspace=0.30)

# (A) reserve of selected individuals: full system vs late approximation
ax = axes[0, 0]
q_idx = np.argsort(Rw)[(np.array([0.05, 0.25, 0.5, 0.75, 0.95]) * (N - 1)).astype(int)]
pt, pR = paths(Rw[q_idx])
S_t = can.sol(pt)[1]
cum_nu = np.concatenate(([0.0], np.cumsum(0.5 * ((nu_0 + nu * S_t[1:]) + (nu_0 + nu * S_t[:-1])) * np.diff(pt))))
cols = plt.cm.Blues(np.linspace(0.45, 0.95, len(q_idx)))
for k, col in enumerate(cols):
    ax.plot(pt, pR[:, k], color=col, lw=1.8)
    approx = Rw[q_idx[k]] * np.exp(-cum_nu)
    ax.plot(pt, np.where(approx >= R_star * 0.8, approx, np.nan), color=col, lw=1.0, ls='--')
ax.axhline(R_star, color='red', ls='--', lw=1.4)
ax.text(108, R_star + 0.012, r'$R^*$', color='red', ha='right', fontsize=10)
ax.set_xlim(T_LATE, 110.0)
ax.set_ylim(0, None)
ax.set_xlabel('Age (yr)')
ax.set_ylabel(r'Repair reserve $R(t)$')
ax.set_title('(A) Reserve: full system (solid)\nand late approximation (dashed)')

# (B) distributions of death times
ax = axes[0, 1]
bins = np.arange(T_LATE, np.nanmax(np.concatenate([tW, tL])) + 1.0, 1.0)
ax.hist(tW[np.isfinite(tW)], bins=bins, density=True, histtype='stepfilled', alpha=0.35, color=cW,
        label=r'Weibull spread of $R_{late}$')
ax.hist(tL[np.isfinite(tL)], bins=bins, density=True, histtype='step', lw=2.0, color=cL,
        label=r'log-normal spread of $R_{late}$')
ax.axvline(t_B_can, color='0.3', ls=':', lw=1.4, label='canonical $t_B$')
ax.set_xlabel(r'Death time $t_B$ (yr)')
ax.set_ylabel('Probability density')
ax.set_title('(B) Death times on the full system')
ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
ax.legend(fontsize=9, loc='upper right')

# (C) mortality rate: Weibull vs log-normal spread
ax = axes[1, 0]
ax.semilogy(aW, mW, 'o', color=cW, ms=4, label='Weibull spread')
ax.semilogy(aL, mL, 's', color=cL, ms=4, mfc='none', label='log-normal spread')
aa = np.linspace(lo_a, hi_a, 100)
ax.semilogy(aa, np.exp(ln_a + b_fit * aa), color=cW, lw=1.6, ls='--',
            label='Gompertz fit, $b=%.3f$ yr$^{-1}$' % b_fit)
ax.semilogy(ts[ok_late], mu_late[ok_late], color='0.25', lw=1.2, ls=':',
            label='late approximation (Weibull)')
ax.set_xlabel('Age (yr)')
ax.set_ylabel(r'Mortality rate $\mu(t)$ (yr$^{-1}$)')
ax.set_title('(C) Mortality rate (life tables)')
ax.legend(fontsize=9, loc='lower right')

# (D) a spread of nu_eff slows the growth of mortality
ax = axes[1, 1]
ax.semilogy(aW, mW, 'o', color=cW, ms=4, label=r'common $\nu_{eff}$')
ax.semilogy(aZ, mZ, '^', color=cZ, ms=4, label=r'spread of $\nu_{eff}$ (CV 0.3)')
ax.set_xlabel('Age (yr)')
ax.set_ylabel(r'Mortality rate $\mu(t)$ (yr$^{-1}$)')
ax.set_title(r'(D) A spread of $\nu_{eff}$ slows the growth of mortality')
ax.legend(fontsize=9, loc='lower right')

out = 'gompertz_full_system.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('\nSaved:', out)
