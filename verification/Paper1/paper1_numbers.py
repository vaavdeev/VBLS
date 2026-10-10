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
VBLS — Paper 1: numbers and figures.

Integrates the exact vector field of the canonical system (S) (Appendix A.1 of
the monograph): the restoration term carries max(0, 1 - Sigma), g(E) is clipped
to [0, 1], and no projection of the state onto [0, 1] is applied.

Memory is written in the kernel form eta = eta0 * exp(-alpha * H) with
dH/dt = Sigma - kappa * H. kappa = 0 is the permanent trace (K = 1, eq. 5.1);
kappa > 0 is an exponentially decaying kernel K(u) = exp(-kappa * u).

Outputs (printed and written to paper1_numbers.out.txt):
  1. illustrative trajectory: t_B, channel, state at t_B, branch t_B^E on the
     continuation, Sigma_crit, the explicit E-channel bound;
  2. memory-sensitivity scan t_B(alpha), both branches, Brent maximisation;
  3. memory-kernel example in a constant environment: permanent versus
     decaying kernel, same parameters and initial state; the bound (9) in this
     environment; a damage-free start of the canonical system;
  4. an example satisfying the conditions (P1)-(P4) of Proposition 3 (start at
     E(t_A) = 0.12): crossings of the production-export kink, finite-difference
     slopes of t_B(alpha) at zero, and the same slope from the variational system;
  5. the setting of 3 with Phi_cap alternating around the same mean: the
     worst-phase margin of Proposition 1(b) and the resulting lifetimes;
  6. the two counterexamples of Section 6 of the paper (also in Appendix G of
     the monograph): unsaturated restoration, where t_B(alpha) decreases from
     alpha = 0, and a maximum of t_B inside the reserve branch, at a kink of
     the field rather than at the switch of the death channel.
Figures: paper1_fig1.png and paper1_fig2.png, written next to this script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def save_fig(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=200)

# ---------------------------------------------------------------------------
# Illustrative parameter set (monograph, Section 17.1); time in model units
# ---------------------------------------------------------------------------
P = dict(mu=1.50, lam=0.30, sigma0=0.0162, alpha=0.030, nu0=0.006, nu=0.060,
         rho=0.002, R_star=0.15, E_star=0.04, E0=0.10)
IC = dict(E=0.10, S=0.01, eta=0.96, R=0.99)


def F(E, p):
    return p['mu'] * E * E + p['lam'] * E if E > 0.0 else 0.0


def g(E, p):
    return min(1.0, max(0.0, (E - p['E_star']) / (p['E0'] - p['E_star'])))


def phi_in_canonical(t):
    return (1.0 - np.exp(-t / 10.0)) * np.exp(-max(0.0, t - 60.0) / 30.0)


def rhs(t, y, p, phi_in, phi_cap, kappa):
    E, S, H, R = y
    eta = IC['eta'] * np.exp(-p['alpha'] * H)
    FE = F(E, p)
    dE = eta * phi_in(t) - FE
    dS = max(0.0, p['sigma0'] * FE - R * phi_cap(t))
    dH = S - kappa * H
    dR = -p['nu0'] * R - p['nu'] * S * R + p['rho'] * g(E, p) * max(0.0, 1.0 - S) * (1.0 - R)
    return [dE, dS, dH, dR]


def run(p, phi_in, phi_cap, T, kappa=0.0, max_step=0.05, rtol=1e-9, atol=1e-12, method='RK45', n_out=4000,
        log_grid=False, y0=None):
    if y0 is None:
        y0 = [IC['E'], IC['S'], 0.0, IC['R']]

    def eE(t, y):
        return y[0] - p['E_star']
    eE.direction = -1

    def eR(t, y):
        return y[3] - p['R_star']
    eR.direction = -1

    if log_grid:
        t_eval = np.minimum(np.concatenate(([0.0], np.logspace(-2, np.log10(T), n_out))), T)
    else:
        t_eval = np.linspace(0.0, T, n_out)
    sol = solve_ivp(lambda t, y: rhs(t, y, p, phi_in, phi_cap, kappa), [0.0, T], y0, method=method,
                    events=[eE, eR], rtol=rtol, atol=atol, max_step=max_step, t_eval=t_eval,
                    dense_output=True)
    tE = sol.t_events[0][0] if len(sol.t_events[0]) else np.inf
    tR = sol.t_events[1][0] if len(sol.t_events[1]) else np.inf
    return sol, tE, tR


def eta_of(H, p):
    return IC['eta'] * np.exp(-p['alpha'] * np.asarray(H))


def sigma_crit(p):
    a, b = p['rho'] * (1 - p['R_star']), p['R_star']
    return (a - p['nu0'] * b) / (p['nu'] * b + a)


def explicit_bound(p, phi_max):
    eps = F(p['E_star'], p) / 2.0
    T1 = np.log(max(1.0, IC['eta'] * phi_max / eps)) / (p['alpha'] * IC['S'])
    E_plus = (np.sqrt(p['lam'] ** 2 + 4 * p['mu'] * IC['eta'] * phi_max) - p['lam']) / (2 * p['mu'])
    E_bar = max(IC['E'], E_plus)
    return T1 + (E_bar - p['E_star']) / eps, eps, E_bar


def main():
    out = []

    def say(s):
        print(s)
        out.append(s)

    phi_cap_c = lambda t: 0.003

    # 1. illustrative trajectory ------------------------------------------------
    sol, tE, tR = run(P, phi_in_canonical, phi_cap_c, T=400.0)
    tB = min(tE, tR)
    ch = 'R' if tR < tE else 'E'
    yB = sol.sol(tB)
    tt = np.linspace(0, 400, 40001)
    phi_max = max(phi_in_canonical(t) for t in tt)
    bound, eps, E_bar = explicit_bound(P, phi_max)
    say('== 1. Illustrative trajectory (exact field, rtol 1e-9, max_step 0.05)')
    say('t_B = %.3f  channel = %s   (t_B^R = %.3f, t_B^E on continuation = %.3f)' % (tB, ch, tR, tE))
    say('state at t_B: E = %.4f, Sigma = %.4f, eta = %.4f, R = %.4f' % (yB[0], yB[1], eta_of(yB[2], P), yB[3]))
    say('min E on [0, 2] = %.4f' % sol.sol(np.linspace(0, 2, 2001))[0].min())
    say('Sigma_crit = %.5f' % sigma_crit(P))
    say('Phi_max = sup Phi_in = %.5f; eps* = F(E*)/2 = %.5f; E_bar = %.5f' % (phi_max, eps, E_bar))
    say('explicit E-channel bound = %.1f  (ratio to t_B = %.1f)' % (bound, bound / tB))

    # 2. memory-sensitivity scan -----------------------------------------------
    def branches(a):
        q = dict(P, alpha=a)
        _, e, r = run(q, phi_in_canonical, phi_cap_c, T=400.0, max_step=0.1, rtol=1e-8, atol=1e-11, n_out=200)
        return e, r

    alphas = np.linspace(0.005, 0.30, 119)
    tEs, tRs = [], []
    for a in alphas:
        e, r = branches(a)
        tEs.append(e)
        tRs.append(r)
    tEs, tRs = np.array(tEs), np.array(tRs)
    tBs = np.minimum(tEs, tRs)
    i = int(np.argmax(tBs))
    res = minimize_scalar(lambda a: -min(branches(a)), bounds=(0.10, 0.17), method='bounded',
                          options=dict(xatol=1e-5))
    a_star = res.x
    e_s, r_s = branches(a_star)
    say('')
    say('== 2. Memory-sensitivity scan (exact field, rtol 1e-8, max_step 0.1)')
    say('grid argmax: alpha = %.4f, t_B = %.3f' % (alphas[i], tBs[i]))
    say('Brent: alpha* = %.4f, t_B^max = %.3f (t_B^R = %.3f, t_B^E = %.3f)' % (a_star, min(e_s, r_s), r_s, e_s))
    say('t_B at alpha = 0.005: %.3f (%s); at alpha = 0.30: %.3f (%s)'
        % (tBs[0], 'R' if tRs[0] < tEs[0] else 'E', tBs[-1], 'R' if tRs[-1] < tEs[-1] else 'E'))
    e_0, r_0 = branches(0.0)
    say('alpha = 0 (no memory): t_B^R = %.3f, t_B^E on continuation = %.3f; the declining inflow alone '
        'closes the potential channel' % (r_0, e_0))

    # 3. memory-kernel example ------------------------------------------------
    phi_in_k = lambda t: 1.0
    phi_cap_k = lambda t: 0.1
    T_k = 20000.0
    say('')
    say('== 3. Memory kernel: constant Phi_in = 1, Phi_cap = 0.1, horizon %.0f' % T_k)
    kern = {}
    for name, kappa in (('permanent (K = 1)', 0.0), ('decaying (kappa = 0.01)', 0.01)):
        s, e, r = run(P, phi_in_k, phi_cap_k, T=T_k, kappa=kappa, max_step=2.0, rtol=1e-9, atol=1e-12,
                      method='LSODA', n_out=6000, log_grid=True)
        kern[name] = (s, e, r)
        t_life = min(e, r)
        life = s.t <= t_life if np.isfinite(t_life) else np.ones_like(s.t, dtype=bool)
        Smax = s.y[1][life].max()
        say('%-24s t_B^E = %s, t_B^R = %s, max(Sigma - Sigma0) on [0, t_B] = %.2e, eta(T) = %.4f, E(T) = %.4f, R(T) = %.4f'
            % (name, ('%.1f' % e) if np.isfinite(e) else 'none', ('%.1f' % r) if np.isfinite(r) else 'none',
               Smax - IC['S'], eta_of(s.y[2][-1], P), s.y[0][-1], s.y[3][-1]))
    E_plus = (np.sqrt(P['lam'] ** 2 + 4 * P['mu'] * IC['eta'] * 1.0) - P['lam']) / (2 * P['mu'])
    R_low = P['rho'] * (1 - IC['S']) / (P['rho'] * (1 - IC['S']) + P['nu0'] + P['nu'] * IC['S'])
    say('proof check: E_plus = %.5f, sigma0 F(E_plus) = %.5f, R_low = R_bal(Sigma0) = %.5f, '
        'required Phi_cap > %.5f' % (E_plus, P['sigma0'] * F(E_plus, P), R_low, P['sigma0'] * F(E_plus, P) / R_low))
    bound_k, _, _ = explicit_bound(P, 1.0)
    e_perm = kern['permanent (K = 1)'][1]
    say('bound (9) in this environment (Phi_max = 1, t_1 = t_A) = %.1f; t_B^E of the permanent trace = %.1f; '
        'ratio %.3f' % (bound_k, e_perm, bound_k / e_perm))
    # A damage-free state of the canonical system (K = 1) in the same environment:
    # the argument of Proposition 1(b) with Sigma_0 = 0 (viability kernel not empty).
    s_nd, e_nd, r_nd = run(P, phi_in_k, phi_cap_k, T=T_k, kappa=0.0, max_step=2.0, rtol=1e-9, atol=1e-12,
                           method='LSODA', n_out=6000, log_grid=True, y0=[IC['E'], 0.0, 0.0, IC['R']])
    R_bal0 = P['rho'] / (P['rho'] + P['nu0'])
    say('damage-free start (Sigma_0 = 0, K = 1): t_B^E = %s, t_B^R = %s, max Sigma on [0, %.0f] = %.1e, '
        'R_bal(0) = %.4f, Phi_cap * R_bal(0) = %.4f'
        % (('%.1f' % e_nd) if np.isfinite(e_nd) else 'none', ('%.1f' % r_nd) if np.isfinite(r_nd) else 'none',
           T_k, s_nd.y[1].max(), R_bal0, 0.1 * R_bal0))

    # 4. Proposition 3: an example satisfying (P1)-(P4) -------------------------
    # The illustrative set started above E_0, so that restoration is saturated
    # on the whole life (P2); E_0 = 0.10 stays the constant of the field (5).
    y0_p3 = [0.12, IC['S'], 0.0, IC['R']]
    kw = dict(max_step=0.01, rtol=1e-11, atol=1e-13, n_out=200, y0=y0_p3)
    p0 = dict(P, alpha=0.0)
    s0, e0, r0 = run(p0, phi_in_canonical, phi_cap_c, T=100.0, **kw)
    T0 = min(e0, r0)
    tt0 = np.linspace(0.0, T0, 200001)
    Y0 = s0.sol(tt0)
    h = P['sigma0'] * np.array([F(e, P) for e in Y0[0]]) - 0.003 * Y0[3]
    idx = np.where(np.sign(h[1:]) != np.sign(h[:-1]))[0]
    j = int(np.argmin(Y0[0][1:])) + 1
    say('')
    say('== 4. Proposition 3: illustrative set started at E(t_A) = 0.12 (alpha = 0; rtol 1e-11, max_step 0.01)')
    say('t_B = %.4f, channel = %s; min E on (t_A, t_B] = %.4f at t = %.2f (E_0 = %.2f); max Sigma on [t_A, t_B] = %.3f'
        % (T0, 'R' if r0 < e0 else 'E', Y0[0][j], tt0[j], P['E0'], Y0[1].max()))
    for k in idx:
        tc = tt0[k]
        d = rhs(tc, s0.sol(tc), p0, phi_in_canonical, phi_cap_c, 0.0)
        dh = P['sigma0'] * (2 * P['mu'] * Y0[0][k] + P['lam']) * d[0] - 0.003 * d[3]
        say('production-export kink crossed at t = %.3f, d/dt(production - export) = %.3e' % (tc, dh))
    after = bool(len(idx)) and bool(np.all(h[idx[-1] + 1:] > 0))
    dRT = rhs(T0, s0.sol(T0), p0, phi_in_canonical, phi_cap_c, 0.0)[3]
    say('dR/dt at t_B = %.5f; production exceeds export from the last crossing to t_B: %s' % (dRT, after))
    for a in (1e-5, 1e-4, 1e-3):
        _, ea, ra = run(dict(P, alpha=a), phi_in_canonical, phi_cap_c, T=100.0, **kw)
        say('alpha = %g: t_B = %.6f, (t_B - t_B(0))/alpha = %.1f' % (a, min(ea, ra), (min(ea, ra) - T0) / a))

    # Variational system of Appendix A along the alpha = 0 trajectory; g = 1
    # there, since E > E_0 on the whole life. u = (u_eta, u_E, u_Sigma, u_R).
    def rhs_var(t, y):
        E, S, H, R = y[:4]
        u_eta, u_E, u_S, u_R = y[4:]
        Fp = 2 * P['mu'] * E + P['lam']
        chi = 1.0 if P['sigma0'] * F(E, P) > 0.003 * R else 0.0
        a_c = P['nu0'] + P['nu'] * S + P['rho'] * max(0.0, 1.0 - S)
        b_c = P['nu'] * R + P['rho'] * (1.0 - R) * (1.0 if S < 1.0 else 0.0)
        du = [-IC['eta'] * S,
              -Fp * u_E + phi_in_canonical(t) * u_eta,
              chi * (P['sigma0'] * Fp * u_E - 0.003 * u_R),
              -a_c * u_R - b_c * u_S]
        return list(rhs(t, y[:4], p0, phi_in_canonical, phi_cap_c, 0.0)) + du

    sv = solve_ivp(rhs_var, [0.0, T0], y0_p3 + [0.0] * 4, rtol=1e-11, atol=1e-13, max_step=0.01,
                   dense_output=True)
    U = sv.sol(np.linspace(0.0, T0, 20001))
    say('variational system: u_R(t_B) / |dR/dt(t_B)| = %.2f; max u_E = %.1e, max u_Sigma = %.1e, min u_R = %.1e'
        % (sv.y[7][-1] / abs(dRT), U[5].max(), U[6].max(), U[7].min()))

    # 5. Worst phase of the environment ---------------------------------------
    # Setting of section 3, but Phi_cap alternates between lo and hi with
    # half-period 1 (low phase first); integrated piecewise between its jumps.
    def run_square(lo, hi, kappa, T, half=1.0):
        y = [IC['E'], IC['S'], 0.0, IC['R']]
        t, low = 0.0, True
        while t < T:
            t1 = min(t + half, T)
            cap = lo if low else hi

            def eE(tt, yy):
                return yy[0] - P['E_star']
            eE.direction = -1

            def eR(tt, yy):
                return yy[3] - P['R_star']
            eR.direction = -1
            s = solve_ivp(lambda tt, yy, c=cap: rhs(tt, yy, P, phi_in_k, lambda u: c, kappa), [t, t1], y,
                          rtol=1e-9, atol=1e-12, events=[eE, eR])
            hits = [(s.t_events[i][0], 'ER'[i], s.y_events[i][0]) for i in (0, 1) if len(s.t_events[i])]
            if hits:
                return min(hits, key=lambda z: z[0])
            y = list(s.y[:, -1])
            t, low = t1, not low
        return np.inf, '-', np.array(y)

    say('')
    say('== 5. Worst phase: Phi_in = 1, Phi_cap alternating with half-period 1, low phase first')
    for lo, hi in ((0.06, 0.14), (0.08, 0.12)):
        say('Phi_cap %.2f/%.2f: worst phase Phi_cap_min * R_low = %.5f, sigma0 F(E_bar) = %.5f'
            % (lo, hi, lo * R_low, P['sigma0'] * F(E_plus, P)))
        for name, kappa in (('decaying (kappa = 0.01)', 0.01), ('permanent (K = 1)', 0.0)):
            if lo == 0.08 and kappa == 0.0:
                continue
            tb, ch, yb = run_square(lo, hi, kappa, T_k)
            if np.isfinite(tb):
                say('  %-24s t_B = %.1f via %s, Sigma(t_B) = %.4f, Sigma(t_B) - Sigma0 = %.1e'
                    % (name, tb, ch, yb[1], yb[1] - IC['S']))
            else:
                say('  %-24s no exit up to %.0f, Sigma - Sigma0 = %.1e, eta = %.4f, E = %.4f, R = %.4f'
                    % (name, T_k, yb[1] - IC['S'], eta_of(yb[2], P), yb[0], yb[3]))

    # 6. counterexamples of Section 6 -----------------------------------------
    # Initial state E(t_A) = E0 = 0.427, Sigma0 = 0.01, eta0 = 0.96, R0 = 0.99,
    # Phi_cap = 0.01; the remaining values as in Table 1 unless set here.
    say('')
    say('== 6. Counterexamples of Section 6 (exact field, RK45, rtol 1e-9, atol 1e-12, max_step 0.5)')
    y0_ex = [0.427, 0.01, 0.0, 0.99]
    cap_ex = lambda t: 0.01
    ex1 = dict(P, E0=0.427, nu=0.02, rho=0.0055, E_star=0.20)
    phi_ex1 = lambda t: 0.234 + 0.266 * np.exp(-max(0.0, t - 20.0) / 20.0)
    ex2 = dict(P, E0=0.427, nu=0.06, rho=0.004, E_star=0.04)
    phi_ex2 = lambda t: 0.5

    def tb_ex(p, phi, a):
        s, e, r = run(dict(p, alpha=a), phi, cap_ex, T=1500.0, max_step=0.5, y0=y0_ex, n_out=3001)
        return (r, 'R', s) if r <= e else (e, 'E', s)

    say('Example 1 (restoration not saturated): nu = 0.02, rho = 0.0055, E* = 0.20, '
        'Phi_in(t) = 0.234 + 0.266 exp(-max(0, t - 20)/20)')
    t0, ch0, s0 = tb_ex(ex1, phi_ex1, 0.0)
    tt = np.linspace(200.0, t0, 2001)
    say('  alpha = 0: potential on [200, t_B] between %.4f and %.4f (E0 = 0.427)'
        % (s0.sol(tt)[0].min(), s0.sol(tt)[0].max()))
    for a in (0.0, 1e-4, 4.96e-4, 2.46e-3, 0.0122, 0.0605, 0.3):
        tb, ch, _ = tb_ex(ex1, phi_ex1, a)
        say('  alpha = %-8g t_B = %.3f via %s' % (a, tb, ch))
    say('Example 2 (maximum inside the reserve branch): rho = 0.004, Phi_in = 0.5')
    A = np.arange(0.25, 0.45, 0.0025)
    v = [tb_ex(ex2, phi_ex2, a)[:2] for a in A]
    k = int(np.argmax([x[0] for x in v]))
    say('  scan of [0.25, 0.45] with step 0.0025: max t_B = %.2f at alpha = %.4f via %s' % (v[k][0], A[k], v[k][1]))
    for a in (A[k] - 0.0025, A[k], A[k] + 0.0025):
        tb, ch, s = tb_ex(ex2, phi_ex2, a)
        E, S, H, R = s.sol(np.linspace(0.0, tb, 20001))
        net = ex2['sigma0'] * (ex2['mu'] * E * E + ex2['lam'] * E) - R * 0.01
        say('  alpha = %.4f: t_B = %.2f via %s, max over the life of production - export = %+.2e'
            % (a, tb, ch, net.max()))
    A2 = np.arange(0.80, 1.20, 0.01)
    sw = next((a for a in A2 if tb_ex(ex2, phi_ex2, a)[1] == 'E'), np.nan)
    say('  first death through the potential channel on [0.80, 1.20] with step 0.01: alpha = %.2f' % sw)

    with open(os.path.join(HERE, 'paper1_numbers.out.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')

    # Figure 1 ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    t = np.linspace(0, 130, 2601)
    Y = sol.sol(t)
    alive = t <= tB
    for k, (lab, col) in enumerate((('$E$', 'C0'), ('$\\Sigma$', 'C3'), ('$\\eta$', 'C2'), ('$R$', 'C1'))):
        v = eta_of(Y[2], P) if k == 2 else Y[k]
        ax[0].plot(t[alive], v[alive], color=col, lw=1.8, label=lab)
        ax[0].plot(t[~alive], v[~alive], color=col, lw=1.0, ls=':')
    ax[0].axhline(P['R_star'], color='C1', lw=0.8, ls='--')
    ax[0].axhline(P['E_star'], color='C0', lw=0.8, ls='--')
    ax[0].text(124, P['R_star'] + 0.015, '$R^*$', fontsize=9, color='C1')
    ax[0].text(124, P['E_star'] + 0.015, '$E^*$', fontsize=9, color='C0')
    ax[0].axvline(tB, color='k', lw=0.8)
    ax[0].text(tB + 1.5, 0.98, '$t_B$', fontsize=10)
    ax[0].set_xlabel('time (model units)')
    ax[0].set_ylabel('state variables')
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(frameon=False, fontsize=9, loc='upper right', ncol=2, bbox_to_anchor=(1.0, 0.92))
    ax[0].set_title('(a) trajectory and its continuation', fontsize=10, loc='left')

    ax[1].plot(alphas, tRs, color='C1', ls='--', lw=1.2, label='$t_B^R(\\alpha)$')
    ax[1].plot(alphas, tEs, color='C0', ls='--', lw=1.2, label='$t_B^E(\\alpha)$')
    ax[1].plot(alphas, tBs, color='k', lw=1.8, label='$t_B=\\min$')
    ax[1].axvline(a_star, color='0.4', lw=0.8, ls=':')
    ax[1].axvline(P['alpha'], color='0.6', lw=0.8, ls='-.')
    ax[1].text(a_star + 0.004, 8, '$\\alpha^*$', fontsize=10, color='0.3')
    ax[1].text(P['alpha'] + 0.004, 8, 'illustrative $\\alpha$', fontsize=9, color='0.4')
    ax[1].set_ylim(0, 150)
    ax[1].set_xlabel('$\\alpha$ (per model time unit)')
    ax[1].set_ylabel('$t_B$ (model units)')
    ax[1].legend(frameon=False, fontsize=9, loc='upper right')
    ax[1].set_title('(b) lifetime against memory sensitivity', fontsize=10, loc='left')
    fig.tight_layout()
    save_fig(fig, 'paper1_fig1.png')
    plt.close(fig)

    # Figure 2 ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    styles = {'permanent (K = 1)': 'k', 'decaying (kappa = 0.01)': 'C2'}
    for name, (s, e, r) in kern.items():
        col = styles[name]
        z = 3 if col == 'k' else 2   # permanent-trace curves drawn on top: they coincide with the other run early on
        t_end = min(e, r)
        tt_k = s.t[1:]
        alive = tt_k <= t_end
        short = name.split(' ')[0]
        for a, v, ls, lw, lab in ((ax[0], eta_of(s.y[2][1:], P), '-', 1.6, name.replace('kappa', '$\\kappa$')),
                                  (ax[1], s.y[0][1:], '-', 1.6, '$E$, ' + short),
                                  (ax[1], s.y[3][1:], '--', 1.2, '$R$, ' + short)):
            a.semilogx(tt_k[alive], v[alive], color=col, ls=ls, lw=lw, label=lab, zorder=z)
            if (~alive).any():
                a.semilogx(tt_k[~alive], v[~alive], color=col, ls=':', lw=1.0, zorder=z)
        if np.isfinite(t_end):
            for a in ax:
                a.axvline(t_end, color=col, lw=0.8, ls='-.')
            ax[1].text(t_end * 0.62, 0.92, '$t_B$', fontsize=12, color=col)
    e_perm = kern['permanent (K = 1)'][1]
    ax[1].axvline(e_perm, color='0.5', lw=0.8, ls=':')
    ax[1].text(e_perm * 1.08, 0.80, '$t_B^E$', fontsize=12, color='0.35')
    ax[1].axhline(P['E_star'], color='0.5', lw=0.8, ls=':')
    ax[1].axhline(P['R_star'], color='0.5', lw=0.8, ls=':')
    ax[1].text(0.012, P['E_star'] + 0.015, '$E^*$', fontsize=11, color='0.35')
    ax[1].text(0.012, P['R_star'] + 0.015, '$R^*$', fontsize=11, color='0.35')
    ax[0].set_xlabel('time (model units)')
    ax[0].set_ylabel('$\\eta$')
    ax[0].legend(frameon=False, fontsize=9, loc='lower left')
    ax[0].set_title('(a) assimilation efficiency', fontsize=10, loc='left')
    ax[1].set_xlabel('time (model units)')
    ax[1].set_ylabel('$E$, $R$')
    ax[1].legend(frameon=False, fontsize=9, loc='center left')
    ax[1].set_title('(b) potential and reserve', fontsize=10, loc='left')
    fig.tight_layout()
    save_fig(fig, 'paper1_fig2.png')
    plt.close(fig)
    print('figures written to', HERE)


if __name__ == '__main__':
    main()
