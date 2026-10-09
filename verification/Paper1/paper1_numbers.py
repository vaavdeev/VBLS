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
     decaying kernel, same parameters and initial state.
Figures: paper1_fig1.png and paper1_fig2.png, written next to this script
(and copied to the monograph's picture/ folder when that folder exists).
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
# Figures are written next to this script; inside the monograph project they are
# also copied to its picture/ folder, from which the manuscripts include them.
FIG_DIRS = [HERE]
_PROJECT_PIC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'picture')
if os.path.isdir(_PROJECT_PIC):
    FIG_DIRS.append(_PROJECT_PIC)


def save_fig(fig, name):
    for d in FIG_DIRS:
        fig.savefig(os.path.join(d, name), dpi=200)

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
        log_grid=False):
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

    # 3. memory-kernel example ------------------------------------------------
    phi_in_k = lambda t: 1.0
    phi_cap_k = lambda t: 0.1
    T_k = 20000.0
    say('')
    say('== 3. Memory kernel: constant Phi_in = 1, Phi_cap = 0.1, horizon %.0f' % T_k)
    kern = {}
    for name, kappa in (('permanent (K = 1)', 0.0), ('decaying (tau_K = 100)', 0.01)):
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

    with open(os.path.join(HERE, 'paper1_numbers.out.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')

    # Figure 1 ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    t = np.linspace(0, 130, 2601)
    Y = sol.sol(t)
    alive = t <= tB
    for k, (lab, col) in enumerate((('$E$', 'C0'), ('$\\Sigma$', 'C3'), ('$\\eta_{abs}$', 'C2'), ('$R$', 'C1'))):
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
    styles = {'permanent (K = 1)': ('k', '-'), 'decaying (tau_K = 100)': ('C2', '-')}
    for name, (s, e, r) in kern.items():
        col, ls = styles[name]
        tt_k = s.t[1:]
        ax[0].semilogx(tt_k, eta_of(s.y[2][1:], P), color=col, ls=ls, lw=1.6, label=name.replace('tau_K', '$\\tau_K$'))
        ax[1].semilogx(tt_k, s.y[0][1:], color=col, ls='-', lw=1.6, label='$E$, ' + name.split(' ')[0])
        ax[1].semilogx(tt_k, s.y[3][1:], color=col, ls='--', lw=1.2, label='$R$, ' + name.split(' ')[0])
    ax[1].axhline(P['E_star'], color='0.5', lw=0.8, ls=':')
    ax[1].axhline(P['R_star'], color='0.5', lw=0.8, ls=':')
    ax[1].text(0.012, P['E_star'] + 0.015, '$E^*$', fontsize=9, color='0.35')
    ax[1].text(0.012, P['R_star'] + 0.015, '$R^*$', fontsize=9, color='0.35')
    ax[0].set_xlabel('time (model units)')
    ax[0].set_ylabel('$\\eta_{abs}$')
    ax[0].legend(frameon=False, fontsize=9, loc='lower left')
    ax[0].set_title('(a) assimilation efficiency', fontsize=10, loc='left')
    ax[1].set_xlabel('time (model units)')
    ax[1].set_ylabel('$E$, $R$')
    ax[1].legend(frameon=False, fontsize=8, loc='center left')
    ax[1].set_title('(b) potential and reserve', fontsize=10, loc='left')
    fig.tight_layout()
    save_fig(fig, 'paper1_fig2.png')
    plt.close(fig)
    print('figures written to', ', '.join(FIG_DIRS))


if __name__ == '__main__':
    main()
