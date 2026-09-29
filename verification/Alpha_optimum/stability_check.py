"""
VBLS — quick stability check of α* under grid resolution and horizon T_MAX.
Isolates whether the old α*≈0.10 was a coarse-grid / short-horizon artifact.
Terminal-events only (fast). Same calibration as alpha_sensitivity.py.
"""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            rho=0.002, R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0] - BASE['E_star']

def Phi_in(t):  return (1.0 - np.exp(-t/10.0)) * np.exp(-max(0.0, t-60.0)/30.0)
def Phi_cap(t): return 0.003

def rhs(t, y, alpha):
    E, S, eta, R = y
    E = max(E,0.0); S = np.clip(S,0,1); eta = np.clip(eta,0,1); R = np.clip(R,0,1)
    F = BASE['mu']*E**2 + BASE['lam']*E
    g = max(0.0, min(1.0, (E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t)-F,
            max(0.0, BASE['sigma_0']*F - R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R - BASE['nu']*S*R + BASE['rho']*g*(1-S)*(1-R)]

def tB(alpha, T_MAX):
    def eE(t,y): return y[0]-BASE['E_star']
    def eR(t,y): return y[3]-BASE['R_star']
    eE.terminal=eR.terminal=True; eE.direction=eR.direction=-1
    s = solve_ivp(lambda t,y: rhs(t,y,alpha),[0,T_MAX],IC,method='RK45',
                  events=[eE,eR], rtol=1e-8, atol=1e-10, max_step=0.5)
    tE = s.t_events[0][0] if s.t_events[0].size else np.inf
    tR = s.t_events[1][0] if s.t_events[1].size else np.inf
    return min(tE,tR) if (tE<np.inf or tR<np.inf) else T_MAX

print(f"{'T_MAX':>7} {'n_grid':>7} {'α_range':>14} {'α*':>8} {'t_B_max':>9}")
print("-"*52)
for T_MAX in (130.0, 400.0):
    for n, hi in ((60,0.20), (240,0.30), (600,0.30)):
        A = np.linspace(0.005, hi, n)
        T = np.array([tB(a, T_MAX) for a in A])
        i = int(np.argmax(T))
        print(f"{T_MAX:>7.0f} {n:>7d} {'[0.005,%.2f]'%hi:>14} {A[i]:>8.4f} {T[i]:>9.2f}")
print("-"*52)
print("If α* is stable ~0.13 across fine grids/horizons and only the coarse")
print("60-pt/short-horizon case gives ~0.10, then 0.10 was a resolution artifact.")
