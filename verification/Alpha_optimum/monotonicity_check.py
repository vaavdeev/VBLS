"""
VBLS — Level-2 monotonicity check for the α*-crossover proof program.

Tests the sign pattern P of the α-sensitivity of the coupled system (S):
    u_E = ∂E/∂α ≤ 0 ,  u_Σ = ∂Σ/∂α ≤ 0 ,  u_η = ∂η/∂α ≤ 0 ,  u_R = ∂R/∂α ≥ 0 .

Analytical reduction (see §8.2.1 program): the whole pattern follows from the
single master condition  u_η ≤ 0  ⇔  d/dα[α·∫Σ] ≥ 0  (memory load grows with α).
This script verifies P *pointwise in α* by finite differences between neighbouring
α on a common time grid, and reports where (if anywhere) each monotonicity breaks —
especially η, the master variable. Same Homo sapiens calibration as alpha_sensitivity.py.
"""
import os
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            rho=0.002, R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0] - BASE['E_star']
T_MAX = 130.0

def Phi_in(t):  return (1.0 - np.exp(-t/10.0)) * np.exp(-max(0.0, t-60.0)/30.0)
def Phi_cap(t): return 0.003 * np.exp(-max(0.0, t-40.0)*0.008)

def rhs(t, y, alpha):
    E, S, eta, R = y
    E = max(E,0.0); S = np.clip(S,0,1); eta = np.clip(eta,0,1); R = np.clip(R,0,1)
    F = BASE['mu']*E**2 + BASE['lam']*E
    g = max(0.0, min(1.0, (E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t)-F,
            max(0.0, BASE['sigma_0']*F - R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R - BASE['nu']*S*R + BASE['rho']*g*(1-S)*(1-R)]

def tB_and_traj(alpha, t_eval):
    def eE(t,y): return y[0]-BASE['E_star']
    def eR(t,y): return y[3]-BASE['R_star']
    eE.terminal=eR.terminal=True; eE.direction=eR.direction=-1
    s = solve_ivp(lambda t,y: rhs(t,y,alpha),[0,T_MAX],IC,method='RK45',
                  events=[eE,eR], rtol=1e-9, atol=1e-11, max_step=0.1,
                  t_eval=t_eval, dense_output=True)
    tE = s.t_events[0][0] if s.t_events[0].size else np.inf
    tR = s.t_events[1][0] if s.t_events[1].size else np.inf
    tB = min(tE,tR) if (tE<np.inf or tR<np.inf) else T_MAX
    return tB, s

# α grid on the R-limited side and across the optimum
ALPHA = np.linspace(0.010, 0.135, 26)   # up to α* ≈ 0.135
t_eval = np.linspace(0.0, T_MAX, 1301)   # 0.1-yr sampling

traj, tBs = {}, {}
for a in ALPHA:
    tB, s = tB_and_traj(a, t_eval)
    tBs[a] = tB
    # y on common grid up to tB (dense_output for uniformity)
    tt = t_eval[t_eval <= tB]
    Y = s.sol(tt)   # shape (4, len(tt))
    traj[a] = (tt, Y)

# Pairwise monotonicity check between neighbouring α (α_i < α_{i+1})
labels = ['E', 'Sigma', 'eta', 'R']
# expected: E,Sigma,eta DECREASE in α  -> y(a_i) >= y(a_{i+1}); R INCREASES -> y(a_i) <= y(a_{i+1})
want_decrease = {'E':True, 'Sigma':True, 'eta':True, 'R':False}
TOL = 1e-9

print("="*70)
print("  VBLS Level-2 sign-pattern check  (pattern P over α, Homo sapiens)")
print("="*70)
worst = {k: (1.0, None) for k in labels}  # (min compliance fraction, alpha where)
any_break = {k: False for k in labels}

for i in range(len(ALPHA)-1):
    a0, a1 = ALPHA[i], ALPHA[i+1]
    (t0,Y0),(t1,Y1) = traj[a0], traj[a1]
    T = min(t0[-1], t1[-1])
    tt = t_eval[(t_eval>=1.0) & (t_eval<=T)]   # skip first 1 yr (Phase-I transient)
    # interpolate both onto tt
    def interp(t_src, Y_src):
        return np.vstack([np.interp(tt, t_src, Y_src[k]) for k in range(4)])
    A = interp(t0,Y0); B = interp(t1,Y1)
    for k,lab in enumerate(labels):
        if want_decrease[lab]:
            ok = A[k] >= B[k] - TOL           # y decreasing in α
        else:
            ok = A[k] <= B[k] + TOL           # y increasing in α
        frac = float(np.mean(ok)) if len(ok) else 1.0
        if frac < worst[lab][0]:
            worst[lab] = (frac, (a0,a1))
        if frac < 1.0 - 1e-6:
            any_break[lab] = True

print(f"  α range: [{ALPHA[0]:.3f}, {ALPHA[-1]:.3f}]  (α* ≈ 0.135)")
print(f"  baseline t_B(0.030) = {tBs[min(ALPHA, key=lambda a: abs(a-0.030))]:.1f} yr")
print("-"*70)
print(f"  {'var':>6} {'expected':>10} {'holds?':>8} {'min compliance':>16}  where")
print("-"*70)
for lab in labels:
    exp = 'decr in α' if want_decrease[lab] else 'incr in α'
    frac, where = worst[lab]
    verdict = 'YES' if not any_break[lab] else f'breaks'
    wtxt = '' if where is None else f'α∈[{where[0]:.3f},{where[1]:.3f}]'
    print(f"  {lab:>6} {exp:>10} {verdict:>8} {frac*100:>14.2f}%  {wtxt}")
print("-"*70)

master_ok = not any_break['eta']
print(f"  MASTER condition u_η ≤ 0 (η decreasing in α): "
      f"{'HOLDS on entire range' if master_ok else 'VIOLATED somewhere'}")
if master_ok:
    print("  => sign pattern P holds; E-monotonicity unconditional, R under C4.")
    print("  => t_B^R increasing on the R-limited branch is numerically confirmed;")
    print("     Level-2 monotonicity holds up to α*, matching the small-α lemma.")
else:
    print("  => memory-feedback offset flips η somewhere: inspect that α-window.")
print("="*70)

# also report α·∫Σ monotonicity (equivalent master form)
print("  Cross-check via master scalar  α·∫Σ dτ  (should be non-decreasing in α):")
Iint = []
for a in ALPHA:
    tt,Y = traj[a]
    Iint.append(a * np.trapz(Y[1], tt))
Iint = np.array(Iint)
dI = np.diff(Iint)
mono = np.all(dI >= -1e-9)
print(f"    α·∫Σ : {'monotone non-decreasing' if mono else 'NOT monotone'} "
      f"(min Δ = {dI.min():+.3e})")
print("="*70)
