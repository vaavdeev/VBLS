"""
VBLS — diagnostic: is the tiny Σ/R monotonicity blemish the ρ·g' restoration
residual (the 'under C4 modulo ρg'' caveat)? Re-run the sign-pattern check for
decreasing ρ (deeper into C4). If Σ/R compliance -> 100% as ρ -> 0, the residual
is confirmed as the restoration term, and strict-C4 gives the full pattern P.
"""
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(mu=1.50, lam=0.30, sigma_0=0.0162, nu_0=0.006, nu=0.060,
            R_star=0.15, E_star=0.04)
IC = [0.10, 0.01, 0.96, 0.99]
E_RANGE = IC[0]-BASE['E_star']; T_MAX=130.0
def Phi_in(t):  return (1-np.exp(-t/10))*np.exp(-max(0.0,t-60)/30)
def Phi_cap(t): return 0.003

def rhs(t,y,alpha,rho):
    E,S,eta,R=y
    E=max(E,0.0);S=np.clip(S,0,1);eta=np.clip(eta,0,1);R=np.clip(R,0,1)
    F=BASE['mu']*E**2+BASE['lam']*E
    g=max(0.0,min(1.0,(E-BASE['E_star'])/E_RANGE))
    return [eta*Phi_in(t)-F,
            max(0.0,BASE['sigma_0']*F-R*Phi_cap(t)),
            -alpha*S*eta,
            -BASE['nu_0']*R-BASE['nu']*S*R+rho*g*(1-S)*(1-R)]

def traj(alpha,rho,t_eval):
    def eE(t,y):return y[0]-BASE['E_star']
    def eR(t,y):return y[3]-BASE['R_star']
    eE.terminal=eR.terminal=True;eE.direction=eR.direction=-1
    s=solve_ivp(lambda t,y:rhs(t,y,alpha,rho),[0,T_MAX],IC,method='RK45',
                events=[eE,eR],rtol=1e-9,atol=1e-11,max_step=0.1,
                t_eval=t_eval,dense_output=True)
    tE=s.t_events[0][0] if s.t_events[0].size else np.inf
    tR=s.t_events[1][0] if s.t_events[1].size else np.inf
    tB=min(tE,tR) if (tE<np.inf or tR<np.inf) else T_MAX
    return tB,s

ALPHA=np.linspace(0.010,0.135,26)
t_eval=np.linspace(0,T_MAX,1301)
labels=['E','Sigma','eta','R']; want_dec={'E':1,'Sigma':1,'eta':1,'R':0}

print(f"{'rho':>8} {'E':>8} {'Sigma':>8} {'eta':>8} {'R':>8}   (min compliance %)")
print("-"*50)
for rho in (0.002,0.0006,0.0002,0.0):
    T={}
    for a in ALPHA:
        tB,s=traj(a,rho,t_eval); tt=t_eval[t_eval<=tB]; T[a]=(tt,s.sol(tt))
    worst={k:1.0 for k in labels}
    for i in range(len(ALPHA)-1):
        (t0,Y0),(t1,Y1)=T[ALPHA[i]],T[ALPHA[i+1]]
        Tm=min(t0[-1],t1[-1]); tt=t_eval[(t_eval>=1.0)&(t_eval<=Tm)]
        A=np.vstack([np.interp(tt,t0,Y0[k]) for k in range(4)])
        B=np.vstack([np.interp(tt,t1,Y1[k]) for k in range(4)])
        for k,lab in enumerate(labels):
            ok=(A[k]>=B[k]-1e-9) if want_dec[lab] else (A[k]<=B[k]+1e-9)
            worst[lab]=min(worst[lab], float(np.mean(ok)) if len(ok) else 1.0)
    print(f"{rho:>8.4f} " + " ".join(f"{worst[l]*100:>7.2f}" for l in labels))
print("-"*50)
print("If Sigma/R columns -> 100% as rho decreases, the blemish is the ρg'")
print("restoration residual: strict-C4 (ρ→0) yields the full sign pattern P,")
print("while η (master) and E stay 100% throughout, independent of ρ.")
