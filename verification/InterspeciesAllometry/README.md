# InterspeciesAllometry — Cross-species lifecycle curves

**Monograph section:** Appendix D.3 (§D.3.1–D.3.6)
**VBLS prediction tested:** full lifecycle trajectories E(t), R(t) for 11 mammalian species under two allometric scaling conventions (by T_max and by T_ls); the initial dip in E(t) at t≈0 and its scaling with body size; sensitivity of the dip to the initial external flux Φ_in(0).

**Data source:** AnAge Build 14 (Tacutu et al., 2018); physiological literature for species lacking metabolic data in AnAge.

---

## Scripts

| File | What it does |
|---|---|
| `lifecycle_comparison.py` | Integrates the full system (S) for 11 species under two mass-scaling variants (by T_max and by T_ls); compares model t_B against the observed reference lifespan. |
| `lifecycle_dip_analysis.py` | Analyzes the initial dip in E(t) present at the start of the trajectory, for all 11 species and in detail for *Homo sapiens*, under two Φ_in(0) regimes (0 and 0.45). |
| `lifecycle_phi_comparison.py` | Full lifecycle curves for all 11 species under the two Φ_in(0) regimes, scaled by T_ls; quantifies how the resulting shift in t_B grows with body size. |

## Figures

| File | Figure in monograph |
|---|---|
| `lifecycle_comparison.png/.pdf` | Fig. D.3.1 — full lifecycle dynamics, 11 species, two scaling variants |
| `lifecycle_dip_allspecies.png` | Fig. D.3.2 — initial E(t) dip, all 11 species |
| `lifecycle_dip_human.png` | Fig. D.3.3 — initial E(t) dip, detailed analysis for *Homo sapiens* |
| `lifecycle_phi_comparison.png/.pdf` | Fig. D.3.4 — full lifecycle curves under two Φ_in(0) regimes |

## Run

```bash
python lifecycle_comparison.py      # Fig. D.3.1
python lifecycle_dip_analysis.py    # Fig. D.3.2, D.3.3
python lifecycle_phi_comparison.py  # Fig. D.3.4
```
