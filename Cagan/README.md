# Cagan — Somatic mutation accumulation verification

**Monograph section:** §17.3
**VBLS prediction tested:** somatic mutation burden as a direct empirical proxy for structural entropy Σ(t); three-phase dynamics (growth / plateau / collapse) in accumulation trajectories; universal Σ* at end-of-observed-life across 15 species; invariant rate × T_max ≈ const reproduced analytically.

**Data source:** Cagan et al. (2022) Nature 604:517–524
15 species including human, mouse, dog, cat, horse, naked mole-rat and others.
Supplementary Table 4: somatic substitution burden per crypt vs age.

---

## Scripts

| File | What it does |
|---|---|
| `cagan_verification.py` | Main script. Three figures: (1) Σ(t) trajectories — data scatter + regression + VBLS curve for species with sufficient age coverage; (2) universal Σ* bar chart (analogue of R* from Whittemore); (3) human detail — Σ(t) over all 28 crypts + life phases + dΣ/dt. |
| `cagan_rateXtmax.py` | Focused analysis of the rate × T_max invariant; extended species comparison. |

## Data

| File | Description |
|---|---|
| `Table_4.csv` | Cagan et al. (2022) Supplementary Table 4: species, age (yr), substitution burden per crypt. Semicolon-delimited. |

## Figures

| File | Figure in monograph |
|---|---|
| `cagan_fig1_trajectories.png/.pdf` | Fig. 17.3.1 — Σ(t) trajectories per species |
| `cagan_fig2_sigma_star.png/.pdf` | Fig. 17.3.2 — Universal Σ* |
| `cagan_fig3_rateXtmax.png/.pdf` | Fig. 17.3.3 — Rate × T_max invariant |
| `cagan_fig4_human.png/.pdf` | Human Σ(t) detail — not embedded in monograph (deferred by the author for a future figure slot) |

## Run

```bash
python cagan_verification.py    # figures 1, 2, 4
python cagan_rateXtmax.py       # figure 3 (rate × T_max analysis)
```
Both scripts read `Table_4.csv` from the same directory.
