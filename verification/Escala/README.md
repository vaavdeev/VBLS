# Escala — Allometric scaling laws verification

**Monograph section:** Appendix D.1 (§D.1.1–D.1.8)
**VBLS predictions tested:** four scaling laws with zero free parameters —
(1) Kleiber's law P ~ M^0.75;
(2) longevity allometry T_max ~ M^0.25;
(3) criterion test T_ls ~ (P/M)^(−1.0);
(4) Rubner invariant P/M × T_ls ≈ const.
Dataset spans 94 non-human mammal species (cross-order). Also documents the human exceptionalism anomaly (§D.1.4): *Homo sapiens* deviates 5–8× above the Rubner product predicted by other mammals.

**Data source:** Escala A. (2022) Scientific Reports 12:2407.

---

## Scripts

| File | What it does |
|---|---|
| `escala_4tests.py` | Main script. Four-panel figure: one panel per scaling test (§D.1.1–D.1.3). All exponents computed from data — nothing hard-coded. |
| `escala_human_fig.py` | Human exceptionalism analysis (§D.1.4): places Человек_1 (T_ls=75 yr) and Человек_2 (T_ls=122 yr) against the 94-species regression; compares Rubner products across the two T_ls conventions. |

## Data

| File | Description |
|---|---|
| `escala_data.csv` | Primary dataset from Escala (2022): species, body mass (g), metabolic rate (kJ/day), T_ls (yr). |

## Figures

| File | Figure in monograph |
|---|---|
| `escala_4tests.png/.pdf` | Рис. D.1.1 — four scaling tests (2×2 panel) |
| `escala_fig_human.png/.pdf` | Рис. D.1.2 — human exceptionalism: power law + Rubner product distribution + Escala/Cagan T_ls comparison |

## Run

```bash
python escala_4tests.py        # four scaling tests
python escala_human_fig.py     # human exceptionalism detail
```
Both scripts read `escala_data.csv` from the same directory.
