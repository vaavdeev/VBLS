# Cagan — Somatic mutation accumulation verification

**Monograph section:** §17.3
**Tested:** somatic mutation burden as one observable proxy of structural entropy Σ(t) (§17.3.1); a common damage level at death across 15 species — a transfer assumption (Appendix D.5.1.5, item 3), not a prediction of (S) alone; the invariant rate × T_max ≈ const under the consistent scaling of rates with the species (Appendix D.3.1).

**Data source:** Cagan et al. (2022) Nature 604:517–524
15 species including human, mouse, dog, cat, horse, naked mole-rat and others.
Supplementary Table 4: somatic substitution burden per crypt vs age.

---

## Scripts

| File | What it does |
|---|---|
| `cagan_verification.py` | Main script. Three figures: (1) Σ(t) trajectories — data scatter + regression + VBLS curve for species with sufficient age coverage; (2) mutation burden at the end of observation by species (bar chart); (3) human detail — Σ(t) over all 28 crypts + dΣ/dt. |
| `cagan_rateXtmax.py` | Focused analysis of the rate × T_max invariant over all 15 species (no species excluded); extended species comparison. |
| `cagan_robustness.py` | Robustness of the 15-species estimate: leave-one-out, bootstrap interval of φ, sensitivity to the cat's T_max (30 yr in AnAge against 19 and 17 yr). Output: `cagan_robustness.out.txt`. |

T_max for all species is the maximum longevity from AnAge (`../AnAge/anage_data.csv`), without corrections; the scripts read it from that file. The domestic ferret is listed in AnAge under *Mustela putorius*.

## Data

| File | Description |
|---|---|
| `Table_4.csv` | Cagan et al. (2022) Supplementary Table 4: species, age (yr), substitution burden per crypt. Semicolon-delimited. |

## Figures

| File | Figure in monograph |
|---|---|
| `cagan_fig1_trajectories.png/.pdf` | Fig. 17.3.1 — Σ(t) trajectories per species: model shape, amplitude fitted per species to the individual means (shape is compared, not level) |
| `cagan_fig2_sigma_star.png/.pdf` | Fig. 17.3.3 — Mutation burden at the end of observation |
| `cagan_fig3_rateXtmax.png/.pdf` | Fig. 17.3.2 — Rate × T_max invariant |

## Run

```bash
python cagan_verification.py    # figures 1, 2
python cagan_rateXtmax.py       # figure 3 (rate × T_max analysis)
python cagan_robustness.py      # robustness of the 15-species estimate
```
`cagan_verification.py` and `cagan_rateXtmax.py` read `Table_4.csv` from the same directory; all three scripts read T_max from `../AnAge/anage_data.csv`. `cagan_verification.py` also produces a fourth, human-detail figure (Σ(t) over all 28 crypts) that is not embedded anywhere in the monograph text and is not included in this repository.

## Excluded from this repository

- `cagan_fig4_human.pdf`/`.png` — not cited or embedded anywhere in the monograph text.
