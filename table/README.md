# VBLS — Table of Living Structures

Interactive HTML version of the systematic table of the monograph (Appendix C, §C.1–C.9), built on the organism catalog of Section 15 and Appendix B. Section numbers below refer to the monograph. Русская версия — [README_RU.md](README_RU.md).

## What the table shows

The table is a map and a catalog. Each organism, or object resembling a living system, sits in a cell chosen by two questions:

- **Row: how the memory of damage works.** Does the accumulated trace of damage noticeably lower the efficiency of resource assimilation $\eta_{abs}$ within one cycle of the system? Criterion (15.6): $\alpha\bar\Sigma\tau_{div}\gtrsim1$ for full memory, $\alpha\bar\Sigma\tau_{div}\ll1$ for frozen memory. The cycle is the life of one system, that is, of an individual: one division cycle for a unicellular organism and the life of the organism for a multicellular one, since division of cells inside a multicellular organism does not create a new system (Sections 2.6, 15.1.1.2).
- **Column: how far the system feeds itself.** The own inflow $\gamma=\eta_0\Phi_{in}^{(aut)}/F_{diss}(E_0)$ compares the system's own inflow at the start with its dissipation; $\gamma=1$ is a stationary start.

Position relative to the viability boundary $\mathcal{M}_{min}$ is a separate, third attribute. The boundary does not depend on the inflow: it is set by whether the system's own repair keeps its repair reserve above the threshold (Barrier I, $R_{bal}$ against $R^*$). The position is therefore stated in each organism's card (above, near or below $\mathcal{M}_{min}$) rather than by the cell.

The table illustrates the theory; it does not test it. The positions of the organisms are estimates and adopted assignments: no organism in the table has measured repair parameters ($\rho$, $\nu_0$, $\nu$, $R^*$).

Cells come in three kinds:

- cells holding real organisms;
- **×** — expected to be empty. These are expectations from estimates of the energetics of memory and division (Appendix C.4.1), not consequences of the finite-lifetime theorem;
- **◈** — predicted: combinations the theory permits that have not been found (Appendix C.5).

## How the objects are placed

| Memory ↓ / own inflow γ → | I: ≫ 1 | II: > 1 | III: ≈ 1 | IV: 0.1–0.3 | V: 0–0.05 | VI: 0 | VII: outside the theory |
|---|---|---|---|---|---|---|---|
| **R1** full | 26 mammals, tardigrade (active) | *Cladosporium* (ISS) | ◈ | × | × | × | × |
| **R2** frozen | *E. coli*, *D. radiodurans*, mutator *E. coli* ΔmutS | SAR11, *M. genitalium* | JCVI-syn3.0 ★ | | × | × | × |
| **R3** borrowed | | ◈ | ◈ | *Rickettsia*, *Chlamydia*, *N. equitans*, *E. cuniculi*, *M. pneumoniae* | *Tremblaya*, *Carsonella*, *Hodgkinia*, *Nasuia* | | |
| **R4** no memory | | | ◈ | | | mitochondrion | Szostak protocell |
| **R5** anabiosis / outside | *B. subtilis* spores, tardigrade (cryptobiosis) | ◈ | lotus seeds, egg | ◈ | | | viruses, viroids, prions |

★ marks the reference point. An empty cell carries no mark.

**Rows**

- **R1, full memory** — the trace of damage acts noticeably over the life of an individual (mammals, for example).
- **R2, frozen memory** — the trace is recorded but hardly shows within one cycle: either the system divides fast (Mechanism A) or its memory parameter is small, as in bacteria (Mechanism B).
- **R3, borrowed memory** — parasites and endosymbionts. The host's inflow provides their starting level of memory; after the start their trace is as permanent as that of any other system.
- **R4, no memory** ($\alpha\approx0$) — no known organisms; the row holds the mitochondrion and a model protocell.
- **R5, anabiosis and objects outside the theory** — spores, seeds and a tardigrade in cryptobiosis; viruses, viroids and prions, for which the variables of the theory are not defined.

**Columns**

- **I** — a large margin of own inflow.
- **II** — a small margin.
- **III** — a stationary start: the inflow just covers dissipation at the start. Here stands JCVI-syn3.0, the reference point taken to lie at the boundary (Section 11.10).
- **IV–V** — host-dependent systems.
- **VI** — an organelle, with no inflow of its own.
- **VII** — outside the theory.

Moving right along a row follows the path of genome reduction: free-living organism → parasite → endosymbiont → organelle.

The strip in cell R1×I ranks 26 mammals by $\mathcal{B}/\mathcal{B}_{human}$. Because $\mathcal{B}$ grows as $M^{5/4}$, the ranking is an ordering by body mass; the absolute $\mathcal{B}$ of multicellular organisms is not determined (Appendix C.9.2).

## Changes from the previous version

The version published earlier in this repository predates the revision of Section 15 and Appendices B and C. The table now differs from it as follows.

- **Columns.** Previously $\gamma$ was presented as the position relative to $\mathcal{M}_{min}$, so that a step to the right meant crossing the boundary. Now $\gamma$ is the own inflow only. A shortage of inflow acts differently: the potential settles lower, repair weakens, and the reserve may fall below its threshold (Section 11.4.5.1).
- **Full versus frozen memory.** Previously the row was decided by the ratio $\tau_{div}/\tau_{regen}$, the number of times the genome is rewritten per division. Now it is decided by criterion (15.6), and $\tau_{div}/\tau_{regen}$ is an informational indicator. *M. genitalium* has accordingly moved from full to frozen memory, like other bacteria.
- **Empty cells.** Previously marked "forbidden by theorem"; now "expected empty".
- **Borrowed memory.** Previously the host kept adjusting the parasite's memory; now it provides only the starting level.
- **No memory ($\alpha=0$).** Previously "the E-channel is closed". Now only the memory mechanism of the E-channel is off, and with insufficient inflow the potential still falls below its threshold.
- **Seeds and eggs.** Previously they "crossed $\mathcal{M}_{min}$ from below" on awakening. Now their repair is inherited and Barrier I holds in advance; on awakening they cross the resource threshold — the provision of memory and the inflow.
- **Viruses, viroids, prions.** Previously "below $\mathcal{M}_{min}$"; now outside the theory.
- **JCVI-syn3.0 and SAR11.** syn3.0 is a reference point adopted at the boundary, and its position has not been measured; the position of SAR11 is an estimate.
- **The $\mathcal{B}$ scale.** It orders mammals by mass. Absolute values ($\mathcal{B}^*=0.32$ and $\log_{10}(\mathcal{B}/\hbar)\approx30$–$36$ for mammals) and claims of clusters or of deviations from the trend on this scale are removed.
- **Numbers.** *Carsonella* has 182 protein-coding genes and *Hodgkinia* 169; mammals have about 20,000 genes; the $\mathcal{B}/\hbar$ of the mitochondrion is not determined (previously 0). For *Cladosporium*, radiation adds no energy; it changes the parameters $\sigma_0^{eff}$ and $\eta_{abs}$ (Appendix B.4.3).

## Open points of this version

- **Row R5.** In dormancy a system has no inflow of its own, so the column is chosen by convention. Spores and the cryptobiotic tardigrade stand by their active state (column I); lotus seeds and the egg stand in column III, although their potential inflow is large; the predicted "anabiant at the $\mathcal{M}_{min}$ boundary" is described with $\gamma\approx1$ but stands in column II.
- **Szostak protocell.** It stands in column VII, outside the theory, whereas Section 15.4 places it below $\mathcal{M}_{min}$.
- ***Mycoplasma pneumoniae*** is listed among the obligate parasites, although it grows on cell-free media; it is not yet in Appendix B. ***Chlamydia***, with $\gamma\approx0$, stands in column IV ($\gamma$ from 0.1 to 0.3).

## Files

| File | Description |
|---|---|
| `vbls_table.html` | The table itself; needs `vbls_table.css` in the same directory. Click any cell to open the list of its organisms with their cards: $\gamma$, the reserve attractor $R_{bal}$ against the threshold $R^*$, $\tau_{div}/\tau_{regen}$, $\mathcal{B}/\hbar$, and notes. |
| `vbls_table_standalone.html` | The same table as a **single file**, with the stylesheet inlined. Nothing else is required except the Google Fonts link. Use this one for sharing or for opening from a flash drive. |
| `vbls_table.css` | Stylesheet (light/dark theme support via the toggle in the header). |
| `build_standalone.py` | Builds `vbls_table_standalone.html` from `vbls_table.html` and `vbls_table.css`. |

## Usage

Open either HTML file directly in a browser — no build step or server required. `vbls_table.html` needs `vbls_table.css` beside it; `vbls_table_standalone.html` needs nothing.

The two are generated from the same source: the standalone file is `vbls_table.html` with `vbls_table.css` inlined into a `<style>` block by `build_standalone.py`. **Edit `vbls_table.html` and regenerate the standalone version**, never the other way round, or the two will drift apart.

## License

CC BY-NC-ND 4.0 — see [../LICENSE](../LICENSE).
