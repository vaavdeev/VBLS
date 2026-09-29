# VBLS — Table of Living Structures

Interactive HTML version of the systematic classification introduced in the monograph's **Appendix C** (§C.1–C.9): a periodic-table-style classification of biological and sub-biological objects by their position relative to the viability boundary $\mathcal{M}_{min}$.

The two axes of the table are the **type of memory** (rows) and **position relative to $\mathcal{M}_{min}$** — the dimensionless autonomous-flux criterion $\gamma$ (columns). Most cells are populated with real organisms; some are forbidden by the finite-lifetime theorem; the remaining empty cells are VBLS predictions — classes of living structures the theory permits but that have not yet been identified.

## Files

| File | Description |
|---|---|
| `vbls_table.html` | The table itself; needs `vbls_table.css` in the same directory. Click any cell to open a modal with the full VBLS parameters for that organism/class: γ, the reserve attractor $R_{bal}$ against the threshold $R^*$, $\tau_{div}/\tau_{regen}$, $\mathcal{B}/\hbar$, and notes. |
| `vbls_table_standalone.html` | The same table as a **single file**, with the stylesheet inlined. Nothing else is required except the Google Fonts link. Use this one for sharing or for opening from a flash drive. |
| `vbls_table.css` | Stylesheet (light/dark theme support via the toggle in the header). |

## Usage

Open either file directly in a browser — no build step or server required. `vbls_table.html` needs `vbls_table.css` beside it; `vbls_table_standalone.html` needs nothing.

The two are generated from the same source: the standalone file is `vbls_table.html` with `vbls_table.css` inlined into a `<style>` block. **Edit `vbls_table.html` and regenerate the standalone version**, never the other way round, or the two will drift apart.

## License

CC BY-NC-ND 4.0 — see [../LICENSE](../LICENSE).
