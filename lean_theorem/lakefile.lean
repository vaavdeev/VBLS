import Lake
open Lake DSL

package «vbls» where
  moreLeanArgs := #[]

-- Конкретный коммит mathlib4, совместимый с Lean 4.14.0
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "4bbdccd9c5f862bf90ff12f0a9e2c8be032b9a84"

lean_lib «VBLSTheorem» where
  roots := #[`VBLS_theorem]
  moreLeanArgs := #[]