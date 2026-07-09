-- Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
-- LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
-- SPDX-License-Identifier: CC-BY-NC-ND-4.0

import Lake
open Lake DSL

package «vbls» where
  moreLeanArgs := #[]

-- Specific mathlib4 commit, compatible with Lean 4.14.0
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "4bbdccd9c5f862bf90ff12f0a9e2c8be032b9a84"

lean_lib «VBLSTheorem» where
  roots := #[`VBLS_theorem]
  moreLeanArgs := #[]