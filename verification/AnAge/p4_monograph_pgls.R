# =============================================================================
#  VBLS — Viability Boundaries of Living Systems
#  Author:     Vasiliy Avdeev (independent researcher)
#  Monograph:  VBLS_THEORY
# =============================================================================
#
#  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
#  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
#  SPDX-License-Identifier: CC-BY-NC-ND-4.0
# =============================================================================
#  p4_monograph_pgls.R — PGLS for the regressions actually used in the VBLS
#  monograph (§17.4.1, Appendix D.6.7; falsification criterion F5).
#
#  WHY THIS SCRIPT EXISTS
#  ----------------------
#  p4_pgls.py / p4_pgls_molecular.R / p4_pgls_datelife.R tested the P4
#  prediction through M1 (log Rubner ~ log M) and M2. M1 delivers only the SUM
#  a + beta. The monograph reports a, beta and p_scale SEPARATELY, and F5 is
#  stated on a alone. Those separate fits have never been run under PGLS.
#
#  Fitted here, OLS vs PGLS, on three trees:
#
#    a        :  log T_max  ~ log M          F5 requires a in [0.15, 0.35]
#    beta     :  log(P/M)   ~ log M          Kleiber input; beta = chi - 1
#    p_scale  :  log T_max  ~ log(P/M)       VBLS predicts -1.000
#    chi      :  log P      ~ log M          Kleiber; structural law a = 1 - chi
#    a+beta   :  log Rubner ~ log M          CONTROL, must reproduce p4 values
#
#  VBLS predicts a = 1 - chi and beta = chi - 1, hence
#
#        a + beta = 0   identically, for any chi
#
#  which makes the control regression the only Block B quantity independent of
#  the 2/3-vs-3/4 dispute over the metabolic exponent.
#
#  Requires: ape, nlme.  Caches: p4_pgls_cache.RData, p4_datelife_tree.RData.
#  No network access.
#
#  Usage:  Rscript p4_monograph_pgls.R  > p4_monograph_pgls.out.txt
# =============================================================================

suppressMessages({library(ape); library(nlme)})
options(width = 130)

## The taxonomic proxy tree is almost entirely polytomous and multi2di resolves
## polytomies AT RANDOM, so its PGLS slopes move between runs (observed spread
## in `a`: 0.148 vs 0.159, which straddles the F5 boundary). The seed makes a
## single run reproducible; TAXREP quantifies how much the choice matters.
## The OTL and dated trees are read from cache and are not affected.
set.seed(20260728)
TAXREP <- 25

args <- commandArgs(FALSE)
f <- sub("--file=", "", grep("--file=", args, value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

## reference values to check against -----------------------------------------
REF <- list(
  a       = c(val = +0.218, se = 0.011),   # monograph §17.4.1, N=318, OLS
  beta    = c(val = -0.277, se = 0.009),
  p_scale = c(val = -0.656, se = 0.035),
  ab_pgls = -0.1185                        # p4_pgls_datelife.out.txt, n=305
)
F5_LO <- 0.15; F5_HI <- 0.35

# =============================================================================
#  1. DATA
# =============================================================================

load_base <- function() {
  d <- read.csv("anage_data.csv", sep = ";", fileEncoding = "UTF-8-BOM",
                stringsAsFactors = FALSE, check.names = TRUE)
  d$M <- suppressWarnings(as.numeric(d$`Body.mass..g.`))
  d$T <- suppressWarnings(as.numeric(d$`Maximum.longevity..yrs.`))
  d$P <- suppressWarnings(as.numeric(d$`Metabolic.rate..W.`))
  d <- subset(d, Class == "Mammalia" & !is.na(M) & !is.na(T) & !is.na(P) &
                 M > 0 & T > 0 & P > 0 &
                 `Data.quality` != "low" & Order != "Chiroptera")
  d$PM     <- d$P / d$M * 1000          # W/kg  (M is in grams)
  d$Rubner <- d$T * d$PM
  d$binom  <- paste(d$Genus, d$Species)
  d <- d[!duplicated(d$binom), ]
  addlogs(d)
}

addlogs <- function(d) {
  d$logM  <- log10(d$M)
  d$logPM <- log10(d$PM)
  d$logT  <- log10(d$T)
  d$logR  <- log10(d$Rubner)
  d$logP  <- log10(d$P)                 # chi:  logP = logPM + logM + const
  d
}

## taxonomic tree from Order/Family/Genus (proxy; mirrors p4_pgls.py) ---------
build_tax <- function(d) {
  d$tipid <- gsub("[^A-Za-z0-9]+", "_", d$binom)
  d <- d[!duplicated(d$tipid), ]
  fam <- ifelse(is.na(d$Family) | d$Family == "", "unkFamily", d$Family)
  fr  <- data.frame(Order  = factor(d$Order),
                    Family = factor(fam),
                    Genus  = factor(d$Genus),
                    tipid  = factor(d$tipid))
  tr <- as.phylo(~Order/Family/Genus/tipid, data = fr, collapse = FALSE)
  tr <- multi2di(tr)
  tr$edge.length <- NULL
  tr <- compute.brlen(tr, method = "Grafen")
  rownames(d) <- d$tipid
  d <- d[tr$tip.label, ]
  d$sp <- factor(rownames(d), levels = tr$tip.label)
  list(dat = d, tree = tr, tag = sprintf("TAX  n=%d", nrow(d)))
}

## align an arbitrary tree with the cached d2 --------------------------------
align <- function(tree, d2, tag) {
  tree$tip.label <- gsub("_", " ", tree$tip.label)
  tree$edge.length[tree$edge.length <= 0] <- 1e-6
  keep <- intersect(tree$tip.label, rownames(d2))
  tree <- keep.tip(tree, keep)
  g <- d2[tree$tip.label, ]
  g$sp <- factor(rownames(g), levels = tree$tip.label)
  list(dat = g, tree = tree, tag = sprintf("%s n=%d", tag, nrow(g)))
}

cat("== loading ==\n")
base <- load_base()
cat(sprintf("   base sample (Mammalia, quality!=low, excl Chiroptera): n=%d\n", nrow(base)))

stopifnot(file.exists("p4_pgls_cache.RData"))
load("p4_pgls_cache.RData")             # -> tr (OTL, Grafen), d2
d2 <- addlogs(d2)
cat(sprintf("   OTL cache: n=%d\n", nrow(d2)))

## quality strata.  AnAge vocabulary here is high / acceptable / questionable;
## there is no category literally called "good", so criterion F5's wording is
## read two ways and both are reported.
strict111 <- function(x) x[x$Data.quality == "acceptable" &
                           x$Sample.size %in% c("medium", "large"), ]
high35    <- function(x) x[x$Data.quality == "high", ]

SETS <- list()
SETS$TAX <- tryCatch(build_tax(base),
                     error = function(e) { cat("   ! taxonomic tree failed:",
                                               conditionMessage(e), "\n"); NULL })
SETS$OTL <- align(tr, d2, "OTL ")

if (file.exists("p4_datelife_tree.RData")) {
  load("p4_datelife_tree.RData")        # -> dated
  SETS$DATED <- align(dated, d2, "DATED")
  cat(sprintf("   dated tree: root age = %.1f Myr\n",
              max(node.depth.edgelength(SETS$DATED$tree))))
} else {
  dated <- NULL
  cat("   ! p4_datelife_tree.RData absent -- dated tree skipped\n")
}

## strict strata, on the taxonomic tree (self-contained) and on the dated tree
SETS$TAX111   <- tryCatch(build_tax(strict111(base)), error = function(e) NULL)
SETS$TAX35    <- tryCatch(build_tax(high35(base)),    error = function(e) NULL)
if (!is.null(dated)) {
  SETS$DATED111 <- tryCatch(align(dated, strict111(d2), "DAT111"), error = function(e) NULL)
  SETS$DATED35  <- tryCatch(align(dated, high35(d2),    "DAT35 "), error = function(e) NULL)
}
SETS <- SETS[!vapply(SETS, is.null, logical(1))]
SETS <- SETS[vapply(SETS, function(s) nrow(s$dat) >= 20, logical(1))]

# =============================================================================
#  2. FITTING
# =============================================================================

## one response ~ one predictor, OLS and PGLS ---------------------------------
fit1 <- function(dat, tree, resp, pred) {
  fml <- as.formula(sprintf("%s ~ %s", resp, pred))

  o  <- lm(fml, data = dat)
  co <- summary(o)$coefficients[pred, ]

  pg <- NULL
  for (start in c(1, 0.5, 0.1)) {       # corPagel can fail from the boundary
    pg <- tryCatch(gls(fml, data = dat, method = "ML",
                       correlation = corPagel(start, phy = tree,
                                              form = ~sp, fixed = FALSE)),
                   error = function(e) NULL)
    if (!is.null(pg)) break
  }
  if (is.null(pg))
    return(data.frame(ols = co[1], ols_se = co[2], ols_p = co[4],
                      pgls = NA, pgls_se = NA, pgls_p = NA, lambda = NA))

  cp  <- summary(pg)$tTable[pred, ]
  lam <- tryCatch(as.numeric(coef(pg$modelStruct$corStruct,
                                  unconstrained = FALSE)[1]),
                  error = function(e) NA_real_)
  data.frame(ols = co[1], ols_se = co[2], ols_p = co[4],
             pgls = cp["Value"], pgls_se = cp["Std.Error"],
             pgls_p = cp["p-value"], lambda = lam)
}

TARGETS <- list(
  a        = c(resp = "logT",  pred = "logM"),
  beta     = c(resp = "logPM", pred = "logM"),
  p_scale  = c(resp = "logT",  pred = "logPM"),
  chi      = c(resp = "logP",  pred = "logM"),
  `a+beta` = c(resp = "logR",  pred = "logM")     # CONTROL
)

ALL <- list()
for (sname in names(SETS)) {
  S <- SETS[[sname]]
  cat(sprintf("\n\n==================== %s ====================\n", S$tag))
  cat(sprintf("%-8s %22s   %30s\n", "", "OLS", "PGLS"))
  cat(strrep("-", 78), "\n")
  rows <- list()
  for (tn in names(TARGETS)) {
    t <- TARGETS[[tn]]
    r <- fit1(S$dat, S$tree, t[["resp"]], t[["pred"]])
    rows[[tn]] <- r
    cat(sprintf("%-8s %+.4f +- %.4f (p=%7.1e)   %+.4f +- %.4f (p=%7.1e) lam=%s\n",
                tn, r$ols, r$ols_se, r$ols_p,
                r$pgls, r$pgls_se, r$pgls_p,
                ifelse(is.na(r$lambda), "  n/a", sprintf("%.3f", r$lambda))))
  }
  R <- do.call(rbind, rows); R$quantity <- names(rows); R$set <- S$tag
  ALL[[sname]] <- R

  ## ---- internal consistency ------------------------------------------------
  ## beta = chi - 1 is an algebraic identity in BOTH fits: logPM and logP differ
  ## by logM plus a constant, so it must hold to machine precision. A failure
  ## here means the data are misaligned with the tree tips.
  ##
  ## a + beta = control holds exactly only under a SHARED covariance matrix,
  ## i.e. for OLS. Under PGLS each regression estimates its own lambda, so GLS
  ## slopes are not additive and a small residual is expected, not an error.
  cat("\n  consistency:\n")
  for (m in c("ols", "pgls")) {
    d_chi <- R["beta", m] + 1 - R["chi", m]
    cat(sprintf("    %-4s beta + 1 - chi     = %+.2e   (exact identity)\n",
                toupper(m), d_chi))
    if (isTRUE(abs(d_chi) > 1e-6))
      cat("    ! ALIGNMENT ERROR: data rows do not match tree tips\n")
    if (is.na(d_chi))
      cat("    ! PGLS did not converge for at least one of beta / chi\n")
  }
  d_sum <- R["a", "ols"] + R["beta", "ols"] - R["a+beta", "ols"]
  cat(sprintf("    OLS  a + beta - control = %+.2e   (exact identity)\n", d_sum))
  if (isTRUE(abs(d_sum) > 1e-6))
    cat("    ! ALIGNMENT ERROR: OLS additivity broken\n")
  cat(sprintf("    PGLS a + beta - control = %+.2e   (not an identity: lambda differs per fit)\n",
              R["a", "pgls"] + R["beta", "pgls"] - R["a+beta", "pgls"]))
  lams <- R$lambda[!is.na(R$lambda)]
  if (length(lams) && max(lams) > 0.999)
    cat("    ! lambda at the upper boundary in at least one fit -- SE understated there\n")
}

# =============================================================================
#  3. CHECKS AND VERDICT
# =============================================================================

cat("\n\n================== VALIDATION AGAINST PUBLISHED VALUES ==================\n")
T1 <- ALL[[1]]
cat(sprintf("  OLS a       = %+.4f   monograph §17.4.1: %+.3f +- %.3f\n",
            T1["a", "ols"], REF$a["val"], REF$a["se"]))
cat(sprintf("  OLS beta    = %+.4f   monograph §17.4.1: %+.3f +- %.3f\n",
            T1["beta", "ols"], REF$beta["val"], REF$beta["se"]))
cat(sprintf("  OLS p_scale = %+.4f   monograph §17.4.1: %+.3f +- %.3f\n",
            T1["p_scale", "ols"], REF$p_scale["val"], REF$p_scale["se"]))
if (!is.null(ALL$DATED)) {
  ctl <- ALL$DATED["a+beta", "pgls"]
  cat(sprintf("\n  CONTROL  PGLS a+beta (dated) = %+.4f   p4_pgls_datelife: %+.4f  -> %s\n",
              ctl, REF$ab_pgls,
              ifelse(abs(ctl - REF$ab_pgls) < 0.005, "reproduced", "MISMATCH")))
}

cat("\n\n=========================== VERDICT ON F5 ===========================\n")
cat(sprintf("  F5 corridor: a in [%.2f, %.2f]\n\n", F5_LO, F5_HI))
for (sname in names(ALL)) {
  R <- ALL[[sname]]
  for (m in c("ols", "pgls")) {
    v  <- R["a", m]; se <- R["a", paste0(m, "_se")]
    lo <- v - 1.96 * se; hi <- v + 1.96 * se
    cat(sprintf("  %-14s %-4s a = %+.4f  95%% CI [%+.4f, %+.4f]  -> %s\n",
                R$set[1], toupper(m), v, lo, hi,
                if (is.na(v)) "n/a"
                else if (v >= F5_LO && v <= F5_HI) "PASSES"
                else "FAILS"))
  }
}

cat("\n  Rubner invariant (VBLS: a + beta = 0):\n")
for (sname in names(ALL)) {
  R <- ALL[[sname]]
  cat(sprintf("  %-14s OLS %+.4f +- %.4f   PGLS %+.4f +- %.4f\n",
              R$set[1], R["a+beta", "ols"], R["a+beta", "ols_se"],
              R["a+beta", "pgls"], R["a+beta", "pgls_se"]))
}

## ---- how much does random polytomy resolution move the taxonomic result? ---
cat("\n\n============ POLYTOMY-RESOLUTION SPREAD (taxonomic tree, base n=318) ============\n")
va <- vb <- numeric(0)
for (i in seq_len(TAXREP)) {
  S <- tryCatch(build_tax(base), error = function(e) NULL)
  if (is.null(S)) next
  va <- c(va, fit1(S$dat, S$tree, "logT", "logM")$pgls)
  vb <- c(vb, fit1(S$dat, S$tree, "logR", "logM")$pgls)
}
cat(sprintf("  %d resolutions\n", length(va)))
cat(sprintf("  a       PGLS: median %+.4f   range [%+.4f, %+.4f]   sd %.4f\n",
            median(va), min(va), max(va), sd(va)))
cat(sprintf("  a+beta  PGLS: median %+.4f   range [%+.4f, %+.4f]   sd %.4f\n",
            median(vb), min(vb), max(vb), sd(vb)))
cat(sprintf("  F5 (point estimate in [%.2f, %.2f]) passes in %d of %d resolutions\n",
            F5_LO, F5_HI, sum(va >= F5_LO & va <= F5_HI), length(va)))
cat("  -> the taxonomic proxy is not decisive on its own; OTL and dated trees are.\n")

OUT <- do.call(rbind, ALL)
OUT <- OUT[, c("set", "quantity", "ols", "ols_se", "ols_p",
               "pgls", "pgls_se", "pgls_p", "lambda")]
write.csv(OUT, "p4_monograph_pgls_results.csv", row.names = FALSE)
cat("\n  results written to p4_monograph_pgls_results.csv\n")
