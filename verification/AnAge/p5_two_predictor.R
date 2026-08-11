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
#  p5_two_predictor.R — T_max on metabolic power and mass as SEPARATE
#  predictors (Appendix D.6.9).
#
#  WHY THIS SCRIPT EXISTS
#  ----------------------
#  The monograph estimates p_scale from
#
#        log T_max  ~  log(P/M)                                  (restricted)
#
#  which already assumes that only the RATIO matters: the same coefficient,
#  with opposite signs, multiplies log P and log M. That assumption has never
#  been tested. The unrestricted model is
#
#        log T_max  =  c + b1 * log P + b2 * log M               (full)
#
#  and the restriction is b1 + b2 = 0. VBLS predicts the stronger statement
#  b1 = -1, b2 = +1 (i.e. T_max proportional to (P/M)^-1).
#
#  Two tests are reported for every tree and stratum, OLS and PGLS:
#
#     H_ratio : b1 + b2 = 0          does only the ratio matter?
#     H_vbls  : b1 = -1 AND b2 = +1  is the ratio exponent -1?
#
#  H_ratio is the one that decides whether writing the prediction in terms of
#  P/M is legitimate at all. H_vbls is the criterion the monograph already
#  reports through p_scale, restated without the restriction.
#
#  COLLINEARITY. log P and log M are correlated at about r = 0.99 (Kleiber),
#  so b1 and b2 are individually ill-determined and their standard errors are
#  large. This does NOT weaken the test of b1 + b2 = 0: the sum is estimated
#  along the well-determined direction. The orthogonal reparametrization
#
#        log T_max  =  c + A * log M + B * u,    u = resid(log P ~ log M)
#
#  makes this explicit -- A is exactly the simple slope a, B is exactly b1,
#  and the two are uncorrelated by construction. Both are printed so that the
#  reader can see which part of the data each coefficient comes from.
#
#  Requires: ape, nlme.  Caches: p4_pgls_cache.RData, p4_datelife_tree.RData.
#  No network access.  Data loading is identical to p4_monograph_pgls.R.
#
#  Usage:  Rscript p5_two_predictor.R  > p5_two_predictor.out.txt
# =============================================================================

suppressMessages({library(ape); library(nlme)})
options(width = 130)

set.seed(20260728)          # taxonomic polytomies are resolved at random

args <- commandArgs(FALSE)
f <- sub("--file=", "", grep("--file=", args, value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

## published single-predictor values, for orientation -------------------------
REF <- list(a = +0.2173, beta = -0.2737, p_scale = -0.6554, chi = +0.7263)

# =============================================================================
#  1. DATA  (identical to p4_monograph_pgls.R)
# =============================================================================

addlogs <- function(d) {
  d$logM  <- log10(d$M)
  d$logPM <- log10(d$PM)
  d$logT  <- log10(d$T)
  d$logR  <- log10(d$Rubner)
  d$logP  <- log10(d$P)
  d
}

load_base <- function() {
  d <- read.csv("anage_data.csv", sep = ";", fileEncoding = "UTF-8-BOM",
                stringsAsFactors = FALSE, check.names = TRUE)
  d$M <- suppressWarnings(as.numeric(d$`Body.mass..g.`))
  d$T <- suppressWarnings(as.numeric(d$`Maximum.longevity..yrs.`))
  d$P <- suppressWarnings(as.numeric(d$`Metabolic.rate..W.`))
  d <- subset(d, Class == "Mammalia" & !is.na(M) & !is.na(T) & !is.na(P) &
                 M > 0 & T > 0 & P > 0 &
                 `Data.quality` != "low" & Order != "Chiroptera")
  d$PM     <- d$P / d$M * 1000
  d$Rubner <- d$T * d$PM
  d$binom  <- paste(d$Genus, d$Species)
  d <- d[!duplicated(d$binom), ]
  addlogs(d)
}

build_tax <- function(d) {
  d$tipid <- gsub("[^A-Za-z0-9]+", "_", d$binom)
  d <- d[!duplicated(d$tipid), ]
  fam <- ifelse(is.na(d$Family) | d$Family == "", "unkFamily", d$Family)
  fr  <- data.frame(Order  = factor(d$Order),  Family = factor(fam),
                    Genus  = factor(d$Genus),  tipid  = factor(d$tipid))
  tr <- as.phylo(~Order/Family/Genus/tipid, data = fr, collapse = FALSE)
  tr <- multi2di(tr)
  tr$edge.length <- NULL
  tr <- compute.brlen(tr, method = "Grafen")
  rownames(d) <- d$tipid
  d <- d[tr$tip.label, ]
  d$sp <- factor(rownames(d), levels = tr$tip.label)
  list(dat = d, tree = tr, tag = sprintf("TAX  n=%d", nrow(d)))
}

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

strict111 <- function(x) x[x$Data.quality == "acceptable" &
                           x$Sample.size %in% c("medium", "large"), ]
high35    <- function(x) x[x$Data.quality == "high", ]

SETS <- list()
SETS$TAX <- tryCatch(build_tax(base), error = function(e) NULL)
SETS$OTL <- align(tr, d2, "OTL ")

if (file.exists("p4_datelife_tree.RData")) {
  load("p4_datelife_tree.RData")        # -> dated
  SETS$DATED <- align(dated, d2, "DATED")
} else {
  dated <- NULL
  cat("   ! p4_datelife_tree.RData absent -- dated tree skipped\n")
}

SETS$TAX111 <- tryCatch(build_tax(strict111(base)), error = function(e) NULL)
SETS$TAX35  <- tryCatch(build_tax(high35(base)),    error = function(e) NULL)
if (!is.null(dated)) {
  SETS$DATED111 <- tryCatch(align(dated, strict111(d2), "DAT111"), error = function(e) NULL)
  SETS$DATED35  <- tryCatch(align(dated, high35(d2),    "DAT35 "), error = function(e) NULL)
}
SETS <- SETS[!vapply(SETS, is.null, logical(1))]
SETS <- SETS[vapply(SETS, function(s) nrow(s$dat) >= 20, logical(1))]

# =============================================================================
#  2. FITTING AND LINEAR HYPOTHESES
# =============================================================================

## Wald test of L %*% beta = rhs, computed from coef() and vcov() so that the
## same code serves lm and gls. Reported as F on (q, n-k) df; for q = 1 this is
## the square of the usual t statistic.
wald <- function(fit, L, rhs, n, k) {
  b <- coef(fit)
  if (inherits(fit, "gls")) b <- b[names(b)]
  V <- vcov(fit)
  L <- matrix(L, nrow = length(rhs), byrow = TRUE)
  d <- as.vector(L %*% b - rhs)
  M <- L %*% V %*% t(L)
  q <- length(rhs)
  Fs <- as.vector(t(d) %*% solve(M) %*% d) / q
  list(est = d, se = sqrt(diag(M)), F = Fs, q = q,
       p = pf(Fs, q, n - k, lower.tail = FALSE))
}

## unrestricted fit, OLS and PGLS ---------------------------------------------
fit2 <- function(dat, tree) {
  n <- nrow(dat)
  out <- list(n = n)

  out$ols <- lm(logT ~ logP + logM, data = dat)

  pg <- NULL
  for (start in c(1, 0.5, 0.1)) {
    pg <- tryCatch(gls(logT ~ logP + logM, data = dat, method = "ML",
                       correlation = corPagel(start, phy = tree,
                                              form = ~sp, fixed = FALSE)),
                   error = function(e) NULL)
    if (!is.null(pg)) break
  }
  out$pgls <- pg

  ## restricted (what the monograph actually fits)
  out$ols_r <- lm(logT ~ logPM, data = dat)
  pr <- NULL
  for (start in c(1, 0.5, 0.1)) {
    pr <- tryCatch(gls(logT ~ logPM, data = dat, method = "ML",
                       correlation = corPagel(start, phy = tree,
                                              form = ~sp, fixed = FALSE)),
                   error = function(e) NULL)
    if (!is.null(pr)) break
  }
  out$pgls_r <- pr
  out
}

lam_of <- function(g) tryCatch(as.numeric(coef(g$modelStruct$corStruct,
                                               unconstrained = FALSE)[1]),
                               error = function(e) NA_real_)

report <- function(sname, S) {
  cat(sprintf("\n\n==================== %s ====================\n", S$tag))
  d <- S$dat; n <- nrow(d)
  FF <- fit2(d, S$tree)

  ## --- collinearity, stated before the coefficients so it is not a footnote --
  r  <- cor(d$logP, d$logM)
  vif <- 1 / (1 - r^2)
  cat(sprintf("  collinearity: r(logP, logM) = %.4f   VIF = %.1f\n", r, vif))
  cat(sprintf("                Var(logM) = %.4f   Var(u) = %.5f   ratio = %.1f\n",
              var(d$logM), var(resid(lm(logP ~ logM, data = d))),
              var(d$logM) / var(resid(lm(logP ~ logM, data = d)))))

  rows <- list()
  for (m in c("ols", "pgls")) {
    fit <- FF[[m]]; fitr <- FF[[paste0(m, "_r")]]
    if (is.null(fit)) { cat(sprintf("\n  %s: did not converge\n", toupper(m))); next }
    b  <- coef(fit); se <- sqrt(diag(vcov(fit)))
    lam <- if (m == "pgls") lam_of(fit) else NA_real_

    cat(sprintf("\n  %-4s  b1 (logP) = %+.4f +- %.4f     b2 (logM) = %+.4f +- %.4f%s\n",
                toupper(m), b["logP"], se["logP"], b["logM"], se["logM"],
                if (m == "pgls") sprintf("   lam=%.3f", lam) else ""))

    ## H_ratio : b1 + b2 = 0
    h1 <- wald(fit, c(0, 1, 1), 0, n, 3)
    cat(sprintf("        H_ratio  b1 + b2 = 0 : est %+.4f +- %.4f   F(1,%d) = %7.2f   p = %.3g   -> %s\n",
                h1$est, h1$se, n - 3, h1$F, h1$p,
                ifelse(h1$p < 0.05, "REJECTED", "not rejected")))

    ## H_vbls : b1 = -1 and b2 = +1
    h2 <- wald(fit, c(0, 1, 0,
                      0, 0, 1), c(-1, 1), n, 3)
    cat(sprintf("        H_vbls   b1=-1,b2=+1 : F(2,%d) = %7.2f   p = %.3g   -> %s\n",
                n - 3, h2$F, h2$p, ifelse(h2$p < 0.05, "REJECTED", "not rejected")))

    ## restricted slope, i.e. p_scale as the monograph reports it
    if (!is.null(fitr)) {
      br <- coef(fitr)["logPM"]; ser <- sqrt(diag(vcov(fitr)))["logPM"]
      cat(sprintf("        restricted p_scale  = %+.4f +- %.4f   (this is what the monograph fits)\n",
                  br, ser))
    }

    ## orthogonal reparametrization: which data direction carries which slope
    if (m == "ols") {
      u  <- resid(lm(logP ~ logM, data = d))
      o2 <- lm(d$logT ~ d$logM + u)
      cat(sprintf("        orthogonal form: A (logM) = %+.4f   B (u) = %+.4f   [A must equal simple a; B must equal b1]\n",
                  coef(o2)[2], coef(o2)[3]))
      a_simple <- coef(lm(logT ~ logM, data = d))["logM"]
      if (abs(coef(o2)[2] - a_simple) > 1e-9 || abs(coef(o2)[3] - b["logP"]) > 1e-9)
        cat("        ! reparametrization identity broken -- check the fit\n")

      ## the OLS F-test of the nested pair must equal the Wald test above
      av <- anova(fitr, fit)
      cat(sprintf("        nested F-test (restricted vs full) F = %7.2f  p = %.3g   [must match H_ratio]\n",
                  av$F[2], av$`Pr(>F)`[2]))
      if (abs(av$F[2] - h1$F) > 1e-6)
        cat("        ! Wald and F disagree -- numerical problem\n")
    }

    rows[[m]] <- data.frame(set = S$tag, method = toupper(m),
                            n = n, b1 = b["logP"], b1_se = se["logP"],
                            b2 = b["logM"], b2_se = se["logM"],
                            sum = h1$est, sum_se = h1$se, sum_p = h1$p,
                            vbls_F = h2$F, vbls_p = h2$p,
                            p_scale_restr = if (!is.null(fitr)) coef(fitr)["logPM"] else NA,
                            lambda = lam, row.names = NULL)
  }
  do.call(rbind, rows)
}

ALL <- list()
for (sname in names(SETS)) ALL[[sname]] <- report(sname, SETS[[sname]])

# =============================================================================
#  3. SUMMARY
# =============================================================================

OUT <- do.call(rbind, ALL)

cat("\n\n===================== SUMMARY: RATIO RESTRICTION =====================\n")
cat("  VBLS writes the prediction as T ~ (P/M)^p. That is legitimate only if\n")
cat("  b1 + b2 = 0. A rejection means P and M act on lifespan through more\n")
cat("  than their ratio, and p_scale is then a constrained summary, not the\n")
cat("  full description.\n\n")
cat(sprintf("  %-14s %-5s %5s %10s %10s %10s   %s\n",
            "set", "meth", "n", "b1+b2", "se", "p", "verdict"))
cat(" ", strrep("-", 74), "\n")
for (i in seq_len(nrow(OUT)))
  cat(sprintf("  %-14s %-5s %5d %+10.4f %10.4f %10.3g   %s\n",
              OUT$set[i], OUT$method[i], OUT$n[i], OUT$sum[i], OUT$sum_se[i],
              OUT$sum_p[i], ifelse(OUT$sum_p[i] < 0.05, "REJECTED", "not rejected")))

cat("\n\n===================== SUMMARY: VBLS POINT PREDICTION =====================\n")
cat("  H_vbls is b1 = -1 and b2 = +1 jointly, i.e. p_scale = -1 without\n")
cat("  assuming the ratio form in advance.\n\n")
cat(sprintf("  %-14s %-5s %10s %10s %12s %10s\n",
            "set", "meth", "b1", "b2", "F(2,n-3)", "p"))
cat(" ", strrep("-", 66), "\n")
for (i in seq_len(nrow(OUT)))
  cat(sprintf("  %-14s %-5s %+10.4f %+10.4f %12.2f %10.3g\n",
              OUT$set[i], OUT$method[i], OUT$b1[i], OUT$b2[i],
              OUT$vbls_F[i], OUT$vbls_p[i]))

cat("\n\n  Reference single-predictor values (monograph / p4_monograph_pgls.R, OLS):\n")
cat(sprintf("    a = %+.4f   beta = %+.4f   chi = %+.4f   p_scale = %+.4f\n",
            REF$a, REF$beta, REF$chi, REF$p_scale))

# =============================================================================
#  4. WHERE p_scale COMES FROM, AND WHAT WOULD RESCUE b1 = -1
# =============================================================================
#
#  In the orthogonal form the two predictors are log M and u = resid(logP~logM),
#  which are uncorrelated by construction. The restricted slope p_scale is then
#  a variance-weighted average of two quite different quantities:
#
#     along mass          a / beta      (allometric axis; already in a and beta)
#     at fixed mass       b1            (the genuine metabolic-rate effect)
#
#  with weights beta^2 Var(logM) and Var(u). Reporting the average alone hides
#  that the two disagree.
#
#  Attenuation. log M is essentially error-free; recorded metabolic rate is not.
#  Because log M and u are orthogonal, error in log P attenuates b1 by the
#  ordinary single-predictor factor Var(u_true)/Var(u_obs). Var(u_obs) is small
#  (about 3% of Var(log M)), so the SAME absolute measurement error bites far
#  harder here than it does on p_scale. The question worth answering is not
#  "is there attenuation" but "how much would be needed".

cat("\n\n============ DECOMPOSITION OF p_scale AND ATTENUATION REQUIRED ============\n")
for (sname in names(SETS)) {
  S <- SETS[[sname]]; d <- S$dat
  if (nrow(d) < 100) next
  uu   <- resid(lm(logP ~ logM, data = d))
  Vu   <- var(uu); VM <- var(d$logM)
  bet  <- coef(lm(logPM ~ logM, data = d))["logM"]
  aa   <- coef(lm(logT  ~ logM, data = d))["logM"]
  b1   <- coef(lm(logT ~ logP + logM, data = d))["logP"]
  wM   <- bet^2 * VM; wU <- Vu
  pmix <- (wM * (aa / bet) + wU * b1) / (wM + wU)
  pobs <- coef(lm(logT ~ logPM, data = d))["logPM"]

  cat(sprintf("\n  %s\n", S$tag))
  cat(sprintf("    along mass      a/beta = %+.4f   weight %.1f%%\n",
              aa / bet, 100 * wM / (wM + wU)))
  cat(sprintf("    at fixed mass   b1     = %+.4f   weight %.1f%%\n",
              b1, 100 * wU / (wM + wU)))
  cat(sprintf("    weighted mix           = %+.4f   fitted p_scale = %+.4f   (diff %.1e)\n",
              pmix, pobs, abs(pmix - pobs)))

  ## reliability needed for b1 = -1
  kappa <- as.numeric(b1) / (-1)
  if (kappa > 0 && kappa < 1) {
    Ve    <- (1 - kappa) * Vu
    sde   <- sqrt(Ve)
    cvpct <- 100 * sqrt(exp((sde * log(10))^2) - 1)
    cat(sprintf("    for b1 = -1: %.0f%% of the mass-independent variance in log P\n",
                100 * (1 - kappa)))
    cat(sprintf("                 must be measurement error, i.e. sd = %.3f dex, CV = %.0f%%\n",
                sde, cvpct))
  }
}

write.csv(OUT, "p5_two_predictor_results.csv", row.names = FALSE)
cat("\n  results written to p5_two_predictor_results.csv\n")
