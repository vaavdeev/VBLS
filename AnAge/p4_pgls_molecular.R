# =============================================================================
#  P4 — molecular PGLS via Open Tree of Life (rotl) + Pagel's lambda (nlme/ape)
#
#  Tests whether sigma0_eff ∝ 1/Rubner depends on body mass M after the
#  phylogeny is accounted for. Uses the REAL OTL synthesis topology (molecular)
#  with Grafen branch lengths (OTL has topology but no dates), then ML lambda.
#  Compare to OLS and to the taxonomic-PGLS (p4_pgls.py).
#
#  M1 (VBLS-native): log10(Rubner) ~ log10(M)   slope = alpha+beta ; sigma0∝M^-slope
#  M2 (multiple):    log10(T_max)  ~ log10(P/M) + log10(M)   b2 ; sigma0∝M^-b2
#  P4 predicts slope = 0 (M1) and b2 = 0 (M2).
# =============================================================================
suppressMessages({library(rotl); library(ape); library(nlme)})

## ---- run from this script's directory -------------------------------------
args <- commandArgs(FALSE)
f <- sub("--file=", "", grep("--file=", args, value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

cache <- "p4_pgls_cache.RData"
if (file.exists(cache)) {
  load(cache)
  cat(sprintf("Loaded cached OTL tree + data: n=%d\n", nrow(d2)))
} else {
## ---- load & filter (== Python 'base' sample) ------------------------------
d <- read.csv("anage_data.csv", sep = ";", fileEncoding = "UTF-8-BOM",
              stringsAsFactors = FALSE, check.names = TRUE)
d$M  <- suppressWarnings(as.numeric(d$`Body.mass..g.`))
d$T  <- suppressWarnings(as.numeric(d$`Maximum.longevity..yrs.`))
d$P  <- suppressWarnings(as.numeric(d$`Metabolic.rate..W.`))
d <- subset(d, Class == "Mammalia" & !is.na(M) & !is.na(T) & !is.na(P) &
                M > 0 & T > 0 & P > 0 &
                `Data.quality` != "low" & Order != "Chiroptera")
d$PM     <- d$P / d$M * 1000
d$Rubner <- d$T * d$PM
d$binom  <- paste(d$Genus, d$Species)
d <- d[!duplicated(d$binom), ]
cat(sprintf("Base sample (Mammalia, quality!=low, excl Chiroptera): n=%d\n", nrow(d)))

## ---- match names to Open Tree, build induced subtree ----------------------
cat("TNRS matching to OTT ...\n")
mt <- tnrs_match_names(d$binom, context_name = "Mammals", do_approximate_matching = TRUE)
mt <- mt[!is.na(mt$ott_id), ]
mt <- mt[!duplicated(mt$ott_id), ]
cat(sprintf("  matched names: %d\n", nrow(mt)))

intree <- is_in_tree(ott_id(mt))
ids <- mt$ott_id[intree]
cat(sprintf("  in synthetic tree: %d\n", length(ids)))

tr <- tol_induced_subtree(ott_ids = ids, label_format = "name")
tr$tip.label <- gsub("_", " ", strip_ott_ids(tr$tip.label, remove_underscores = TRUE))
cat(sprintf("  tree tips: %d\n", length(tr$tip.label)))

## ---- map data rows to tree tips (via TNRS unique_name) ---------------------
key <- data.frame(search = tolower(d$binom), binom = d$binom, stringsAsFactors = FALSE)
mtk <- data.frame(search = mt$search_string, uname = mt$unique_name, stringsAsFactors = FALSE)
key <- merge(key, mtk, by = "search", all.x = TRUE)
d2  <- merge(d, key[, c("binom", "uname")], by = "binom", all.x = TRUE)
d2  <- d2[d2$uname %in% tr$tip.label, ]
d2  <- d2[!duplicated(d2$uname), ]
rownames(d2) <- d2$uname

## prune tree to matched tips, resolve polytomies, Grafen branch lengths
tr <- keep.tip(tr, intersect(tr$tip.label, d2$uname))
tr <- multi2di(tr)
tr <- compute.brlen(tr, method = "Grafen")
d2 <- d2[tr$tip.label, ]
d2$sp     <- factor(rownames(d2), levels = tr$tip.label)
d2$logM   <- log10(d2$M); d2$logPM <- log10(d2$PM)
d2$logT   <- log10(d2$T); d2$logR  <- log10(d2$Rubner)
cat(sprintf("Final analysis set (data ∩ tree): n=%d\n", nrow(d2)))
save(tr, d2, file = cache)
}

## ---- helper to print a gls fit --------------------------------------------
showfit <- function(tag, m, term) {
  tt <- summary(m)$tTable
  lam <- tryCatch({ v <- coef(m$modelStruct$corStruct, unconstrained = FALSE)
                    if (length(v) >= 1) as.numeric(v[1]) else NA_real_ },
                  error = function(e) NA_real_)
  hasl <- length(lam) == 1 && is.finite(lam)
  cat(sprintf("   %-5s %-8s = %+.4f ± %.4f  p=%.2e%s\n", tag, term,
              tt[term, "Value"], tt[term, "Std.Error"], tt[term, "p-value"],
              if (hasl) sprintf("  lambda=%.3f", lam) else ""))
  invisible(tt)
}

cat("\n=== M1:  log Rubner ~ log M   (slope = alpha+beta ; P4: 0) ===\n")
ols1 <- gls(logR ~ logM, data = d2, method = "ML")
pg1  <- gls(logR ~ logM, data = d2, method = "ML",
            correlation = corPagel(1, phy = tr, form = ~sp, fixed = FALSE))
showfit("OLS",  ols1, "logM")
showfit("PGLS", pg1,  "logM")
s1 <- summary(pg1)$tTable["logM", ]
cat(sprintf("   => sigma0_eff ∝ M^(%+.4f)   [P4: exponent 0]\n", -s1["Value"]))

cat("\n=== M2:  log T_max ~ log(P/M) + log M   (b2 [log M] ; P4: 0) ===\n")
ols2 <- gls(logT ~ logPM + logM, data = d2, method = "ML")
pg2  <- gls(logT ~ logPM + logM, data = d2, method = "ML",
            correlation = corPagel(1, phy = tr, form = ~sp, fixed = FALSE))
showfit("OLS",  ols2, "logPM"); showfit("OLS",  ols2, "logM")
showfit("PGLS", pg2,  "logPM"); showfit("PGLS", pg2,  "logM")
s2 <- summary(pg2)$tTable["logM", ]
cat(sprintf("   => sigma0_eff ∝ M^(%+.4f) at fixed metabolism   [P4: exponent 0]\n", -s2["Value"]))

cat("\n================ VERDICT (molecular PGLS, OTL topology) ================\n")
cat(sprintf("  M1 slope(Rubner~M) = %+.3f  p=%.2e  -> %s ; sigma0∝M^%+.3f\n",
            s1["Value"], s1["p-value"], ifelse(s1["p-value"] > .05, "P4 holds", "P4 rejected"),
            -s1["Value"]))
cat(sprintf("  M2 b2[logM]        = %+.3f  p=%.2e  -> %s ; sigma0∝M^%+.3f\n",
            s2["Value"], s2["p-value"], ifelse(s2["p-value"] > .05, "P4 holds", "P4 rejected"),
            -s2["Value"]))
