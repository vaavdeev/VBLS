# =============================================================================
#  P4 — molecular PGLS on a REAL DATED tree (datelife) — gold finish.
#
#  Takes the OTL synthesis topology (cached from p4_pgls_molecular.R) and dates
#  it with real divergence times via datelife (chronogram database + BLADJ),
#  then re-runs the Pagel-lambda PGLS. This replaces the Grafen (topology-only)
#  dating of §12.6 with real ages.  Compare slopes/lambda to Grafen result.
# =============================================================================
suppressMessages({library(datelife); library(ape); library(nlme)})
args <- commandArgs(FALSE); f <- sub("--file=", "", grep("--file=", args, value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

stopifnot(file.exists("p4_pgls_cache.RData"))
load("p4_pgls_cache.RData")              # tr (OTL topology, Grafen brlen), d2 (data)
cat(sprintf("OTL topology + data: %d tips\n", length(tr$tip.label)))

rcache <- "p4_datelife_result.RData"
dcache <- "p4_datelife_tree.RData"
if (file.exists(dcache)) {
  load(dcache); cat("Loaded cached datelife-dated tree.\n")
} else {
  if (file.exists(rcache)) {
    load(rcache); cat("Loaded cached datelife query result.\n")
  } else {
    cat("Querying chronogram database (datelife) ...\n")
    query  <- make_datelife_query(input = tr$tip.label, use_tnrs = FALSE)
    result <- get_datelife_result(query)
    save(query, result, file = rcache)
  }
  dated <- NULL
  for (fmt in c("phylo_biggest", "phylo_median", "phylo_sdm")) {  # biggest first: fast, real dates
    cat(sprintf("  summary_format = %s ...\n", fmt))
    dd <- tryCatch(summarize_datelife_result(result, summary_format = fmt),
                   error = function(e) { cat("   failed:", conditionMessage(e), "\n"); NULL })
    if (inherits(dd, "multiPhylo")) dd <- dd[[1]]
    if (inherits(dd, "phylo")) {
      ov <- length(intersect(gsub("_", " ", dd$tip.label), rownames(d2)))
      cat(sprintf("   -> %d tips, overlap with data = %d\n", length(dd$tip.label), ov))
      if (ov >= 50) { dated <- dd; cat(sprintf("  using %s\n", fmt)); break }
    }
  }
  stopifnot(!is.null(dated))
  save(dated, file = dcache)
}
dated$tip.label <- gsub("_", " ", dated$tip.label)
dated$edge.length[dated$edge.length <= 0] <- 1e-6
cat(sprintf("Dated tree: %d tips, root age = %.1f Myr\n",
            length(dated$tip.label), max(node.depth.edgelength(dated))))

## align data to dated-tree tips
keep <- intersect(dated$tip.label, rownames(d2))
dated <- keep.tip(dated, keep)
g <- d2[dated$tip.label, ]
g$sp <- factor(rownames(g), levels = dated$tip.label)
cat(sprintf("Analysis set (data ∩ dated tree): n=%d\n", nrow(g)))

showfit <- function(tag, m, term) {
  tt <- summary(m)$tTable
  lam <- tryCatch({ v <- coef(m$modelStruct$corStruct, unconstrained = FALSE)
                    if (length(v) >= 1) as.numeric(v[1]) else NA_real_ },
                  error = function(e) NA_real_)
  hasl <- length(lam) == 1 && is.finite(lam)
  cat(sprintf("   %-5s %-7s = %+.4f ± %.4f  p=%.2e%s\n", tag, term,
              tt[term, "Value"], tt[term, "Std.Error"], tt[term, "p-value"],
              if (hasl) sprintf("  lambda=%.3f", lam) else ""))
}

cat("\n=== M1:  log Rubner ~ log M   (slope = alpha+beta ; P4: 0) ===\n")
pg1 <- gls(logR ~ logM, data = g, method = "ML",
           correlation = corPagel(1, phy = dated, form = ~sp, fixed = FALSE))
showfit("PGLS", pg1, "logM")
s1 <- summary(pg1)$tTable["logM", ]
cat(sprintf("   => sigma0_eff ∝ M^(%+.4f)\n", -s1["Value"]))

cat("\n=== M2:  log T_max ~ log(P/M) + log M   (b2 ; P4: 0) ===\n")
pg2 <- gls(logT ~ logPM + logM, data = g, method = "ML",
           correlation = corPagel(1, phy = dated, form = ~sp, fixed = FALSE))
showfit("PGLS", pg2, "logPM"); showfit("PGLS", pg2, "logM")
s2 <- summary(pg2)$tTable["logM", ]
cat(sprintf("   => sigma0_eff ∝ M^(%+.4f) at fixed metabolism\n", -s2["Value"]))

cat("\n============ VERDICT (real-dated PGLS via datelife) ============\n")
cat(sprintf("  M1 slope(Rubner~M) = %+.3f  p=%.2e -> %s ; sigma0∝M^%+.3f\n",
            s1["Value"], s1["p-value"], ifelse(s1["p-value"] > .05, "P4 holds", "P4 rejected"),
            -s1["Value"]))
cat(sprintf("  M2 b2[logM]        = %+.3f  p=%.2e -> %s ; sigma0∝M^%+.3f\n",
            s2["Value"], s2["p-value"], ifelse(s2["p-value"] > .05, "P4 holds", "P4 rejected"),
            -s2["Value"]))
