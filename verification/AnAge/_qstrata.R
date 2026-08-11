d <- read.csv("anage_data.csv", sep = ";", fileEncoding = "UTF-8-BOM",
              check.names = TRUE, stringsAsFactors = FALSE)
d$M <- suppressWarnings(as.numeric(d$Body.mass..g.))
d$T <- suppressWarnings(as.numeric(d$Maximum.longevity..yrs.))
d$P <- suppressWarnings(as.numeric(d$Metabolic.rate..W.))
d <- subset(d, Class == "Mammalia" & !is.na(M) & !is.na(T) & !is.na(P) &
               M > 0 & T > 0 & P > 0 & Order != "Chiroptera")

strata <- list(
  `all n=318`     = d,
  `strict n=111`  = d[d$Data.quality == "acceptable" &
                      d$Sample.size %in% c("medium", "large"), ],
  `high n=35`     = d[d$Data.quality == "high", ])

cat(sprintf("%-14s %5s %8s %8s %7s %8s %8s\n",
            "stratum", "n", "logM min", "logM max", "range", "a(OLS)", "SE"))
for (nm in names(strata)) {
  s <- strata[[nm]]
  lm_ <- lm(log10(T) ~ log10(M), data = s)
  co  <- summary(lm_)$coefficients[2, ]
  cat(sprintf("%-14s %5d %8.2f %8.2f %7.2f %+8.4f %8.4f\n",
              nm, nrow(s), min(log10(s$M)), max(log10(s$M)),
              diff(range(log10(s$M))), co[1], co[2]))
}

cat("\nOrder composition of the high-quality stratum (n=35):\n")
print(sort(table(strata$`high n=35`$Order), decreasing = TRUE))

cat("\nSame mass window, full sample -- does the slope drop by itself?\n")
hi <- strata$`high n=35`
w  <- subset(d, log10(M) >= min(log10(hi$M)) & log10(M) <= max(log10(hi$M)))
lw <- summary(lm(log10(T) ~ log10(M), data = w))$coefficients[2, ]
cat(sprintf("  full sample restricted to the high stratum's mass window: n=%d  a=%+.4f +- %.4f\n",
            nrow(w), lw[1], lw[2]))
