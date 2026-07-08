pkgs <- c("rotl","ape","nlme","caper","phylolm","geiger","phytools")
inst <- rownames(installed.packages())
cat("R", as.character(getRversion()), "\n")
for (p in pkgs) cat(sprintf("  %-9s %s\n", p, ifelse(p %in% inst, "OK", "MISSING")))
