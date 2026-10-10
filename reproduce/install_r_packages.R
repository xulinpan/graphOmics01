options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)
lib <- .libPaths()[1]
cat("library:", lib, "\n")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = lib)
BiocManager::install(c("pcaMethods", "impute", "proDA"), lib = lib, update = FALSE, ask = FALSE)
for (p in c("norm", "tmvtnorm", "imputeLCMD", "missForest", "jsonlite")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib)
}
if (!requireNamespace("imputeLCMD", quietly = TRUE)) {
  install.packages("https://cran.r-project.org/src/contrib/Archive/imputeLCMD/imputeLCMD_2.1.tar.gz",
                   repos = NULL, type = "source", lib = lib)
}
for (p in c("proDA", "imputeLCMD", "missForest", "pcaMethods", "impute", "norm", "tmvtnorm", "jsonlite")) {
  cat(p, as.character(tryCatch(packageVersion(p), error = function(e) "MISSING")), "\n")
}
