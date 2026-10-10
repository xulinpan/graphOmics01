# Reference R implementations of comparators, called from rbridge.py.
# Input  (in the job directory): protein.csv (samples x proteins, raw log scale, NA = missing),
#        optional covariates.csv (samples x covariates, complete), params.json.
# Output: imputed.csv (samples x proteins, raw log scale) and info.json.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
p <- jsonlite::fromJSON(file.path(dir, "params.json"))
set.seed(p$seed)
X <- as.matrix(read.csv(file.path(dir, "protein.csv"), header = FALSE, na.strings = c("NA", "", "nan", "NaN")))
storage.mode(X) <- "double"
info <- list(method = p$method)
t0 <- Sys.time()
if (p$method %in% c("MinDet", "MinProb", "QRILC")) {
  suppressMessages(library(imputeLCMD))
  Y <- t(X)                                   # imputeLCMD expects features x samples
  out <- switch(p$method,
                MinDet = impute.MinDet(Y, q = 0.01),
                MinProb = impute.MinProb(Y, q = 0.01, tune.sigma = 1),
                QRILC = impute.QRILC(Y, tune.sigma = 1)[[1]])
  R <- t(out)
  info$version <- as.character(packageVersion("imputeLCMD"))
} else if (p$method == "missForest") {
  suppressMessages(library(missForest))
  C <- as.matrix(read.csv(file.path(dir, "covariates.csv"), header = FALSE))
  D <- cbind(C, X)
  colnames(D) <- paste0("V", seq_len(ncol(D)))
  res <- missForest(D, ntree = p$ntree, maxiter = p$maxiter)
  R <- as.matrix(res$ximp)[, (ncol(C) + 1):ncol(D)]
  info$version <- as.character(packageVersion("missForest"))
  info$OOBerror <- unname(res$OOBerror)
} else if (p$method == "proDA") {
  suppressMessages(library(proDA))
  Y <- t(X)                                   # proteins x samples
  fit <- proDA(Y, design = ~ 1, verbose = FALSE)
  mu <- coefficients(fit)[, 1]
  s2 <- feature_parameters(fit)$s2
  hp <- hyper_parameters(fit)
  rho <- hp$dropout_curve_position            # per sample
  zeta <- abs(hp$dropout_curve_scale)         # per sample; P(missing | x) = Phi((rho - x) / zeta)
  n <- nrow(X); P <- ncol(X)
  M <- matrix(mu, n, P, byrow = TRUE)
  V <- matrix(s2, n, P, byrow = TRUE)
  root <- sqrt(V + matrix(zeta ^ 2, n, P))
  a <- (matrix(rho, n, P) - M) / root
  mills <- exp(dnorm(a, log = TRUE) - pnorm(a, log.p = TRUE))
  E <- M - V / root * mills                   # E[x | missing] under the fitted proDA model
  R <- ifelse(is.na(X), E, X)
  info$version <- as.character(packageVersion("proDA"))
  info$dropout_position_median <- median(rho)
  info$dropout_scale_median <- median(zeta)
} else {
  stop(paste("unknown method", p$method))
}
info$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(R, file.path(dir, "imputed.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, digits = NA), file.path(dir, "info.json"))
