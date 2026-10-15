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
} else if (p$method %in% c("proDA", "proDA_design")) {
  suppressMessages(library(proDA))
  Y <- t(X)                                   # proteins x samples
  n <- nrow(X); P <- ncol(X)
  if (p$method == "proDA") {
    fit <- proDA(Y, design = ~ 1, verbose = FALSE)
    M <- matrix(coefficients(fit)[, 1], n, P, byrow = TRUE)
  } else {
    # tumour-versus-normal design (downstream scenarios): groups.csv holds 1 for normal samples
    g <- as.numeric(read.csv(file.path(dir, "groups.csv"), header = FALSE)[[1]])
    cond <- factor(ifelse(g > 0.5, "normal", "tumour"), levels = c("tumour", "normal"))
    fit <- proDA(Y, design = ~ cond, col_data = data.frame(cond = cond), verbose = FALSE)
    M <- model.matrix(~ cond) %*% t(as.matrix(coefficients(fit)))   # samples x proteins
    info$design <- "~ cond"
  }
  s2 <- feature_parameters(fit)$s2
  hp <- hyper_parameters(fit)
  rho <- hp$dropout_curve_position            # per sample
  zeta <- abs(hp$dropout_curve_scale)         # per sample; P(missing | x) = Phi((rho - x) / zeta)
  V <- matrix(s2, n, P, byrow = TRUE)
  root <- sqrt(V + matrix(zeta ^ 2, n, P))
  a <- (matrix(rho, n, P) - M) / root
  mills <- exp(dnorm(a, log = TRUE) - pnorm(a, log.p = TRUE))
  E <- M - V / root * mills                   # E[x | missing] under the fitted proDA model
  R <- ifelse(is.na(X), E, X)
  info$version <- as.character(packageVersion("proDA"))
  info$dropout_position_median <- median(rho)
  info$dropout_scale_median <- median(zeta)
} else if (p$method %in% c("Pirat", "PiratT", "PiratS")) {
  # Pirat (Etourneau et al., Biostatistics 2025). Protein-level input: every protein is its own
  # protein group. PiratT adds each protein's cognate transcript (rna.csv, samples x proteins,
  # log scale, complete) as a fully observed member of its group (paired samples).
  suppressMessages(library(Pirat))
  n <- nrow(X); P <- ncol(X)
  cn <- paste0("P", seq_len(P))
  dimnames(X) <- list(paste0("S", seq_len(n)), cn)
  adj <- diag(P) == 1
  dimnames(adj) <- list(cn, cn)
  d <- list(peptides_ab = X, adj = adj)
  # Pirat estimates its inverse-gamma prior on variances from fully observed peptides only. With
  # random misses on top of censoring (all benchmark scenarios), few or no proteins are fully
  # observed, and Pirat stops. In that case the prior is estimated from the proteins with the fewest
  # missing values: those with at most c missing values, c being the smallest count giving at least
  # 30 proteins (c = 0 reproduces Pirat's own rule). Everything else is Pirat's own pipeline.
  nmiss <- colSums(is.na(X))
  cmax <- min(sort(nmiss)[min(30, P)])
  info$psi_max_missing <- cmax
  info$psi_n_proteins <- sum(nmiss <= cmax)
  args <- list(extension = "base")
  if (p$method == "PiratS") args <- list(extension = "S")   # sample-wise correlations for singleton groups
  if (p$method == "PiratT") {
    Rn <- as.matrix(read.csv(file.path(dir, "rna.csv"), header = FALSE))
    dimnames(Rn) <- list(rownames(X), paste0("G", seq_len(P)))
    ar <- diag(P) == 1
    dimnames(ar) <- list(colnames(Rn), cn)
    d$rnas_ab <- Rn
    d$adj_rna_pg <- ar
    args <- list(extension = "T", rna.cond.mask = seq_len(n), pep.cond.mask = seq_len(n),
                 max.pg.size.pirat.t = 1)
  }
  # full.activation = FALSE: on Windows, conda 24.11's cmd.exe activation hands its temporary
  # activation script (*.env) to the desktop's file handler instead of running it, which opens an
  # editor window and adds nothing; basilisk still sets PYTHONNOUSERSITE and clears PYTHONPATH.
  proc <- basilisk::basiliskStart(envPirat, full.activation = FALSE)
  out <- basilisk::basiliskRun(proc, fun = function(d, args, keep) {
    ns <- asNamespace("Pirat")
    orig <- get("estimate_psi_df", envir = ns)
    calls <- 0
    patched <- function(pep.ab.table) {
      # first call: peptide (protein) prior; a later call (Pirat-T) estimates the mRNA prior as usual
      calls <<- calls + 1
      if (calls == 1) orig(d$peptides_ab[, keep, drop = FALSE]) else orig(pep.ab.table)
    }
    unlockBinding("estimate_psi_df", ns)
    assign("estimate_psi_df", patched, envir = ns)
    lockBinding("estimate_psi_df", ns)
    do.call(Pirat::pipeline_llkimpute, c(list(d), args))
  }, d = d, args = args, keep = which(nmiss <= cmax))
  basilisk::basiliskStop(proc)
  if (is.list(out) && !is.data.frame(out)) {                          # Pirat >= 1.2 returns a list
    if (!is.null(out$params)) info$params <- lapply(out$params, function(v) as.numeric(v)[1])
    out <- out$data.imputed
  }
  R <- as.matrix(out)[, cn]
  info$version <- as.character(packageVersion("Pirat"))
  info$remaining_na <- sum(is.na(R))
} else {
  stop(paste("unknown method", p$method))
}
info$seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(R, file.path(dir, "imputed.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, digits = NA), file.path(dir, "info.json"))
