#' Fit survivalFM with specified parameters
#'
#' This function fits a survivalFM model, given specified input parameters. If
#' you wish to automatically optimize the regularization parameters
#' \code{lambda1} and \code{lambda2}, use fit.survivalfm().
#'
#' @param x Input data matrix; observations as rows, covariates/features as
#'   columns.
#' @param y Response variable; \code{Surv} object from the survival package.
#' @param rank Rank of the factorization for the interaction terms.
#' @param lambda1 Regularization strength for the linear terms.
#' @param lambda2 Regularization strength for the factorized interaction terms.
#' @param maxiter Maximum number of iterations over the data for all
#'   regularization parameter values. Default is 1000.
#' @param reltol Relative convergence tolerance for the optimization method in
#'   \code{stats::optim}. Default is \code{sqrt(.Machine$double.eps)}.
#' @param trace If trace=1, will display messages of the progress.
#' @param optimization_method The optimization method used by
#'   \code{stats::optim}. Default is "BFGS".
#'
#' @return Returns an object of class \code{"survivalfm"}. It is a list
#'   containing the coefficients of the fitted model \code{beta} (linear effects)
#'   and \code{P} (factorized interaction parameter matrix). Returns also the
#'   approximated interaction effects in \code{PP}, given by the inner product
#'   <p,p>. All model coefficients, including both the linear effects and
#'   interaction effects are contained in \code{coefficients}.
#'
#' @export survivalfm
#' @importFrom stats optim
survivalfm <- function(
    x,
    y,
    rank,
    lambda1,
    lambda2,
    maxiter = 1000,
    reltol = sqrt(.Machine$double.eps),
    trace = 0,
    optimization_method = "BFGS"
    ) {

  inputs <- .process_input(x, y)
  X <- inputs$X
  time <- inputs$time
  status <- inputs$status

  interaction_terms <- if (rank > 0) {T} else {F}

  n_features <- ncol(X)

  # Initialize parameters
  beta <- matrix(stats::rnorm(n_features, mean = 0, sd = 0.001), nrow = n_features, ncol = 1)
  P <- if (interaction_terms) matrix(stats::rnorm(n_features * rank, mean = 0, sd = 0.001), nrow = n_features, ncol = rank)

  # Cache computations shared by the gradient and loss function
  cache <<- new.env()
  params <- .flatten_parameters(beta, P, interaction_terms)
  cache$last_params <- NULL

  optim.res <-
    optim(
      par =  params,
      fn = function(params) .loss_function(params, X, time, status, lambda1, lambda2, interaction_terms),
      gr = function(params) .gradient_function(params, X, time, status, lambda1, lambda2, interaction_terms),
      control = list(trace = trace, maxit = maxiter, reltol = reltol),
      method = optimization_method)

  optimized_params <- optim.res$par
  
  parameters <- .restore_parameters(optimized_params, n_features, interaction_terms)

  beta <- parameters$beta
  P <- parameters$P

  feat_names = colnames(X)
  names(beta) = feat_names

  if (!is.null(P)) {
    rownames(P) = feat_names
    colnames(P) = paste0("rank", 1:rank)
    PP <- P %*% t(P)
    diag(PP) = NA
    P_list <- PP[upper.tri(PP, diag = FALSE)]
    PP_names <- outer(feat_names, feat_names, FUN = function(x, y) paste(x, "*", y, sep = ""))
    PP_names <- PP_names[upper.tri(PP_names, diag = FALSE)]
    names(PP) = PP_names
    names(P_list) = PP_names
    coefficients <- c(beta, P_list)
  } else {
    coefficients <- beta
    PP <- NULL
  }
  
  if (optim.res$convergence != 0) {
    warning("Convergence not reached, consider increasing maxiter. ")
  }
  
  return(
    structure(
      list(
        Call = match.call(),
        beta = beta,
        P = P,
        PP = PP,
        coefficients = coefficients,
        optim.res = optim.res
      ),
      class = c("survivalfm")
    )
  )

}

