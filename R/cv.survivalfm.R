#' Fit survivalFM and optimize regularization parameters using cross-validation
#'
#' This function automatically optimizes the regularization parameters
#' \code{lambda1} and \code{lambda2} for survivalFM. It automatically partitions
#' the input dataset into cross-validation folds.
#'
#' The optimization process seeks to identify the values of \code{lambda1} and
#' \code{lambda2} that maximize the concordance index (C-index) on the
#' validation folds. After determining the optimal regularization parameters, the
#' function retrains the model on the entire dataset using these parameters.
#'
#' By default, the function generates cross-validation folds randomly, ensuring that
#' events are evenly distributed between the training and validation folds.
#'
#' Note that this function does not search for the optimal rank of the
#' factorization \code{k} and it needs to be provided by the user. It is
#' recommended to start with low values for \code{k} (e.g. 5 or 10).
#'
#'
#' @param x Input data matrix; observations as rows, covariates/features as
#'   columns.
#' @param y Response variable; \code{Surv} object from the survival package.
#' @param rank Rank of the factorization for the interaction terms.
#' @param nfolds Number of cross-validation folds (default \code{nfolds} = 10). 
#' @param foldid An optional vector of values between 1 and nfolds identifying
#'   what fold each observation is in to use custom cross-validation folds. 
#' @param nlambda Number of values for \code{lambda1} and \code{lambda2} to
#'   search in the logarithmic space between \code{lambda.min} and
#'   \code{lambda.max}. If \code{nlambda} is set to its default value of 20, the
#'   function will automatically generate 20 logarithmically spaced values for
#'   each lambda within the specified range. Alternatively, one can provide
#'   custom sequences for \code{lambda1} and \code{lambda2} using the
#'   \code{lambda1_range} and \code{lambda2_range} parameters.
#' @param lambda.min The minimum boundary of the range within which the
#'   regularization parameters \code{lambda1} and \code{lambda2} are searched.
#'   This is the lower limit for generating lambda values unless a custom range
#'   is provided. It must be a positive number.
#' @param lambda.max The maximum boundary of the range within which the
#'   regularization parameters \code{lambda1} and \code{lambda2} are searched.
#'   This is the upper limit for generating lambda values unless a custom range
#'   is provided. It must be a positive number greater than \code{lambda.min}.
#' @param lambda_equal Defines whether only equally strong regularization for
#'   the linear and the interaction part should be considered, i.e. set
#'   \code{lambda1} = \code{lambda2}. Considering only equal values for the two
#'   regularization parameters reduces the number of potential parameter
#'   combinations to search and can therefore speed up the optimization process.
#'   However, in some cases it might be desired to allow differing
#'   regularization strengths for the linear and the interaction part, in which
#'   case this should be set to \code{FALSE}. Default is \code{FALSE}.
#' @param lambda1_range Custom range of values for the regularization parameter
#'   \code{lambda1} to be optimized (regularization for the linear terms;
#'   optional). By default, the function will create 50 equally spaced values in
#'   a logarithmic scale between 1 and 0.0001.
#' @param lambda2_range Custom range of values for the regularization parameter
#'   \code{lambda2} to be optimized (regularization for the factorized
#'   interaction terms; optional).
#' @param maxiter Maximum number of iterations over the data for all
#'   regularization parameter values. Default is 1000.
#' @param reltol Relative convergence tolerance for the optimization method in
#'   \code{stats::optim}. Default is \code{sqrt(.Machine$double.eps)}.
#' @param parallel If TRUE, use parallel foreach to fit the models with
#'   different regularization strengths. The parallel backend must be registered
#'   beforehand. Default is \code{FALSE}.
#' @param trace If trace=1, will display messages of the progress.
#' @param optimization_method The optimization method used by
#'   \code{stats::optim}. Default is "BFGS".
#'
#' @return Returns an object of class \code{"cv.survivalfm"}. It is a list
#'   containing the coefficients of the final fitted model \code{beta} (linear
#'   effects) and \code{P} (factorized interaction parameter matrix). Returns
#'   also the interaction effects in \code{PP}, given by the inner product
#'   <p,p>. All model coefficients, including both the linear effects and
#'   interaction effects are contained in \code{coefficients}. The validation
#'   results obtained for different values of the regularization parameters are
#'   provided in \code{val.results}.
#'
#' @export cv.survivalfm
#'
#' @importFrom survival concordance
#' @importFrom foreach foreach
#' @importFrom foreach `%dopar%`
#' @importFrom utils setTxtProgressBar
#' @importFrom utils txtProgressBar
#' @importFrom stats aggregate
#' @importFrom stats sd
cv.survivalfm <- function(
    x,
    y,
    rank,
    nfolds = 10,
    foldid = NULL,
    nlambda = 20,
    lambda.min = 1e-4,
    lambda.max = 1,
    lambda_equal = FALSE,
    lambda1_range = NULL,
    lambda2_range = NULL,
    maxiter = 1000,
    reltol = sqrt(.Machine$double.eps),
    parallel = FALSE,
    trace = 0,
    optimization_method = "BFGS") {
  
  if (parallel) {
    if (!requireNamespace("foreach", quietly = TRUE)) {
      stop("The 'foreach' package is needed for the parallel computation to work. Please install it using install.packages('foreach')")
    }
  }
  
  stopifnot("x and y must have the same number of samples" = nrow(x) == length(y))
  stopifnot("lambda.min must be positive" = lambda.min > 0)
  stopifnot("lambda.max must be positive" = lambda.max > 0)
  stopifnot("lambda.max must be greater than lambda.min" = lambda.max > lambda.min)
  stopifnot("rank must be a positive integer" = (rank%%1 == 0 & rank > 0))
  stopifnot("nfolds must be a positive integer" = (nfolds%%1 == 0 & nfolds > 0))
  stopifnot("x must be complete" = sum(stats::complete.cases(x)) == nrow(x))
  
  
  if (is.null(lambda1_range)) {
    lambda1_range <- exp(seq(log(lambda.max), log(lambda.min), length.out = nlambda))
  }
  
  if (is.null(lambda2_range)) {
    lambda2_range <- exp(seq(log(lambda.max), log(lambda.min), length.out = nlambda))
  }
  
  param_combinations <-
    expand.grid(
      lambda1 = lambda1_range,
      lambda2 = lambda2_range
    )
  
  if (lambda_equal) {
    param_combinations <- param_combinations[param_combinations$lambda1 == param_combinations$lambda2,]
  }
  
  param_combinations <- param_combinations[order(-(param_combinations$lambda1 + param_combinations$lambda2)),]
  
  if (is.null(foldid)) {
    folds <- .stratified_folds(y[,"status"], nfolds)
  }
  
  if (trace == 1) {
    print(paste0("Using ", nfolds, "-fold cross-validation to optimize regularization parameters..."))
    pb = txtProgressBar(min = 0, max = nrow(param_combinations), initial = 0)
  }
  
  cv.results <- data.frame()
  
  for (param_idx in 1:nrow(param_combinations)) {
    
    
    if (trace == 1) {setTxtProgressBar(pb, param_idx)}
    
    param_comb <- param_combinations[param_idx,]
    
    fold_results <- foreach(fold_idx = seq_along(folds), .combine = 'rbind', .packages = c("survival", "survivalfm")) %dopar% {
      
      val_idx <- folds[[fold_idx]]
      train_idx <- setdiff(1:nrow(x), val_idx)
      
      fit <- survivalfm(
        x = x[train_idx, ],
        y = y[train_idx, ],
        lambda1 = param_comb$lambda1,
        lambda2 = param_comb$lambda2,
        rank = rank,
        trace = 0,
        maxiter = maxiter,
        reltol = reltol,
        optimization_method = optimization_method
      )
      
      lp <- predict.survivalfm(fit, as.matrix(x[val_idx,]), type = "link")
      cindex <- survival::concordance(y[val_idx, ] ~ lp, reverse = T)$concordance
      
      return(
        data.frame(
          "lambda1" = param_comb$lambda1,
          "lambda2" = param_comb$lambda2,
          "cindex" = cindex,
          "fold_idx" = fold_idx,
          "fit" = I(list(fit))
        )
      )
    }
    
    cv.results <- rbind(cv.results, fold_results)
    
  }
  
  aggregate_results <- aggregate(cindex ~ lambda1 + lambda2, data = cv.results, function(x) c(mean = mean(x), sd = sd(x)))
  
  cv_results <- data.frame(
    lambda1 = aggregate_results$lambda1,
    lambda2 = aggregate_results$lambda2,
    mean_cindex = aggregate_results$cindex[,"mean"],
    sd_cindex = aggregate_results$cindex[,"sd"]
  )
  
  best_params <- cv_results[with(cv_results, order(-mean_cindex, sd_cindex)), ]
  best_params <- best_params[1, ]
  
  best_lambda1 <- best_params$lambda1
  best_lambda2 <- best_params$lambda2
  
  if (trace == 1) {close(pb)}
  
  if (trace == 1) {
    print(paste0("Best lambda1: ", best_lambda1))
    print(paste0("Best lambda2: ", best_lambda2))
    print("Training final model...")
  }
  
  fit <- survivalfm(
    x = x,
    y = y,
    lambda1 = best_lambda1,
    lambda2 = best_lambda2,
    rank = rank,
    trace = 0,
    maxiter = maxiter,
    reltol = reltol,
    optimization_method = optimization_method
  )
  
  return(
    structure(
      list(
        Call = match.call(),
        beta = fit$beta,
        P = fit$P,
        PP = fit$PP,
        coefficients = fit$coefficients,
        cv.result = cv_results
      ),
      class = c("cv.survivalfm", "survivalfm")
    )
  )
  
}
