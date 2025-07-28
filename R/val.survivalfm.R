#' Fit survivalFM and optimize regularization parameters using a validation set
#'
#' This function automatically optimizes the regularization parameters
#' \code{lambda1} and \code{lambda2} for survivalFM. It automatically partitions
#' the input dataset into a training set and a validation set.
#'
#' The optimization process seeks to identify the values of \code{lambda1} and
#' \code{lambda2} that maximize the concordance index (C-index) on the
#' validation set. After determining the optimal regularization parameters, the
#' function retrains the model on the entire dataset using these parameters.
#'
#' By default, the function generates validation sets randomly, ensuring that
#' events are evenly distributed between the validation and training sets.
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
#' @param frac Fraction of data to use for validation data set. Default is 0.2
#'   (20%).
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
#' @param early_stopping Stop early if further decreasing lambda2 does not
#'    improve performance (default = TRUE).
#' @param val_idx Row indices to be used as the validation set. If not set, the
#'   function will randomly sample a validation set from the training data.
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
#' @param seed Optional integer. If provided, sets the random seed for
#'   reproducible parameter initialization. If NULL, random initialization is
#'   used.  
#'   
#'
#' @return Returns an object of class \code{"val.survivalfm"}. It is a list
#'   containing the coefficients of the final fitted model \code{beta} (linear
#'   effects) and \code{P} (factorized interaction parameter matrix). Returns
#'   also the interaction effects in \code{PP}, given by the inner product
#'   <p,p>. All model coefficients, including both the linear effects and
#'   interaction effects are contained in \code{coefficients}. The validation
#'   results obtained for different values of the regularization parameters are
#'   provided in \code{val.results}.
#'
#' @export val.survivalfm
#'
#' @importFrom survival concordance
#' @importFrom foreach foreach
#' @importFrom foreach `%dopar%`
#' @importFrom utils setTxtProgressBar
#' @importFrom utils txtProgressBar
val.survivalfm <- function(
    x,
    y,
    rank,
    frac = 0.2,
    nlambda = 20,
    lambda.min = 1e-4,
    lambda.max = 1,
    lambda_equal = FALSE,
    lambda1_range = NULL,
    lambda2_range = NULL,
    early_stopping = TRUE,
    val_idx = NULL,
    maxiter = 1000,
    reltol = sqrt(.Machine$double.eps),
    parallel = FALSE,
    trace = 0,
    optimization_method = "BFGS",
    seed = NULL
) {
  
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
  stopifnot("frac must be a positive number between 0 and 1" = (frac > 0 & frac <= 1))
  stopifnot("x must be complete" = sum(stats::complete.cases(x)) == nrow(x))
  
  if (is.null(val_idx)) {
    split <- .stratified_split(y[,"status"], frac)
    train_idx <- split$train
    val_idx <- split$val
  } else {
    train_idx <- setdiff(1:nrow(x), val_idx)
  }
  
  if (is.null(lambda1_range)) {
    lambda1_range <- exp(seq(log(lambda.max), log(lambda.min), length.out = nlambda))
  }
  
  if (is.null(lambda2_range)) {
    lambda2_range <- exp(seq(log(lambda.max), log(lambda.min), length.out = nlambda))
  }
  
  if (trace == 1) {
    print(paste0("Using ", frac * 100, "% validation set to optimize regularization parameters..."))
    pb = txtProgressBar(min = 0, max = length(lambda2_range), initial = 0)
  }
  
  lambda2_range <- lambda2_range[order(-lambda2_range)]
  
  val.results <- data.frame()
  
  if (parallel) {
    
    for (lambda2 in lambda2_range) {
      
      sub.results <- foreach(
        lambda1 = lambda1_range, 
        .combine = 'rbind', 
        .inorder = FALSE
      ) %dopar% {
        
        fit <- survivalfm(
          x = x[train_idx, ],
          y = y[train_idx, ],
          lambda1 = lambda1,
          lambda2 = lambda2,
          rank = rank,
          maxiter = maxiter,
          reltol = reltol,
          optimization_method = optimization_method,
          seed = seed
        )
        
        lp <- predict.survivalfm(fit, as.matrix(x[val_idx,]), type = "link")
        cindex <- survival::concordance(y[val_idx, ] ~ lp, reverse = T)$concordance
        
        return(
          data.frame(
            "lambda1" = lambda1,
            "lambda2" = lambda2,
            "cindex.val" = cindex,
            "fit" = I(list(fit))
          )
        )
      }
      
      if (trace == 1) {setTxtProgressBar(pb, which(lambda2_range == lambda2))}
      
      if (early_stopping & nrow(val.results) > 0) {
        # Stop early if decreasing lambda2 did not improve performance on the last round
        if (round(max(sub.results$cindex.val), 2) < round(max(val.results$cindex.val), 2)) {
          val.results <- rbind(val.results, sub.results)
          break
        }
      }
      
      val.results <- rbind(val.results, sub.results)
      
      
    }
    
  } else {
    
    val.results <- data.frame()
    
    for (lambda2 in lambda2_range) {
      
      sub.results <- data.frame()
      
      for (lambda1 in lambda1_range) {
        
        fit <- survivalfm(
          x = x[train_idx, ],
          y = y[train_idx, ],
          lambda1 = lambda1,
          lambda2 = lambda2,
          rank = rank,
          maxiter = maxiter,
          reltol = reltol,
          optimization_method = optimization_method,
          seed = seed
        )
        
        lp <- predict.survivalfm(fit, as.matrix(x[val_idx,]), type = "link")
        cindex <- survival::concordance(y[val_idx, ] ~ lp, reverse = T)$concordance
        
        
        sub.results <-
          rbind(
            sub.results,
            data.frame(
              "lambda1" = lambda1,
              "lambda2" = lambda2,
              "cindex.val" = cindex,
              "fit" = I(list(fit))
            )
          )
        
      }
      
      if (trace == 1) {setTxtProgressBar(pb, which(lambda2_range == lambda2))}
      
      if (early_stopping & nrow(val.results) > 1) {
        # Stop early if decreasing lambda2 did not improve performance on the last round
        if (round(max(sub.results$cindex.val), 2) < round(max(val.results$cindex.val), 2)) {
          val.results <- rbind(val.results, sub.results)
          break
        }
      }
      
      val.results <- rbind(val.results, sub.results)
    }
  }
    
  if (trace == 1) {close(pb)}
  
  best_params <- val.results[which.max(val.results$cindex.val)[1],]
  best_lambda1 <- best_params$lambda1
  best_lambda2 <- best_params$lambda2
  
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
    trace = F,
    maxiter = maxiter,
    reltol = reltol,
    optimization_method = optimization_method,
    seed = seed
  )
  
  return(
    structure(
      list(
        Call = match.call(),
        beta = fit$beta,
        P = fit$P,
        PP = fit$PP,
        coefficients = fit$coefficients,
        optim.res = fit$optim.res,
        val.results = val.results
      ),
      class = c("val.survivalfm", "survivalfm")
    )
  )
  
}

