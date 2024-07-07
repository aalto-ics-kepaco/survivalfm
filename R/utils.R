

basesurv <- function(time, event, lp, times.eval = NULL){
  if (is.null(times.eval))
    times.eval <- sort(unique(time))
  t.unique <- sort(unique(time[event == 1L]))
  alpha <- length(t.unique)
  for (i in seq_along(t.unique)) {
    alpha[i] <- sum(time[event == 1L] == t.unique[i])/sum(exp(lp[time >= t.unique[i]]))
  }
  obj <- stats::approx(t.unique, cumsum(alpha), yleft = 0, xout = times.eval, rule = 2)
  obj$z <- exp(-obj$y)
  names(obj) <- c("times", "cumulative_base_hazard", "base_surv")
  obj
}


.flatten_parameters <- function(w, V, interaction_terms) {
  if (interaction_terms) {
    return(c(w, as.vector(V)))
  } else {
    return(w)
  }
}

.restore_parameters <- function(params, n_features, interaction_terms) {
  if (interaction_terms) {
    w <- params[1:n_features]
    V <- matrix(params[(n_features+1):length(params)], nrow = n_features)
    return(list(w = w, V = V))
  } else {
    return(list(w = params))
  }
}


.cache_shared_computations <- function(X, params, interaction_terms) {

  if (!is.null(cache$last_params) && all(identical(cache$last_params, params))) {
    return(cache)
  }

  n_features <- ncol(X)
  parameters <- .restore_parameters(params, n_features, interaction_terms)
  w <- parameters$w
  V <- parameters$V

  if (!interaction_terms) {
    V <- matrix(0, nrow = ncol(X), ncol = 1)
  }

  shared <- shared_computations(X, w, V, interaction_terms)

  cache$w <- w
  cache$V <- V
  cache$last_params <- params
  cache$XV <- shared$XV
  cache$lp <- shared$lp
  cache$exp_lp <- shared$exp_lp
  cache$cumsum_exp_lp <- shared$cumsum_exp_lp

  return(cache)
}



.loss_function <- function(params, X, time, status, lambda1, lambda2, interaction_terms) {

  shared <- .cache_shared_computations(X, params, interaction_terms)
  w <- shared$w
  V <- shared$V
  lp <- shared$lp
  cumsum_exp_lp <- shared$cumsum_exp_lp

  if (!interaction_terms) {V <- matrix(0, nrow = ncol(X), ncol = 1)}

  loss <- regularized_negative_log_likelihood(status, time, lp, cumsum_exp_lp, V, w, lambda1, lambda2, interaction_terms)
  if (is.nan(loss)) {loss <- NA}

  return(loss)
}


.gradient_function <- function(params, X, time, status, lambda1, lambda2, interaction_terms) {
  shared <- .cache_shared_computations(X, params, interaction_terms)
  w <- shared$w
  V <- shared$V
  XV <- shared$XV
  exp_lp <- shared$exp_lp
  cumsum_exp_lp <- shared$cumsum_exp_lp
  if (interaction_terms) {
    grads <- compute_gradients(X, status, time, V, w, XV, exp_lp, cumsum_exp_lp, lambda1, lambda2, interaction_terms)
    grad_w <- grads$grad_w
    grad_V <- grads$grad_V
  } else {
    V = matrix(0, nrow = ncol(X), ncol = 1)
    grads <- compute_gradients(X, status, time, V, w, XV, exp_lp, cumsum_exp_lp, lambda1, lambda2, interaction_terms)
    grad_w <- grads$grad_w
    grad_V <- NULL
  }
  grads <- .flatten_parameters(grad_w, grad_V, interaction_terms)
  grads[is.nan(grads)] <- NA
  return(grads)
}

.process_input <- function(X, y) {

  stopifnot(attr(y, "type") %in% c("right", "interval"))

  if (attr(y, "type") == "right") {
    status <- y[,"status"]
    time <- y[,"time"]
  } else if (attr(y, "type") == "interval") {
    status <- y[,"status"]
    time <- y[,"time2"] - y[,"time"]
  }

  X <- as.matrix(X)

  time_order <- order(-time)
  X <- X[time_order,]
  time <- time[time_order]
  status = status[time_order]

  return(
    list(
      X = X,
      time = time,
      status = status
    )
  )
}


.stratified_split <- function(y, val_frac) {

  train_frac <- 1 - val_frac
  train_indices <- numeric(0)
  val_indices <- numeric(0)

  for(level in unique(y)) {
    indices <- which(y == level)
    train_size <- floor(length(indices) * train_frac)
    train_indices_level <- sample(indices, train_size)
    val_indices_level <- setdiff(indices, train_indices_level)
    train_indices <- c(train_indices, train_indices_level)
    val_indices <- c(val_indices, val_indices_level)
  }

  list(train = train_indices, val = val_indices)
}

