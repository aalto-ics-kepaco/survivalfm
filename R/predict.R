
#' Predict using a survivalFM object
#'
#' Given a fitted survivalFM object, this function makes predictions on new observations. 
#'
#' @param object Fitted "survivalfm" model object.
#' @param newx Matrix of new observations for which predictions are to be made.
#' @param type Type of prediction. Currently supports only "link", which gives the linear predictors. 
#' 
#' @return Returns a vector of the linear predictors. 
#'   
#' @export predict.survivalfm 
predict.survivalfm <- function(object, newx, type = "link") {
  stopifnot("must be a survivalFM object" = class(object) %in% c("survivalfm", "val.survivalfm", "cv.survivalfm"))
  
  beta = object$beta
  P= object$P
  linear_terms <- newx %*% beta
  if (!is.null(P)) {
    interactions <-  0.5 * rowSums((newx %*% P)^2 - (newx^2) %*% (P^2))
  } else {
    interactions <- 0
  }
  if (type == "link") {
    return(linear_terms + interactions)
  }
}
