Poissondeviance <- function(y_true, y_pred){
  dev <- sum(y_pred) - sum(y_true) - sum(log((y_pred/y_true)^(y_true)))
  return(dev*(2/length(y_pred)))
}

square_loss <- function(y_true, y_pred){mean((y_true-y_pred)^2)}

weighted_square_loss <- function(y_true, y_pred){mean((y_true-y_pred)^2/y_pred)}
