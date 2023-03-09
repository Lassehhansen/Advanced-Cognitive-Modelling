PVLAgent_Function <- function(give_choices, take_choices, shape, LOSSW, LR, IH, E_right, E_left){
  
  # Calculate the payoffs for the previous round
  last_give_payoff <- ifelse(take_choices[length(take_choices)] == give_choices[length(give_choices)], -2, 2)
  last_take_payoff <- ifelse(take_choices[length(take_choices)] == give_choices[length(give_choices)], 2, -2)
  
  # Calculate the utility parameter U based on the last round's payoff
  if (last_take_payoff == 2) {
    U <- (last_take_payoff)^shape
  } else {
    U <- -LOSSW*abs(last_take_payoff)^shape
  }
  
  # Calculate the expected utility E based on the previous round's choice
  if (take_choices[length(take_choices)] == 1) {
    E_right <- IH*((1-LR)*E_right + LR*U)
  } else {
    E_left <- IH*((1-LR)*E_left + LR*U)
  }
  
  # Calculate the probabilities of choosing 0 or 1 based on E
  p <- exp(E_right) / (exp(E_left) + exp(E_right))
  
  # Make a choice based on the calculated probabilities
  choice <- rbinom(1, 1, p)
  output_vector = c(choice, E_left, E_right)
  return(output_vector)
}


