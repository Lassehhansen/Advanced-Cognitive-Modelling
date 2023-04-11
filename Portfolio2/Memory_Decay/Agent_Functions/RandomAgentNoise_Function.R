RandomAgentNoise_Function <- function(bias, noise) {
  choice <-rbinom(1, 1, bias)
  
  if (rbinom(1, 1, noise) == 1) 
  
    {choice = rbinom(1, 1, 0.5)}
  return(choice)}