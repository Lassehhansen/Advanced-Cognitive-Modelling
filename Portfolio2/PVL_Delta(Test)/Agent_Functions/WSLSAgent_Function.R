WSLSAgent_Function <- function(prevChoice, agent_give_payoff) {
  
  if (agent_give_payoff == 1) {
    choice <- prevChoice
  } 
  
  else if (agent_give_payoff == -1) {
    choice <- 1 - prevChoice
  }
  return(choice)
  
  } 