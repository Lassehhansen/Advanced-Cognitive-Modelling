Matching_Pennies_Game_PVL = function(n_agents, 
                                 n_trials, 
                                 function_give, 
                                 bias, 
                                 noise,
                                 shape,
                                 LOSSW, 
                                 LR, 
                                 IH, 
                                 E_right, 
                                 E_left){
  
  # Sourcing functions
  source("RandomAgentNoise_Function.R")
  source("WSLSAgent_Function.R")
  source("PVL_Delta.R")
  
  
  
  # Making a counter 
  
  counter = 0
  
  # Making payoff df 
  
  payoff_df <- data.frame()
  
  
  for (a in 1:n_agents){
    
    counter = counter + 1
    
    # Creating give and take choices
    
    
    agent_give_choices <- vector(mode = "numeric", length = n_trials)
    agent_take_choices <- vector(mode = "numeric", length = n_trials)
    agent_take_thetas <- vector(mode = "numeric", length = n_trials)
    
    # Creating give & take payoffs
    
    agent_give_payoff_m <- vector(mode = "numeric", length = n_trials)
    agent_take_payoff_m <- vector(mode = "numeric", length = n_trials)
    
    # Creating trial and agent vectors
    
    trial <- vector(mode = "numeric", length = n_trials)
    agent <- vector(mode = "numeric", length = n_trials)
    
    for (t in 1:n_trials) {
      
      # Get the agent 1 choice
      if (function_give == "RandomAgentNoise_Function"){
      agent_give_choice <- RandomAgentNoise_Function(bias, noise)
      }
      else if (function_give == "WSLSAgent_Function" & t == 1){
      agent_give_choice = rbinom(1, 1, bias)
      }
      else if (function_give == "WSLSAgent_Function" & t > 1){
      agent_give_choice <- WSLSAgent_Function(agent_give_choice, agent_give_payoff)
      }
        
      # If this is the first round, choose randomly # REMEMBER TO SET BETA * MEMORY TO 0 FOR TRIAL 1 IN STAN!
      if (t == 1) {
        agent_take_choice <- rbinom(1, 1, bias)
      }
      else {
        # Get the agent 2 choice based on its memory
        agent_take_results <- PVLAgent_Function(give_choices = agent_give_choices, 
                                                take_choices = agent_take_choices, 
                                                shape, 
                                                LOSSW, 
                                                LR, 
                                                IH, 
                                                E_right, 
                                                E_left)
        agent_take_choice <- agent_take_results[1]
        E_left <- agent_take_results[2]
        E_right <- agent_take_results[3]
      }
      
      # Determine the payoff for each agent
      if (agent_give_choice == agent_take_choice) {
        agent_give_payoff <- -1
        agent_take_payoff <- 1
      } 
      else {
        agent_give_payoff <- 1
        agent_take_payoff <- -1
      }
      
      # Determine the payoff for each agent
      if (agent_give_choice == agent_take_choice) {
        agent_give_payoff <- 1
        agent_take_payoff <- -1
      } 
      else {
        agent_give_payoff <- -1
        agent_take_payoff <- 1
      }
      
      # Save the agents' choices for this round
      agent_give_choices[t] <- agent_give_choice
      agent_take_choices[t] <- agent_take_choice
      
      agent_give_payoff_m[t] <- agent_give_payoff
      agent_take_payoff_m[t] <- agent_take_payoff
      
      trial[t] <-  t
      agent[t] <- counter
      
      payoff_temp = as.data.frame(cbind(agent_give_choices, agent_take_choices, agent_give_payoff_m, agent_take_payoff_m, trial, agent))
      
    }
    
    payoff_df <- rbind(payoff_df, payoff_temp)
    
  }
  return(payoff_df)
}