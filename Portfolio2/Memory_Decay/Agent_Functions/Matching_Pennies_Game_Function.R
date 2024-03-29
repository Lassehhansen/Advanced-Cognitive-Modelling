Matching_Pennies_Game = function(n_agents, 
                                 n_trials, 
                                 function_give, 
                                 function_take, 
                                 noise, 
                                 msize, 
                                 bias,
                                 power, 
                                 alpha,
                                 beta){
  
  # Sourcing functions
  
  setwd("/Users/lassehyldighansen/Desktop/Lasse/Cognitive Science /Masters Degree/Masters - 2. Sem/ACM/Portfolios/Advanced-Cognitive-Modelling/Portfolio2/Memory_Decay/Agent_Functions")
  source("RandomAgentNoise_Function.R")
  source("WSLSAgent_Function.R")
  getwd()
  source("MemDecay_Function.R")
  
  # Making a counter 
  
  counter = 0
  
  # Making payoff df 
  
  payoff_df <- data.frame()
  
  
  for (a in 1:n_agents){
    
    counter = counter + 1
    
    # Creating give and take choices
    
    
    agent_give_choices <- vector(mode = "numeric", length = n_trials)
    agent_take_choices <- vector(mode = "numeric", length = n_trials)
    
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
      
      if (function_take == "MemAgent_RW") {
        if (t <= msize) {
          agent_take_choice <- rbinom(1, 1, bias)
        }
        else {
          # Get the agent 2 choice based on its memory
          agent_take_choice <- MemAgent_RW(agent_give_choices, msize, bias, power, alpha)
        }
      }
      
      else if(function_take == "MemDecay_Function"){
        # If this is the first round or the short-term memory agent has not yet accumulated enough memory, choose randomly
        if (t <= 1) {
          agent_take_choice <- rbinom(1, 1, bias)
        }
        else {
          # Get the agent 2 choice based on its memory
          agent_take_choice <- MemDecay_Function(agent_give_choices, agent_take_choice, alpha, beta, power, noise)
        }
      }
      
      else if(function_take == "MemAgent_Function"){
        # If this is the first round or the short-term memory agent has not yet accumulated enough memory, choose randomly
        if (t <= msize) {
          agent_take_choice <- rbinom(1, 1, bias)
        }
        else {
          # Get the agent 2 choice based on its memory
          agent_take_choice <- MemAgent_Function(agent_give_choices, msize, bias, power)
        }
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