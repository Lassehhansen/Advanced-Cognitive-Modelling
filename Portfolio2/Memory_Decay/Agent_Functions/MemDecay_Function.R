pacman::p_load(LaplacesDemon)

MemDecay_Function <- function(give_choices, take_choices, alpha, beta, power, noise){
  
  
  for(i in 1:length(take_choices)){
    
    if (take_choices[i] == 0){ # 
      take_choices[i] = -1
    }
    
    
    if (give_choices[i] == 0){ # 
      give_choices[i] = -1
    }
    
  } 
  # We're doing this because weighting a 0 is impossible, and our parameters are centered around 0.
  
  
  take_choices = take_choices[!is.na(take_choices)]
  

  
  weights_won <- (1:length(give_choices)) ^ (-power)
  weights_won = rev(weights_won)
    
  weighted_won_memory = give_choices*weights_won
  weighted_won_memory = sum(weighted_won_memory)
  
  theta = invlogit(alpha + beta * weighted_won_memory)
    
  choice <- rbinom(1,1,theta)
  
  if (rbinom(1, 1, noise) == 1) {
    choice = rbinom(1, 1, 0.5)
    }
  
  return(choice) 
  
  
}
