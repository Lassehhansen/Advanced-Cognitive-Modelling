MemAgent_Function <- function(input, msize, noise, power){
  if (power == 0) {
    
    input = input[!is.na(input)]
    input = tail(input, n = msize)
    choice = round(mean(input))
    
    if (rbinom(1, 1, noise) == 1) 
    {choice = rbinom(1, 1, 0.5)}
    
  }
  else if (power > 0){
    
    # 0.5 is an uninformative memory. This means that something affects the memory more
    # if it is further away from 0.5.
    
    input = input[!is.na(input)]
    input = tail(input, n = msize)
    # Create weighted memory store from inputs
    weighted_memory <- rep(0.5, length(input))
    
    weights <- (1:length(input)) ^ (-power)
    weights = rev(weights)
    
    for (i in 1:length(input)){
      prevChoice = input[i]
      
      if (prevChoice == 1){
        weighted_memory[i] = weighted_memory[i] + weights[i]}
      
      if (prevChoice == 0){
        weighted_memory[i] = weighted_memory[i] - weights[i]
      }
    }
    choice <- round(sum(weighted_memory)/length(weighted_memory))
    # Ensuring choice cannot be larger than 1 or smaller than 0 (due to strange interactions with weights)
    if (choice > 1){
      choice = 1
    }
    else if (choice < 0){
      choice = 0
    }
    # Add noise to the agent's choice
    if (rbinom(1, 1, noise) == 1) { 
      choice <- rbinom(1, 1, 0.5)
    }
  }
  return(choice) 
}
