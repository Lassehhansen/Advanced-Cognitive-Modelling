data{
  int<lower=0> N; // ntrials.
  vector[N] give_choices; // memory of opponent's choices (the giver)
  array[N] int take_choices; // list of own choices 
  real<lower = 0> power; // power of the exponential decay
}

transformed data{
  vector[N-1] memory; // Vector containing, for each trial, the agent's memory.
  
  for(i in 1:N-1){ // For each entry in memory
  
    row_vector[i] weights; // Create weights of i trials
    
    for(j in 1:i){
    
      weights[j] = j^(-power); //Exponentiate weights
      
      }
    weights = reverse(weights); // More recent trials have higher weights.
    // memory on trial i is summed weighted memory of opposing agent's choices
    // for the last i trials.
    memory[i] = weights*give_choices[1:i]; 
    
  }
}

parameters{ // Initialise parameters
// In the simpler model, there is only one beta parameter.
// Here, there is one for wins and losses.
  real alpha;
  real beta_loss; 
  real beta_win;
}

transformed parameters{
  vector[N] theta; // Initialise theta, which encodes decision probabilities.
  
  
  theta[1] = inv_logit(alpha);
  // theta of trial 1 is equivalent to alpha.
  
  for(i in 2:N){ // From trial 2 and forwards:
  
  // in the simpler model, the following if statement is absent.
  
      if (take_choices[i-1] == give_choices[i-1]){ // If previous trial was
      // a win or loss, update theta accordingly:
        theta[i] = inv_logit(alpha + beta_win*memory[i-1]); 
      } else {
        theta[i] = inv_logit(alpha + beta_loss*memory[i-1]);
      }
  
    
    }
}

model{
  //Priors 
  target += normal_lpdf(alpha | 0, 1);
  target += normal_lpdf(beta_loss | 1, 1);
  target += normal_lpdf(beta_win | 0.5, 1);
  // Decision function
  target += bernoulli_lpmf(take_choices | theta);
}

generated quantities{
  real alpha_prior;
  real beta_loss_prior;
  real beta_win_prior;
  
  int<lower=0, upper=N> prior_preds_loss1;
  int<lower=0, upper=N> prior_preds_loss3;
  int<lower=0, upper=N> prior_preds_loss5;
  int<lower=0, upper=N> prior_preds_loss7;
  int<lower=0, upper=N> prior_preds_loss9;
  
  int<lower=0, upper=N> prior_preds_win1;
  int<lower=0, upper=N> prior_preds_win3;
  int<lower=0, upper=N> prior_preds_win5;
  int<lower=0, upper=N> prior_preds_win7;
  int<lower=0, upper=N> prior_preds_win9;
  
  int<lower=0, upper=N> post_preds_loss1;
  int<lower=0, upper=N> post_preds_loss3;
  int<lower=0, upper=N> post_preds_loss5;
  int<lower=0, upper=N> post_preds_loss7;
  int<lower=0, upper=N> post_preds_loss9;
  
  int<lower=0, upper=N> post_preds_win1;
  int<lower=0, upper=N> post_preds_win3;
  int<lower=0, upper=N> post_preds_win5;
  int<lower=0, upper=N> post_preds_win7;
  int<lower=0, upper=N> post_preds_win9;

  alpha_prior = normal_rng(0, 0.3);
  beta_loss_prior = normal_rng(0.5, 0.5);
  beta_win_prior = normal_rng(0.2, 0.3);

  prior_preds_loss1 = binomial_rng(N, inv_logit(alpha_prior + beta_loss_prior * logit(0.1)));
  prior_preds_loss3 = binomial_rng(N, inv_logit(alpha_prior + beta_loss_prior * logit(0.3)));
  prior_preds_loss5 = binomial_rng(N, inv_logit(alpha_prior + beta_loss_prior * logit(0.5)));
  prior_preds_loss7 = binomial_rng(N, inv_logit(alpha_prior + beta_loss_prior * logit(0.7)));
  prior_preds_loss9 = binomial_rng(N, inv_logit(alpha_prior + beta_loss_prior * logit(0.9)));
  
  prior_preds_win1 = binomial_rng(N, inv_logit(alpha_prior + beta_win_prior * logit(0.1)));
  prior_preds_win3 = binomial_rng(N, inv_logit(alpha_prior + beta_win_prior * logit(0.3)));
  prior_preds_win5 = binomial_rng(N, inv_logit(alpha_prior + beta_win_prior * logit(0.5)));
  prior_preds_win7 = binomial_rng(N, inv_logit(alpha_prior + beta_win_prior * logit(0.7)));
  prior_preds_win9 = binomial_rng(N, inv_logit(alpha_prior + beta_win_prior * logit(0.9)));
  
  post_preds_loss1 = binomial_rng(N, inv_logit(alpha + beta_loss * logit(0.1)));
  post_preds_loss3 = binomial_rng(N, inv_logit(alpha + beta_loss * logit(0.3)));
  post_preds_loss5 = binomial_rng(N, inv_logit(alpha + beta_loss * logit(0.5)));
  post_preds_loss7 = binomial_rng(N, inv_logit(alpha + beta_loss * logit(0.7)));
  post_preds_loss9 = binomial_rng(N, inv_logit(alpha + beta_loss * logit(0.9)));
  
  post_preds_win1 = binomial_rng(N, inv_logit(alpha + beta_win * logit(0.1)));
  post_preds_win3 = binomial_rng(N, inv_logit(alpha + beta_win * logit(0.3)));
  post_preds_win5 = binomial_rng(N, inv_logit(alpha + beta_win * logit(0.5)));
  post_preds_win7 = binomial_rng(N, inv_logit(alpha + beta_win * logit(0.7)));
  post_preds_win9 = binomial_rng(N, inv_logit(alpha + beta_win * logit(0.9)));


}

