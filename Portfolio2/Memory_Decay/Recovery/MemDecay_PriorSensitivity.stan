data{
  int<lower=0> N; // ntrials.
  vector[N] give_choices; // memory of the giver. SHOULD BE CODED AS -1's and 1's.
  array[N] int take_choices; // memory of of the taker. SHOULD BE CODED AS -1's and 1's.
  real<lower = 0> power;
  
  real prior_mean_alpha;
  real<lower=0> prior_sd_alpha;
  
  real prior_mean_beta;
  real<lower=0> prior_sd_beta;
}

transformed data{
  vector[N] memory; // consider logit here. Either that, or just encode choices as -1's and 1's in the data-generating stuff.
  
  for(i in 1:N){ // For every trial
  
    row_vector[i] weights;
    
    for(j in 1:i){ // Create weights
    
      weights[j] = j^(-power);
      
      }
    weights = reverse(weights);
    memory[i] = weights*give_choices[1:i]; // memory on trial i is summed (NOT AVERAGED) weighted memory.
  }
}

parameters {
  real alpha;
  real beta;
}

transformed parameters{
  vector[N] theta;
  
  theta[1] = inv_logit(alpha);
  
  for (i in 2:N){
    theta[i] = inv_logit(alpha + beta*memory[i-1]); // Update theta based on weighted memory.
}}

model {
  //priors
  target += normal_lpdf(alpha | prior_mean_alpha, prior_sd_alpha);
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta);
  
  target += bernoulli_lpmf(take_choices | theta);
}

generated quantities{
  real alpha_prior;
  real beta_prior;
  
  int<lower=0, upper=N> prior_preds1;
  int<lower=0, upper=N> post_preds1;
  int<lower=0, upper=N> prior_preds3;
  int<lower=0, upper=N> post_preds3;
  int<lower=0, upper=N> prior_preds5;
  int<lower=0, upper=N> post_preds5;
  int<lower=0, upper=N> prior_preds7;
  int<lower=0, upper=N> post_preds7;
  int<lower=0, upper=N> prior_preds9;
  int<lower=0, upper=N> post_preds9;
  
  alpha_prior = normal_rng(prior_mean_alpha, prior_sd_alpha);
  beta_prior = normal_rng(prior_mean_beta, prior_sd_beta);
    
  prior_preds1 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.1)));
  prior_preds3 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.3)));
  prior_preds5 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.5)));
  prior_preds7 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.7)));
  prior_preds9 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.9)));
  
  post_preds1 = binomial_rng(N, inv_logit(alpha + beta * logit(0.1)));
  post_preds3 = binomial_rng(N, inv_logit(alpha + beta * logit(0.3)));
  post_preds5 = binomial_rng(N, inv_logit(alpha + beta * logit(0.5)));
  post_preds7 = binomial_rng(N, inv_logit(alpha + beta * logit(0.7)));
  post_preds9 = binomial_rng(N, inv_logit(alpha + beta * logit(0.9)));

}


