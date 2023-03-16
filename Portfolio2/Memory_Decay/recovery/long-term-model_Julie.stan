data {
  int<lower=0> N; // ntrials.
  array[N] int choice; // choices that the agent makes
  vector[N] others_choice; // memory of opposing agent choices. SHOULD BE CODED AS -1's and 1's.
  real<lower = 0> power;
}

transformed data {
  vector[N] memory; // consider logit here. Either that, or just encode choices as -1's and 1's in the data-generating stuff.
  for(i in 1:N){ // For every trial
    row_vector[i] weights;
    int exponent;
    for(j in 1:i){ // Create weights
      exponent = i - j + 1;
      weights[j] = exponent^(-power);}
    memory[i] = weights*others_choice[1:i]; // memory on trial i is summed (NOT AVERAGED) weighted memory.
  }
}

parameters {
  real alpha;
  real beta;
}

transformed parameters{
  vector[N] theta;
    theta = inv_logit(alpha + beta*memory); // Update theta based on weighted memory.
}

model {
  //priors
  target += normal_lpdf(alpha | 0, 1);
  target += normal_lpdf(beta | 0, .3);
  
  
  target += bernoulli_lpmf(choice | theta);
}

generated quantities{
  real alpha_prior;
  real beta_prior;
  
  int<lower=0, upper=N> prior_preds5;
  int<lower=0, upper=N> post_preds5;
  int<lower=0, upper=N> prior_preds7;
  int<lower=0, upper=N> post_preds7;
  int<lower=0, upper=N> prior_preds9;
  int<lower=0, upper=N> post_preds9;
  
  alpha_prior = normal_rng(0, 0.3);
  beta_prior = normal_rng(0, 0.5);
  prior_preds5 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.5)));
  prior_preds7 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.7)));
  prior_preds9 = binomial_rng(N, inv_logit(alpha_prior + beta_prior * logit(0.9)));
  post_preds5 = binomial_rng(N, inv_logit(alpha + beta * logit(0.5)));
  post_preds7 = binomial_rng(N, inv_logit(alpha + beta * logit(0.7)));
  post_preds9 = binomial_rng(N, inv_logit(alpha + beta * logit(0.9)));

}


