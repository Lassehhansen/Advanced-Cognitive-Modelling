data{
  int<lower=0> N; // ntrials.
  vector[N] give_choices; // memory of the giver. SHOULD BE CODED AS -1's and 1's.
  array[N] int take_choices; // memory of of the taker. SHOULD BE CODED AS -1's and 1's.
  real<lower = 0> power;
  
  real prior_mean_alpha;
  real<lower=0> prior_sd_alpha;
  
  real prior_mean_beta_loss;
  real<lower=0> prior_sd_beta_loss;
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

parameters{
  real alpha;
  real beta_loss;
  real beta_win;
}

transformed parameters{
  vector[N] theta;
  
  theta[1] = inv_logit(alpha);
  // theta of trial 1 is equivalent to alpha.
  
  for(i in 2:N){ // Should start at 2
  
      if (take_choices[i-1] == give_choices[i-1]){ // Should be i -1
        theta[i] = inv_logit(alpha + beta_win*memory[i-1]); // Update theta based on weighted memory.
      } else {
        theta[i] = inv_logit(alpha + beta_loss*memory[i-1]); // Update theta based on weighted memory.
      }
  
    
    }
}

model{
  //Priors 
  target += normal_lpdf(alpha | prior_mean_alpha, prior_sd_alpha);
  target += normal_lpdf(beta_loss | prior_mean_beta_loss, prior_sd_beta_loss);
  target += normal_lpdf(beta_win | 0, 0.5);  
  
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

  alpha_prior = normal_rng(prior_mean_alpha, prior_sd_alpha);
  beta_loss_prior = normal_rng(prior_mean_beta_loss, prior_sd_beta_loss);
  beta_win_prior = normal_rng(0, 0.5);


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

