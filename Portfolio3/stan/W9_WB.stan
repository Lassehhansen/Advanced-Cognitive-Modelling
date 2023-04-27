
data {
  int<lower=0> N;
  array[N] real <lower = 0, upper = 1> y;
  array[N] real <lower = 0, upper = 1> Source1; 
  array[N] real <lower = 0, upper = 1> Source2; 
}

transformed data {
  array[N] real l_Source1;
  array[N] real l_Source2;
  array[N] real l_Discrete_Choice;
  l_Source1 = logit(Source1);
  l_Source2 = logit(Source2);
  l_Discrete_Choice = logit(y);
}
parameters {
  real bias;
  real<lower = 0.0001> sd;
  // meaningful weights are btw 0.5 and 1 (theory reasons)
  real<lower = 0.5, upper = 1> w1; 
  real<lower = 0.5, upper = 1> w2;
}
transformed parameters {
  real<lower = 0, upper = 1> weight1;
  real<lower = 0, upper = 1> weight2;
  // weight parameters are rescaled to be on a 0-1 scale (0 -> no effects; 1 -> face value)
  weight1 = (w1 - 0.5) * 2;  
  weight2 = (w2 - 0.5) * 2;
}
model {

  target += normal_lpdf(bias | 0, 0.5);
  target += beta_lpdf(weight1 | 2, 2);
  target += beta_lpdf(weight2 | 2, 2);
  target += lognormal_lpdf(sd | 0, 0.5);
  
  target += normal_lpdf(to_vector(l_Discrete_Choice) | bias + weight1 * to_vector(l_Source1) + weight2 * to_vector(l_Source2), sd);

}

generated quantities{
  array[N] real log_lik;
  real bias_prior;
  real sd_prior;
  real w1_prior;
  real w2_prior;
  bias_prior = normal_rng(0, 0.5) ;
  sd_prior = lognormal_rng(0, 0.5);
  w1_prior = 0.5 + inv_logit(normal_rng(0, 1))/2 ;
  w2_prior = 0.5 + inv_logit(normal_rng(0, 1))/2 ;
  for (n in 1:N)
    log_lik[n]= normal_lpdf(l_Discrete_Choice[n] | bias + weight1 * l_Source1[n] + weight2 * l_Source2[n], sd);
}


