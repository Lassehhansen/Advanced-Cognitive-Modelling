
data {
  int<lower=0> N;
  array[N] real<lower=0, upper = 1> y;
  array[N] real<lower=0, upper = 1> Source1;
  array[N] real<lower=0, upper = 1> Source2;
}

transformed data{
  array[N] real l_Source1;
  array[N] real l_Source2;
  array[N] real l_Discrete_Belief;
  l_Discrete_Belief = logit(y);
  l_Source1 = logit(Source1);
  l_Source2 = logit(Source2);
  
}

parameters {
  real bias;
  real<lower = 0.0001> sd;
}

model {
  target +=  normal_lpdf(bias | 0, 0.5);
  target +=  lognormal_lpdf(sd | 0, 0.5);
  target +=  normal_lpdf(to_vector(l_Discrete_Belief) | bias + to_vector(l_Source1) + to_vector(l_Source2), sd);
}

generated quantities{
  real bias_prior;
  real sd_prior;

  array[N] real log_lik;
  
  bias_prior = normal_rng(0, 0.5);
  sd_prior = lognormal_rng(0, 0.5);
  
  for (n in 1:N){  
    log_lik[n] = normal_lpdf(l_Discrete_Belief[n] | bias + l_Source1[n] +  l_Source2[n], sd);
  }
  
}


