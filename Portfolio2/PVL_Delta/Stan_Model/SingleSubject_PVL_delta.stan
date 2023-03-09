// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; // N.
  array[N] int x; // choices that the agent makes
  vector[N] payoff; // Payoff for each x (win or loss).
  real E_left_1;
  real E_right_1;
}


parameters {
  real a;
  real A;
  real theta;
  real w;

}

model {
  array[N, 2] real u;
  array[N, 2] real E;
  E[1, 1] = E_left_1;
  E[1, 2] = E_right_1;
  array[N, 2] real exp_lr; // components of the softmax choice rule for left and right.  
  array[N, 2] real p;
  for (t in 2:N) {

    
      for (h in 1:2) {
        if(payoff[t - 1] < 0)  { u[t, h] = -w * abs(payoff[t-1]);
        }
          
        else{ u[t,h] = payoff[t-1]^A;
        }
        
        if(x[t-1] == h){
          E[t, h] = E[t-1, h] + (a*(u[t, h] - E[t-1, h]));
        }
        else{ E[t, h] = E[t-1, h];
        }
        
        exp_lr[t, h] = exp(theta*E[t, h]); # first step of softmax
      }
    
      for (h in 1:2) {
        
        p[t, h] = exp_lr[ t, h]/sum(exp_lr[ t, ]); # second step of softmax (convertin to probability space)
      }
    
      
  }
  target += bernoulli_lpmf(x | p[,2]);
}
