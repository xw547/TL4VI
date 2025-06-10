library(MASS)  # For mvrnorm
library(randomForest)
library(mgcv)

tol = 1e-2
set.seed(20044)
tol = 1e-2
n    = 1e5
N = 1e4
p = 10
#sig  = diag(rep(1, p))
#sig[1, 2] = rho
#sig[2, 1] = rho
#create sig such that each entry would be rho^|i-j| using built in function

calculate_condition_exp<- function(X1, X210, sig){
  samples = rnorm(1e4, mean = sig[1, 2:10] %*% solve(sig[2:10, 2:10], X210), sd = sqrt(1-rho^2))
  return(mean(sin(samples)))
}

full_vals = rep(NA, 5)
rhos = c(0.1, 0.3, 0.5, 0.7, 0.9)
results = rep(NA, n)
# value of variable importance
for (j in 1:5){
  rho = rhos[j]
  sig = rho^toeplitz(0:(p-1))
  # Generate multivariate normal samples
  X <- mvrnorm(n, mu = rep(0, p), Sigma = sig)
  X_reduced = X[, -1]
  # Generate random noise
  epsilon <- rnorm(n)
  
  # Compute Y using the formula
  Y <- 3*sin(X[, 1]) +
    3*cos(X[, 2]) +
    3*X[, 3] * X[, 6] +
    3*X[, 7]^2 +
    3*X[, 10] +
    epsilon
  
  for (i in 1:n){
    X210 = X[i, -1]
    X1 = X[i, 1]
    value = calculate_condition_exp(X1, X210, sig)
    results[i] = (Y[i] - 3 * value- 3 * (cos(X[i, 2]) +
                                        X[i, 3] * X[i, 6] +
                                        X[i, 7]^2 +
                                        X[i, 10])- epsilon[i])^2
  }
  full_vals[j] = mean(results)
}
full_vals
