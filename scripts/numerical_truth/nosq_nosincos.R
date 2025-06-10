# -- Install needed packages if not already installed --
# install.packages(c("MASS", "randomForest", "mgcv", "doParallel"))

library(MASS)       # For mvrnorm
library(randomForest)
library(mgcv)
library(parallel)   # For detectCores, makeCluster
library(doParallel) # For foreach %dopar%

# Set seed for reproducibility
set.seed(20044)

# Problem parameters
tol   <- 1e-2
n     <- 1e5   # Be cautious: this is large in combination with calculate_condition_exp!
N     <- 1e4
p     <- 10
rhos  <- c(0.1, 0.3, 0.5, 0.7, 0.9)

# Function to compute conditional expectation E[ sin(X1) | X2,...,X10 ] approximation
# Note: We now explicitly pass 'rho' to avoid scoping issues in parallel workers.
calculate_condition_exp <- function(X210, sig, rho) {
  # Mean of conditional distribution
  mu_cond <- sig[1, 2:10] %*% solve(sig[2:10, 2:10], X210)
  # Generate many samples from the conditional distribution
  # (sd depends on sqrt(1 - rho^2))
  samples <- rnorm(N, mean = mu_cond, sd = sqrt(1 - rho^2))
  return(mean((samples)^2))
}

# Setup parallel backend
numCores <- detectCores()       # Number of CPU cores on your machine
cl <- makeCluster(numCores)     # Create cluster
registerDoParallel(cl)          # Register parallel backend

# We'll store the final results here
full_vals <- numeric(length(rhos))

# Outer loop over different 'rho' values
for (j in seq_along(rhos)) {
  rho <- rhos[j]
  
  # Covariance matrix: each entry is rho^|i-j|
  sig <- rho^toeplitz(0:(p-1))
  
  # Generate multivariate normal samples
  X <- mvrnorm(n, mu = rep(0, p), Sigma = sig)
  
  # Generate random noise
  epsilon <- rnorm(n)
  
  # Compute Y using the formula
  Y <- 3*(X[, 1])^2 +
    3*X[, 3] * X[, 6] +
    3*X[, 10] +
    epsilon
  
  # Parallelize the inner loop using foreach
  results <- foreach(i = 1:n, .combine = 'c') %dopar% {
    # Extract X2,...,X10 for this row
    X210  <- X[i, -1]
    # Compute the conditional expectation term
    value <- calculate_condition_exp(X210, sig, rho)
    # Return the squared difference
    ( Y[i]
      - 3 * value
      - 3*(X[i, 3] * X[i, 6] + X[i, 10])
      - epsilon[i]
    )^2
  }
  
  # Mean over all i
  full_vals[j] <- mean(results)
}

# Shutdown cluster
stopCluster(cl)

# Print final results
full_vals
