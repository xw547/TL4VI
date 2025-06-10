tol = 1e-2
set.seed(20044)
tol = 1e-2
n    = 1e6
N = 5*1e2
p = 10
rho = 0.3
#sig  = diag(rep(1, p))
#sig[1, 2] = rho
#sig[2, 1] = rho

#create sig such that each entry would be rho^|i-j| using built in function


sig = rho^toeplitz(0:(p-1))
  



X = rmvnorm(n, sigma = sig)
X_reduced = X[, -1]
beta = c(3, 1, 1, 1, 1, 0, 0.5, 0.8, 1.2, 1.5)

Y = c(X%*%beta + rnorm(n, 0, 1))

full_data = data.frame(y = Y, X = X)
colnames(full_data) <- c("output", paste0("X", 1:10))
reduced_data = data.frame(y = Y, X_reduced = X_reduced)
colnames(reduced_data) <- c("output", paste0("X", 2:10))

full_model = lm(output~., data = full_data)
reduced_model = lm(output~., data = reduced_data)
reduced_x  = lm(X1~ 0+ ., data = full_data[,-1])
mean(reduced_model$residuals^2) - mean(full_model$residuals^2)
