# Load necessary library for vectorized operations
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Number of samples
n_samples <- 1e6

# Generate random variables
X <- matrix(rnorm(n_samples * 15, mean = 0, sd = 2), ncol = 15) # Each row represents one sample of X1, X2, ..., X15
epsilon <- rnorm(n_samples, mean = 0, sd = 2) # epsilon ~ N(0, 1) for each sample

# Define the indicator functions (vectorized for efficiency)
I_neg2_to_2 <- function(x) as.numeric(x >= -2 & x <= 2)
I_negInf_to_0 <- function(x) as.numeric(x <= 0)
I_0_to_Inf <- function(x) as.numeric(x > 0)

# Calculate Y for each sample
Y <- I_neg2_to_2(X[, 1]) * floor(X[, 1]) + # I(-2,+2)(X1) * |X1|
  I_negInf_to_0(X[, 2]) +                # I(-inf,0](X2)
  I_0_to_Inf(X[, 3]) * X[, 3] +          # I(0,+inf)(X3) * X3
  abs(X[, 6] / 4)^3 +                    # |X6/4|^3
  abs(X[, 7] / 4)^5 +                    # |X7/4|^5
  (7 / 3) * cos(X[, 11] / 2) +           # 7/3 * cos(X11/2)
  epsilon                                # + epsilon


Y_full <- I_neg2_to_2(X[, 1]) * floor(X[, 1]) + # I(-2,+2)(X1) * |X1|
  I_negInf_to_0(X[, 2]) +                # I(-inf,0](X2)
  I_0_to_Inf(X[, 3]) * X[, 3] +          # I(0,+inf)(X3) * X3
  abs(X[, 6] / 4)^3 +                    # |X6/4|^3
  abs(X[, 7] / 4)^5 +                    # |X7/4|^5
  (7 / 3) * cos(X[, 11] / 2) 

Y_red <- I_negInf_to_0(X[, 2]) +                # I(-inf,0](X2)
  I_0_to_Inf(X[, 3]) * X[, 3] +          # I(0,+inf)(X3) * X3
  abs(X[, 6] / 4)^3 +                    # |X6/4|^3
  abs(X[, 7] / 4)^5 +                    # |X7/4|^5
  (7 / 3) * cos(X[, 11] / 2) 

# Summary of Y
summary(Y)

mean((Y - Y_red)^2) - mean((Y - Y_full)^2) 
 
full_data = data.frame(y = Y, X = X)
colnames(full_data) <- c("output", paste0("X", 1:15))
reduced_data = data.frame(y = Y, X_reduced = X[,-1])
colnames(reduced_data) <- c("output", paste0("X", 2:15))

full_model = lm(output~., data = full_data)
reduced_model = lm(output~., data = reduced_data)
reduced_x  = lm(X1~ 0+ ., data = full_data[,-1])
mean(reduced_model$residuals^2) - mean(full_model$residuals^2)


# 1) Install/load xgboost (if not already installed)
# install.packages("xgboost")
library(xgboost)

# ------------------------------------------------------------------------------
# 2) Prepare data for xgboost
#    xgboost requires numeric matrices (for predictors) and numeric vectors (for response).
#    We'll assume 'output' is the name of your response variable.

# a) Full data
full_x <- as.matrix(full_data[, !(names(full_data) %in% c("output"))])  # all columns except 'output'
full_y <- full_data$output

# b) Reduced data (fewer predictors)
reduced_x <- as.matrix(reduced_data[, !(names(reduced_data) %in% c("output"))])
reduced_y <- reduced_data$output

# ------------------------------------------------------------------------------
# 3) Train xgboost models
#    This is a minimal example using default hyperparameters and 50 boosting rounds.
#    In practice, you should tune these hyperparameters (e.g., using cross-validation).

full_xgb <- xgboost(data = full_x, label = full_y,
                    nrounds = 50,               # number of boosting iterations
                    verbose = 0)               # suppress printing

reduced_xgb <- xgboost(data = reduced_x, label = reduced_y,
                       nrounds = 50,
                       verbose = 0)

# ------------------------------------------------------------------------------
# 4) Get predictions on the same data used for training
#    (If you have a separate test set, use that instead to get unbiased estimates.)

pred_full    <- predict(full_xgb,   newdata = full_x)
pred_reduced <- predict(reduced_xgb, newdata = reduced_x)

# ------------------------------------------------------------------------------
# 5) Calculate and compare MSEs
mse_full    <- mean((pred_full    - full_y)^2)
mse_reduced <- mean((pred_reduced - reduced_y)^2)

mse_diff <- mse_reduced - mse_full

mse_full
mse_reduced
mse_diff  # difference in MSE between reduced and full model
