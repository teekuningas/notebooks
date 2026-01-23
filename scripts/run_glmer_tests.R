#!/usr/bin/env nix-shell
#!nix-shell -i Rscript -p R rPackages.lme4

# R script for running glmer (random effects) models
# Called by Python's run_random_effects_tests() function
# 
# Usage: ./run_glmer_tests.R <input_csv> <output_csv> <n_predictors> <n_outcomes>

library(lme4)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: run_glmer_tests.R <input_csv> <output_csv> <n_predictors> <n_outcomes>")
}

input_file <- args[1]
output_file <- args[2]
n_predictors <- as.integer(args[3])
n_outcomes <- as.integer(args[4])

# Load data (check.names=FALSE preserves spaces in column names)
data <- read.csv(input_file, row.names=1, check.names=FALSE)

# Extract predictor and outcome column names
pred_cols <- grep("^pred_", colnames(data), value=TRUE)
out_cols <- grep("^out_", colnames(data), value=TRUE)

# Strip prefixes for clean names
pred_names <- sub("^pred_", "", pred_cols)
out_names <- sub("^out_", "", out_cols)

cat(sprintf("Running %d tests (%d predictors × %d outcomes)\n", 
            length(pred_cols) * length(out_cols),
            length(pred_cols), 
            length(out_cols)))

# Run models
results <- data.frame()
n_converged <- 0
n_failed <- 0

for (i in seq_along(pred_cols)) {
  pred_col <- pred_cols[i]
  pred_name <- pred_names[i]
  
  for (j in seq_along(out_cols)) {
    out_col <- out_cols[j]
    out_name <- out_names[j]
    
    # Progress indicator every 100 tests
    n_total <- (i-1) * length(out_cols) + j
    if (n_total %% 100 == 0) {
      cat(sprintf("  Progress: %d/%d (converged: %d, failed: %d)\n",
                  n_total, length(pred_cols) * length(out_cols),
                  n_converged, n_failed))
    }
    
    # Prepare data for this test
    df <- data.frame(
      outcome = data[[out_col]],
      predictor = data[[pred_col]],
      user = data$user
    )
    df <- na.omit(df)
    
    # Skip if no variation
    if (length(unique(df$outcome)) < 2 || length(unique(df$predictor)) < 2) {
      results <- rbind(results, data.frame(
        Outcome = out_name,
        Predictor = pred_name,
        Coefficient = NA,
        p_value = NA,
        'P(Outcome|Pred)' = NA,
        'P(Outcome|~Pred)' = NA,
        Difference = NA,
        check.names = FALSE
      ))
      n_failed <- n_failed + 1
      next
    }
    
    # Calculate conditional probabilities
    pred_mask <- df$predictor == 1
    p_with <- mean(df$outcome[pred_mask])
    p_without <- mean(df$outcome[!pred_mask])
    diff <- p_with - p_without
    
    # Fit random effects model
    tryCatch({
      model <- glmer(outcome ~ predictor + (1|user),
                     data = df,
                     family = binomial,
                     control = glmerControl(optimizer = "bobyqa",
                                          calc.derivs = FALSE,
                                          optCtrl = list(maxfun = 100000)))
      
      coef_summary <- summary(model)$coefficients
      
      # Extract results
      coef <- coef_summary['predictor', 'Estimate']
      p_val <- coef_summary['predictor', 'Pr(>|z|)']
      
      results <- rbind(results, data.frame(
        Outcome = out_name,
        Predictor = pred_name,
        Coefficient = coef,
        p_value = p_val,
        'P(Outcome|Pred)' = p_with * 100,
        'P(Outcome|~Pred)' = p_without * 100,
        Difference = diff * 100,
        check.names = FALSE
      ))
      
      n_converged <- n_converged + 1
      
    }, error = function(e) {
      # Model failed - record with NA p-value
      results <<- rbind(results, data.frame(
        Outcome = out_name,
        Predictor = pred_name,
        Coefficient = NA,
        p_value = NA,
        'P(Outcome|Pred)' = p_with * 100,
        'P(Outcome|~Pred)' = p_without * 100,
        Difference = diff * 100,
        check.names = FALSE
      ))
      n_failed <<- n_failed + 1
    })
  }
}

cat(sprintf("\nCompleted: converged=%d, failed=%d\n", n_converged, n_failed))

# Save results
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))
