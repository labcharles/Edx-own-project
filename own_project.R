## Section 2.1
# Installing and loading the required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(RKaggle)) install.packages("RKaggle", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(RKaggle)

set.seed(2025)

### Section 2.1.1
# Downloading the dataset
datasets <- get_dataset("uciml/biomechanical-features-of-orthopedic-patients")
df_two_class <- datasets[[1]]

### Section 2.1.2
# Adding a "target" column
ortho <- df_two_class %>%
  mutate(target = factor(if_else(class == "Normal", 0, 1),
                         levels = c(0, 1), 
                         labels = c("Normal", "Abnormal")))

# Adding an underscore
ortho <- ortho %>%
  rename(pelvic_tilt_numeric = `pelvic_tilt numeric`)

## Section 2.2
# Exploring the data
glimpse(ortho)

# Data wrangling, from wide data to tidy data
ortho_plot <- ortho %>%
  select(-class) %>% 
  pivot_longer(-target, names_to = "feature", values_to = "value") 

head(ortho_plot)

# Data visualization
ortho_plot %>% ggplot(aes(x = target, y = value, fill = target)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~feature, scales = "free_y") +
  theme_minimal() +
  labs(title = "Feature Distributions by Clinical Condition", 
       x = "Diagnosis",
       y = "Value",
       fill = "Diagnosis") +
  scale_fill_brewer(palette = "Set1")

## Section 2.3
### Section 2.3.1
# Splitting the data
test_index <- createDataPartition(y = ortho$target, times = 1, p = 0.2, list = FALSE)
train_set <- ortho[-test_index, ]
test_set  <- ortho[test_index, ]

# Checking the proportion of the data
prop.table(table(train_set$target))
prop.table(table(test_set$target))

### Section 2.3.2
# Generalized linear models
glm_fit <- train_set %>%
  glm(target ~ pelvic_incidence + pelvic_tilt_numeric + 
        lumbar_lordosis_angle + sacral_slope + 
        pelvic_radius + degree_spondylolisthesis, 
      data = ., 
      family = "binomial")
  
# Summary of glm model
summary(glm_fit)

### Section 2.3.3
# Cross validation
control <- trainControl(method = "cv", number = 5) # five-fold cross validation

# Tune mtry
mtry_grid <- data.frame(mtry = 1:6)   

set.seed(2025)
train_rf <- train(target ~ pelvic_incidence + pelvic_tilt_numeric +
                    lumbar_lordosis_angle + sacral_slope + 
                    pelvic_radius + degree_spondylolisthesis,
                  data = train_set,
                  method = "rf",
                  tuneGrid = mtry_grid,
                  trControl = control,
                  importance = TRUE)

plot(train_rf)
best_mtry <- train_rf$bestTune
best_mtry

# Random forest
fit_rf <- randomForest(target ~ pelvic_incidence + pelvic_tilt_numeric +
                         lumbar_lordosis_angle + sacral_slope + 
                         pelvic_radius + degree_spondylolisthesis,
                       data = train_set,
                       mtry = best_mtry$mtry,
                       importance = TRUE)

# Section 3
## Section 3.1 Prection models
# Logistic regression predictions using test_set
pred_glm_prob <- predict(glm_fit, newdata = test_set, type = "response")
pred_glm <- factor(if_else(pred_glm_prob > 0.5, "Abnormal", "Normal"),
                   levels = c("Normal", "Abnormal"))

# Random forest predictions
pred_rf_prob <- predict(fit_rf, newdata = test_set, type = "prob")[, "Abnormal"]
pred_rf <- predict(fit_rf, newdata = test_set)

## Section 3.2 Confusion matrices
# Confusion matrices
cm_glm <- confusionMatrix(data = pred_glm, 
                          reference = test_set$target, 
                          positive = "Abnormal")

cm_rf  <- confusionMatrix(data = pred_rf, 
                          reference = test_set$target, 
                          positive = "Abnormal")

# AUC
glm_auc <- pROC::auc(response = test_set$target, predictor = pred_glm_prob)
rf_auc <- pROC::auc(response = test_set$target, predictor = pred_rf_prob)

# Summarizing the results 
results <- tibble(
  Model = c("Logistic Regression", "Random Forest"),
  Accuracy = c(cm_glm$overall["Accuracy"], cm_rf$overall["Accuracy"]),
  Sensitivity = c(cm_glm$byClass["Sensitivity"], cm_rf$byClass["Sensitivity"]),
  Specificity = c(cm_glm$byClass["Specificity"], cm_rf$byClass["Specificity"]),
  F1 = c(cm_glm$byClass["F1"], cm_rf$byClass["F1"]),
  AUC = c(as.numeric(glm_auc), as.numeric(rf_auc))) %>%
  mutate(across(where(is.numeric), ~round(., 3)))

results %>%
  knitr::kable(caption = "Comparison of test set performance")
  
# ROC curves 
roc_glm <- pROC::roc(test_set$target, pred_glm_prob) 
roc_rf <- pROC::roc(test_set$target, pred_rf_prob)

# Extracting sensitivities and specificities for the ROC objects
roc_data <- bind_rows(tibble(sensitivities = roc_glm$sensitivities,
                             specificities = roc_glm$specificities,
                             model = "Logistic Regression"),
                      tibble(sensitivities = roc_rf$sensitivities,
                             specificities = roc_rf$specificities,
                             model = "Random Forest"))

# Side-by-side ROC curves
roc_data %>% ggplot(aes(x = 1 - specificities, y = sensitivities, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "black", alpha = 1) +
  theme_minimal() +
  labs(title = "ROC curves comparison", 
       subtitle = paste("Logistic AUC =", round(glm_auc, 3),
                        " | Random Forest AUC =", round(rf_auc, 3)),
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)") +
  scale_color_brewer(palette = "Set1")
