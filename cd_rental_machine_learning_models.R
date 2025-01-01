#importing libraries 
library(dplyr)
library(rsample)
library(tidymodels)
library(lubridate)
library(caret)
library(glmnet)

df_rental <- read.csv("rental_info.csv")
glimpse(df_rental)
df_rental$rental_date <- ymd_hms(df_rental$rental_date)
df_rental$return_date <- ymd_hms(df_rental$return_date)
df_rental <- df_rental %>% 
  mutate(rental_length = as.numeric(difftime(return_date, rental_date, units = "hours")))
df_rental$rental_length_days <- df_rental$rental_length/24

#add variables from the special features column
df_rental$deleted_scenes <- as.numeric(grepl("Deleted Scenes", df_rental$special_features))
df_rental$behind_the_scenes  <- as.numeric(grepl("Behind the Scenes", df_rental$special_features))

#keep relevant columns 
X <- df_rental %>% 
  select(-return_date, -rental_date, -rental_length, -special_features)
set.seed(9)
split <- initial_split(X, prop = 0.8)
X_train <- training(split)
X_test <- testing(split)
y_train <- as.numeric(X_train$rental_length_days)
y_test <- as.numeric(X_test$rental_length_days)
X_train$rental_length_days <- NULL
X_test$rental_length_days <- NULL

#center and scale the training and testing sets
#this standardization makes the model less sensative to the scale of features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train <- predict(preProcValues, X_train)
X_test <- predict(preProcValues, X_test)

#perform features selection: here we are using the Lasso model to identify the 
#features to be subsequently used in other regression models
lasso_model <- glmnet(as.matrix(X_train), y_train, alpha = 1, lambda = 0.3)
#extract coefficients at the specified lambda value 
non_zero_coef <- coef(lasso_model, s = 0.3)[,1]
#exclude the intercept 
non_zero_coef <- non_zero_coef[2:length(non_zero_coef)]
#select non_zero coefficients 
features_selected <- names(non_zero_coef[non_zero_coef != 0])
X_train_selected <- X_train[, features_selected, drop = FALSE]
X_test_selected <- X_test[, features_selected, drop = FALSE]

#try a couple of models and choose the best MSE score
#linear regression
lm_model <- lm(y_train ~ ., data = as.data.frame(X_train_selected))
predictions <- predict(lm_model, newdata = X_test)
mse_lr <- mean((predictions - y_test)^2)

#Decision Tree
##train the tree model using cross-validation to select the best tree complexity
dt_model <- train(x = as.matrix(X_train), 
                  y = y_train, 
                  method = "rpart", 
                  trControl = trainControl(method = "cv", number = 10), 
                  tuneLength = 10)
dt_pred <- predict(dt_model, newdata = X_test)
mse_dt <- mean((dt_pred - y_test)^2)

model_verdict <- data.frame(linear_regression = mse_lr, decision_tree = mse_dt)
model_verdict
best_mse <- max(mse_lr, mse_dt)
best_model <- dt_model






