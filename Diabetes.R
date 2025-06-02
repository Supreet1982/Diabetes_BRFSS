head(diabetes_012_health_indicators_BRFSS2015)

df <- diabetes_012_health_indicators_BRFSS2015

library(tidyverse)
library(caret)
library(corrplot)

#Target Variable

table(df$Diabetes_012/ncol(df))
mean(df$Diabetes_012)
str(df)

#Correlation

cor_matrix <- cor(df, use = 'complete.obs')
diag(cor_matrix) <- NA

top_corr <- as.data.frame(as.table(cor_matrix)) %>%
  mutate(Var1 = as.character(Var1), Var2 = as.character(Var2)) %>%
  filter(!is.na(Freq), Var1 < Var2) %>%
  arrange(desc(abs(Freq)))

head(top_corr, 10)

corrplot(cor_matrix, method = 'color', tl.cex = 0.5)

#Split

set.seed(1234)

partition <- createDataPartition(y=as.factor(df$Diabetes_012), p=0.7, 
                                 list = FALSE)
df.train <- df[partition,]
df.test <- df[-partition,]

mean(df.train$Diabetes_012)
mean(df.test$Diabetes_012)

#Set the controls

ctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 3)

#Set up the training grid

rf.grid <- expand.grid(mtry=1:21)

#Set up x and y variables

target <- factor(df.train$Diabetes_012)
predictors <- df.train[,-1]

#Train Random Forest

rf1 <- train(y=target, x=predictors, method = 'rf', ntree=5, importance=TRUE,
             trControl = ctrl, tuneGrid = rf.grid)

rf1

plot(rf1)

varImp(rf1)

plot(varImp(rf1))

imp2 <- varImp(rf1, scale = FALSE)
imp_df <- imp2$importance
str(imp_df)
top_class1 <- imp_df[order(-imp_df[['1']]), , drop = FALSE]
head(top_class1, 5)
top_class2 <- imp_df[order(-imp_df[['2']]), , drop = FALSE]
head(top_class2, 5)
top_class0 <- imp_df[order(-imp_df[['0']]), , drop = FALSE]
head(top_class0, 5)

randomForest::varImpPlot(rf1$finalModel)
rf1

pred <- predict(rf1, newdata = df.test, type = 'raw')

confusionMatrix(pred, as.factor(df.test$Diabetes_012))

table(df.test$Diabetes_012)


prop.table(table(df$Diabetes_012))

###############################################################################

#Set up tuning grid for XGBoost

xgb.grid <- expand.grid(max_depth=7, min_child_weight = 1, gamma = 0,
                        nrounds = c(50, 100, 150, 200, 250),
                        eta = c(0.001, 0.002, 0.01, 0.02, 0.1),
                        colsample_bytree = 0.6, subsample = 0.6)

xgb.grid

xgb.ctrl <- trainControl(method = 'cv', number = 5)

set.seed(42)

xgb.tuned <- train(as.factor(Diabetes_012) ~ ., data = df.train,
                   method = 'xgbTree', trControl=xgb.ctrl,
                   tuneGrid=xgb.grid)


xgb.tuned
ggplot(xgb.tuned)

xgb.pred <- predict(xgb.tuned, newdata = df.test, type='raw')
confusionMatrix(xgb.pred, as.factor(df.test$Diabetes_012))

prop.table(table(df.train$Diabetes_012))


################################################################################

#Class imbalance

table(df$Diabetes_012)

df1 <- df 
df1$Diabetes_012 <- as.factor(df1$Diabetes_012)

min_count <- min(table(df1$Diabetes_012))

df_balanced <- df1 %>%
  group_by(Diabetes_012) %>%
  slice_sample(n = min_count) %>%
  ungroup()

table(df_balanced$Diabetes_012)

library(DMwR2)
library(ROSE)
library(smotefamily)

x <- df1[, -which(names(df1) == 'Diabetes_012')]
y <- df1$Diabetes_012

smote_result <- SMOTE(x, y, K=5)
df_smote <- smote_result$data
df_smote$class <- as.factor(df_smote$class)
table(df_smote$class)

str(df_smote)
prop.table(table(df_smote$class))

df_downsampled <- df_smote %>%
  group_by(class) %>%
  slice_sample(n = 35346) %>%
  ungroup()

table(df_downsampled$class)

table(df1$Diabetes_012)

################################################################################

#XGBoost on balanced data

str(df_downsampled)

partition2 <- createDataPartition(y=df_downsampled$class, p=0.7, 
                                 list = FALSE)

df_downsampled.train <- df_downsampled[partition2,]
df_downsampled.test <- df_downsampled[-partition2,]

set.seed(41)

xgb.tuned2 <- train(class ~ ., data = df_downsampled.train,
                   method = 'xgbTree', trControl=xgb.ctrl,
                   tuneGrid=xgb.grid)
ggplot(xgb.tuned2)

ggplot(xgb.tuned)

xgb.pred2 <- predict(xgb.tuned2, newdata = df_downsampled.test, type='raw')
confusionMatrix(xgb.pred2, df_downsampled.test$class)

print(varImp(xgb.pred2))

xgb.tuned2

set.seed(42)

xgb.tuned3 <- train(as.factor(class) ~ ., data = df_downsampled.train,
                    method = 'xgbTree', trControl=xgb.ctrl,
                    tuneGrid=xgb.grid)

ggplot(xgb.tuned3)
xgb.pred3 <- predict(xgb.tuned3, newdata = df_downsampled.test, type = 'raw')
confusionMatrix(xgb.pred3, as.factor(df_downsampled.test$class))

ggplot(varImp(xgb.tuned3))

################################################################################

#SHAP calculations

X <- data.matrix(df_downsampled.train[ 
  ,-which(names(df_downsampled.train)=='class')])

model <- xgb.tuned3$finalModel

# SHAP contributions for all classes
# Predict SHAP with predcontrib = TRUE and reshape output

shap_array <- predict(model, X, predcontrib = TRUE, reshape = TRUE)
str(shap_array)
dim(shap_array)

# Choose class (1 = class 0, 2 = class 1, 3 = class 2)

class_index <- 1
shap_matrix <- shap_array[[class_index]]
shap_matrix
X <- data.matrix(df_downsampled.train
                 [, -which(names(df_downsampled.train) == "class")])
colnames(shap_matrix) <- c(colnames(X), 'BIAS')

# Remove BIAS term (assume it's the last column)

feature_names <- colnames(shap_matrix)[-ncol(shap_matrix)]
shap_df <- as.data.frame(shap_matrix)[, feature_names]

feature_names <- colnames(shap_df)[-ncol(shap_df)]  # exclude bias
importance <- data.frame(
  Feature = feature_names,
  MeanAbsSHAP = apply(abs(shap_df[, feature_names]), 2, mean)
)
library(ggplot2)

ggplot(head(importance, 20), aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 SHAP Features for Class 0",
       x = "Feature", y = "Mean Absolute SHAP Value") +
  theme_minimal()

importance

################################################################################

class_index <- 2
shap_matrix <- shap_array[[class_index]]
shap_matrix
X <- data.matrix(df_downsampled.train
                 [, -which(names(df_downsampled.train) == "class")])
colnames(shap_matrix) <- c(colnames(X), 'BIAS')

# Remove BIAS term (assume it's the last column)

feature_names <- colnames(shap_matrix)[-ncol(shap_matrix)]
shap_df <- as.data.frame(shap_matrix)[, feature_names]

feature_names <- colnames(shap_df)[-ncol(shap_df)]  # exclude bias
importance <- data.frame(
  Feature = feature_names,
  MeanAbsSHAP = apply(abs(shap_df[, feature_names]), 2, mean)
)
library(ggplot2)

ggplot(head(importance, 20), aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 SHAP Features for Class 1",
       x = "Feature", y = "Mean Absolute SHAP Value") +
  theme_minimal()

importance

################################################################################

class_index <- 3
shap_matrix <- shap_array[[class_index]]
shap_matrix
X <- data.matrix(df_downsampled.train
                 [, -which(names(df_downsampled.train) == "class")])
colnames(shap_matrix) <- c(colnames(X), 'BIAS')

# Remove BIAS term (assume it's the last column)

feature_names <- colnames(shap_matrix)[-ncol(shap_matrix)]
shap_df <- as.data.frame(shap_matrix)[, feature_names]

feature_names <- colnames(shap_df)[-ncol(shap_df)]  # exclude bias
importance <- data.frame(
  Feature = feature_names,
  MeanAbsSHAP = apply(abs(shap_df[, feature_names]), 2, mean)
)
library(ggplot2)

ggplot(head(importance, 20), aes(x = reorder(Feature, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 SHAP Features for Class 2",
       x = "Feature", y = "Mean Absolute SHAP Value") +
  theme_minimal()

importance

################################################################################







