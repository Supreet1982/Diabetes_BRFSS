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











