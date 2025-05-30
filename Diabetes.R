head(diabetes_012_health_indicators_BRFSS2015)

df <- diabetes_012_health_indicators_BRFSS2015

library(tidyverse)
library(caret)
library(corrplot)


cor_matrix <- cor(df, use = 'complete.obs')
diag(cor_matrix) <- NA

top_corr <- as.data.frame(as.table(cor_matrix)) %>%
  mutate(Var1 = as.character(Var1), Var2 = as.character(Var2)) %>%
  filter(!is.na(Freq), Var1 < Var2) %>%
  arrange(desc(abs(Freq)))

head(top_corr, 10)

corrplot(cor_matrix, method = 'color', tl.cex = 0.5)

