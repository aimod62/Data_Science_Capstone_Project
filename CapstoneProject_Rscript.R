
title: "Rscript - Capstone Project"
author: "Astrid Melhado Dyer"
date: "9/18/2019"


##load packages and data
if (!require("pacman")) install.packages("pacman"); library(pacman)
p_load("mlbench", "caret", "dplyr", "psych", "knitr", "kableExtra", "corrplot",
              "rpart", "kernlab", "gbm", "randomForest", "Cubist")
data("BostonHousing2")

##Create data partition
train_index <- createDataPartition(BostonHousing2$cmedv, p=0.80, list = FALSE)
train_set <- BostonHousing2[train_index,]
test_set <- BostonHousing2[-train_index,]


## Descriptive Statistcs
### Quantitative Summary
sapply(train_set, class)
###The first 6 rows of our data set.
train_set <- train_set  %>% mutate(chas = as.numeric(as.character(chas)),
                                   town = as.numeric(as.factor(town))) %>%
  select(-medv) %>% select(cmedv, everything())
kable(head(train_set),format="latex", booktabs=TRUE) %>% 
  kable_styling(latex_options="scale_down")
###data dimension 
dim(train_set)
###Summary statitics
tab <- describe(train_set[,2:18])
tab %>% mutate(SKEW = cell_spec(skew, "latex", color = ifelse(skew > 0.5 | skew < -0.5, "red", "black")),
               KURTOSIS = cell_spec(kurtosis, "latex", color = ifelse(kurtosis > 3 | kurtosis < -3, "blue", "black"))) %>%
  select(-skew, -kurtosis) %>%
  kable(format = "latex", escape = FALSE, booktabs=TRUE, linesep ="") %>%
  kable_styling(latex_options="scale_down") 

### Visualizations
#### Density Plots
par(mfrow=c(3,6))
for(i in 2:18) {
  plot(density(train_set[,i]), main=names(train_set)[i])}
#### Boxplots
par(mfrow=c(3,6))
for(i in 2:18) {
  boxplot(train_set[,i], main=names(train_set)[i])
}

## Looking for Collinearity
### Correlation Plot
correlations <- cor(train_set[,2:18])
corrplot(correlations, method="circle")
### Pruning Highly Correlated Features

#A threshold of 0.75 is established, above that the features will be removed by the following piece of code:
  
set.seed(13)
cutoff <- 0.75
correlations <- cor(train_set[,2:18])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(train_set)[value])
}
#The new dimensions of our data set:
train_set_1 <- train_set[,-highlyCorrelated]
dim(train_set_1)

## In search of the Right Algorithm
### Test Harness Design
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)#Standard approach
metric <- "RMSE"
#lm
set.seed(13)
fit.lm <- train(cmedv~., data=train_set_1, method="lm", metric=metric, preProc=c("center",
                                                                                 "scale", "BoxCox"), trControl=trainControl)
#glm
set.seed(13)
fit.glm <- train(cmedv~., data=train_set_1, method="glm", metric=metric, preProc=c("center",
                                                                                   "scale", "BoxCox"), trControl=trainControl)
#rpart
set.seed(13)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.rpart<- train(cmedv~., data=train_set_1, method="rpart", metric=metric, tuneGrid = grid, preProc=c("center",
                                                                                                       "scale", "BoxCox"), trControl=trainControl)
#svm
set.seed(13)
fit.svm <- train(cmedv~., data=train_set_1, method="svmRadial", metric=metric, preProc=c("center",
                                                                                         "scale", "BoxCox"), trControl=trainControl)
#KNN
set.seed(13)
fit.knn <- train(cmedv~., data=train_set_1, method="knn", metric=metric, preProc=c("center",
                                                                                   "scale", "BoxCox"), trControl=trainControl)
#GBM
set.seed(13)
fit.gbm <- train(cmedv~., data=train_set_1, method="gbm", metric=metric, preProc=c("center",
                                                                                   "scale", "BoxCox"), trControl=trainControl, verbose = FALSE)
#RF
set.seed(13)
fit.rf <- train(cmedv~., data=train_set_1, method="rf", metric=metric, preProc=c("center",
                                                                                 "scale", "BoxCox"),trControl=trainControl)
#Cubist
set.seed(13)
fit.cubist <- train(cmedv~., data=train_set_1, method="cubist", metric=metric,
                    preProc=c("center","scale", "BoxCox"), trControl=trainControl)
results <- resamples(list(LM=fit.lm, GLM=fit.glm, CART=fit.rpart, SVM=fit.svm, KNN = fit.knn,
                          GBM=fit.gbm, RF=fit.rf, CUBIST=fit.cubist))

summary(results)

### Visualizing the results  
#Comparing the models: box and whisker plots
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)
# Comparing selected models: xyplot
xyplot(results, models=c("CUBIST", "RF"))

## Improving the Selected Models
### Tuning the Random Forest Algorithm
# Manual Search
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
metric <- "RMSE"
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(train_set_1))))
modellist <- list()
for (ntree in c(500, 1000, 1500, 2000)) {
  set.seed(13)
  fit <- train(cmedv~., data = train_set_1, method="rf", metric=metric, tuneGrid=tunegrid,
               trControl=trainControl, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

### Tune the Cubist Algorithm
#Tune the Cubist model
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(13)
tunegrid <- expand.grid(.committees=seq(10, 25, by=1), .neighbors=c(2, 3, 5, 8))
cubist_tuning <- train(cmedv~., data=train_set_1, method="cubist", metric=metric,
                       preProc=c("center", "scale", "BoxCox"), tuneGrid=tunegrid, trControl=trainControl)
print(cubist_tuning)
plot(cubist_tuning)

#The results are self-explanatory: the optimal combination occurs at committee 24, neighbor 2.

## Final Model
### Train the Final Model
set.seed(13)
x <- train_set_1[,2:14]
y <- train_set_1[,1]
preprocessed_data <- preProcess(x, method=c("center", "scale", "BoxCox"))
x_preprocessed <- predict(preprocessed_data, x)
finalModel <- cubist(x=x_preprocessed, y=y, committees=24)
summary(finalModel)

#We perform same transformation on the test set as the ones applied to our train set, and reassure ourselves that both sets have the same shape

# transform the validation dataset
set.seed(13)
test_set_1 <- test_set %>% mutate(chas = as.numeric(as.character(chas)),
                                  town = as.numeric(as.factor(town))) %>%
  select(-medv, -rad, -town, -chas, -rm) %>% select(cmedv, everything())
new_df <- setdiff(train_set_1, test_set_1)
new_df

### Making Predictions on the Test Dataset

test_set_x <- test_set_1[,2:14]
test_set_y <- test_set_1[,1]
test_set_x_preprocessed <- predict(preprocessed_data, test_set_x)
# use final model to make predictions on the validation dataset
predictions <- predict(finalModel, newdata=test_set_x_preprocessed, neighbors=2)
# calculate Metrics
rmse <- RMSE(predictions, test_set_y)
r2 <- R2(predictions, test_set_y)

#**Rsquared** = 0.8073963

#**RMSE** = 4.140558


