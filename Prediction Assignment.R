##Install the packages echo=FALSE
```{r, include=FALSE, echo=FALSE}
destination <- getwd()
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
install.packages("data.table")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("lmtest")
install.packages("flextable")
install.packages("caret")
```

##Load the libraries
```{r, Load Libraries}
library("data.table")
library("ggplot2")
library("lmtest")
library("flextable")
library("dplyr")
library("tinytex")
library("caret")
```

##Download all needed data
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","training.csv",method = "curl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","testing.csv",method = "curl")

training_set <- tibble::as_tibble(fread("training.csv",na.strings=c('#DIV/0!', '', 'NA')))
testing_set  <- tibble::as_tibble(fread("testing.csv",na.strings=c('#DIV/0!', '', 'NA')))

## Analyze the Data
# Testing Set
exit_values_1 <- tibble(
  "Observations" = ncol(testing_set),
  "Variables" = nrow(testing_set))
flextable(exit_values_1)

# Training Set
exit_values_2 <- tibble(
  "Observations" = ncol(training_set),
  "Variables" = nrow(training_set))
flextable(exit_values_2)

##Split the Data
set.seed(171)
training_sub_set <- createDataPartition( y = training_set$classe,p = 0.7,list = FALSE)
real_training <- training_set[training_sub_set,]
real_validation <- training_set[-training_sub_set,]

##Pre-process the data
# Remove variables with mostly N/A values
NA_vals <- sapply(real_training,function(x) mean(is.na(x))) > 0.95
real_training <- real_training[,NA_vals==FALSE]
real_validation <- real_validation[,NA_vals==FALSE]

# Remove variables with low variance
nzv <- nearZeroVar(real_training) #Using the training across both datasets as there has to be conformity
real_training <- real_training[,-nzv]
real_validation <- real_validation[,-nzv]

# Remove columns that will have no bearing on the results
# (V1, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)
real_training_clean = select(real_training, -V1, -user_name, -raw_timestamp_part_1, -raw_timestamp_part_2, -cvtd_timestamp)
real_validation_clean = select(real_validation, -V1, -user_name, -raw_timestamp_part_1, -raw_timestamp_part_2, -cvtd_timestamp)

# Testing Set
exit_values_3 <- tibble(
  "Observations" = ncol(real_validation_clean),
  "Variables" = nrow(real_validation_clean))
flextable(exit_values_3)

# Training Set
exit_values_4 <- tibble(
  "Observations" = ncol(real_training_clean),
  "Variables" = nrow(real_training_clean))
flextable(exit_values_4)

## Create validation partitions of sample observations from the Training set
control <- trainControl(method="cv",number = 10)

##Algorithm Performance Check
# Gradient Boosting Model
set.seed(171)
model_GBM <- train(classe ~.,
                  data = real_training_clean,
                  method = "gbm",
                  trControl = control,
                  verbose = FALSE)
# Random Forest
set.seed(171)
model_RF  <- train(classe ~.,
                   data = real_training_clean,
                   method = "rf",
                   trControl = control)

# k-Nearest Neighbors
set.seed(171)
model_KNN <- train(classe~., 
                   data=real_training_clean, 
                   method="knn", 
                   metric="Accuracy", 
                   trControl=control)

# Linear Discriminant Analysis
set.seed(171)
model_LDA <- train(classe~., 
                   data=real_training_clean, 
                   method="lda", 
                   metric="Accuracy",
                   trControl=control)

###Checking performance for all the models
model_results <- resamples(list(lda=model_LDA, knn=model_KNN, gbm=model_GBM, rf=model_RF))
summary(model_results)

## Check performance using training dataset

# Gradient Boosting Model
GBM_Pred <- predict(model_GBM, newdata=real_validation_clean)
conf_Mat_GBM <- confusionMatrix(GBM_Pred, as.factor(real_validation_clean$classe))
print(conf_Mat_GBM)

# Random Forest
RF_Pred <- predict(model_RF, newdata=real_validation_clean)
conf_Mat_RF <- confusionMatrix(RF_Pred, as.factor(real_validation_clean$classe))
print(conf_Mat_RF)

# k-Nearest Neighbors
LDA_Pred <- predict(model_LDA, newdata=real_validation_clean)
conf_Mat_LDA <- confusionMatrix(LDA_Pred, as.factor(real_validation_clean$classe))
print(conf_Mat_LDA)

# Linear Discriminant Analysis
KNN_Pred <- predict(model_KNN, newdata=real_validation_clean)
conf_Mat_KNN <- confusionMatrix(KNN_Pred, as.factor(real_validation_clean$classe))
print(conf_Mat_KNN)

##Summary of the performance
performance <- matrix(round(c(conf_Mat_GBM$overall,conf_Mat_RF$overall,conf_Mat_LDA$overall,conf_Mat_KNN$overall),4), ncol=4)
colnames(performance)<-c('Linear Discrimination Analysis', 'K- Nearest Neighbors','Gradient Boosting','Random Forest')
performance.df <- as.data.frame(performance)
print(qflextable(performance.df))

##Check the variables with the most influence
print(summary(model_GBM))

print(qplot(num_window, roll_belt, data = real_training, col = classe))
print(qplot(pitch_forearm, roll_belt, data = real_training, col = classe))
print(qplot(num_window, pitch_forearm, data = real_training, col = classe))

# The Random Forest has the higher accuracy rate of 0.998 and that will be the best algorithm to use.

## Prediction on the test dataset
model_prediction <- predict(model_RF, testing_set)
table(model_prediction,testing_set$problem_id)


## Summary
# The best model based on the output in the model_results summary is the random forests model