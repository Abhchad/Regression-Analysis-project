## An Analysis of the Residential Building Dataset


### Packages Required
library('e1071')
library('readxl')
library('ROCR')
library('randomForest')
library('xgboost')
library('readxl')
library('caret')

###Let's import data from excel 

Residential_Building_Data_Set <- read_excel("residential-building-data-set.xlsx")

mydata = Residential_Building_Data_Set 


###Let's do some Preprocessing

#1.Data Cleaning

# Let's see any null value is present in this: 

# Check for null
sum( is.na(mydata) ) > 0

# Great, There is no null value in this dataset.

#now, As Column name is not readable so let's do some changes in that:

# Reset colomn names
colnames(mydata) = mydata[1,]
mydata = mydata[-1,]

names(mydata)

# We will exam only the first lag for this project
# Remove Lag 2 to Lag 5
mydata = mydata[, c(seq(5,31),108, 109)]
names(mydata)
mydata = apply(mydata, 2, as.numeric )
mydata = data.frame(mydata)
class(mydata)

# Explore the dataset
# Actual sales prices - Preliminary estimated construction cost based on the prices at the beginning of the project
mydata$V.10 - mydata$V.5
# Price of the unit at the beginning of the project per m2 - Preliminary estimated construction cost based on the prices at the beginning of the project
mydata$V.8 - mydata$V.5

##  Create Response

# Generate a new response based on the Actual sales prices (V.9) and the Actual construction costs(V.10)
# Response Profitability is a binary variable. 
# It has value of 1 if the the Actual construction costs is more than five times of the construction price;
# It has value of 0 if the the Actual construction costs is less than five times of the construction price.

profitability.5 = ifelse(mydata$V.9/mydata$V.10 >5, 'Y','N')
sum(profitability.5 == 'Y')
sum(profitability.5 == 'N')
mydata$prof = as.factor(profitability.5)
names(mydata)

# drop V.9 and V.10
mydata = mydata[ , !(names(mydata) %in% c('V.9', 'V.10'))]

# After cleaning
str(mydata)



# Now , Let's Do some Exploratory Data Analysis (EDA)
# Summary of dataset
summary(mydata)

# Bar plot for the response variable
ggplot(mydata, aes(x = prof)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  ggtitle("Distribution of Response Variable") +
  xlab("Profitability") +
  ylab("Count")

# Histogram for a continuous variable (replace 'V.8' with any other continuous variable)
ggplot(mydata, aes(x = V.8)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  theme_minimal() +
  ggtitle("Distribution of Variable V.8") +
  xlab("V.8") +
  ylab("Frequency")



# Boxplot for a continuous variable (replace 'V.8' with any other continuous variable)
ggplot(mydata, aes(x = prof, y = V.8, fill = prof)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Boxplot of V.8 by Profitability") +
  xlab("Profitability") +
  ylab("V.8")


# Pair plot for the first few variables
library(GGally)
ggpairs(mydata[, c(1:5, ncol(mydata))], aes(color = prof, alpha = 0.5)) +
  theme_minimal() +
  ggtitle("Pair Plot of First Five Variables")

# Correlation heatmap
library(corrplot)
numeric_vars <- mydata[, sapply(mydata, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.8)

# Density plot for a continuous variable (replace 'V.8' with any other continuous variable)
ggplot(mydata, aes(x = V.8, fill = prof)) +
  geom_density(alpha = 0.5) +
  theme_minimal() +
  ggtitle("Density Plot of V.8 by Profitability") +
  xlab("V.8") +
  ylab("Density")

# Bar plot for a categorical variable (replace 'V.5' with any other categorical variable)
ggplot(mydata, aes(x = factor(V.5), fill = prof)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  ggtitle("Bar Plot of V.5 by Profitability") +
  xlab("V.5") +
  ylab("Count")


# Let's Play with all three Models:Random Forest,XGBoost, SVM
# Random Forest

set.seed(100)
train_idx = sample(1:nrow(mydata), 200)
train_data = mydata[train_idx, ]
test_data = mydata[-train_idx, ]

rf_model = randomForest(prof ~ ., data = train_data, importance = TRUE)
rf_pred = predict(rf_model, test_data)

# Confusion Matrix and Accuracy
rf_cm = table(predict = rf_pred, truth = test_data$prof)
rf_accuracy = sum(diag(rf_cm)) / sum(rf_cm)
print(rf_accuracy)


# ROC Curve
rf_prob = predict(rf_model, test_data, type = "prob")[,2]
predob_rf = prediction(rf_prob, test_data$prof)
perf_rf = performance(predob_rf, 'tpr', 'fpr')
plot(perf_rf, main = "Random Forest ROC")




# XGBoost
dtrain = xgb.DMatrix(data = as.matrix(train_data[,-ncol(train_data)]), label = as.numeric(train_data$prof) - 1)
dtest = xgb.DMatrix(data = as.matrix(test_data[,-ncol(test_data)]), label = as.numeric(test_data$prof) - 1)

params = list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, max_depth = 6, gamma = 1)
xgb_model = xgb.train(params, dtrain, nrounds = 100)

xgb_pred_prob = predict(xgb_model, dtest)
xgb_pred = ifelse(xgb_pred_prob > 0.5, 'Y', 'N')

# Confusion Matrix and Accuracy
xgb_cm = table(predict = xgb_pred, truth = test_data$prof)
xgb_accuracy = sum(diag(xgb_cm)) / sum(xgb_cm)
print(xgb_accuracy)

# ROC Curve
predob_xgb = prediction(xgb_pred_prob, test_data$prof)
perf_xgb = performance(predob_xgb, 'tpr', 'fpr')
plot(perf_xgb, main = "XGBoost ROC")




## Support Vector Machine - non-linear kernel


set.seed(100)
train = sample(372, 200, replace = FALSE)
train = sort(train, decreasing = FALSE)


# ROC Curve
rocplot = function(pred, truth, ...){
  predob = prediction(pred, truth)
  perf = performance(predob,'tpr', 'fpr')
  plot(perf, ...)
}



## radial kernel

# use tune on training data set to find the best cost and gamma
tune.out = tune(svm, prof~., data = mydata[train, ], kernel ='radial',
                ranges = 
                  list(cost = c(0.1, 1, 10, 100, 1000),
                       gamma = c(0.5, 1, 2, 3, 4)),
                decision.values = TRUE)
summary(tune.out)

# the model with the lowest error
bestmod.radial = tune.out$best.model
summary(bestmod.radial)
bestmod.radial$index

par(mfrow =c(1,1))
# ROC curve - training data
fitted = attributes(predict(bestmod.radial, mydata[train, ], decision.values = TRUE))$decision.values
#rocplot(fitted, mydata[train, 'prof'], main = "Training Data - Radial Kernel of SVM")

# roc curve step by step

predob = prediction(fitted, mydata[train, 'prof'])
perf = performance(predob,'tpr', 'fpr')
plot(perf, main = "Training Data - Radial Kernel of SVM")





# predict on the test data with the best model
bestmod.rad.pred = predict(bestmod.radial, mydata[-train, ],  decision.values = TRUE)
table1 = table(predict = bestmod.rad.pred, truth = mydata[-train,]$prof )
table1

# accuracy rate
(table1[1,1] + table1[2,2])/sum(table1)
# percentage of obs that are misclassified by this svm
(table1[1,2] + table1[2,1])/sum(table1)

# ROC curve - test data
fitted = attributes(bestmod.rad.pred)$decision.values
#rocplot(fitted, mydata[-train, ], 
#        main = "Test Data - Radial Kernel of SVM", col = 'red')

predob = prediction(fitted, mydata[-train, 'prof'])
perf = performance(predob,'tpr', 'fpr')
plot(perf, main = 'Test Data - Radial Kernel of SVM', col = 'red')


## polynomial kernel 

# use tune on training data set to find the best cost and gamma
tune.out = tune(svm, prof~., data = mydata[train,], kernel = 'polynomial',
                ranges = 
                  list(degree =c(1, 2, 3, 4, 5), 
                       cost = c(0.1, 1, 10, 100, 1000)),
                decision.values = TRUE)
summary(tune.out)

# the model with the lowest error
bestmod.poly = tune.out$best.model
summary(bestmod.poly)
bestmod.poly$index

# ROC curve - training data
fitted = attributes(predict(bestmod.poly, mydata[train, ], decision.values = TRUE))$decision.values
#rocplot(fitted, mydata[train, "prof"], 
 #       main = "Training Data - Polynomial Kernel of SVM")

predob = prediction(fitted, mydata[train, 'prof'])
perf = performance(predob,'tpr', 'fpr')
plot(perf, main = 'Training Data - Polynomial Kernel of SVM')



# predict on the test data with the best model
bestmod.poly.pred = predict(bestmod.poly, mydata[-train,], decision.values = TRUE)
table2 = table(predict = bestmod.poly.pred, truth = mydata[-train,]$prof )
table2

# accuracy rate
svm_acc=(table2[1,1] + table2[2,2])/sum(table2)
# percentage of obs that are misclassified by this svm
svm_mis=(table2[1,2] + table2[2,1])/sum(table2)

# ROC curve - test data
fitted = attributes(bestmod.poly.pred)$decision.values
#rocplot(fitted, mydata[order(mydata$prof,decreasing = TRUE), ][-train, "prof"], 
#        main = "Test Data - Polynomial Kernel of SVM", col ='red')
predob = prediction(fitted, mydata[-train, 'prof'])
perf = performance(predob,'tpr', 'fpr')
plot(perf, main = 'Test Data - Polynomial Kernel of SVM', col = 'red')




# Create a bar plot for accuracy comparison
accuracy_data <- data.frame(
  Model = c("Random Forest", "XGBoost", "SVM"),
  Accuracy = c(rf_accuracy, xgb_accuracy, svm_acc)
)

library(ggplot2)
ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("Accuracy Comparison of Models") +
  ylab("Accuracy") +
  xlab("Model")



# Plot ROC curves for all models on the same graph
plot(perf_rf, col = "blue", main = "ROC Curve Comparison", lwd = 2)
plot(perf_xgb, add = TRUE, col = "green", lwd = 2)
plot(perf, add = TRUE, col = "red", lwd = 2) # for SVM, use the appropriate perf variable

legend("bottomright", legend = c("Random Forest", "XGBoost", "SVM"), 
       col = c("blue", "green", "red"), lwd = 2)





library(caret)
library(e1071)
library(randomForest)
library(xgboost)

# Function to calculate metrics
calculate_metrics <- function(confusion_matrix) {
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  precision <- diag(confusion_matrix) / rowSums(confusion_matrix)
  recall <- diag(confusion_matrix) / colSums(confusion_matrix)
  f1_score <- 2 * precision * recall / (precision + recall)
  metrics <- data.frame(
    Accuracy = accuracy,
    Precision_Y = precision[1],
    Precision_N = precision[2],
    Recall_Y = recall[1],
    Recall_N = recall[2],
    F1_Score_Y = f1_score[1],
    F1_Score_N = f1_score[2]
  )
  return(metrics)
}

# Random Forest
rf_pred <- predict(rf_model, test_data)
rf_cm <- table(Predicted = rf_pred, Actual = test_data$prof)
rf_metrics <- calculate_metrics(rf_cm)

# XGBoost
xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 'Y', 'N')
xgb_cm <- table(Predicted = xgb_pred, Actual = test_data$prof)
xgb_metrics <- calculate_metrics(xgb_cm)

# SVM
svm_pred <- predict(bestmod.poly, mydata[-train,], decision.values = TRUE)
svm_cm <- table(predict = bestmod.poly.pred, truth = mydata[-train,]$prof )
svm_metrics <- calculate_metrics(svm_cm)

# Combine all metrics
all_metrics <- rbind(
  cbind(Model = "Random Forest", rf_metrics),
  cbind(Model = "XGBoost", xgb_metrics),
  cbind(Model = "SVM", svm_metrics)
)

# Save to CSV
write.csv(all_metrics, "model_metrics.csv", row.names = FALSE)

# Displaying the accuracies
print(all_metrics)

# Displaying the accuracies
cat("Random Forest Accuracy: ", rf_metrics$Accuracy, "\n")
cat("XGBoost Accuracy: ", xgb_metrics$Accuracy, "\n")
cat("SVM Accuracy: ", svm_metrics$Accuracy, "\n")


#########Result #######################################
#Hence We have SVM as the best Model having 86.62% of accuracy.