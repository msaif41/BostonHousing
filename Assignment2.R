# ISLR package for the dataset
#install.packages('ISLR') 
#install.packages('caret') 
#install.packages('randomForest')

library(ISLR) 
library(car)
library(caret) 
library(randomForest)

# Import BostonHousing.csv file
# Specify missing values
dfSM <- read.csv("Downloads/BostonHousing.csv",na.strings=c("NA",""))

# Initial exploration
summary(dfSM) 
str(dfSM)

# Remove MEDV variable
dfSM <- subset(dfSM,select=-c(MEDV))

# Change data type and rename if needed
dfSM$CAT.MEDV <- factor(dfSM$CAT..MEDV)
dfSM$CHAS <- factor(dfSM$CHAS)

# Remove misnamed CAT..MEDV variable
dfSM <- subset(dfSM,select=-c(CAT..MEDV))

summary(dfSM)
str(dfSM)

# Check multicollinearity
vif(glm(formula=CAT.MEDV ~ . , family = binomial(link='logit'),data = dfSM))

# Random Forest

# Data partition with the Caret package
set.seed(101)
trainIndex <- createDataPartition(dfSM$CAT.MEDV,
                                  p=0.7, list=FALSE, times=1)
# Create Training data
dfSM.train <- dfSM[trainIndex,] 
# Create Test data
dfSM.test <-dfSM[-trainIndex,]

# Create a default random forest model
rf_defaultSM <- train(CAT.MEDV~., data=dfSM.train,
                    method='rf', metric='Accuracy', ntree=100)
print(rf_defaultSM)

# More detailed model tuning to search the best mtry
tuneGrid <- expand.grid(.mtry=c(1:12))

rf_mtrySM <- train(CAT.MEDV~., 
                   data=dfSM.train,
                   method='rf', 
                   metric='Accuracy', 
                   tuneGrid=tuneGrid, 
                   importance=TRUE, 
                   ntree=100)
print(rf_mtrySM)

# Evaluate model performance
prediction <- predict(rf_mtrySM,dfSM.test) 
confusionMatrix(prediction,dfSM.test$CAT.MEDV)
# Variable importance
varImp(rf_mtrySM)

#SVM

# Create 10-fold cross validation with trainControl() function
trControl <- trainControl(method='cv', number=10,
                          search='grid')

# SVM Model with the linear Kernel function
# Pre-processing data with centering and scaling 
svm_linearSM <- train(CAT.MEDV~.,
                      data=dfSM.train, 
                      method='svmLinear', 
                      trControl=trControl, 
                      preProcess=c('center','scale'))
print(svm_linearSM)

# Evaluate the linear SVM model performance
linear_pred <- predict(svm_linearSM,dfSM.test) 
confusionMatrix(linear_pred,dfSM.test$CAT.MEDV)

# SVM Model with the Radial Kernel function
svm_radialSM <- train(CAT.MEDV~., 
                      data=dfSM.train,
                      method='svmRadial', 
                      trControl=trControl, 
                      preProcess=c('center','scale'))
print(svm_radialSM)

# Evaluate the radial SVM model performance
radial_pred <- predict(svm_radialSM,dfSM.test) 
confusionMatrix(radial_pred,dfSM.test$CAT.MEDV)

# Additional model tuning for the linear SVM model

grid_linear <- expand.grid(C = c(0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5, 0.75, 1))

svm_linear_tuneSM <- train(CAT.MEDV~., 
                           data=dfSM.train,
                           method='svmLinear', 
                           trControl=trControl, 
                           preProcess=c('center','scale'), 
                           tuneGrid=grid_linear)

print(svm_linear_tuneSM)

# Evaluate the linear SVM model performance
linear_tune_pred <- predict(svm_linear_tuneSM,dfSM.test) 
confusionMatrix(linear_tune_pred,dfSM.test$CAT.MEDV)

# Additional model tuning for the radial SVM model
grid_radial <- expand.grid(sigma = c(0.0, 0.05, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20), 
                           C = c(0, 0.10, 0.25, 0.5, 0.75, 1))
svm_radial_tuneSM <- train(CAT.MEDV~., 
                           data=dfSM.train,
                           method='svmRadial', 
                           trControl=trControl, 
                           preProcess=c('center','scale'), 
                           tuneGrid=grid_radial)
print(svm_radial_tuneSM)

# Evaluate the radial SVM model performance
radial_tune_pred <- predict(svm_radial_tuneSM,dfSM.test) 
confusionMatrix(radial_tune_pred,dfSM.test$CAT.MEDV)