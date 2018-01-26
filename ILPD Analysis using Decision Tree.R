#Akash Gairola

#Make sure to install this package once before using it.
library(psych)
library(rpart)
library(caret)
library(rpart.plot)
library(ROCR)


#Directly set the working directly usinf file pane or use code setwd("D:/CS422 Data Mining/Homework-2")
ilpd <- read.csv("ILPD.csv", sep=",", header=T)


set.seed(100)

# Split into 60-40 (train-test).
index<- sample(1:nrow(ilpd),size =0.4*nrow(ilpd))
test<- ilpd[index,]
train<-ilpd[-index,]

#(A)
#Produces correlation scatterplot
#the diagonal is the attribute value, the upper triangle contains the numeric correlation coefficient and the lower triangle contains the correlation graphs.
pairs.panels(train[1:10])

#We can also check correlation of 2 attributes using cor()

# (i) (tb,db) is having strongest correlation having correlation value 0.97
cor(train$tb,train$db)

# (ii) (age,sgoaa) and (age,tb) are having weakest or no correlation having correlation value 0.0023 and 0.0032 respectively
cor(train$age,train$sgoaa)
cor(train$age,train$tb)

# (iii) (tb,alb) and (age,alb) are most negatively correlated having correaltion value 0.24
cor(train$tb,train$alb)
cor(train$age,train$alb)

# (iv)After looking at the the diagonal in correlation scatterplot, Age appear to follow a Gaussian distribution.
#     We can also confirm using plot(density(train$age))

plot(density(train$age))

#(B)Normalizationa or Scaling the attributes are not useful in scale-invariant algorithms like decision tree as in decison 
#   tree we just comparing  and branching down the tree, so normalization would not help.
#   Normalization is useful in scale-variant algorithms like KNN algorithm, egression, SVMs, perceptrons, neural networks etc.

#(C)
model <- rpart(label ~ ., method="class", data=train)
rpart.plot(model, extra=104, fallen.leaves = T, type=4, tweak = 3)

pred <- predict(model, test, type="class")
table(pred)       # Show predicted class distribution

#Accuracy is 0.6824
#TPR(Sensitivity) is 0.8098
#TNR(specificity) is 0.3857
#PPV(Precion) is 0.7543
confusionMatrix(pred, test$label)

#(F) (i) 
#ROC Curve
pred.rocr <- predict(model, newdata=test, type="prob")[,2]
f.pred <- prediction(pred.rocr, test$label)
f.perf <- performance(f.pred, "tpr", "fpr")
plot(f.perf, colorize=T, lwd=3)
abline(0,1)
auc <- performance(f.pred, measure = "auc")
auc@y.values[[1]]




#(D)

plotcp(model)
printcp(model)
model.pruned <- prune(model, cp=0.027491)
rpart.plot(model.pruned, extra=104, fallen.leaves = T, type=4, tweak=3)
pred1 <- predict(model.pruned, test, type="class")
table(pred1)       # Show predicted class distribution
#Accuracy of pruned tree is better(which is 0.7082) because Pruning reduces the complexity, and therefores improves accuracy by the reduction of overfitting. 
confusionMatrix(pred1, test$label)

#(E)

#Attributes pairs that have high correlation with each other can be reduced to one predictor

#we can remove attribute tb as cor(tb,db) is equivalent to 1 
train_new <- train[,-3]
test_new <- test[,-3]
##we can remove attribute sgpaa as cor(sgpaa,sgoaa) is high
train_new <- train_new[,-5]
test_new <- test_new[,-5]
model <- rpart(label ~ ., method="class", data=train_new)
rpart.plot(model, extra=104, fallen.leaves = T, type=4, tweak = 3)

pred1 <- predict(model, test_new, type="class")
table(pred1)       # Show predicted class distribution

#Accuracy is 0.6953
#TPR(Sensitivity) is 0.8834
#TNR(specificity) is 0.2571
#PPV(Precion) is 0.7347
confusionMatrix(pred, test_new$label)
#This model is better as we have reduced the number of dimensions as compare to the original and have increased the accuracy too.


#(F) (ii)
# ROC curve
pred1.rocr <- predict(model, newdata=test_new, type="prob")[,2]
f.pred <- prediction(pred1.rocr, test_new$label)
f.perf <- performance(f.pred, "tpr", "fpr")
plot(f.perf, colorize=T, lwd=3)
abline(0,1)
auc <- performance(f.pred, measure = "auc")
auc@y.values[[1]]

