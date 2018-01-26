#2.1 Naive Bayes Classification

library(psych)
library(rpart)
library(caret)
library(rpart.plot)
library(ROCR)
library(e1071)


#Directly set the working directly using file pane 
ilpd <- read.csv("ILPD.csv", sep=",", header=T)


set.seed(100)

# Split into 60-40 (train-test).
index<- sample(1:nrow(ilpd),size =0.4*nrow(ilpd))
test<- ilpd[index,]
train<-ilpd[-index,]

#Tree-Based Classifier
model <- rpart(label ~ tb+db+aap+alb+age, method="class", data=train)
rpart.plot(model, extra=104, fallen.leaves = T, type=4, tweak = 3)
pred <- predict(model, test, type="class")

#ROCR Curve
pred.rocr <- predict(model, newdata=test, type="prob")[,2]
f.pred <- prediction(pred.rocr, test$label)
f.perf <- performance(f.pred, "tpr", "fpr")
plot(f.perf, colorize=T, lwd=3)
abline(0,1)
auc <- performance(f.pred, measure = "auc")
auc@y.values[[1]]



#Naive Bayes Classification 
model <- naiveBayes(as.factor(label) ~ tb+db+aap+alb+age, method="class", data=train)

# More information about the model
print(model)


# Run the prediction on the test dataset.
pred <- predict(model, test, type="class")
table(pred)


#ROCR curve 
pred.rocr <- predict(model, newdata=test, type="raw") # Posterior probabilities
f.pred <- prediction(pred.rocr[,2], test$label)
f.perf <- performance(f.pred, "tpr", "fpr")
plot(f.perf, colorize=T, lwd=3)
abline(0,1)
auc <- performance(f.pred, measure = "auc")
auc@y.values[[1]]

#As area under curve of Naive Bayes Classifier is more as compare to 
#tree-based classifier, hence Niave Bayes performs better
