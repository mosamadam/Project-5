library(imager)
library(dplyr)
library(kernlab)
library(caret) 

#read the dataset
data <- read.csv("WDBC.csv", stringsAsFactors = FALSE)

#convert the 'diagnosis' column to a factor
data$diagnosis <- as.factor(data$diagnosis)

#split the data
set.seed(123) # for reproducible results
index <- createDataPartition(data$diagnosis, p = 0.8, list = FALSE)
train_data <- data[index,]
test_data <- data[-index,]

#scale and center features
preproc <- preProcess(train_data[,3:ncol(train_data)], method = c("center", "scale"))
train_data_norm <- predict(preproc, train_data[,3:ncol(train_data)])
test_data_norm <- predict(preproc, test_data[,3:ncol(test_data)])

train_data_norm <- cbind(train_data[,2], train_data_norm)
colnames(train_data_norm)[1] <- "diagnosis"
test_data_norm <- cbind(test_data[,2], test_data_norm)
colnames(test_data_norm)[1] <- "diagnosis"


#find best k
k_val <- seq(2, 10, 1)
test_accuracies <- numeric(length(k_val))

for (i in seq_along(k_val)) {
  k <- k_val[i]
  set.seed(123)
  pred <- knn(train = train_data_norm[, -1], test = test_data_norm[, -1], cl = train_data_norm[, 1], k = k)
  
  #calc and store test accuracy
  test_accuracies[i] <- sum(pred == test_data_norm[, 1]) / nrow(test_data_norm)
}

#find best k
best_k <- k_val[which.max(test_accuracies)]
best_accuracy <- max(test_accuracies)

final_pred <- knn(train=train_data_norm[, -1], test=test_data_norm[, -1], cl=train_data_norm[, 1], k=best_k)

confusion_matrix <- table(Predicted = final_pred, Actual = test_data_norm[, 1])
print(confusion_matrix)

print(paste("Best K:", best_k))
print(paste("Best test accuracy:", best_accuracy))

accuracy_df <- data.frame(K = k_val, TestAccuracy = test_accuracies)

#plot test accuracies for each k
wbdc_acc <- ggplot(accuracy_df, aes(x = K, y = TestAccuracy)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "WBDC Dataset",
       x = "Number of Neighbors (K)",
       y = "Test Accuracy")

ggsave("wbdc_acc.png", plot = wbdc_acc, width = 7, height = 3, dpi = 300)

