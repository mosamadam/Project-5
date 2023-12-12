library(imager)
library(dplyr)
library(kernlab)

load_images <- function(folder_name, response_value) {
  file_paths <- list.files(path = folder_name, pattern = "\\.png$", full.names = TRUE)
  image_list <- lapply(file_paths, function(file) {
    image <- load.image(file) # Use load.image from imager
    data_frame <- data.frame(matrix(as.numeric(image), ncol = length(image)))
    data_frame$response <- response_value
    return(data_frame)
  })
  return(image_list)
}

images_c <- load_images("C", "c")
images_g <- load_images("G", "g")

# combine the data frames into a single data set
data <- bind_rows(images_c, images_g)

data$response <- as.factor(data$response)

write.csv(data, "noMNIST_CG.csv", row.names = FALSE)

#split the data into training and testing sets
set.seed(123) 
index <- createDataPartition(data$response, p = 0.8, list = FALSE)
train_data <- data[index,]
test_data <- data[-index,]

train_data_norm <- train_data[,1:(ncol(train_data)-1)]
test_data_norm <- test_data[,1:(ncol(train_data)-1)]

train_data_norm <- cbind(train_data[,ncol(train_data)], train_data_norm)
colnames(train_data_norm)[1] <- "response"
test_data_norm <- cbind(test_data[,ncol(train_data)], test_data_norm)
colnames(test_data_norm)[1] <- "response"

head(test_data_norm)


#finf best K using cross-validation on the training set
k_val <- seq(2, 10, 1)
test_accuracies <- numeric(length(k_val))

for (i in seq_along(k_val)) {
  k <- k_val[i]
  set.seed(123)
  pred <- knn(train = train_data_norm[, -1], test = test_data_norm[, -1], cl = train_data_norm[, 1], k = k)
  
  #calc and store test accuracy
  test_accuracies[i] <- sum(pred == test_data_norm[, 1]) / nrow(test_data_norm)
}

#find the best k
best_k <- k_val[which.max(test_accuracies)]
best_accuracy <- max(test_accuracies)

print(paste("Best K:", best_k))
print(paste("Best test accuracy:", best_accuracy))

accuracy_df <- data.frame(K = k_val, TestAccuracy = test_accuracies)

#plot test accuracies
mnist_acc <- ggplot(accuracy_df, aes(x = K, y = TestAccuracy)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "noMNIST Dataset",
       x = "Number of Neighbors (K)",
       y = "Test Accuracy")

ggsave("mnist_acc.png", plot = mnist_acc, width = 7, height = 3, dpi = 300)

