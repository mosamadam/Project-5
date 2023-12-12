library(imager)
library(dplyr)
library(kernlab)
library(class)

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


# Convert the 'diagnosis' column to a factor, which is necessary for classification
data$response <- as.factor(data$response)

###################################################################################
# KNN FUNCTION 

find_best_k <- function(train_data, test_data) {
  k_val <- seq(2, 10, 1)
  test_accuracies <- numeric(length(k_val))
  
  for (i in seq_along(k_val)) {
    set.seed(123)
    pred <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_data[, 1], k = k_val[i])
    test_accuracies[i] <- sum(pred == test_data[, 1]) / nrow(test_data)
  }
  
  best_k <- k_val[which.max(test_accuracies)]
  best_accuracy <- max(test_accuracies)
  list(best_k = best_k, best_accuracy = best_accuracy)
}


########################################################################################
# PCA Linear

set.seed(123)

data_pca <- data[,1:(ncol(data)-1)]

kpca_model <- kpca(~., data = data_pca, kernel = "polydot", 
                   features = 6, kpar = list(degree = 2, scale = 1, offset = 0))

transformed_data <- as.data.frame(rotated(kpca_model))

feature_dim <- seq(from = 2,to = 6, by = 1)

accuracy_h <- c()

for (best_num in feature_dim) {
  
  principal_components <- transformed_data[, 1:best_num]
  
  # Create the dataset with the principal components and response
  pca_data <- data.frame(cbind(response = data$response, principal_components))
  
  # Split the data into training and testing sets
  set.seed(123)
  index <- createDataPartition(pca_data[,1], p = 0.8, list = FALSE)
  train_data_norm <- pca_data[index, ]
  test_data_norm <- pca_data[-index, ]
  
  # Find the best K for the transformed data
  best_k_result <- find_best_k(train_data_norm, test_data_norm)

  accuracy_h <- c(accuracy_h ,best_k_result$best_accuracy)



}

resultsF <- data.frame(cbind(feature_dim ,accuracy_h))

pca_size <- ggplot(resultsF, aes(x = feature_dim, y = accuracy_h)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "noMNIST Dataset - PCA",
       x = "Feature size",
       y = "Test Accuracy")

ggsave("pca_size_mnist.png", plot = pca_size, width = 4, height = 3, dpi = 300)



