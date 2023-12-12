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

###################################################################################
# PCA RADIAL

# Initialize a data frame to store the best accuracies for each sigma
sigma_values <- c(0.1, 1, 10)
resultsR <- data.frame(sigma = sigma_values, best_k = integer(length(sigma_values)), best_accuracy = numeric(length(sigma_values)))

# Apply kPCA with different sigma values and find the best K for each
for (i in seq_along(sigma_values)) {
  # Perform Kernel PCA
  set.seed(123)
  
  data_pca <- data[,1:(ncol(data)-1)]
  
  kpca_model <- kpca(~., data = data_pca, kernel = "rbfdot",features = 6, kpar = list(sigma = sigma_values[i]))
  
  transformed_data <- as.data.frame(rotated(kpca_model))
  
  best_num = 6
  
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
  
  # Store the results
  resultsR[i, "best_k"] <- best_k_result$best_k
  resultsR[i, "best_accuracy"] <- best_k_result$best_accuracy
}

# Output the results
print(resultsR)

pcaRadial_acc <- ggplot(resultsR, aes(x = sigma, y = best_accuracy)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "noMNIST Dataset - PCA Radial Kernal",
       x = "Sigma",
       y = "Test Accuracy")

ggsave("pcaRadial_acc.png", plot = pcaRadial_acc, width = 4, height = 3, dpi = 300)

###################################################################################
# PCA POLYNOMIAL

# Initialize a data frame to store the best accuracies for each sigma
poly_values <- c(2,3,4)
resultsP <- data.frame(sigma = poly_values, best_k = integer(length(poly_values)), best_accuracy = numeric(length(poly_values)))

# Apply kPCA with different sigma values and find the best K for each
for (i in seq_along(poly_values)) {
  # Perform Kernel PCA
  set.seed(123)
  
  data_pca <- data[,1:(ncol(data)-1)]
  
  kpca_model <- kpca(~., data = data_pca, kernel = "polydot", 
               features = 6, kpar = list(degree = poly_values[i], scale = 1, offset = 0))
  
  transformed_data <- as.data.frame(rotated(kpca_model))
  
  best_num = 6
  
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
  
  # Store the results
  resultsP[i, "best_k"] <- best_k_result$best_k
  resultsP[i, "best_accuracy"] <- best_k_result$best_accuracy
}

# Output the results
print(resultsP)

pcaPoly_acc <- ggplot(resultsP, aes(x = sigma, y = best_accuracy)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "noMNIST Dataset - PCA Polynomial Kernal",
       x = "Degree",
       y = "Test Accuracy")

ggsave("pcaPoly_acc.png", plot = pcaPoly_acc, width = 4, height = 3, dpi = 300)

## lets see the scatterplot distribution for the polynomial (2) case

kpca_model <- kpca(~., data = data_pca, kernel = "polydot", 
                   features = 2, kpar = list(degree = 2, scale = 1, offset = 0))

transformed_data <- as.data.frame(rotated(kpca_model))

principal_components <- transformed_data

principal_components <- data.frame(cbind(data$response,principal_components))

polyScat <- ggplot(principal_components, aes(x = V1, y = V2, color = data.response)) + 
  geom_point() +
  labs(title = "Polynomial (2nd degree)",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

ggsave("polyScat_mnist.png", plot = polyScat, width = 4, height = 3, dpi = 300)

########################################################################################
# PCA Linear

# Split the data into training and testing sets
set.seed(123) 

data_pca <- data[,1:(ncol(data)-1)]

pca_model <- prcomp(data_pca)

best_num <- 6

principal_components <- pca_model$x[, 1:best_num]

# Create the dataset with the principal components and response
pca_data <- data.frame(cbind(response = data$response, principal_components))

# Split the data into training and testing sets
set.seed(123) 
index <- createDataPartition(pca_data[,1], p = 0.8, list = FALSE)
train_data_norm <- pca_data[index, ]
test_data_norm <- pca_data[-index, ]

# Find the best K for the transformed data
best_k_result <- find_best_k(train_data_norm, test_data_norm)

## lets see the scatterplot distribution for the liner case


principal_components <- data.frame(cbind(factor(data$response),principal_components))

principal_components$V1 <- as.factor(principal_components$V1)

linScat <-ggplot(principal_components, aes(x = PC1, y = PC2, color = V1)) + 
  geom_point() +
  labs(title = "Linear PCA",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

ggsave("linScat_mnist.png", plot = linScat, width = 4, height = 3, dpi = 300)


