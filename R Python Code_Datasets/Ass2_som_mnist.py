import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

np.random.seed(123)
random.seed(123)

data = pd.read_csv("noMNIST_CG.csv")

label_encoder = LabelEncoder()
data['response'] = label_encoder.fit_transform(data['response'])

X = data.iloc[:, data.columns != 'response'].values
y = data['response'].values

X_scaled = (X)

#learning rates
learning_rates = [0.01, 0.001, 0.0001]

results = {'learning_rate': [], 'knn_accuracy': []}

#set fixed grid size
grid_size = (6,6)

for lr in learning_rates:
    
    som = MiniSom(x=grid_size[0], y=grid_size[1], input_len=X_scaled.shape[1], sigma=1, learning_rate=lr)
    som.random_weights_init(X_scaled)
    som.train_random(data=X_scaled, num_iteration=500)

    #get bmu for each point
    BMUs = np.array([som.winner(x) for x in X_scaled])

    BMUs_df = pd.DataFrame(BMUs)

   #print(BMUs_df.head())
    
    # Split the BMU coordinates into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        BMUs, y, test_size=0.2, random_state=123)

    #init knn algo
    best_k = 1
    best_score = 0
    for k in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_k = k
    
    
    results['learning_rate'].append(lr)
    results['knn_accuracy'].append(best_score)

results = pd.DataFrame(results)

plt.figure(figsize=(4, 3))
plt.plot(results.learning_rate, results.knn_accuracy, marker='o')
plt.title('noMNIST Dataset')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.xticks(results.learning_rate)
plt.grid(True)
plt.savefig('som_lr_mnist.png') 
plt.show()
