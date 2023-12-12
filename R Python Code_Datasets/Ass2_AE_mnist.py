import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import random

tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)


data = pd.read_csv("noMNIST_CG.csv")

label_encoder = LabelEncoder()
data['response'] = label_encoder.fit_transform(data['response'])

X = data.iloc[:, data.columns != 'response']
y = data['response']

dimensions = data.shape 

num_rows = data.shape[0]
num_columns = data.shape[1]

print(f"The DataFrame has {num_rows} rows and {num_columns} columns.")


#standerdize
# scaler = StandardScaler().fit(X)
X_scaled = X # scaler.transform(X)

test_accuracy = []


hidden_layer = [128, 256, 384]

for hidden_1 in hidden_layer: 
    encoding_dim = 6
    
    #autoencoder
    input_img = tf.keras.Input(shape=(X_scaled.shape[1],))
    hidden = layers.Dense(hidden_1, activation='relu')(input_img)
    encoded = layers.Dense(encoding_dim, activation='relu')(hidden)
    hidden = layers.Dense(hidden_1, activation='relu')(encoded)
    decoded = layers.Dense(X_scaled.shape[1], activation='sigmoid')(hidden)
    autoencoder = models.Model(input_img, decoded)

    #compile and train ae
    tf.random.set_seed(123)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_scaled, X_scaled,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0)

    #encode data
    encoder = models.Model(input_img, encoded)
    X_encoded = encoder.predict(X_scaled)

    #split data for train and test
    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123)

    # Find the best K
    best_k = 1
    best_score = 0
    for k in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_encoded, y_train)
        y_pred = knn.predict(X_test_encoded)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_k = k

    #with best k get accuracy
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_encoded, y_train)
    y_pred = knn.predict(X_test_encoded)
    test_accuracy.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(4, 3))
plt.plot(hidden_layer, test_accuracy, marker='o')
plt.title('noMNIST Dataset')
plt.xlabel('Number of Neurons in Hidden Layer')
plt.ylabel('Test Accuracy')
plt.xticks(hidden_layer)
plt.grid(True)
plt.savefig('ae_neu_mnist.png') 
plt.show()


test_accuracy = []


l2_reg = [0.00001, 0.0001, 0.001]

for l2_1 in l2_reg: 
    encoding_dim = 6
    hidden_dim = 256

    input_img = tf.keras.Input(shape=(X_scaled.shape[1],))
    hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_1))(input_img)
    encoded = layers.Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(l2_1))(hidden)
    hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_1))(encoded)
    decoded = layers.Dense(X_scaled.shape[1], activation='sigmoid')(hidden)
    autoencoder = models.Model(input_img, decoded)

    tf.random.set_seed(123)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_scaled, X_scaled,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0)

    encoder = models.Model(input_img, encoded)
    X_encoded = encoder.predict(X_scaled)

    X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=123)

    best_k = 1
    best_score = 0
    for k in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_encoded, y_train)
        y_pred = knn.predict(X_test_encoded)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_k = k

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_encoded, y_train)
    y_pred = knn.predict(X_test_encoded)
    test_accuracy.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(4, 3))
plt.plot(l2_reg, test_accuracy, marker='o')
plt.title('noMNIST Dataset')
plt.xlabel('Lambda')
plt.ylabel('Test Accuracy')
plt.xticks(l2_reg)
plt.grid(True)
plt.savefig('ae_l2_mnist.png') 
plt.show()
 



 

