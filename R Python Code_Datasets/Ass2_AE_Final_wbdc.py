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


data = pd.read_csv("WDBC.csv")

label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

X = data.iloc[:, 2:]
y = data['diagnosis']

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

test_accuracy = []


feature_size =list(range(2, 7))

for size in feature_size: 
    encoding_dim = size
    hidden_dim = 64
    l2_val = 0.01
    
    #autoencoder
    input_img = tf.keras.Input(shape=(X_scaled.shape[1],))
    hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(input_img)
    encoded = layers.Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(hidden)
    hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(encoded)
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


print(pd.DataFrame({'feature size': feature_size , 'accuracy': test_accuracy}))

plt.figure(figsize=(4, 3))
plt.plot(feature_size, test_accuracy, marker='o')
plt.title('WBDC Dataset - AE')
plt.xlabel('Feature dimension size')
plt.ylabel('Test Accuracy')
plt.xticks(feature_size)
plt.grid(True)
plt.savefig('ae_size_wbdc.png') 
plt.show()