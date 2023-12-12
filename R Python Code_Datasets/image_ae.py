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

encoding_dim = 5
hidden_dim = 256
l2_val = 0.00001
input_img = tf.keras.Input(shape=(X_scaled.shape[1],))
hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(input_img)
encoded = layers.Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(hidden)
hidden = layers.Dense(hidden_dim, activation='relu', activity_regularizer=regularizers.l2(l2_val))(encoded)
decoded = layers.Dense(X_scaled.shape[1], activation='sigmoid')(hidden)
autoencoder = models.Model(input_img, decoded)


autoencoder = models.Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

X_scaled = X_scaled.reset_index(drop=True)

image_index = 1905
sample_image = X_scaled.iloc[image_index].values.reshape(1, -1) 


reconstructed_img = autoencoder.predict(sample_image)

reconstructed_img = reconstructed_img.reshape(28, 28)

plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image - AE')
plt.axis('off')  
plt.savefig('recon_ae_1905.png') 
plt.show()