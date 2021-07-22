import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# loading dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# scaling training and test data
X_train = X_train / 255
X_test = X_test / 255


# ANN model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)
print(model.evaluate(X_test, y_test))


# Model gave an accuracy of around 93% on the training data and around 93% on the test data

'''
Trying out multiple hidden layers, epochs, neurons in the hiddden layers, optimizer, loss function etc. can vary the accuracy
'''