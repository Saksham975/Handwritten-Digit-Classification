from tensorflow import keras

# loading dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# scaling training and test data
X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ANN model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
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

# Model gave an accuracy of around 98% on the training data and around 98% on the test data

'''
CNN model gave better accuracy than ANN model on training and testing data
'''
