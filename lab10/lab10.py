# -*- coding: utf-8 -*-
"""
    Exercises with neural network - sequental models 
"""
# %%

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28,28 )
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000, )
assert y_test.shape == (10000, )

# %%

X_train = X_train/255.0
X_test = X_test/255.0

# %%
plt.imshow(X_train[142], cmap="binary")
plt.axis('off')
plt.show()

# %%
class_names=["koszulka","spodnie","pulower","sukienka","kurtka",
             "sandał","koszula","but","torba","kozak"]

# %%
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

#
print(model.summary())
tf.keras.utils.plot_model(model,"fashion_mnist.png", show_shapes=True)

# %%
model.compile(loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./image_logs")

history = model.fit(X_train, y_train, epochs=20,
validation_split=0.1, callbacks= [tensorboard_callback])

# %%
image_index=np.random.randint(len(X_test))
image=np.array([X_test[image_index]])
confidences=model.predict(image)
confidence=np.max(confidences[0])
prediction=np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()

# %%
model.save("fashion_clf.h5")

# %%
housing=fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=1)

X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)

# %%
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(1, activation="softmax"))

# %%
model.compile(loss="mse",
    optimizer="sgd",
    metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./housing_logs")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.1)

history = model.fit(X_train, y_train, epochs=50,
validation_data=(X_valid, y_valid), callbacks= [callback])

# %%
model.save("reg_housing_1.h5")

# %%
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(1, activation="softmax"))

# %%
model.compile(loss="mse",
    optimizer="sgd",
    metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./housing_logs/log1")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.1)

history = model.fit(X_train, y_train, epochs=50,
validation_data=(X_valid, y_valid), callbacks= [callback])

# %%
model.save("reg_housing_2.h5")

# %%
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dense(1, activation="softmax"))

# %%
model.compile(loss="mse",
    optimizer="sgd",
    metrics=["accuracy"])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./housing_logs/log2")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.1)

history = model.fit(X_train, y_train, epochs=50,
validation_data=(X_valid, y_valid), callbacks= [callback])

# %%
model.save("reg_housing_3.h5")

