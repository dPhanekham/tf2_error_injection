import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt # Dataset visualization.
import numpy as np              # Low-level numerical Python library.
import time

from tensorflow.keras import layers
import error_inject_layer


print(tf.executing_eagerly())

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  error_inject_layer.DenseErrorLayer(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

print("FIT DONE")

model.evaluate(x_test,  y_test, verbose=2)


print("HELLLOOO")