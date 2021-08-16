# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import error_inject_layer
import dense_error_injection
import error_inject_optimizer
import error_inject_model2
from tensorflow.python.util import deprecation
print(tf.__version__)

#Load Data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Preprocess data
# Scale values
train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.shape)
print(train_labels.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)


model = tf.keras.Sequential([
              tf.keras.layers.Flatten(input_shape=(28, 28)),
              # tf.keras.layers.Dense(128),
              dense_error_injection.Dense_Error_Injection(128, activation=tf.nn.relu,
                                                          error_rate=0.00000,
                                                          error_type='random_bit_flip_percentage',
                                                          error_inject_phase='training',
                                                          error_element='weight',
                                                          verbose=0,
                                                          error_persistence=True
                                                          ),
              tf.keras.layers.Dense(10)
])

###########################################################################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  # print("call")
  y_hat = model(x, training=True)
  # print("Y:")
  # print(y)
  # print("Y_HAT:")
  # print(y_hat)

  return loss_object(y_true=y, y_pred=y_hat)

l = loss(model, train_images, train_labels)
# print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.keras.optimizers.Adam()

loss_value, grads = grad(model, train_images, train_labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, train_images, train_labels).numpy()))


print("\n\n\n\n\n")

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 50
#
print("TRAINING STARTS HERE")
for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    # print("LOSS VALUE")
    # print(loss_value)
    # print("GRADS")
    # print(grads)
    # print("APPY GRADIENTS")
    # print(model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    epoch_accuracy(y, model(x))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 1 == 0:
    print("RESULTS")
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
###################################################################################################################################

# model.compile(optimizer='adam',
#                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                     metrics=['accuracy'])

# # print(model.layers)
# # print(model.layers[1])

# model.train_function = model.make_train_function()

# model.train_step(train_images)

# # train_history = model.fit(train_images, train_labels, epochs=5, verbose=1, validation_data=(test_images,  test_labels))

# print(model.layers[1].inject_times_count)
# plt.plot(train_history.history['accuracy'])
# plt.plot(train_history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['no_error', '0.1 error rate'], loc='upper left')
plt.show()