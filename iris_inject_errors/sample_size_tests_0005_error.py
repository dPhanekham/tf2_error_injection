# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
#import matplotlib.pyplot as plt
import os
import error_inject_layer
import dense_error_injection
import error_inject_optimizer
from tensorflow.keras.utils import to_categorical
from tensorflow.python.util import deprecation
print(tf.__version__)

# LOAD DATA
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Scale values
train_images = train_images / 255.0
test_images = test_images / 255.0

# train_labels = to_categorical(train_labels, 10)
# test_labels = to_categorical(test_labels, 10)


def loss(model, loss_object, x, y):
  # print("call")
  y_hat = model(x, training=True)
  # print("Y:")
  # print(y)
  # print("Y_HAT:")
  # print(y_hat)

  return loss_object(y_true=y, y_pred=y_hat)

def grad(model, loss_object, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, loss_object, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_step(model, optimizer, loss_object, x, y):
  # Optimize the model
  loss_value, grads = grad(model, loss_object, x, y)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  
  return loss_value
    
    
def train(model, optimizer, loss_object, inputs, epochs=50, validation_set=None):
  train_loss_results = []
  train_accuracy_results = []
  validation_loss_results = []
  validation_accuracy_results = []
  
  if validation_set:
    val_data, val_labels = validation_set
  
  for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # Training loop - using batches of 32
    for x, y in inputs:
      loss = train_step(model, optimizer, loss_object, x, y)
      epoch_loss_avg(loss)
      epoch_accuracy(y, model(x))
    
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    
    if validation_set:
      test_loss, test_acc = model.evaluate(val_data,  val_labels, verbose=0)
      validation_accuracy_results.append(test_acc)
      validation_loss_results.append(test_loss)
    
    if epoch % 100 == 0:

      print("RESULTS")
      print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
      
  train_history = (train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results)
  return train_history



### TRAIN WITH NO ERROR

sample_size_0005_error_train_history = {}
try:
  sample_size_0005_error_train_history = np.load('batch_results/sample_size_train_history_error_rate_0005.npy', allow_pickle=True, fix_imports=True)
except:
  sample_size_0005_error_train_history = {}

train_size = 60000
error_rate = 0.0005
num_epochs = 500
iterations = 20

sample_sizes = [
                1000, 
                # 2000, 
                # 3000,
                # 4000,
                5000,
                # 6000,
                # 7000,
                # 8000,
                # 9000, 
                10000,
                # 20000,
                30000, 
                # 40000, 
                # 50000,
                60000
               ]

for sample_size in sample_sizes:

  if sample_size not in sample_size_0005_error_train_history:
    sample_size_0005_error_train_history[sample_size] = {}
    sample_size_0005_error_train_history[sample_size]['training_accuracy'] = []
    sample_size_0005_error_train_history[sample_size]['training_loss'] = []
    sample_size_0005_error_train_history[sample_size]['validation_accuracy'] = []
    sample_size_0005_error_train_history[sample_size]['validation_loss'] = []

  for i in range(0,iterations):
    train_dataset = None
    if sample_size != train_size:
      random_indices = np.random.choice(train_size, size=sample_size, replace=False)
      train_dataset = tf.data.Dataset.from_tensor_slices((train_images[random_indices], train_labels[random_indices])).batch(32)
    else:
      train_dataset = tf.data.Dataset.from_tensor_slices((train_images[0:sample_size,:,:], train_labels[0:sample_size])).batch(32)

    print(f'sample size: {sample_size}')
    test_model = tf.keras.Sequential([
                 tf.keras.layers.Flatten(input_shape=(28, 28)),
                 tf.keras.layers.Dense(512, activation='relu'),
                 dense_error_injection.Dense_Error_Injection(512, activation=tf.nn.relu,
                                                             error_rate=error_rate, 
                                                             error_type='random_bit_flip_percentage',
                                                             error_inject_phase='training',
                                                             error_element='weight',
                                                             verbose=0,
                                                             error_persistence=True
                                                             ),
                 tf.keras.layers.Dense(10)
    ])
    
    test_model.compile(optimizer='adam',
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    
    
    sample_size_history_tmp = train(test_model, optimizer, loss_object, train_dataset, epochs=num_epochs, validation_set=(test_images, test_labels))

    
#     train_history_tmp = test_model.fit(x=train_images, y=train_labels, epochs=num_epochs,
#                                        validation_data=(test_images,  test_labels),
#                                        batch_size=32, verbose=0)
    # (train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results)
    sample_size_0005_error_train_history[sample_size]['training_loss'].append(sample_size_history_tmp[0])
    sample_size_0005_error_train_history[sample_size]['training_accuracy'].append(sample_size_history_tmp[1])
    sample_size_0005_error_train_history[sample_size]['validation_loss'].append(sample_size_history_tmp[2])
    sample_size_0005_error_train_history[sample_size]['validation_accuracy'].append(sample_size_history_tmp[3])

#  plt.figure(figsize=(10, 8))
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][0])
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][1])
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][2])
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][3])
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][4])
#  plt.plot(sample_size_0005_error_train_history[sample_size]['validation_accuracy'][5])
#  plt.title('Validation accuracy per training run')
#  plt.ylabel('Accuracy')
#  plt.xlabel('Epoch')
  # plt.show()
#  plt.savefig(f'batch_results/figures/sample_size_{sample_size}_model_training_accuracy_1_small_error_rate_0005.png', bbox_inches='tight')


np.save('batch_results/sample_size_train_history_error_rate_0005', sample_size_0005_error_train_history, allow_pickle=True, fix_imports=True)

# sample_size = 60000
