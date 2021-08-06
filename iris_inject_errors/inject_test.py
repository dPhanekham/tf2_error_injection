import tensorflow as tf
import os
import matplotlib.pyplot as plt
import error_inject_layer
import error_inject_optimizer
from tensorflow.python.util import deprecation
import tensorflow_datasets as tfds
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import dense_error_injection

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

# smash warnings at beginning
deprecation._PRINT_DEPRECATION_WARNINGS = False

# tf.enable_eager_execution()
# print(tf.executing_eagerly())

# grab iris dataset from cloud
# train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
# train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
#                                            origin=train_dataset_url)
# test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
# test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
#                                   origin=test_url)
# print("Local copy of the dataset file: {}".format(train_dataset_fp))

# already have dataset downloaded. comment this out and uncomment above if otherwise
train_dataset_fp = "/home/derek/.keras/datasets/iris_training.csv"

test_fp = "/home/derek/.keras/datasets/iris_test.csv"

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

#print dataset info
print("\n\n\n\n\n")
print("DATASET INFO:")
print(type(train_dataset))
print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

# This gets the next batch of data, randomly pulled from the training dataset
features, labels = next(iter(train_dataset))
print("FEATURES:")
print(features)
print(len(features))
print("LABELS:")
print(labels)

# creates map dataset
train_dataset = train_dataset.map(pack_features_vector)


features, labels = next(iter(train_dataset))
print("dataset type: ", type(train_dataset))
print("FEATURES, FIRST FIVE:")
print(features[:5])
print("labels")
print(labels[:5])
print(train_dataset)

print("CHANGE TO NUMPY")
train_numpy = tfds.as_numpy(train_dataset)
print(train_numpy)
feature_array = np.empty((0,4), dtype=np.float32)
label_array = np.empty((0,), dtype=np.int32)
for ex in train_numpy:
  feature_array = np.concatenate((feature_array,ex[0]), axis=0)
  label_array = np.append(label_array, ex[1])

# print(feature_array)
print(label_array)
print(feature_array)
print(feature_array[:,0])

# note, difficult to convert tensor to float and back again
print("features type: ", type(features))
print("features data type", type(features[0]))
print(features[0])
fl1 = features[0][0]
print(fl1)


print("\n\n\n\n\n")


error_node_weight_bit_tuples = [(0,0,3)]

# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(4,)),
  # tf.keras.layers.Dense(10, activation=tf.nn.relu),
  dense_error_injection.Dense_Error_Injection(10, activation=tf.nn.relu,
                                        error_rate=0.1,
                                        error_type='random_bit_flip_percentage',
                                        error_inject_phase='training',
                                        error_node_weight_bit_tuples=error_node_weight_bit_tuples,
                                        error_persistence=True,
                                        verbose=0),  # input shape required
  # error_inject_layer.DenseErrorLayer(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


predictions = model(features)
print("\n\n")
print("PREDICTIONS")
print(predictions[:5])

# convert to probability
tf.nn.softmax(predictions[:5])
print(tf.nn.softmax(predictions[:5]))

# predict class
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# define loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("\n\n\n\n\n")

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 50

# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(4,)),
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  # dense_error_injection.Dense_Error_Injection(10, activation=tf.nn.relu,
  #                                       error_rate=0.0,
  #                                       error_type='random_bit_flip_percentage',
  #                                       error_inject_phase='training',
  #                                       error_node_weight_bit_tuples=error_node_weight_bit_tuples,
  #                                       error_persistence=True,
  #                                       error_element='weight',
  #                                       verbose=1),  # input shape required
  # error_inject_layer.DenseErrorLayer(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

model.compile(optimizer=optimizer,
              loss=loss_object,
              metrics=['accuracy'])

num_epochs = 300

train_history_tmp = model.fit(train_dataset, epochs=num_epochs,
                              validation_data=(test_dataset),
                              batch_size=32, verbose=1)

# error_rate_train_history.append((error_rate, train_history_tmp))

print(train_history_tmp)


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_history_tmp.history['loss'])

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_history_tmp.history['accuracy'])
plt.show()


test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

print("DONE")