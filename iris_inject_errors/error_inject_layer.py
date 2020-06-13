import copy
import sys
import types as python_types
import warnings

import numpy as np
import tensorflow as tf

# from tensorflow.eager import context
from tensorflow import dtypes
# from tensorflow.framework import ops
from tensorflow import TensorShape
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import InputSpec
# from tensorflow.keras.utils import conv_utils
# from tensorflow.keras.utils import generic_utils
# from tensorflow.keras.utils import tf_utils
# from tensorflow.ops import array_ops
# from tensorflow.ops import gen_math_ops
# from tensorflow.ops import math_ops
# from tensorflow.ops import nn
# from tensorflow.ops import sparse_ops
# from tensorflow.ops import standard_ops
# from tensorflow.ops import variable_scope
# from tensorflow.util import nest
# from tensorflow.util import tf_inspect
# from tensorflow.util.tf_export import keras_export


from tensorflow.keras import layers

import bitstring
from random import randint

ERROR_TYPES = ['random_bit_flip_percentage', 'random_bit_flip_number', 'stuck_at_0', 'stuck_at_1',
               'bit_flip_at_location', 'missing_node', 'missing_connection', 'zero_weight']

# inject errors per each of these elements
# ex. if node is chosen, random_bit_flip with a rate of 0.3, it will inject an error in 30% of nodes
# if layer is chosen, it will inject an error in 30% of layers (or this layer with a 30% chance)
ERROR_ELEMENTS = ['layer', 'node', 'weight']

ERROR_INJECT_PHASE = ['training', 'inference', 'both']


#based off of Dense Layer, but randomly injects errors into the weights
class DenseErrorLayer(layers.Layer):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               error_rate=0.0,
               error_number=0,
               error_element=None,
               error_weight_bit_tuples=None,
               error_type=None,
               error_inject_phase='training',
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(DenseErrorLayer, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)
    self.error_rate = error_rate
    self.error_type = error_type
    self.error_number = error_number
    self.error_element = error_element
    self.error_weight_bit_tuples = error_weight_bit_tuples
    self.error_inject_phase = error_inject_phase

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = TensorShape(input_shape)
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = input_shape[-1]
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units, ],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, training=False):

    # if(tf.executing_eagerly() == True):
    #   print("EXECUTE EAGER")
    #   print(tf.__version__)
    # print("INPUTS")
    # print(inputs.shape)
    if training and self.error_inject_phase in ['training', 'both']:
      print("TRAINING")
      # print("CALLLLL")
      # print(inputs)

      # print("KERNEL")
      # print(self.kernel)
      
      # print(type(self.kernel))
      # print("KERNAL [0]")
      # print(type(self.kernel[0]))
      # print(self.kernel[0])
      # np.array(self.kernel[0])

      self.inject_errors()

      # print("KERNEL AFTER INJECT")
      # print(self.kernel)
    elif not training and self.error_inject_phase in ['inference', 'both']:
      print("INFERENCE")
      self.inject_errors()
      # print(self.kernel)

    rank = len(inputs.shape)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      inputs = tf.cast(inputs, self._compute_dtype)
      if K.is_sparse(inputs):
        outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.kernel)
      else:
        outputs = tf.matmul(inputs, self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



  # TODO: should error rate be a percentage of weights, or percentage of bits
  # NOTE: this probably won't work without eager execution
  def inject_errors(self):
    # convert
    # t = tf.Variable(lambda: self.kernel[0])

    if self.error_rate <= 0.0 or self.error_type is None:
      return

    # convert weights to numpy array
    numpy_kernel = K.get_value(self.kernel)

    # print(type(numpy_kernel))
    # print(numpy_kernel.shape)
    # print(numpy_kernel)
    # print(numpy_kernel[0]) 
    # print(type(numpy_kernel[0]))

    # ta[0,0] = 1.0
    shape = numpy_kernel.shape
    # print(shape)
    numpy_kernel = numpy_kernel.flatten()
    # print(numpy_kernel.shape)
    # print(type(numpy_kernel.shape))
    # print(numpy_kernel.shape[0])
    # print(numpy_kernel.shape[0] * self.error_rate)

    # ERROR_TYPES = ['random_bit_flip_percentage', 'random_bit_flip_number', 'stuck_at_0', 'stuck_at_1',
    #                'bit_flip_at_location', 'missing_node', 'missing_connection', 'zero_weight']

    if self.error_type == "random_bit_flip_percentage":
      error_amount = int(numpy_kernel.shape[0] * self.error_rate)
      numpy_kernel = self.inject_random_bit_flips(numpy_kernel, error_amount, self.error_element)
    elif self.error_type == "random_bit_flip_number":
      error_amount = self.error_number
      numpy_kernel = self.inject_random_bit_flips(numpy_kernel, error_amount, self.error_element)
    elif self.error_type == "stuck_at_1":
      numpy_kernel = self.inject_stuck_at(numpy_kernel, starting_bit=32, number_of_bits=1,stuck_at=1)
    elif self.error_type == "stuck_at_0":
      numpy_kernel = self.inject_stuck_at(numpy_kernel, starting_bit=32, number_of_bits=1, stuck_at=0) 
    elif self.error_type == "bit_flip_at_location":
      numpy_kernel = self.inject_bit_flip_at_location(numpy_kernel, starting_bit, number_of_bits=1)

    # print(numpy_kernel)
    numpy_kernel = numpy_kernel.reshape(shape)
    self.kernel.assign(numpy_kernel)

  def inject_random_bit_flips(self, array, error_amount, error_element):
    error_locations = []
    # np.set_printoptions(precision=200)
    # f = array[0,]
    # print(type(f))
    # print(f)
    # TODO need to check datatype first, 32 or 64
    # b = bitstring.BitArray(float=f, length=32)
    # print(b)
    # b.invert(0)
    # myFloat = b.float
    # print(type(myFloat))
    # print(myFloat)
    
    # inject errors
    error_count = 0
    while error_count < error_amount:
      weight_index = randint(0, array.shape[0]-1)
      # print(weight_index)
      error_locations.append(weight_index)

      weight = array[weight_index,]
      print("WEIGHT: ", weight)
      b = bitstring.BitArray(float=weight, length=32)
      location_in_weight = randint(2, 31)
      b.invert(location_in_weight)
      adjusted_weight = b.float
      print("ADJUSTED: ", adjusted_weight)
      array[weight_index, ] = adjusted_weight
      error_count += 1

    return array

  # TODO add verfication for weight/bit locations
  def inject_stuck_at(self, array, starting_bit, number_of_bits=1, stuck_at=0):

    for weight_index,bit_index in self.error_weight_bit_tuples:
      stuck_bit = '0b0'
      if stuck_at == 1:
        stuck_bit = '0b1'

      weight = array[weight_index,]
      b = bitstring.BitArray(float=weight, length=32)
      b.overwrite(stuck_bit, bit_index)
      adjusted_weight = b.float
      array[weight_index, ] = adjusted_weight

  def inject_bit_flip_at_location(self, array, starting_bit, number_of_bits=1):
    pass

  def inject_positional_bit_flip(self, array):
    for weight_index,bit_index in self.error_weight_bit_tuples:
      weight = array[weight_index,]
      b = bitstring.BitArray(float=weight, length=32)
      b.invert(bit_index)
      adjusted_weight = b.float
      array[weight_index, ] = adjusted_weight

  def inject_random_element(self):
    pass

  def inject_zero(self):
    pass

  def inject_zero_weight(self):
    pass


# error layer
class ErrorLayer(layers.Layer):
  """Applies Dropout to the input.
  Dropout consists in randomly setting
  a fraction `rate` of input units to 0 at each update during training time,
  which helps prevent overfitting.
  Arguments:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = tf_utils.smart_cond(training,
                                 dropped_inputs,
                                 lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
