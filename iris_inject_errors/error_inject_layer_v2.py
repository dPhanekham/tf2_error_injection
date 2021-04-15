import copy
import sys
import types as python_types
import warnings
import random

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
# if weight is chosen, it will inject in 30% of the weights
ERROR_ELEMENTS = ['layer', 'node', 'weight', 'bit']

ERROR_INJECT_PHASE = ['training', 'inference', 'both']


#based off of Dense Layer, but randomly injects errors into the weights
class DenseErrorLayerV2(layers.Layer):
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
               error_node_weight_bit_tuples=None,
               error_type=None,
               error_inject_phase='training',
               error_pattern=None,
               error_persistence=False,
               verbose = 0,
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
    self.error_node_weight_bit_tuples = error_node_weight_bit_tuples
    self.error_inject_phase = error_inject_phase
    self.error_pattern = error_pattern
    self.error_pattern_counter = 0
    self.error_persistence = error_persistence
    self.error_inject_locations = []
    if self.error_pattern:
      self.error_pattern_length = len(self.error_pattern)
    else:
      self.error_pattern_length = 0
    self.verbose = verbose

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
    if training and self.error_inject_phase in ['training', 'both']:
      if self.verbose:
        print("TRAINING")
      self.inject_errors()
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      self.inject_errors()
      
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

    if training and self.error_inject_phase in ['training', 'both']:
      if self.verbose:
        print("TRAINING")
      if not self.error_persistence:
        self.remove_errors()
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      if not self.error_persistence:
        self.remove_errors()

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
    """[summary]
    
    [description]
    """

    # convert
    # t = tf.Variable(lambda: self.kernel[0])

    if self.error_rate <= 0.0 or self.error_type is None:
      return

    # convert weights to numpy array
    numpy_kernel = K.get_value(self.kernel)

    shape = numpy_kernel.shape
    
    # print(numpy_kernel)
    numpy_kernel = numpy_kernel.flatten()
    if self.verbose:
      print("KERNEL SHAPE")
      print(shape)
      print(numpy_kernel)

    #TODO implement error elements
    if self.error_type == "random_bit_flip_percentage":
      error_amount = int(numpy_kernel.shape[0] * self.error_rate)
      if error_amount == 0:
        if random.random() < self.error_rate:
          error_amount = 1
      if self.verbose:
        print(f'inserting {error_amount} errors')
      numpy_kernel = self.inject_random_bit_flips(numpy_kernel, error_amount, self.error_element)
    elif self.error_type == "random_bit_flip_number":
      error_amount = self.error_number
      print(f'inserting {error_amount} errors')
      numpy_kernel = self.inject_random_bit_flips(numpy_kernel, error_amount, self.error_element)
    elif self.error_type == "stuck_at_1":
      numpy_kernel = self.inject_stuck_at(numpy_kernel, stuck_at=1)
    elif self.error_type == "stuck_at_0":
      numpy_kernel = self.inject_stuck_at(numpy_kernel, stuck_at=0) 
    elif self.error_type == "bit_flip_at_location":
      numpy_kernel = self.inject_bit_flip_at_location(numpy_kernel, shape=shape,
                                                      error_rate=self.error_rate,
                                                      error_pattern=self.error_pattern)

    numpy_kernel = numpy_kernel.reshape(shape)
    if self.verbose:
      print("POST INJECT")
      print(numpy_kernel)
    self.kernel.assign(numpy_kernel)

  def inject_random_bit_flips(self, array, error_amount, error_element):
    error_locations = []
    # TODO need to check datatype first, 32 or 64
    # b = bitstring.BitArray(float=f, length=32)

    # inject errors
    error_count = 0
    while error_count < error_amount:
      weight_index = randint(0, array.shape[0]-1)
      # print(weight_index)

      weight = array[weight_index,]
      if self.verbose:
        print("WEIGHT: ", weight)
      b = bitstring.BitArray(float=weight, length=32)
      location_in_weight = randint(2, 31)
      b.invert(location_in_weight)
      error_locations.append((weight_index,location_in_weight))
      adjusted_weight = b.float
      if self.verbose:
        print("ADJUSTED: ", adjusted_weight)
      array[weight_index, ] = adjusted_weight
      error_count += 1

    self.error_inject_locations = error_locations
    return array

  # TODO add verfication for weight/bit locations
  def inject_stuck_at(self, array, stuck_at=0):

    for node_index, weight_index, bit_index in self.error_node_weight_bit_tuples:
      if stuck_at == 0:
        stuck_bit = '0b0'
      else:
        stuck_bit = '0b1'

      weight = array[node_index*shape[-1] + weight_index]
      b = bitstring.BitArray(float=weight, length=32)
      b.overwrite(stuck_bit, bit_index)
      adjusted_weight = b.float
      array[node_index*shape[-1] + weight_index] = adjusted_weight

  def inject_bit_flip_at_location(self, array, shape, error_rate=None, error_pattern=None):
    """Injects bit flips at specified locations in the weight matrix

    Injects bit flips at locations specified in self.error_node_weight_bit_tuples.
    
    Arguments:
      array {numpy vector} -- flattened weight matrix to change
      shape {tuple} -- The original shape of the weight matrix/kernel
    
    Keyword Arguments:
      error_rate {float} -- [description] (default: {None})
      error_pattern {list} -- [description] (default: {None})
    
    Returns:
      numpy vector -- flattened weight matrix with injected bit flips
    """

    if error_rate and error_rate > 0.0 and random.uniform(0,1) >= error_rate:
      return array
    elif error_pattern:
      if self.error_pattern_counter >= self.error_pattern_length:
        self.error_pattern_counter = 0
      if error_pattern[self.error_pattern_counter] == 0:
        self.error_pattern_counter += 1
        return array
      else:
        self.error_pattern_counter += 1

    for node_index, weight_index, bit_index in self.error_node_weight_bit_tuples:
      weight = array[node_index*shape[-1] + weight_index]
      b = bitstring.BitArray(float=weight, length=32)
      b.invert(bit_index)
      adjusted_weight = b.float
      array[node_index*shape[-1] + weight_index] = adjusted_weight
    return array

  def inject_positional_bit_flip(self, array):
    for weight_index,bit_index in self.error_node_weight_bit_tuples:
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

  def remove_errors(self):
    """Remove errors that were previously inserted
    
    Remove previously inserted errors
    """
    if self.error_rate <= 0.0 or self.error_type is None:
      return

    # convert weights to numpy array
    numpy_kernel = K.get_value(self.kernel)

    shape = numpy_kernel.shape
    
    # print(numpy_kernel)
    numpy_kernel = numpy_kernel.flatten()
    if self.verbose:
      print("KERNEL SHAPE")
      print(shape)
      print(numpy_kernel)

    # ERROR_TYPES = ['random_bit_flip_percentage', 'random_bit_flip_number', 'stuck_at_0', 'stuck_at_1',
    #                'bit_flip_at_location', 'missing_node', 'missing_connection', 'zero_weight']

    if self.error_type in ["random_bit_flip_percentage", "random_bit_flip_number"]:
      for weight_index, location_in_weight in self.error_inject_locations:
        weight = numpy_kernel[weight_index,]
        if self.verbose:
          print("WEIGHT: ", weight)
        b = bitstring.BitArray(float=weight, length=32)
        b.invert(location_in_weight)
        adjusted_weight = b.float
        if self.verbose:
          print("UNADJUSTED: ", adjusted_weight)
        numpy_kernel[weight_index, ] = adjusted_weight

    # elif self.error_type == "random_bit_flip_number":
    #   error_amount = self.error_number
    #   numpy_kernel = self.inject_random_bit_flips(numpy_kernel, error_amount, self.error_element)
    # elif self.error_type == "stuck_at_1":
    #   numpy_kernel = self.inject_stuck_at(numpy_kernel, stuck_at=1)
    # elif self.error_type == "stuck_at_0":
    #   numpy_kernel = self.inject_stuck_at(numpy_kernel, stuck_at=0) 
    # elif self.error_type == "bit_flip_at_location":
    #   numpy_kernel = self.inject_bit_flip_at_location(numpy_kernel, shape=shape,
    #                                                   error_rate=self.error_rate,
    #                                                   error_pattern=self.error_pattern)
    self.error_inject_locations = []
    numpy_kernel = numpy_kernel.reshape(shape)
    if self.verbose:
      print("POST REMOVE ERRORS")
      print(numpy_kernel)
    self.kernel.assign(numpy_kernel)

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
