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
import error_inject_util

ERROR_TYPES = ['random_bit_flip_percentage', 'random_bit_flip_number', 'stuck_at_0', 'stuck_at_1',
               'bit_flip_at_location', 'missing_node', 'missing_connection', 'zero_weight']

# inject errors per each of these elements
# ex. if node is chosen, random_bit_flip with a rate of 0.3, it will inject an error in 30% of nodes
# if layer is chosen, it will inject an error in 30% of layers (or this layer with a 30% chance)
# if weight is chosen, it will inject in 30% of the weights
ERROR_ELEMENTS = ['layer', 'node', 'weight', 'bit']

ERROR_INJECT_PHASE = ['training', 'inference', 'both']


#based off of Dense Layer, but randomly injects errors into the weights
class Dense_Error_Injection(layers.Dense):
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
               error_element='bit',
               error_node_weight_bit_tuples=None,
               error_type=None,
               error_inject_phase='training',
               error_pattern=None,
               error_persistence=False,
               verbose = 0,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(Dense_Error_Injection, self).__init__(
               units=units,
               activation=activation,
               use_bias=use_bias,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer,
               kernel_regularizer=kernel_regularizer,
               bias_regularizer=bias_regularizer,
               activity_regularizer=activity_regularizer,
               kernel_constraint=kernel_constraint,
               bias_constraint=bias_constraint,
               **kwargs)
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


  def call(self, inputs, training=False):
    if training and self.error_inject_phase in ['training', 'both']:
      if self.verbose:
        print("TRAINING")
      error_inject_util.inject_errors(self)
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      error_inject_util.inject_errors(self)

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
        error_inject_util.remove_errors(self)
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      if not self.error_persistence:
        error_inject_util.remove_errors(self)

    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable

    return outputs
