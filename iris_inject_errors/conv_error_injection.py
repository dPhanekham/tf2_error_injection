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
from tensorflow.keras import layers
import error_inject_util

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
class Conv_Error_Injection(layers.Conv):
  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               error_rate=0.0,
               error_number=0,
               error_element=None,
               error_node_weight_bit_tuples=None,
               error_type=None,
               error_inject_phase='training',
               error_pattern=None,
               error_persistence=False,
               verbose=False,
               **kwargs):
    super(Conv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
      filters = int(filters)
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

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


  def call(self, inputs):
    input_shape = inputs.shape


    # print(type(inputs))
    if training and self.error_inject_phase in ['training', 'both']:
      if self.verbose:
        print("TRAINING")
      error_inject_util.inject_errors(self)
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      error_inject_util.inject_errors(self)

    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self._convolution_op(inputs, self.kernel)

    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

          outputs = conv_utils.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)

    if training and self.error_inject_phase in ['training', 'both']:
      if self.verbose:
        print("TRAINING")
      if not self.error_persistence:
        error_inject_util.inject_errors(self)
    elif not training and self.error_inject_phase in ['inference', 'both']:
      if self.verbose:
        print("INFERENCE")
      if not self.error_persistence:
        error_inject_util.inject_errors(self)


    if self.activation is not None:
      return self.activation(outputs)
    return outputs




class Conv2D_Error_Injection(Conv_Error_Injection)
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
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
               verbose=False,
               **kwargs):
    super(Conv2D_Error_Injection, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        error_rate=error_rate,
        error_number=error_number,
        error_element=error_element,
        error_node_weight_bit_tuples=error_node_weight_bit_tuples,
        error_type=error_type,
        error_inject_phase=error_inject_phase,
        error_pattern=error_pattern,
        error_persistence=error_persistence,
        verbose=verbose,
        **kwargs)
