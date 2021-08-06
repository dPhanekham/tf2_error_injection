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

import bitstring
from random import randint
import error_inject_util

def inject_errors(obj):
  """[summary]

  [description]
  """

  # convert
  # t = tf.Variable(lambda: obj.kernel[0])

  if obj.error_rate <= 0.0 or obj.error_type is None:
    return

  # convert weights to numpy array
  numpy_kernel = K.get_value(obj.kernel)
  shape = numpy_kernel.shape
#   print(f'shape {shape}')
#   print(numpy_kernel)
  numpy_kernel = numpy_kernel.flatten()
  if obj.verbose >= 2:
    print("KERNEL SHAPE")
    print(shape)
    print(numpy_kernel)

  if obj.error_type == "random_bit_flip_percentage":
    error_amount = int(numpy_kernel.shape[0] * obj.error_rate)
    # TODO check datatype first
    if obj.error_element == 'bit':
      error_amount = error_amount * 32
    if obj.error_element == 'byte':
      error_amount = error_amount * 4
    if error_amount == 0:
      if random.random() < obj.error_rate:
        error_amount = 1
    if obj.verbose >= 1:
      print(f'inserting {error_amount} errors')
    numpy_kernel = inject_random_bit_flips(obj, numpy_kernel, obj.error_rate, error_amount, obj.error_element)

  elif obj.error_type == "random_bit_flip_number":
    error_amount = obj.error_number
    # if obj.verbose >= 1:
    #   print(f'inserting {error_amount} errors')
    numpy_kernel = inject_random_bit_flips(obj, numpy_kernel, error_amount, obj.error_element)
  elif obj.error_type == "stuck_at_1":
    numpy_kernel = inject_stuck_at(obj, numpy_kernel, stuck_at=1)
  elif obj.error_type == "stuck_at_0":
    numpy_kernel = inject_stuck_at(obj, numpy_kernel, stuck_at=0)
  elif obj.error_type == "bit_flip_at_location":
    numpy_kernel = inject_bit_flip_at_location(obj, numpy_kernel, shape=shape,
                                                    error_rate=obj.error_rate,
                                                    error_pattern=obj.error_pattern)

  numpy_kernel = numpy_kernel.reshape(shape)
  if obj.verbose >= 2:
    print("POST INJECT")
    print(numpy_kernel)
  obj.kernel.assign(numpy_kernel)

def inject_random_bit_flips(obj, array, error_rate, error_amount, error_element):
  """Inject random bit flips into the given array

  Args:
      obj (layers.Layer): tensorflow Layer that we are injecting errors into
      array (numpy.ndarray): flattened kernel from Layer that we are injecting errors into
      error_amount (int): The number of errors to inject
      error_element (str): The level of element [bit,byte,weight,node,kernel]

  Returns:
      [type]: [description]
  """
  error_locations = []
  # TODO need to check datatype first, 32 or 64
  # b = bitstring.BitArray(float=f, length=32)

  # inject errors
  error_count = 0
  if error_rate > 0.0 and error_amount == 0:
    if random.random() < error_rate:
      error_count = 1
  while error_count < error_amount:
    weight_index = randint(0, array.shape[0]-1)
    # print(weight_index)
    weight = array[weight_index,]
    if obj.verbose >= 2:
      print("WEIGHT: ", weight)
    b = bitstring.BitArray(float=weight, length=32)
    location_in_weight = randint(2, 31)
    b.invert(location_in_weight)
    error_locations.append((weight_index,location_in_weight))
    adjusted_weight = b.float
    if obj.verbose >= 2:
      print("ADJUSTED: ", adjusted_weight)
    array[weight_index, ] = adjusted_weight
    error_count += 1
  if obj.verbose >= 1:
    print(f'Injected {len(error_locations)} errors')
  obj.error_inject_locations = error_locations
  return array


def inject_random_bit_flips_into_array(array, error_rate, error_amount, error_element, verbose=1):
  """Inject random bit flips into the given array

  Args:
      obj (layers.Layer): tensorflow Layer that we are injecting errors into
      array (numpy.ndarray): flattened kernel from Layer that we are injecting errors into
      error_amount (int): The number of errors to inject
      error_element (str): The level of element [bit,byte,weight,node,kernel]

  Returns:
      [type]: [description]
  """
  error_locations = []
  # TODO need to check datatype first, 32 or 64
  # b = bitstring.BitArray(float=f, length=32)
  print("HERE2")
  # inject errors
  error_count = 0
  if error_rate > 0.0 and error_amount == 0:
    if random.random() < error_rate:
      error_count = 1
  while error_count < error_amount:
    weight_index = randint(0, array.shape[0]-1)
    # print(weight_index)
    weight = array[weight_index,]
    if verbose >= 2:
      print("WEIGHT: ", weight)
    b = bitstring.BitArray(float=weight, length=32)
    location_in_weight = randint(2, 31)
    b.invert(location_in_weight)
    error_locations.append((weight_index,location_in_weight))
    adjusted_weight = b.float
    if verbose >= 2:
      print("ADJUSTED: ", adjusted_weight)
    array[weight_index, ] = adjusted_weight
    error_count += 1
  if verbose >= 1:
    print(f'Injected {len(error_locations)} errors')
  return array, error_locations


# TODO add verfication for weight/bit locations
def inject_stuck_at(obj, array, stuck_at=0):

  for node_index, weight_index, bit_index in obj.error_node_weight_bit_tuples:
    if stuck_at == 0:
      stuck_bit = '0b0'
    else:
      stuck_bit = '0b1'

    weight = array[node_index*shape[-1] + weight_index]
    b = bitstring.BitArray(float=weight, length=32)
    b.overwrite(stuck_bit, bit_index)
    adjusted_weight = b.float
    array[node_index*shape[-1] + weight_index] = adjusted_weight

def inject_bit_flip_at_location(obj, array, shape, error_rate=None, error_pattern=None):
  """Injects bit flips at specified locations in the weight matrix

  Injects bit flips at locations specified in obj.error_node_weight_bit_tuples.

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
    if obj.error_pattern_counter >= obj.error_pattern_length:
      obj.error_pattern_counter = 0
    if error_pattern[obj.error_pattern_counter] == 0:
      obj.error_pattern_counter += 1
      return array
    else:
      obj.error_pattern_counter += 1

  for node_index, weight_index, bit_index in obj.error_node_weight_bit_tuples:
    weight = array[node_index*shape[-1] + weight_index]
    b = bitstring.BitArray(float=weight, length=32)
    b.invert(bit_index)
    adjusted_weight = b.float
    array[node_index*shape[-1] + weight_index] = adjusted_weight
  return array

def inject_positional_bit_flip(obj, array):
  for weight_index,bit_index in obj.error_node_weight_bit_tuples:
    weight = array[weight_index,]
    b = bitstring.BitArray(float=weight, length=32)
    b.invert(bit_index)
    adjusted_weight = b.float
    array[weight_index, ] = adjusted_weight

def inject_random_element(obj):
  pass

def inject_zero(obj):
  pass

def inject_zero_weight(obj):
  pass

def remove_errors(obj):
  """Remove errors that were previously inserted

  Remove previously inserted errors
  """
  if obj.error_rate <= 0.0 or obj.error_type is None:
    return

  # convert weights to numpy array
  numpy_kernel = K.get_value(obj.kernel)

  shape = numpy_kernel.shape

  # print(numpy_kernel)
  numpy_kernel = numpy_kernel.flatten()
  if obj.verbose >= 2:
    print("KERNEL SHAPE")
    print(shape)
    print(numpy_kernel)

  # ERROR_TYPES = ['random_bit_flip_percentage', 'random_bit_flip_number', 'stuck_at_0', 'stuck_at_1',
  #                'bit_flip_at_location', 'missing_node', 'missing_connection', 'zero_weight']

  if obj.error_type in ["random_bit_flip_percentage", "random_bit_flip_number"]:
    if obj.verbose >= 1:
      print(f'removing {len(obj.error_inject_locations)} errors')
    print(obj.error_inject_locations)
    for weight_index, location_in_weight in obj.error_inject_locations:
      weight = numpy_kernel[weight_index,]
      if obj.verbose >= 2:
        print("WEIGHT: ", weight)
      b = bitstring.BitArray(float=weight, length=32)
      b.invert(location_in_weight)
      adjusted_weight = b.float
      if obj.verbose >= 2:
        print("UNADJUSTED: ", adjusted_weight)
      numpy_kernel[weight_index, ] = adjusted_weight

  # elif obj.error_type == "random_bit_flip_number":
  #   error_amount = obj.error_number
  #   numpy_kernel = obj.inject_random_bit_flips(numpy_kernel, error_amount, obj.error_element)
  # elif obj.error_type == "stuck_at_1":
  #   numpy_kernel = obj.inject_stuck_at(numpy_kernel, stuck_at=1)
  # elif obj.error_type == "stuck_at_0":
  #   numpy_kernel = obj.inject_stuck_at(numpy_kernel, stuck_at=0)
  # elif obj.error_type == "bit_flip_at_location":
  #   numpy_kernel = obj.inject_bit_flip_at_location(numpy_kernel, shape=shape,
  #                                                   error_rate=obj.error_rate,
  #                                                   error_pattern=obj.error_pattern)
  obj.error_inject_locations = []
  numpy_kernel = numpy_kernel.reshape(shape)
  if obj.verbose >= 2:
    print("POST REMOVE ERRORS")
    print(numpy_kernel)
  obj.kernel.assign(numpy_kernel)