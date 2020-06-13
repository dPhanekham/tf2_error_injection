import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

class SequentialErrorInject(Model):
  """Linear stack of layers.
  Arguments:
      layers: list of layers to add to the model.
  Example:
  ```python
  # Optionally, the first layer can receive an `input_shape` argument:
  model = Sequential()
  model.add(Dense(32, input_shape=(500,)))
  # Afterwards, we do automatic shape inference:
  model.add(Dense(32))
  # This is identical to the following:
  model = Sequential()
  model.add(Dense(32, input_dim=500))
  # And to the following:
  model = Sequential()
  model.add(Dense(32, batch_input_shape=(None, 500)))
  # Note that you can also omit the `input_shape` argument:
  # In that case the model gets built the first time you call `fit` (or other
  # training and evaluation methods).
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.compile(optimizer=optimizer, loss=loss)
  # This builds the model for the first time:
  model.fit(x, y, batch_size=32, epochs=10)
  # Note that when using this delayed-build pattern (no input shape specified),
  # the model doesn't have any weights until the first call
  # to a training/evaluation method (since it isn't yet built):
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.weights  # returns []
  # Whereas if you specify the input shape, the model gets built continuously
  # as you are adding layers:
  model = Sequential()
  model.add(Dense(32, input_shape=(500,)))
  model.add(Dense(32))
  model.weights  # returns list of length 4
  # When using the delayed-build pattern (no input shape specified), you can
  # choose to manually build your model by calling `build(batch_input_shape)`:
  model = Sequential()
  model.add(Dense(32))
  model.add(Dense(32))
  model.build((None, 500))
  model.weights  # returns list of length 4
  ```
  """

  def __init__(self, layers=None, name=None):
    super(Sequential, self).__init__(name=name)
    self.supports_masking = True
    self._build_input_shape = None
    self._compute_output_and_mask_jointly = True

    self._layer_call_argspecs = {}

    # Add to the model any layers passed to the constructor.
    if layers:
      if not isinstance(layers, (list, tuple)):
        layers = [layers]
      tf_utils.assert_no_legacy_layers(layers)
      for layer in layers:
        self.add(layer)

  @property
  def layers(self):
    # Historically, `sequential.layers` only returns layers that were added
    # via `add`, and omits the auto-generated `InputLayer` that comes at the
    # bottom of the stack.
    # `Trackable` manages the `_layers` attributes and does filtering
    # over it.
    layers = super(Sequential, self).layers
    if layers and isinstance(layers[0], input_layer.InputLayer):
      return layers[1:]
    return layers[:]

  @property
  def dynamic(self):
    return any(layer.dynamic for layer in self.layers)

  def add(self, layer):
    """Adds a layer instance on top of the layer stack.
    Arguments:
        layer: layer instance.
    Raises:
        TypeError: If `layer` is not a layer instance.
        ValueError: In case the `layer` argument does not
            know its input shape.
        ValueError: In case the `layer` argument has
            multiple output tensors, or is already connected
            somewhere else (forbidden in `Sequential` models).
    """
    # If we are passed a Keras tensor created by keras.Input(), we can extract
    # the input layer from its keras history and use that without any loss of
    # generality.
    if hasattr(layer, '_keras_history'):
      origin_layer = layer._keras_history[0]
      if isinstance(origin_layer, input_layer.InputLayer):
        layer = origin_layer

    if not isinstance(layer, base_layer.Layer):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))

    tf_utils.assert_no_legacy_layers([layer])

    self.built = False
    set_inputs = False
    if not self._layers:
      if isinstance(layer, input_layer.InputLayer):
        # Corner case where the user passes an InputLayer layer via `add`.
        assert len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) == 1
        set_inputs = True
      else:
        batch_shape, dtype = training_utils.get_input_shape_and_dtype(layer)
        if batch_shape:
          # Instantiate an input layer.
          x = input_layer.Input(
              batch_shape=batch_shape, dtype=dtype, name=layer.name + '_input')
          # This will build the current layer
          # and create the node connecting the current layer
          # to the input layer we just created.
          layer(x)
          set_inputs = True

      if set_inputs:
        # If an input layer (placeholder) is available.
        if len(nest.flatten(layer._inbound_nodes[-1].output_tensors)) != 1:
          raise ValueError('All layers in a Sequential model '
                           'should have a single output tensor. '
                           'For multi-output layers, '
                           'use the functional API.')
        self.outputs = [
            nest.flatten(layer._inbound_nodes[-1].output_tensors)[0]
        ]
        self.inputs = layer_utils.get_source_inputs(self.outputs[0])

    elif self.outputs:
      # If the model is being built continuously on top of an input layer:
      # refresh its output.
      output_tensor = layer(self.outputs[0])
      if len(nest.flatten(output_tensor)) != 1:
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]

    if self.outputs:
      # True if set_inputs or self._is_graph_network or if adding a layer
      # to an already built deferred seq model.
      self.built = True

    if set_inputs or self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
    else:
      self._layers.append(layer)
    if self._layers:
      self._track_layers(self._layers)

    self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

  def pop(self):
    """Removes the last layer in the model.
    Raises:
        TypeError: if there are no layers in the model.
    """
    if not self.layers:
      raise TypeError('There are no layers in the model.')

    layer = self._layers.pop()
    self._layer_call_argspecs.pop(layer)
    if not self.layers:
      self.outputs = None
      self.inputs = None
      self.built = False
    elif self._is_graph_network:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

  def build(self, input_shape=None):
    if self._is_graph_network:
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
    else:
      if input_shape is None:
        raise ValueError('You must provide an `input_shape` argument.')
      input_shape = tuple(input_shape)
      self._build_input_shape = input_shape
      super(Sequential, self).build(input_shape)
    self.built = True

  def call(self, inputs, training=None, mask=None):  # pylint: disable=redefined-outer-name
    if self._is_graph_network:
      if not self.built:
        self._init_graph_network(self.inputs, self.outputs, name=self.name)
      return super(Sequential, self).call(inputs, training=training, mask=mask)

    outputs = inputs  # handle the corner case where self.layers is empty
    for layer in self.layers:
      # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
      # are the outputs of `layer` applied to `inputs`. At the end of each
      # iteration `inputs` is set to `outputs` to prepare for the next layer.
      kwargs = {}
      argspec = self._layer_call_argspecs[layer].args
      if 'mask' in argspec:
        kwargs['mask'] = mask
      if 'training' in argspec:
        kwargs['training'] = training

      outputs = layer(inputs, **kwargs)

      # `outputs` will be the inputs to the next layer.
      inputs = outputs
      mask = outputs._keras_mask

    return outputs