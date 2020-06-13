# If you intend to create your own optimization algorithm, simply inherit from this class and override the following methods:

# resource_apply_dense (update variable given gradient tensor is dense)
# resource_apply_sparse (update variable given gradient tensor is sparse)
# create_slots (if your optimizer algorithm requires additional variables)
# get_config (serialization of the optimizer, include all hyper parameters)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.training import gen_training_ops


class SGDErrorInject(Optimizer):
  """Stochastic gradient descent and momentum optimizer.
  Computes:
  ```
  theta(t+1) = theta(t) - learning_rate * gradient
  gradient is evaluated at theta(t).
  ```
  or Computes (if `nesterov = False`):
  ```
  v(t+1) = momentum * v(t) - learning_rate * gradient
  theta(t+1) = theta(t) + v(t+1)
  if `nesterov` is False, gradient is evaluated at theta(t).
  if `nesterov` is True, gradient is evaluated at theta(t) + momentum * v(t),
    and the variables always store theta + m v instead of theta
  ```
  Some of the args below are hyperparameters, where a hyperparameter is
  defined as a scalar Tensor, a regular Python value, or a callable (which
  will be evaluated when `apply_gradients` is called) returning a scalar
  Tensor or a Python value.
  @compatibility(eager)
  When eager execution is enabled, learning_rate can be a callable that takes
  no arguments and returns the actual value to use. This can be useful for
  changing these values across different invocations of optimizer functions.
  @end_compatibility
  # References
      nesterov = True, See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
  """

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               **kwargs):
    """Construct a new Stochastic Gradient Descent or Momentum optimizer.
    Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      momentum: float hyperparameter >= 0 that accelerates SGD in the relevant
        direction and dampens oscillations.
      nesterov: boolean. Whether to apply Nesterov momentum.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'SGD'.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super(SGDErrorInject, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGDErrorInject, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
        self._get_hyper("momentum", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # print("RESOURCE APPLY DENSE")
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    # print(var)
    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      v =  tf.resource_apply_keras_momentum(
          var.handle,
          momentum_var.handle,
          coefficients["lr_t"],
          grad,
          coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
      # print("AFTER1")
      # print(v)
      return v
    else:
      v = gen_training_ops.resource_apply_gradient_descent(
          var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

      # print("AFTER2")
      # print(v)
      return v


  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    if self._momentum:
      return super(SGD, self)._resource_apply_sparse_duplicate_indices(
          grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      return resource_variable_ops.resource_scatter_add(
          var.handle, indices, -grad * coefficients["lr_t"])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    print("RESOURCE APPLY SPARSE")
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    momentum_var = self.get_slot(var, "momentum")
    return gen_training_ops.resource_sparse_apply_keras_momentum(
        var.handle,
        momentum_var.handle,
        coefficients["lr_t"],
        grad,
        indices,
        coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    config = super(SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
    })
    return config