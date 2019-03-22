import tensorflow as tf
from coordConv import CoordConv
class ConvGRUCell(tf.contrib.rnn.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""
  def __init__(self, shape, filters, kernel, activation=tf.tanh, data_format='channels_last', reuse=None, is_training=False, name='', padding='same'):
    super(ConvGRUCell, self).__init__()
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = True
    self._name = name
    self._shape = shape
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = 'NHWC'
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 1
        self._data_format = 'NCHW'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, x, h):
    channels = x.shape[self._feature_axis].value
    with tf.variable_scope('gates' + self._name,reuse=tf.AUTO_REUSE):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tf.get_variable('kernel', self._kernel + [n, m])
      y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._data_format == 'NHWC':
        y = CoordConv(x_dim=self._shape[0], y_dim=self._shape[1], with_r=False,
                       filters = m, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = None, name='coordConv')(inputs)
      else:
        y = CoordConv(x_dim=self._shape[0], y_dim=self._shape[1], with_r=False,
                       filters = m, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = None, name='coordConv')(tf.transpose(inputs,[0,2,3,1]))
        y = tf.transpose(y,[0,3,1,2])
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = tf.contrib.layers.layer_norm(r)
        u = tf.contrib.layers.layer_norm(u)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
        r, u = tf.split(y, 2, axis=self._feature_axis)
      r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.variable_scope('candidate' + self._name,reuse=tf.AUTO_REUSE):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = tf.get_variable('kernel', self._kernel + [n, m])
      # y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._data_format == 'NHWC':
        y = CoordConv(x_dim=self._shape[0], y_dim=self._shape[1], with_r=False,
                       filters = m, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = None, name='coordConv')(inputs)
      else:
        y = CoordConv(x_dim=self._shape[0], y_dim=self._shape[1], with_r=False,
                       filters = m, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = None, name='coordConv')(tf.transpose(inputs,[0,2,3,1]))
        y = tf.transpose(y,[0,3,1,2])
      if self._normalize:
        y = tf.contrib.layers.layer_norm(y)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h
