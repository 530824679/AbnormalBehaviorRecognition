import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

def ConvLayer(input, filter, kernel, stride, padding='SAME', scope="conv"):
    with tf.name_scope(scope):
        return tf.layers.conv2d(inputs=input,
                               use_bias=False,
                               filters=filter,
                               kernel_size=kernel,
                               strides=stride,
                               padding=padding)

def AveragePool(input, pool_size=[2, 2], stride=2, padding='SAME', scope="average_pool"):
    with tf.name_scope(scope):
        return tf.layers.average_pooling2d(inputs=input,
                                           pool_size=pool_size,
                                           strides=stride,
                                           padding=padding)

def MaxPool(input, pool_size=[2, 2], stride=2, padding='SAME', scope="max_pool"):
    with tf.name_scope(scope):
        return tf.layers.max_pooling2d(inputs=input,
                                           pool_size=pool_size,
                                           strides=stride,
                                           padding=padding)

def GlobalAveragePool(input, data_format='channels_last', scope="gap"):
    """
    Global Average Pooling
    arguments:
        input(tf.Tensor): a 4D tensor
            If `data_format='channels_last'`:
          4D tensor with shape:
          `(batch_size, rows, cols, channels)`
            - If `data_format='channels_first'`:
          4D tensor with shape:
          `(batch_size, channels, rows, cols)`
        data_format(string):one of 'channels_last'(default) or 'channels_last'
            'channels_last' corresponds to inputs with shape (batch, height, width, channels)
            'channels_first' corresponds to inputs with shape (batch, channels, height, width)
        :return: tf.Tensor: a NC tensor named output.
            2D tensor with shape:
            (batch_size, channels)
    """
    assert input.shape.ndims == 4
    with tf.name_scope(scope):
        if data_format == 'channels_last':
            axis = [1, 2]
        else:
            axis = [2, 3]
        return tf.reduce_mean(input, axis, name='global_average_pool', keep_dims=True)

def BatchNormalization(input, is_train, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(is_train,
                       lambda: batch_norm(inputs=input, is_training=is_train, reuse=None),
                       lambda: batch_norm(inputs=input, is_training=is_train, reuse=True))

def Relu(input, scope="relu"):
    with tf.name_scope(scope):
        return tf.nn.relu(input)

def Sigmoid(input, scope="sigmoid"):
    with tf.name_scope(scope):
        return tf.nn.sigmoid(input)

def Concat(layers, scope="concat"):
    with tf.name_scope(scope):
        return tf.concat(layers, axis=3)

def Fully_connected(input, num_classes, scope='fully_connected'):
    with tf.name_scope(scope):
        return tf.layers.dense(inputs=input, use_bias=False, units=num_classes)