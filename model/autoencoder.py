import tensorflow as tf

from model.common import new_biases, new_weights


def build_car_autoencoder(x_in):
    num_filters_list = [100, 200, 300, 200]
    filter_size_list = [3, 3, 3, 3]

    # Normalize
    x = (x_in - 127.5) / 127.5

    # Build the encoder
    shape_list = []
    weight_list = []
    current_input = x
    for i in range(len(num_filters_list)):
        input_shape = current_input.get_shape().as_list()
        shape_list.append(input_shape)
        shape = [filter_size_list[i], filter_size_list[i], input_shape[-1], num_filters_list[i]]
        weights = new_weights(shape=shape)
        biases = new_biases(length=num_filters_list[i])
        layer = tf.nn.conv2d(input=current_input, filter=weights,
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        layer += biases
        current_input = tf.nn.relu(layer)

    # The latent representation
    z = current_input
    shape_list.reverse()
    weight_list.reverse()

    # Build the decoder using the different weights
    for i in range(len(shape_list)):
        shape = shape_list[i]
        weights_i = weight_list[i]
        biases = new_biases(length=shape[3])
        layer = tf.nn.conv2d_transpose(current_input, weights_i,
                                       tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                       strides=[1, 2, 2, 1],
                                       padding='SAME')
        layer += biases
        current_input = tf.nn.relu(layer)

    y = current_input

    return x, y, z
