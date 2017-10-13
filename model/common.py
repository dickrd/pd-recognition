import tensorflow as tf


def build_cnn(input_tensor, num_class, image_size, image_channel=3):

    # Hyper-parameters.
    filter_size1 = 3
    num_filters1 = 32

    filter_size2 = 3
    num_filters2 = 64

    filter_size3 = 3
    num_filters3 = 128

    filter_size4 = 3
    num_filters4 = 48

    fc_size = 1024

    print "Building cnn: \n" \
          "\tinput  :\t{0}x{0}x{1}\n" \
          "\tlayer 1:\t{2}x{3}\n" \
          "\tlayer 2:\t{4}x{5}\n" \
          "\tlayer 3:\t{6}x{7}\n" \
          "\tlayer 4:\t{8}\n" \
          "\toutput :\t{9}" \
        .format(image_size, image_channel,
                filter_size1, num_filters1,
                filter_size2, num_filters2,
                filter_size3, num_filters3,
                fc_size,
                num_class)

    # Make input shape like [batch_size, img_height, img_width, img_channel]
    layer_input_image = tf.reshape(input_tensor, [-1, image_size, image_size, image_channel])

    # Building layers.
    layer_conv1, weights_conv1 = new_conv_layer(layer_last=layer_input_image,
                                                num_input_channels=image_channel,
                                                filter_size=filter_size1,
                                                num_filters=num_filters1,
                                                use_pooling=True)

    layer_conv2, weights_conv2 = new_conv_layer(layer_last=layer_conv1,
                                                num_input_channels=num_filters1,
                                                filter_size=filter_size2,
                                                num_filters=num_filters2,
                                                use_pooling=True)

    layer_conv3, weights_conv3 = new_conv_layer(layer_last=layer_conv2,
                                                num_input_channels=num_filters2,
                                                filter_size=filter_size3,
                                                num_filters=num_filters3,
                                                use_pooling=True)

    layer_conv4, weights_conv4 = new_conv_layer(layer_last=layer_conv3,
                                                num_input_channels=num_filters3,
                                                filter_size=filter_size4,
                                                num_filters=num_filters4,
                                                use_pooling=True)

    layer_latest_conv = layer_conv4

    # The shape of this layer is assumed to be:
    # layer_last_conv_shape == [num_images, img_height, img_width, num_channels]
    layer_last_conv_shape = layer_latest_conv.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_last_conv_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer_latest_conv, [-1, num_features])

    layer_fc1 = new_fc_layer(layer_last=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             use_relu=True)
    layer_fc2 = new_fc_layer(layer_last=layer_fc1,
                             num_inputs=fc_size,
                             num_outputs=num_class,
                             use_relu=False)

    # Output layer.
    y = layer_fc2
    # Use softmax to normalize the output.
    y_pred = tf.nn.softmax(y)
    # Use the most likely prediction as class label.
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    return y, y_pred_cls


def new_fc_layer(layer_last, num_inputs, num_outputs, use_relu=True):
    """
    Create a new fully connected layer.
    :param layer_last: The previous layer.
    :param num_inputs: Num. inputs from prev. layer.
    :param num_outputs: Num. outputs.
    :param use_relu: Use Rectified Linear Unit (ReLU)?
    :return: 
    """

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(layer_last, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def new_conv_layer(layer_last, num_input_channels, filter_size, num_filters, use_pooling=True):
    """
    Create a new convolution layer.
    :param layer_last: The previous layer.
    :param num_input_channels: Num. channels in prev. layer.
    :param filter_size: Width and height of each filter.
    :param num_filters: Number of filters.
    :param use_pooling: Use 2x2 max-pooling.
    :return: Result layer and layer weights.
    """

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=layer_last,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))