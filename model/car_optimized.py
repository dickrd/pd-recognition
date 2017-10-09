import tensorflow as tf

from model.common import new_conv_layer, new_fc_layer


def build_car_cnn(input_tensor, num_class, image_size, image_channel=3):

    # Hyper-parameters.
    filter_size1 = 3
    num_filters1 = 100

    filter_size2 = 3
    num_filters2 = 200

    filter_size3 = 3
    num_filters3 = 300

    filter_size4 = 3
    num_filters4 = 200

    fc_size = 1500

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
