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
          "\toutput :\t{9}"\
        .format(image_size, image_channel,
                filter_size1, num_filters1,
                filter_size2, num_filters2,
                filter_size3, num_filters3,
                fc_size,
                num_class)

    # Make input shape like [batch_size, img_height, img_width, img_channel]
    layer_input_image = tf.reshape(input_tensor, [-1, image_size, image_size, image_channel])

    # Building layers.
    layer_conv1, weights_conv1 = _new_conv_layer(layer_last=layer_input_image,
                                                 num_input_channels=image_channel,
                                                 filter_size=filter_size1,
                                                 num_filters=num_filters1,
                                                 use_pooling=True)

    layer_conv2, weights_conv2 = _new_conv_layer(layer_last=layer_conv1,
                                                 num_input_channels=num_filters1,
                                                 filter_size=filter_size2,
                                                 num_filters=num_filters2,
                                                 use_pooling=True)

    layer_conv3, weights_conv3 = _new_conv_layer(layer_last=layer_conv2,
                                                 num_input_channels=num_filters2,
                                                 filter_size=filter_size3,
                                                 num_filters=num_filters3,
                                                 use_pooling=True)

    layer_conv4, weights_conv4 = _new_conv_layer(layer_last=layer_conv3,
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

    layer_fc1 = _new_fc_layer(layer_last=layer_flat,
                              num_inputs=num_features,
                              num_outputs=fc_size,
                              use_relu=True)
    layer_fc2 = _new_fc_layer(layer_last=layer_fc1,
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


def optimize(layer_output, true_class, save_path="./", report_rate=100,
             learning_rate=1e-4):

    print "Learning model parameters:\n" \
          "\tmodel save path:\t{0}\n" \
          "\treport     rate:\t{1}\n" \
          "\tlearning   rate:\t{2}\n".format(save_path, report_rate, learning_rate)

    # Calculate cross-entropy for each image.
    # This function calculates the softmax internally, so use the output layer directly.
    # The input label is of type int in [0, num_class).
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_output,
                                                                   labels=true_class)

    # Calculate average cost across all images.
    cost = tf.reduce_mean(cross_entropy)

    # Optimizer that optimize the cost.
    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step_op)

    # Run supervised session.
    supervisor = tf.train.Supervisor(logdir=save_path)
    with supervisor.managed_session() as sess:
        global_step = -1
        try:
            while not supervisor.should_stop():
                _, global_step, current_cost = sess.run([optimizer, global_step_op, cost])
                if global_step % report_rate == 0:
                    print "{0} steps passed with current cost: {1}.".format(global_step, current_cost)
        except tf.errors.OutOfRangeError:
            print "All images used in {0} steps.".format(global_step)
        finally:
            supervisor.request_stop()


def _new_fc_layer(layer_last, num_inputs, num_outputs, use_relu=True):
    """
    Create a new fully connected layer.
    :param layer_last: The previous layer.
    :param num_inputs: Num. inputs from prev. layer.
    :param num_outputs: Num. outputs.
    :param use_relu: Use Rectified Linear Unit (ReLU)?
    :return: 
    """

    # Create new weights and biases.
    weights = _new_weights(shape=[num_inputs, num_outputs])
    biases = _new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(layer_last, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def _new_conv_layer(layer_last, num_input_channels, filter_size, num_filters, use_pooling=True):
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
    weights = _new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = _new_biases(length=num_filters)

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


def _new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def _new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def use_model():
    import argparse
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="cnn model util.")
    parser.add_argument("action",
                        help="action to perform, including: train, predict")
    parser.add_argument("-i", "--image",
                        help="path to unclassified image")
    parser.add_argument("-d", "--data",
                        help="path to formatted tfrecords data")
    parser.add_argument("-l", "--class-label",
                        help="path to json file that contains a map of readable name to class label.")
    parser.add_argument("-g", "--googlenet", action="store_true",
                        help="use googlenet model")
    parser.add_argument("-m", "--model-path", default="./",
                        help="path to stored model")
    parser.add_argument("-n", "--class-count", default=0, type=int,
                        help="number of classes")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="resized image size")
    args = parser.parse_args()

    if args.googlenet:
        from model.googlenet import build_googlenet
        build = build_googlenet
    else:
        build = build_cnn

    if args.action == "train":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        class_count = args.class_count
        if args.class_label:
            import json
            with open(args.class_label, 'r') as label_file:
                label_names = json.load(label_file)
                class_count = len(label_names)

        if class_count == 0:
            print "Must specify number of classes(--class-count) or class label file(--class-label)!"
            return

        train(model_path=args.model_path, train_data_path=args.data, class_count=class_count, build=build,
              image_size=args.resize)

    elif args.action == "test":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        class_count = args.class_count
        if args.class_label:
            import json
            with open(args.class_label, 'r') as label_file:
                label_names = json.load(label_file)
                class_count = len(label_names)

        if class_count == 0:
            print "Must specify number of classes(--class-count) or class label file(--class-label)!"
            return
        test(model_path=args.model_path, test_data_path=args.data, class_count=class_count, build=build,
             image_size=args.resize)

    elif args.action == "predict":
        if not args.image:
            print "Must specify image to predict(--image)!"
            return

        if not args.model_path:
            print "Must specify directory of trained model(--model-path)!"
            return

        if not args.class_label:
            print "Must specify class label file(--class-label)!"
            return
        import json
        with open(args.class_label, 'r') as label_file:
            label_names = json.load(label_file)
            predict(model_path=args.model_path, name_dict=label_names, build=build,
                    feed_image=args.image, feed_image_size=args.resize)


def train(model_path, train_data_path, class_count, image_size, image_channel=3, build=build_cnn,
          num_epoch=50, batch_size=100, capacity=3000, min_after_dequeue=800):
    from data.convert import TfReader
    with tf.Graph().as_default():
        # Read training data.
        train_data = TfReader(data_path=train_data_path,
                              size=(image_size, image_size),
                              num_epochs=num_epoch)
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        y, y_pred_cls = build(input_tensor=images, num_class=class_count,
                              image_size=image_size, image_channel=image_channel)
        optimize(layer_output=y, true_class=classes,
                 save_path=model_path)


def test(model_path, test_data_path, class_count, image_size, image_channel=3, report_rate=10, build=build_cnn,
         batch_size=100, capacity=3000, min_after_dequeue=800):
    from data.convert import TfReader
    import os
    with tf.Graph().as_default():
        # Read test data.
        train_data = TfReader(data_path=test_data_path, size=(image_size, image_size))
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        y, y_pred_cls = build(input_tensor=images, num_class=class_count,
                              image_size=image_size, image_channel=image_channel)
        correct_prediction = tf.equal(y_pred_cls, classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Run session.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        with open(os.path.join(model_path, "checkpoint")) as checkpoint_file:
            checkpoint_name = checkpoint_file.readline().strip().split(':', 1)[1].strip()[1:-1]
        print "Using model: " + checkpoint_name
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, os.path.join(model_path, checkpoint_name))
            tf.train.start_queue_runners(sess)
            overall_accuracy = 0.0
            step_count = 0.0
            try:
                while True:
                    step_count += 1.0
                    predictions, current_accuracy = sess.run([y_pred_cls, accuracy])
                    overall_accuracy += current_accuracy
                    if step_count % report_rate == 0:
                        print "{0} steps passed with accuracy of {1}/{2}."\
                            .format(step_count, current_accuracy, overall_accuracy / step_count)
                        print "Sample predictions: {0}".format(predictions)
            except tf.errors.OutOfRangeError:
                print "All images used in {0} steps.".format(step_count)
            finally:
                print "Final accuracy: {0}.".format(overall_accuracy / step_count)


def predict(model_path, name_dict, feed_image, feed_image_size, feed_image_channel=3, build=build_cnn):
    from data.convert import load_image_data
    from data.convert import generate_name
    import re
    print "Loading image from {0} with size: {1}x{1}.".format(feed_image, feed_image_size)
    image = load_image_data(image_file_path=feed_image,
                            resize=(feed_image_size, feed_image_size))
    name = generate_name(feed_image, re.compile(r".*([FM])(\d\d\d\d).*"))
    if name:
        print "True class is: " + name
    with tf.Graph().as_default():
        # Input placeholder.
        x = tf.placeholder(tf.float32, shape=[feed_image_size, feed_image_size, 3], name='image_flat')
        y, y_pred_cls = build(input_tensor=x, num_class=len(name_dict),
                              image_size=feed_image_size, image_channel=feed_image_channel)
        y_pred = tf.nn.softmax(y)
        feed_dict = {x: image}

        # Run supervised session.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        import os
        with open(os.path.join(model_path, "checkpoint")) as checkpoint_file:
            checkpoint_name = checkpoint_file.readline().strip().split(':', 1)[1].strip()[1:-1]
        print "Using model: " + checkpoint_name
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, os.path.join(model_path, checkpoint_name))
            predictions, predicted_class = sess.run([y_pred, y_pred_cls], feed_dict=feed_dict)
            print "The prediction is: {0}".format(predictions)
            name = next(key for key, value in name_dict.items() if value == predicted_class)
            print "The predicted class is: " + name


if __name__ == "__main__":
    use_model()
