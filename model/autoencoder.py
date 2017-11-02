import tensorflow as tf

from model.common import new_biases, new_weights, new_fc_layer, optimize


def build_decoder_classifier(input_tensor, num_class, image_size, image_channel=3):
    print "Using image: {0}x{0}x{1}".format(image_size, image_channel)

    fc_size = 1500
    print "Fc layer: ", fc_size

    y, z = build_autoencoder(input_tensor)

    with tf.variable_scope("custom_classifier"):
        layer_latest_conv = z

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


def build_autoencoder(x_in, corruption=False):
    num_filters_list = [100, 200, 300, 200]
    filter_size_list = [3, 3, 3, 3]
    print "Autoencoder configuration: \n" \
          "\t{0}\n" \
          "\t{1}".format(num_filters_list, filter_size_list)

    # Add noise.
    if corruption:
        x_in = _corrupt(x_in)
    # Normalize
    x = x_in - 127.5

    # Build the encoder
    shape_list = []
    weight_list = []
    current_input = x
    for i in range(len(num_filters_list)):
        input_shape = current_input.get_shape().as_list()
        shape_list.append(input_shape)
        shape = [filter_size_list[i], filter_size_list[i], input_shape[-1], num_filters_list[i]]
        weights = new_weights(shape=shape)
        weight_list.append(weights)
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

    y = current_input + 127.5

    return y, z


def train_autoencoder(model_path, train_data_path, image_size,
                      num_epoch=50, batch_size=10, capacity=3000, min_after_dequeue=800):
    from data.common import TfReader
    with tf.Graph().as_default():
        # Read training data.
        train_data = TfReader(data_path=train_data_path,
                              size=(image_size, image_size),
                              num_epochs=num_epoch)
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        y, z = build_autoencoder(images)

        # Cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - images))
        optimize(cost=cost,
                 scope=None, save_path=model_path)

def test_autoencoder(model_path, test_data_path, image_size, report_rate=10,
                     batch_size=100, capacity=3000, min_after_dequeue=800):
    from PIL import Image
    import numpy as np
    import os
    from data.common import TfReader
    with tf.Graph().as_default():
        # Read test data.
        test_data = TfReader(data_path=test_data_path, size=(image_size, image_size))
        images, classes = test_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        y, z = build_autoencoder(images)

        # Cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - images))

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
            overall_cost = 0.0
            step_count = 0.0
            try:
                while True:
                    step_count += 1.0
                    original, decoded, predictions, current_cost = sess.run([images, y, z, cost])
                    overall_cost += current_cost
                    if step_count % report_rate == 0:
                        Image.fromarray(np.uint8(decoded[0]), "RGB") \
                            .save("decoded_step_{0}.jpg".format(step_count))
                        Image.fromarray(np.uint8(original[0]), "RGB") \
                            .save("original_step_{0}.jpg".format(step_count))
                        print "{0} steps passed with cost of {1}/{2}." \
                            .format(step_count, current_cost, overall_cost / step_count)
            except tf.errors.OutOfRangeError:
                print "All images used in {0} steps.".format(step_count)
            finally:
                print "Final cost: {0}.".format(overall_cost / step_count)

def _corrupt(x):
    """
    Take an input tensor and add uniform masking.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                    minval=0,
                                                    maxval=2,
                                                    dtype=tf.int32), tf.float32))


def _use_model():
    import argparse
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="autoencoder model util.")
    parser.add_argument("action",
                        help="action to perform, including: train, test")
    parser.add_argument("-d", "--data", nargs='*',
                        help="path to formatted tfrecords data")
    parser.add_argument("-m", "--model-path", default="./",
                        help="path to stored model")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="resized image size")
    args = parser.parse_args()

    if args.action == "train":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        train_autoencoder(model_path=args.model_path, train_data_path=args.data, image_size=args.resize)

    elif args.action == "test":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        test_autoencoder(model_path=args.model_path, test_data_path=args.data, image_size=args.resize)

    else:
        print "Unsupported action: " + args.action


if __name__ == "__main__":
    _use_model()
