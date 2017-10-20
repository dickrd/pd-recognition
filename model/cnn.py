import tensorflow as tf

from model.common import build_cnn


def optimize(layer_output, true_class, save_path="./", report_rate=100, scope=None,
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
    if scope:
        var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                 global_step=global_step_op,
                                                                                 var_list=var_to_train)
        print "Optimizing variables: " + var_to_train
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                 global_step=global_step_op)
        print "Optimizing all trainable variables."

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


def use_model():
    import argparse
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="cnn model util.")
    parser.add_argument("action",
                        help="action to perform, including: train, test, predict")
    parser.add_argument("-i", "--image",
                        help="path to unclassified image")
    parser.add_argument("-d", "--data", nargs='*',
                        help="path to formatted tfrecords data")
    parser.add_argument("-l", "--class-label",
                        help="path to json file that contains a map of readable name to class label.")
    parser.add_argument("-t", "--model-type", default="general",
                        help="which type of model to use(general, google, car, vgg)")
    parser.add_argument("-m", "--model-path", default="./",
                        help="path to stored model")
    parser.add_argument("-n", "--class-count", default=0, type=int,
                        help="number of classes")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="resized image size")
    args = parser.parse_args()

    scope = None
    if args.model_type == "google":
        from model.cnn_googlenet import build_googlenet
        build = build_googlenet
    elif args.model_type == "car":
        from model.cnn_car import build_car_cnn
        build = build_car_cnn
    elif args.model_type == "vgg":
        from model.cnn_vggface import build_custom_vgg
        build = build_custom_vgg
        scope = "custom_vgg"
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

        train(model_path=args.model_path, train_data_path=args.data, class_count=class_count, build=build, scope=scope,
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
    else:
        print "Unsupported action: " + args.action


def train(model_path, train_data_path, class_count, image_size, image_channel=3, build=build_cnn, scope=None,
          num_epoch=50, batch_size=10, capacity=3000, min_after_dequeue=800):
    from data.common import TfReader
    with tf.Graph().as_default():
        # Read training data.
        train_data = TfReader(data_path=train_data_path,
                              size=(image_size, image_size),
                              num_epochs=num_epoch)
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        y, y_pred_cls = build(input_tensor=images, num_class=class_count,
                              image_size=image_size, image_channel=image_channel)
        optimize(layer_output=y, true_class=classes,
                 scope=scope, save_path=model_path)


def test(model_path, test_data_path, class_count, image_size, image_channel=3, report_rate=10, build=build_cnn,
         batch_size=100, capacity=3000, min_after_dequeue=800):
    from data.common import TfReader
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
    from data.common import load_image_data
    print "Loading image from {0} with size: {1}x{1}.".format(feed_image, feed_image_size)
    image = load_image_data(image_file_path=feed_image,
                            resize=(feed_image_size, feed_image_size))
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
