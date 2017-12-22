import tensorflow as tf
import numpy as np

from model.common import build_cnn, optimize, ConfusionMatrix, RegressionBias


def run_cluster(model_path, train_data_path, class_count, image_size,
                regression=False, build=build_cnn, scope=None,
                ps_hosts=None, worker_hosts=None, job_name=None, task_index=0):
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{0}".format(task_index), cluster=cluster)):
            train(model_path=model_path, train_data_path=train_data_path, class_count=class_count, image_size=image_size,
                  regression=regression, build=build, scope=scope,
                  master=server.target, task_index=task_index)
    else:
        print "Unknown job: " + job_name


def train(model_path, train_data_path, class_count, image_size, image_channel=3, report_rate=100, build=build_cnn,
          regression=False, scope=None, learning_rate=1e-4,
          num_epoch=50, batch_size=10, capacity=3000, min_after_dequeue=800,
          master="", task_index=0):
    from data.common import TfReader
    with tf.Graph().as_default():
        # Read training data.
        train_data = TfReader(data_path=train_data_path, regression=regression,
                              size=(image_size, image_size),
                              num_epochs=num_epoch)
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)


        y, y_pred_cls = build(input_tensor=images, num_class=class_count,
                              image_size=image_size, image_channel=image_channel)

        if regression:
            cost = tf.reduce_sum(tf.pow(tf.transpose(y) - classes, 2)) / (2 * batch_size)
        else:
            # Calculate cross-entropy for each image.
            # This function calculates the softmax internally, so use the output layer directly.
            # The input label is of type int in [0, num_class).
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                           labels=classes)

            # Calculate average cost across all images.
            cost = tf.reduce_mean(cross_entropy)

        optimize(cost=cost, save_path=model_path, report_rate=report_rate,
                 scope=scope, learning_rate=learning_rate,
                 master=master, task_index=task_index)


def test(model_path, test_data_path, class_count, image_size, image_channel=3, report_rate=100, build=build_cnn,
         regression=False,
         batch_size=1, capacity=3000, min_after_dequeue=800):
    from data.common import TfReader
    import os
    with tf.Graph().as_default():

        # Read test data.
        train_data = TfReader(data_path=test_data_path, regression=regression, size=(image_size, image_size))
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        y, y_pred_cls = build(input_tensor=images, num_class=class_count,
                              image_size=image_size, image_channel=image_channel)

        if regression:
            prediction_op = tf.squeeze(y)
            correct_prediction = tf.abs(tf.transpose(y) - classes)
            statistics = RegressionBias()
        else:
            prediction_op = y_pred_cls
            correct_prediction = tf.equal(y_pred_cls, classes)
            statistics = ConfusionMatrix(class_count=class_count)

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
                    predictions, truth, current_accuracy = sess.run([prediction_op, classes, accuracy])
                    step_count += 1.0
                    overall_accuracy += current_accuracy
                    statistics.update(predictions=predictions, truth=truth)
                    if step_count % report_rate == 0:
                        print "{0} steps passed with result of {1}/{2}."\
                            .format(step_count, current_accuracy, overall_accuracy / step_count)
                        print "Sample predictions: {0}".format(predictions)
            except tf.errors.OutOfRangeError:
                print "All images used in {0} steps.".format(step_count)
            finally:
                print "Final result: {0}.".format(overall_accuracy / step_count)
                statistics.print_result()

        return overall_accuracy / step_count


def predict(model_path, name_dict, class_count, feed_image, feed_image_size, feed_image_channel=3, build=build_cnn,
            regression=False):
    from data.image_loading import load_image_data
    print "Loading image from {0} with size: {1}x{1}.".format(feed_image, feed_image_size)
    image = load_image_data(image_path=feed_image,
                            resize=(feed_image_size, feed_image_size))

    with tf.Graph().as_default():
        # Input placeholder.
        x = tf.placeholder(tf.float32, shape=[1, feed_image_size, feed_image_size, 3], name='image_flat')
        y, y_pred_cls = build(input_tensor=x, num_class=class_count,
                              image_size=feed_image_size, image_channel=feed_image_channel)

        y_pred = tf.nn.softmax(y)
        feed_dict = {x: np.expand_dims(image, axis=0)}

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
            if regression:
                predictions = sess.run([y], feed_dict=feed_dict)
            else:
                predictions, predicted_class = sess.run([y_pred, y_pred_cls], feed_dict=feed_dict)
                name = next(key for key, value in name_dict.items() if value == predicted_class)
                print "The predicted class is: " + name

            print "The prediction is: {0}".format(predictions)


def _use_model():
    import argparse
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="cnn classify model util.")
    parser.add_argument("action",
                        help="action to perform, including: auto, train, test, predict")
    parser.add_argument("-i", "--image",
                        help="path to unclassified image")
    parser.add_argument("-d", "--data", nargs='*',
                        help="path to formatted tfrecords data")
    parser.add_argument("-v", "--test-data", nargs='*',
                        help="path to formatted tfrecords test data (used only with auto action)")
    parser.add_argument("-l", "--class-label",
                        help="path to json file that contains a map of readable name to class label.")
    parser.add_argument("-t", "--model-type", default="general",
                        help="which type of model to use (general, autoencoder, googlecar, vggface, resnet)")
    parser.add_argument("-m", "--model-path", default="./",
                        help="path to stored model")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="resized image size")

    parser.add_argument("--regression", action="store_true",
                        help="set to make labels as float numbers parsed from names")
    parser.add_argument("--ps", nargs='*',
                        help="ps hosts in address:port format")
    parser.add_argument("--worker", nargs='*',
                        help="worker hosts in address:port format")
    parser.add_argument("--job-name",
                        help="name of this job (ps, worker)")
    parser.add_argument("--task-index", type=int,
                        help="index of this task")
    args = parser.parse_args()

    scope = None
    if args.model_type == "googlecar":
        from model.cnn_googlenetcar import build_googlecar
        build = build_googlecar
    elif args.model_type == "autoencoder":
        from model.autoencoder import build_decoder_classifier
        build = build_decoder_classifier
    elif args.model_type == "vggface":
        from model.cnn_vggface import build_custom_vgg
        build = build_custom_vgg
        scope = "custom_vgg"
    elif args.model_type == "resnet":
        from model.cnn_resnet import build_custom_resnet
        build = build_custom_resnet
        scope = "custom_resnet"
    elif args.model_type == "general":
        build = build_cnn
    else:
        print "Unsupported model type: " + args.model_type
        return

    if args.regression:
        label_names = None
        class_count = 1
        print "Regression set."
    else:
        if not args.class_label:
            print "Must specify class label file(--class-label)!"
            return

        import json
        with open(args.class_label, 'r') as label_file:
            label_names = json.load(label_file)
            class_count = len(label_names)

    action(build=build, scope=scope,
           label_names=label_names, class_count=class_count,
           args=args)


def action(build, scope,
           label_names, class_count,
           args):
    if args.action == "train":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        has_cluster_config = False
        cluster_config_count = 0
        if args.ps:
            has_cluster_config = True
            cluster_config_count += 1
            print "Using ps: {0}".format(args.ps)
        if args.worker:
            has_cluster_config = True
            cluster_config_count += 1
            print "Using worker: {0}".format(args.worker)
        if args.job_name:
            has_cluster_config = True
            cluster_config_count += 1
            print "Job is " + args.job_name
        if args.task_index or args.task_index == 0:
            has_cluster_config = True
            cluster_config_count += 1
            print "Task index is {0}".format(args.task_index)

        if has_cluster_config:
            if cluster_config_count == 4:
                run_cluster(model_path=args.model_path, train_data_path=args.data, class_count=class_count,
                            regression=args.regression, build=build, scope=scope,
                            image_size=args.resize,
                            ps_hosts=args.ps, worker_hosts=args.worker, job_name=args.job_name, task_index=args.task_index)
            else:
                print "Cluster config incomplete, --ps --worker --job-name, or --task-index is missing!"
        else:
            train(model_path=args.model_path, train_data_path=args.data, class_count=class_count,
                  regression=args.regression, build=build, scope=scope,
                  image_size=args.resize)

    elif args.action == "test":
        if not args.data:
            print "Must specify data path(--data)!"
            return

        test(model_path=args.model_path, test_data_path=args.data, class_count=class_count,
             regression=args.regression, build=build,
             image_size=args.resize)

    elif args.action == "predict":
        import json
        if not args.image:
            print "Must specify image to predict(--image)!"
            return

        if not args.model_path:
            print "Must specify directory of trained model(--model-path)!"
            return

        predict(model_path=args.model_path, name_dict=label_names, feed_image=args.image, class_count=class_count,
                regression=args.regression, build=build,
                feed_image_size=args.resize)

    elif args.action == "auto":
        from model.autotune import tune_cnn
        if not args.test_data:
            print "Must specify test data path(--test-data)!"

        if not args.class_label and not args.regression:
            print "Must specify class label file(--class-label)!"
            return

        tune_cnn(save_path=args.model_path, train_data_path=args.data, test_data_path=args.test_data, class_count=class_count,
                 regression=args.regression, build=build, scope=scope,
                 image_size=args.resize)

    else:
        print "Unsupported action: " + args.action


if __name__ == "__main__":
    _use_model()
