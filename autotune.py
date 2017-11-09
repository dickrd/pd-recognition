import json
import os
import tensorflow as tf

from data.common import TfReader
from model.common import build_cnn


class TuningSave(object):

    def __init__(self, save_path="./"):
        self._save_path=os.path.join(save_path, "tuning.json")

        if os.path.exists(self._save_path):
            with open(self._save_path, "r") as saved_file:
                self.status = json.load(saved_file)
        else:
            self.status = {
                "best_accuracy": 0,
                "at_step": 0,
                "learning_rate": 1e-1,
                "iteration": 0
            }

    def update(self, accuracy, step, learning_rate):
        self.status["learning_rate"] = learning_rate

        if accuracy > self.status["best_accuracy"]:
            self.status["best_accuracy"] = accuracy
            self.status["at_step"] = step
            return True
        return False

    def next_iteration(self):
        self.status["iteration"] += 1

    def save(self):
        with open(self._save_path, "w") as save_file:
            save_file.write(json.dumps(self.status))


def tune_cnn(train_data_path, test_data_path, class_count, image_size=512, image_channel=3,
         batch_size=10, capacity=3000, min_after_dequeue=800,
         build=build_cnn, tuning_rate=100,
         save_path="./", scope=None):
    tuning_save = TuningSave(save_path=save_path)
    print "Tuning model parameters:\n" \
          "\tmodel save path:\t{0}\n" \
          "\ttuning     rate:\t{1}\n" \
          "\ttuning   status:\t{2}".format(save_path, tuning_rate, tuning_save.status)

    # Network structure.
    x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='image_input')
    y, y_pred_cls = build(input_tensor=x, num_class=class_count, image_size=image_size, image_channel=image_channel)
    global_step_op = tf.Variable(0, trainable=False, name="global_step")

    if scope:
        var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        print "Tuning variables: {0}".format(var_to_train)
    else:
        var_to_train = None
        print "Tuning all trainable variables."

    # Run supervised session.
    accurate_saver = tf.train.Saver()
    learning_rate = tuning_save.status["learning_rate"]
    with tf.Session() as sess:
        global_step = -1
        test_accuracy = -1.0

        tuning_save.next_iteration()
        train_images, train_classes = TfReader(data_path=train_data_path, size=(image_size, image_size)) \
            .read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        train_feed = {x: train_images.eval()}
        tf.train.start_queue_runners(sess)

        optimizer = _get_optimizer(logits=y, labels=train_classes,
                                   learning_rate=learning_rate, global_step_op=global_step_op, var_to_train=var_to_train)
        while True:
            try:
                _, global_step = sess.run([optimizer, global_step_op], feed_dict=train_feed)

                if global_step % tuning_rate == 0:
                    print "{0} steps passed\t".format(global_step),
                    test_images, test_classes = TfReader(data_path=test_data_path, size=(image_size, image_size)) \
                        .read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
                    test_feed = {x: test_images.eval()}
                    tf.train.start_queue_runners(sess)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, test_classes), tf.float32))

                    overall_accuracy = 0.0
                    step_count = 0.0
                    try:
                        while True:
                            step_count += 1.0
                            current_accuracy = sess.run([accuracy], feed_dict=test_feed)
                            overall_accuracy += current_accuracy
                    except tf.errors.OutOfRangeError:
                        pass
                    finally:
                        print "accuracy: {0}".format(overall_accuracy / step_count)

                    learning_rate /= global_step * 10
                    optimizer = _get_optimizer(logits=y, labels=train_classes,
                                               learning_rate=learning_rate, global_step_op=global_step_op, var_to_train=var_to_train)
                    print "Learning rate changed to :", learning_rate

                    previous_status = "(previous best is {0} at step {1})"\
                        .format(tuning_save.status["best_accuracy"], tuning_save.status["at_step"])
                    if tuning_save.update(accuracy=test_accuracy, step=global_step, learning_rate=learning_rate):
                        print "New best accuracy found: {0} {1}"\
                            .format(tuning_save.status["best_accuracy"], previous_status)
                        accurate_saver.save(sess=sess,
                                               save_path=os.path.join(save_path,
                                                                      "new_best_model.{0}.ckpt".format(global_step)))
                    tuning_save.save()

            except tf.errors.OutOfRangeError:
                print "At step {0}, iteration {1} complete. Current accuracy: {2}, best accuracy: {3} (at step {4})"\
                    .format(global_step, tuning_save.status["iteration"],
                            test_accuracy, tuning_save.status["best_accuracy"], tuning_save.status["at_step"])

                tuning_save.next_iteration()
                train_images, train_classes = TfReader(data_path=train_data_path, size=(image_size, image_size)) \
                    .read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
                train_feed = {x: train_images.eval()}
                tf.train.start_queue_runners(sess)
                optimizer = _get_optimizer(logits=y, labels=train_classes,
                                           learning_rate=learning_rate, global_step_op=global_step_op, var_to_train=var_to_train)

                tuning_save.save()


def _get_optimizer(logits, labels, learning_rate, global_step_op, var_to_train):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels)
    cost = tf.reduce_mean(cross_entropy)

    if var_to_train:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                 global_step=global_step_op,
                                                                                 var_list=var_to_train)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                 global_step=global_step_op)
    return optimizer


def _start_tuning():
    import argparse
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="cnn classify model tuning util.")
    parser.add_argument("-d", "--data", nargs='*',
                        help="path to formatted tfrecords data for training")
    parser.add_argument("-v", "--test-data", nargs='*',
                        help="path to formatted tfrecords data for testing")
    parser.add_argument("-l", "--class-label",
                        help="path to json file that contains a map of readable name to class label.")
    parser.add_argument("-t", "--model-type", default="general",
                        help="which type of model to use(general, autoencoder, googlecar, vggface, resnet)")
    parser.add_argument("-m", "--model-path", default="./",
                        help="path to stored model")
    parser.add_argument("-n", "--class-count", default=0, type=int,
                        help="number of classes")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="resized image size")
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

    if not args.data or not args.test_data:
        print "Must specify data path(--data and --test-data)!"
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

    tune_cnn(train_data_path=args.data, test_data_path=args.test_data, class_count=class_count, image_size=args.resize,
             build=build, save_path=args.model_path, scope=scope)


if __name__ == '__main__':
    _start_tuning()
