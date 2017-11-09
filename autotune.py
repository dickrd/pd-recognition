import json
import os

from model.common import build_cnn
from model.cnn import train, test


class TuningSave(object):

    def __init__(self, save_path="./"):
        self.model_path = os.path.join(save_path, "tuned")
        self._save_path=os.path.join(self.model_path, "tuning.json")

        if os.path.exists(self._save_path):
            with open(self._save_path, "r") as saved_file:
                self.status = json.load(saved_file)
        else:
            self.status = {
                "best_accuracy": 0,
                "checkpoint": "",
                "iteration": 0
            }
            os.mkdir(self.model_path)

    def update(self, accuracy, checkpoint):

        if accuracy > self.status["best_accuracy"]:
            self.status["best_accuracy"] = accuracy
            self.status["checkpoint"] = checkpoint
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
          "\ttuning   status:\t{2}".format(save_path, tuning_rate, tuning_save.status)

    while True:
        tuning_save.next_iteration()
        print "Start training: ", tuning_save.status["iteration"]
        train(model_path=save_path, train_data_path=train_data_path, class_count=class_count, image_size=image_size, image_channel=image_channel,
              build=build, scope=scope, num_epoch=1,
              batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        print "Start testing: ", tuning_save.status["iteration"]
        accuracy = test(model_path=save_path, test_data_path=test_data_path, class_count=class_count, image_size=image_size, image_channel=image_channel,
                        build=build, report_rate=100,
                        batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        previous_status = "(previous accuracy is {0} at {1})"\
            .format(tuning_save.status["best_accuracy"], tuning_save.status["checkpoint"])
        with open(os.path.join(save_path, "checkpoint")) as checkpoint_file:
            checkpoint_name = checkpoint_file.readline().strip().split(':', 1)[1].strip()[1:-1]
        if tuning_save.update(accuracy=accuracy, checkpoint=checkpoint_name):
            print "Found new best model with accuracy: {0} {1}."\
                .format(tuning_save.status["best_accuracy"], previous_status)
            import glob
            import shutil
            for a_file in glob.glob(os.path.join(save_path, checkpoint_name) + "*"):
                shutil.copy2(a_file, tuning_save.model_path)

        tuning_save.save()


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
