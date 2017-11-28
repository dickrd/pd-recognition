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
                "checkpoint": "no model",
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


def tune_cnn(save_path, train_data_path, test_data_path, class_count, image_size, image_channel=3, tuning_rate=100, build=build_cnn,
             scope=None,
             batch_size=10, capacity=3000, min_after_dequeue=800):
    tuning_save = TuningSave(save_path=save_path)
    print "Tuning model parameters:\n" \
          "\tmodel save path:\t{0}\n" \
          "\ttuning   status:\t{2}".format(save_path, tuning_rate, tuning_save.status)

    while True:
        tuning_save.next_iteration()
        print "Start training: ", tuning_save.status["iteration"]
        train(model_path=save_path, train_data_path=train_data_path, class_count=class_count, image_size=image_size, image_channel=image_channel, build=build,
              scope=scope,
              num_epoch=1, batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        print "Start testing on training set: ", tuning_save.status["iteration"]
        test(model_path=save_path, test_data_path=train_data_path, class_count=class_count, image_size=image_size, image_channel=image_channel, report_rate=100, build=build,
             batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
        print "Start testing on testing set: ", tuning_save.status["iteration"]
        accuracy = test(model_path=save_path, test_data_path=test_data_path, class_count=class_count, image_size=image_size, image_channel=image_channel, report_rate=100, build=build,
                        batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        previous_status = "(previous accuracy is {0} with {1})"\
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
