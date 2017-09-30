import os

import tensorflow as tf


class AutoList(list):
    def __setitem__(self, index, value):
        missing = index - len(self) + 1
        if missing > 0:
            self.extend([0] * missing)
        list.__setitem__(self, index, value)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return 0


class TfWriter(object):
    def __init__(self, write_path):
        self.writer = tf.python_io.TFRecordWriter(write_path)

    def close(self):
        self.writer.close()

    def write(self, image, label):
        """
        Write image and label to a tfrecords file.
        :param image: Image data.
        :param label: Label of this image.
        :return: None.
        """
        an_feature = {
            "label": _int64_feature(label),
            "image": _bytes_feature(tf.compat.as_bytes(image.tobytes()))
        }
        an_example = tf.train.Example(features=tf.train.Features(feature=an_feature))
        self.writer.write(an_example.SerializeToString())


class TfReader(object):
    def __init__(self, data_path, size=(512, 512), num_epochs=1):
        self.reader = tf.TFRecordReader()
        self.filename_queue = tf.train.string_input_producer([data_path], num_epochs=num_epochs)
        self.size = size

        self.data_path = data_path
        self.num_epoch = num_epochs

    def read(self, batch_size=50, num_threads=4, capacity=1000, min_after_dequeue=100):
        """
        Read tfrecords file containing image and label.
        :param batch_size: Size of bath.
        :param num_threads: Number of threads used.
        :param capacity: Maximal examples in memory.
        :param min_after_dequeue: Minimal examples for starting training.
        :return: Images and labels array of bath size.
        """

        print "Reading configuration:\n" \
              "\tdata path:\t{0}\n" \
              "\timage size:\t{1}x{2}\n" \
              "\treadings:\t{3}x{4}/{5}\n" \
              "\tcapacity:\t{6}~{7}"\
            .format(self.data_path,
                    self.size[0], self.size[1],
                    self.num_epoch, batch_size, num_threads,
                    min_after_dequeue, capacity)

        feature_structure = {'image': tf.FixedLenFeature([], tf.string),
                             'label': tf.FixedLenFeature([], tf.int64)}

        # Read serialized data.
        _, serialized_example = self.reader.read(self.filename_queue)
        features = tf.parse_single_example(serialized=serialized_example, features=feature_structure)

        # Cast to original format.
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [self.size[0], self.size[1], 3])
        label = tf.cast(features['label'], tf.int64)

        # Random bathing.
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)
        return images, labels


def load_image_data(image_file_path,
                    resize=(512, 512)):
    """
    Load image data in RGB format.
    :param image_file_path: Path to a image file.
    :param resize: Resize the car image to the size.
    :return: PIL.Image object of the car's image and the label(car brand).
    """
    from PIL import Image

    # Pre-process image.
    image = Image.open(image_file_path).convert(mode="RGB")

    # Resize image.
    image = image.resize(resize, Image.LANCZOS)

    return image


def generate_name_by_path(image_file_path, pattern):
    matches = pattern.match(image_file_path)
    if matches:
        return matches.group(1)
    else:
        return None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    """
    Tool to convert saved json data to tfrecords.
    
    The random chance is for example [0.2, 0.3] indicating that there is a 20% chance a
    specific image will be written to the second tfrecord file and if it fails, there will be a 
    30% chance it will be written to the third tfrecord file and if all these fails, it will, 
    default to be written to the first file.
    :return: None 
    """
    import argparse
    import re
    import random
    import json

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(description="convert image to tfrecords file.")

    parser.add_argument("-i", "--input-path", nargs='+',
                        help="path to image files")
    parser.add_argument("-l", "--limit", type=int,
                        help="limit data set size of every label to a fixed number")
    parser.add_argument("-o", "--output-path", default="./",
                        help="path to store result tfrecords")
    parser.add_argument("-r", "--random-chance", default=[], type=float, nargs='*',
                        help="chance to split an example to a new tfrecord file")
    parser.add_argument("-s", "--resize", default=512, type=int,
                        help="size to resize image files to")
    args = parser.parse_args()

    print "Image will be resized to: " + str(args.resize) + "x" + str(args.resize)

    # Vars for write images to different files and count.
    # File path list of tfrecord files.
    result_files = [os.path.join(args.output_path, "train.tfrecords")]
    # Writers list.
    writers = [TfWriter(result_files[0])]
    # Count of each tfrecord file.
    count = [0]
    # Count for each name.
    name_wrote_count = AutoList()

    print "Default output file: " + result_files[0]
    for index, item in enumerate(args.random_chance):
        result_files.append(os.path.join(args.output_path, "test_" + str(index) + ".tfrecords"))
        writers.append(TfWriter(result_files[index + 1]))
        count.append(0)
        print "Chance for example to file " + result_files[index + 1] + ": " + str(item)

    # Compile regex for name generation.
    pattern = re.compile(r".*([FM])(\d\d\d\d).*")

    # Name to label conversion.
    name_labels = {}
    for directory in args.input_path:
        # Walk through given path, to find car images.
        for path, subdirs, files in os.walk(directory):
            print "In: " + path
            for a_file in files:
                # Try to read image.
                # noinspection PyBroadException
                try:
                    current_file_path = os.path.join(path, a_file)

                    img = load_image_data(current_file_path,
                                          resize=(args.resize, args.resize))
                    name = generate_name_by_path(current_file_path, pattern)
                    if not name:
                        continue

                    # Convert readable name to int label. Save the conversion for converting back.
                    if name not in name_labels:
                        label = len(name_labels)
                        name_labels[name] = label
                    else:
                        label = name_labels[name]

                    # Skip extra images.
                    if args.limit and name_wrote_count[label] > args.limit:
                        continue

                    # If this car image has been write to a tfrecord file.
                    write = False

                    # Determine which tfrecord file to write this image to.
                    for index, item in enumerate(args.random_chance):
                        if random.random() < item:
                            writers[index + 1].write(img, label)
                            count[index + 1] += 1
                            write = True
                            break
                    # Default write to this file.
                    if not write:
                        writers[0].write(img, label)
                        count[0] += 1

                    name_wrote_count[label] += 1
                except Exception as e:
                    print "Error reading " + a_file + ": " + repr(e)

            # Count example for every directory read.
            for index, file_name in enumerate(result_files):
                print "Current examples in " + file_name + ": " + str(count[index])

    print "All images processed."

    # Close the writers.
    for writer in writers:
        writer.close()

    print "Labels wrote: {0}".format(name_wrote_count)
    # Save name label conversion.
    with open(os.path.join(args.output_path, "name_labels.json"), 'w') as label_file:
        label_file.write(json.dumps(name_labels))
    print "Done."


if __name__ == "__main__":
    main()
