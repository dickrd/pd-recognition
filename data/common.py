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


# noinspection PyUnusedLocal
class DummyWriter(object):
    def __init__(self, write_path, regression=False):
        pass
    def close(self):
        pass
    def write(self, image, label):
        pass


class TfWriter(object):
    def __init__(self, write_path, regression=False):
        self.writer = tf.python_io.TFRecordWriter(write_path)
        self.regression = regression

    def close(self):
        self.writer.close()

    def write(self, image, label):
        """
        Write image and label to a tfrecords file.
        :param image: Image data.
        :param label: Label of this image.
        :return: None.
        """
        if self.regression:
            an_feature = {
                "label": _float_feature(label),
                "image": _bytes_feature(tf.compat.as_bytes(image.tobytes()))
            }
        else:
            an_feature = {
                "label": _int64_feature(label),
                "image": _bytes_feature(tf.compat.as_bytes(image.tobytes()))
            }

        an_example = tf.train.Example(features=tf.train.Features(feature=an_feature))
        self.writer.write(an_example.SerializeToString())


class TfReader(object):
    def __init__(self, data_path, regression=False, size=(512, 512), num_epochs=1):
        self.reader = tf.TFRecordReader()
        self.filename_queue = tf.train.string_input_producer(data_path, num_epochs=num_epochs)
        self.size = size

        self.data_path = data_path
        self.num_epoch = num_epochs

        if regression:
            self.label_type = tf.float32
        else:
            self.label_type = tf.int64

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
              "\tcapacity:\t{6}~{7}" \
            .format(self.data_path,
                    self.size[0], self.size[1],
                    self.num_epoch, batch_size, num_threads,
                    min_after_dequeue, capacity)

        feature_structure = {'image': tf.FixedLenFeature([], tf.string),
                             'label': tf.FixedLenFeature([], self.label_type)}

        # Read serialized data.
        _, serialized_example = self.reader.read(self.filename_queue)
        features = tf.parse_single_example(serialized=serialized_example, features=feature_structure)

        # Cast to original format.
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [self.size[0], self.size[1], 3])
        label = tf.cast(features['label'], self.label_type)

        # Random bathing.
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)
        return images, labels


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
