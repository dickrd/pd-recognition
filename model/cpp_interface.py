import tensorflow as tf
import numpy as np
import io

from PIL import Image
from model.common import build_cnn
from multiprocessing.pool import ThreadPool


class Config():
    pass


config = Config()
config.pool = ThreadPool(processes=50)
config.name_labels = "name_labels.json"
config.feed_image_size = 256
config.feed_image_channel=3
config.build=build_cnn

def load(model_path):
    import json, os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(os.path.join(model_path, config.name_labels), 'r') as label_file:
        config.name_dict = json.load(label_file)
        class_count = len(config.name_dict)

    with tf.Graph().as_default():
        # Input placeholder.
        config.x = tf.placeholder(tf.float32, shape=[1, config.feed_image_size, config.feed_image_size, 3], name='image_flat')
        config.y, config.y_pred_cls = config.build(input_tensor=config.x, num_class=class_count,
                              image_size=config.feed_image_size, image_channel=config.feed_image_channel)
        config.y_pred = tf.nn.softmax(config.y)

        # Run supervised session.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver = tf.train.Saver()
        import os
        with open(os.path.join(model_path, "checkpoint")) as checkpoint_file:
            checkpoint_name = checkpoint_file.readline().strip().split(':', 1)[1].strip()[1:-1]
        print "Using model: " + checkpoint_name
        config.sess = tf.Session()
        config.sess.run(init_op)
        saver.restore(config.sess, os.path.join(model_path, checkpoint_name))

def run(feed_image):
    #async_result = config.pool.apply_async(actual_run, (feed_image, ))
    #return async_result.get()
    return actual_run(feed_image)

def actual_run(feed_image):
    from data.image_loading import load_image_data, normalize_image
    #image = load_image_data(feed_image)
    image = Image.open(io.BytesIO(feed_image)).convert(mode="RGB")
    image = normalize_image(image, resize=(config.feed_image_size, config.feed_image_size))
    feed_dict = {config.x: np.expand_dims(image, axis=0)}

    predictions, predicted_class = config.sess.run([config.y_pred, config.y_pred_cls], feed_dict=feed_dict)
    name = next(key for key, value in config.name_dict.items() if value == predicted_class)
    return name
