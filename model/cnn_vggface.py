import tensorflow as tf
import numpy as np
from scipy.io import loadmat

from model.common import new_fc_layer


def build_custom_vgg(input_tensor, num_class, image_size, image_channel=3,
                     original_model="vgg-face.mat"):
    """
    Build custom vgg classification layers using downloaded original model in .mat format
    here: http://www.vlfeat.org/matconvnet/pretrained/#face-recognition
    """
    fc_size = 2000
    print "Custom fc layer: ", fc_size

    # Pre-trained image size and channel.
    assert image_size == 224
    assert image_channel == 3

    network, average_image = vgg_face(original_model, input_tensor)

    layer_latest_conv = network['pool5']

    with tf.variable_scope("custom_vgg"):
        num_features = layer_latest_conv.get_shape()[1:].num_elements()
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


def vgg_face(param_path, input_maps):
    """
    Code from https://github.com/ZZUTK/Tensorflow-VGG-face
    """
    data = loadmat(param_path)

    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, (image_size[0], image_size[1]))

    # read layer info
    layers = data['layers']
    current = input_maps - average_image
    network = {}
    weights = {}
    for layer in layers[0]:
        name = layer[0]['name'][0][0]
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer[0]['stride'][0][0]
            kernel, bias = layer[0]['weights'][0][0]
            weight = tf.Variable(kernel)
            bias = tf.Variable(np.squeeze(bias).reshape(-1))
            conv = tf.nn.conv2d(current, weight,
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)

            weights[name] = [weight, bias]
        elif layer_type == 'relu':
            current = tf.nn.relu(current)

            weights[name] = []
        elif layer_type == 'pool':
            stride = layer[0]['stride'][0][0]
            pool = layer[0]['pool'][0][0]
            current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')

            weights[name] = []
        elif layer_type == 'softmax':
            current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

            weights[name] = []

        network[name] = current

    print "Vgg model loaded."
    return network, weights
