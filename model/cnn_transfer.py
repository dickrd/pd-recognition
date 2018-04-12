import tensorflow as tf

from model.cnn_vggface import vgg_face
from model.common import new_fc_layer, EndSavingHook


def build_transfer_vgg(input_tensor, num_class, image_size=224, image_channel=3,
                       compatible=True, last_name="pool5", original_model="vgg-face.mat"):
    """
    Build custom vgg classification layers using downloaded original model in .mat format
    here: http://www.vlfeat.org/matconvnet/pretrained/#face-recognition
    """
    print "Training from layer: ", last_name
    assert image_size == 224
    assert image_channel == 3

    network, average_image, class_names = vgg_face(original_model, input_tensor)

    opt_layers = []
    past_last = False
    for name, layer in network:
        if past_last:
            opt_layers.append(layer)

        if name == last_name:
            past_last = True

    layer_latest_conv = network['relu7']

    with tf.variable_scope("vgg_output"):
        num_features = layer_latest_conv.get_shape()[1:].num_elements()
        layer_flat = tf.reshape(layer_latest_conv, [-1, num_features])

        layer_fc = new_fc_layer(layer_last=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=num_class,
                                 use_relu=False)

        # Output layer.
        y = layer_fc
        # Use softmax to normalize the output.
        y_pred = tf.nn.softmax(y)
        # Use the most likely prediction as class label.
        y_pred_cls = tf.argmax(y_pred, dimension=1)

    if compatible:
        return y, y_pred_cls
    else:
        return y, y_pred_cls, opt_layers

def train_transfer(model_path, train_data_path, class_count,
                   regression=False, last_name="pool5",
                   image_size = 224, report_rate=100,
                   learning_rate=1e-4,
                   num_epoch=50, batch_size=10, capacity=3000, min_after_dequeue=800):
    from data.common import TfReader
    with tf.Graph().as_default():
        # Read training data.
        train_data = TfReader(data_path=train_data_path, regression=regression,
                              size=(image_size, image_size),
                              num_epochs=num_epoch)
        images, classes = train_data.read(batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)


        y, y_pred_cls, opt_layers = build_transfer_vgg(input_tensor=images, num_class=class_count,
                                                       last_name=last_name)

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

        print "Learning model parameters:\n" \
              "\tmodel save path:\t{0}\n" \
              "\treport     rate:\t{1}\n" \
              "\tlearning   rate:\t{2}".format(model_path, report_rate, learning_rate)

    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vgg_output")
    var_to_train += opt_layers
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                             global_step=global_step_op,
                                                                             var_list=var_to_train,
                                                                             colocate_gradients_with_ops=True)
    print "Optimizing variables: {0}".format(var_to_train)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    hooks = [EndSavingHook(save_path=model_path, global_step=global_step_op)]
    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_path,
                                           hooks=hooks) as mon_sess:
        global_step = -1
        try:
            while not mon_sess.should_stop():
                _, global_step, current_cost = mon_sess.run([optimizer, global_step_op, cost])
                if global_step % report_rate == 0:
                    print "{0} steps passed with current cost: {1}.".format(global_step, current_cost)
        except tf.errors.OutOfRangeError:
            print "All images used in {0} steps.".format(global_step)
