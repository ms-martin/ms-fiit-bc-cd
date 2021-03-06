import tensorflow as tf


class Siamese:

    def __init__(self, training):
        self.input1 = tf.placeholder(tf.float32, [None, 24576])
        self.input2 = tf.placeholder(tf.float32, [None, 24576])

        with tf.variable_scope("siamese") as scope:
            self.out1 = self.convnet(self.input1, training)
            scope.reuse_variables()
            self.out2 = self.convnet(self.input2, training)

        self.labels = tf.placeholder(tf.float32, [None])
        self.loss = self.contrastive_loss()
        self.distance = self.euclidian_distance()

    def convnet(self, inputx, training):
        input_reshaped = tf.reshape(inputx, [-1, 128, 64, 3])
        conv1 = self.conv_layer(input_reshaped, [5, 5, 3, 32], [32], "conv1")

        max1 = tf.layers.max_pooling2d(inputs=conv1,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max1")

        conv2 = self.conv_layer(max1, [5, 5, 32, 64], [64], "conv2")

        max2 = tf.layers.max_pooling2d(inputs=conv2,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max2")

        conv3 = self.conv_layer(max2, [5, 5, 64, 128], [128], "conv3")

        max3 = tf.layers.max_pooling2d(inputs=conv3,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max3")

        max3flat = tf.reshape(max3, [-1, 16 * 8 * 128], name="max3flat")

        dense1 = self.fc_layer(max3flat, 1024, "dense1")

        dropout1 = tf.layers.dropout(inputs=dense1,
                                     rate=0.4,
                                     name="dropout1",
                                     training=training)

        dense2 = self.fc_layer(dropout1, 512, "dense2")

        dropout2 = tf.layers.dropout(inputs=dense2,
                                     rate=0.4,
                                     name="dropout2",
                                     training=training)

        out = self.fc_layer(dropout2, 3, "dense3")
        return out

    def conv_layer(self, inputx, kernel_shape, bias_shape, name):
        weights = tf.get_variable(name=name + "_weights",
                                  dtype=tf.float32,
                                  shape=kernel_shape,
                                  initializer=tf.random_normal_initializer)
        biases = tf.get_variable(name=name + "_biases",
                                 dtype=tf.float32,
                                 shape=bias_shape,
                                 initializer=tf.constant_initializer)
        conv = tf.nn.conv2d(input=inputx,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        return tf.nn.relu(conv + biases)

    def fc_layer(self, _input, units, name):
        assert len(_input.get_shape()) == 2
        n_prev_weight = _input.get_shape()[1]
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        weights = tf.get_variable(name=name + '_weights',
                                  dtype=tf.float32,
                                  shape=[n_prev_weight, units],
                                  initializer=initializer)
        biases = tf.get_variable(name=name + '_biases',
                                 dtype=tf.float32,
                                 shape=[units],
                                 initializer=tf.constant_initializer)
        return tf.nn.bias_add(tf.matmul(_input, weights), biases)

    def euclidian_distance(self):
        euclidian2 = tf.pow(tf.subtract(self.out1, self.out2), 2)
        euclidian2 = tf.reduce_sum(euclidian2, 1)
        return tf.sqrt(euclidian2 + 1e-6, name="distance")

    def contrastive_loss(self):
        margin = 5.0
        c = tf.constant(margin)
        labels_true = self.labels
        labels_false = tf.subtract(1.0, self.labels, name="1-y")
        euclidian = self.euclidian_distance()
        pos = tf.multiply(labels_true, euclidian, name="y_x_distance")
        neg = tf.multiply(labels_false, tf.pow(tf.maximum(tf.subtract(c, euclidian), 0), 2), name="ny_x_c-distance_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
