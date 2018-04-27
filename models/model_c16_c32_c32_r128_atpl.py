import tensorflow as tf


class Siamese:

    def __init__(self, training, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = 128
        self.input1 = tf.placeholder(tf.float32, [None, 24576])
        self.input2 = tf.placeholder(tf.float32, [None, 24576])

        with tf.variable_scope("siamese") as scope:
            self.leg_out1 = self.siamese_leg(self.input1, training)
            scope.reuse_variables()
            self.leg_out2 = self.siamese_leg(self.input2, training)

        self.out1, self.out2 = self.atpl_layer(self.leg_out1, self.leg_out2)
        self.labels = tf.placeholder(tf.float32, [None])
        self.loss = self.contrastive_loss(self.out1, self.out2)
        self.distance = self.euclidian_distance(self.out1, self.out2)

    def siamese_leg(self, inputx, training):
        input_reshaped = tf.reshape(inputx, [-1, 128, 64, 3])
        conv1 = self.conv_layer(input_reshaped, [5, 5, 3, 16], [16], "conv1")

        max1 = tf.layers.max_pooling2d(inputs=conv1,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max1")

        conv2 = self.conv_layer(max1, [5, 5, 16, 32], [32], "conv2")

        max2 = tf.layers.max_pooling2d(inputs=conv2,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max2")

        conv3 = self.conv_layer(max2, [5, 5, 32, 32], [32], "conv3")

        flatten = tf.reshape(conv3, [self.batch_size, self.seq_len, 32 * 16 * 32], name="max3flat")

        rnn = self.rnn_layers(flatten, 1, self.hidden_size)
        rnn = tf.reshape(rnn, [self.batch_size, self.seq_len, self.hidden_size], name="rnn_flat")
        return rnn

    def atpl_layer(self, input1, input2):
        temp_mat = tf.get_variable(name="temp_mat",
                                   shape=[self.batch_size, self.hidden_size, self.hidden_size],
                                   dtype=tf.float32,
                                   initializer=tf.random_normal_initializer)

        in1_temp_mat = tf.matmul(input1, temp_mat, name="in1_temp_mat")
        in1_temp_mat_in2 = tf.matmul(in1_temp_mat, input2, transpose_b=True, name="in1_temp_mat_in2")
        atpl_mat = tf.tanh(in1_temp_mat_in2, name="atpl_mat")

        max_row = tf.reduce_max(atpl_mat, axis=1, name="max_row")
        max_col = tf.reduce_max(atpl_mat, axis=2, name="max_col")
        return max_row, max_col

    def rnn_layers(self, inputs, num_layers, hidden_size):
        rnn_cells = []

        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,
                                                state_is_tuple=True)
            rnn_cells.append(cell)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells,
                                                 state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(cell=multi_cell,
                                       inputs=inputs,
                                       dtype=tf.float32)

        return outputs

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

    def euclidian_distance(self, out1, out2):
        euclidian2 = tf.pow(tf.subtract(out1, out2), 2)
        euclidian2 = tf.reduce_sum(euclidian2, 1)
        return tf.sqrt(euclidian2 + 1e-6, name="distance")

    def contrastive_loss(self, out1, out2):
        margin = 5.0
        c = tf.constant(margin)
        labels_true = self.labels
        labels_false = tf.subtract(1.0, self.labels, name="1-y")
        euclidian = self.euclidian_distance(out1, out2)
        pos = tf.multiply(labels_true, euclidian, name="y_x_distance")
        neg = tf.multiply(labels_false, tf.pow(tf.maximum(tf.subtract(c, euclidian), 0), 2), name="ny_x_c-distance_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
