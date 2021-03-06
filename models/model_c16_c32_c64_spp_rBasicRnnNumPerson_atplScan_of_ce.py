import tensorflow as tf
import numpy as np


class Siamese:

    def __init__(self, training, optical_flow, augment, margin, batch_size, seq_len, num_of_persons):
        self.training = training
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.optical_flow = optical_flow
        self.margin = margin
        self.augment = augment
        self.channels = 5 if optical_flow else 3
        self.num_persons = num_of_persons
        self.hidden_size = num_of_persons
        self.input1 = tf.placeholder(tf.float32, [None, 128 * 64 * self.channels])
        self.input2 = tf.placeholder(tf.float32, [None, 128 * 64 * self.channels])

        with tf.variable_scope("siamese") as scope:
            self.leg_out1 = self.siamese_leg(self.input1)
            scope.reuse_variables()
            self.leg_out2 = self.siamese_leg(self.input2)

        self.out1, self.out2 = self.atpl_layer(self.leg_out1, self.leg_out2)
        self.similarity_labels = tf.placeholder(tf.float32, [None])
        self.seq1_labels = tf.placeholder(tf.int32, [None])
        self.seq2_labels = tf.placeholder(tf.int32, [None])

        self.seq1_labels_one_hot = tf.one_hot(indices=self.seq1_labels,
                                              depth=self.num_persons)

        self.seq2_labels_one_hot = tf.one_hot(indices=self.seq2_labels,
                                              depth=self.num_persons)

        self.seq1_cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.seq1_labels_one_hot,
                                                                       logits=self.out1)
        self.seq2_cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.seq2_labels_one_hot,
                                                                       logits=self.out2)

        self.contrastive_loss = self.contrastive_loss(self.out1, self.out2)

        self.loss = tf.add_n([self.seq1_cross_entropy_loss,
                              self.seq2_cross_entropy_loss,
                              self.contrastive_loss])

        self.distance = self.euclidian_distance(self.out1, self.out2)

    def siamese_leg(self, input_x):
        input_reshaped = tf.reshape(input_x, [self.batch_size * self.seq_len, 128, 64, self.channels])
        self.conv1 = self.conv_layer(input_reshaped, [5, 5, self.channels, 16], [16], "conv1")

        max1 = tf.layers.max_pooling2d(inputs=self.conv1,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max1")

        self.conv2 = self.conv_layer(max1, [5, 5, 16, 32], [32], "conv2")

        max2 = tf.layers.max_pooling2d(inputs=self.conv2,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name="max2")

        self.conv3 = self.conv_layer(max2, [5, 5, 32, 64], [64], "conv3")

        self.spp = self.spp_layer(self.conv3, [8, 4, 2, 1], "spp")

        self.rnn = self.rnn_layers(self.spp, self.hidden_size)
        self.rnn = tf.reshape(self.rnn, [self.batch_size, self.seq_len, self.hidden_size], name="rnn_flat")
        return self.rnn

    def spp_layer(self, input_, levels, name):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(name):
            pool_outputs = []
            for level in levels:
                kernel = [1,
                          np.ceil(shape[1] * 1.0 / level).astype(np.int32),
                          np.ceil(shape[2] * 1.0 / level).astype(np.int32),
                          1]
                stride = [1,
                          np.floor(shape[1] * 1.0 / level).astype(np.int32),
                          np.floor(shape[2] * 1.0 / level).astype(np.int32),
                          1]
                poll = tf.nn.max_pool(value=input_,
                                      ksize=kernel,
                                      strides=stride,
                                      padding='SAME',
                                      name="spp_pool")
                pool_outputs.append(tf.reshape(poll, [shape[0], -1]))
            spp_pool = tf.concat(pool_outputs, 1)
            spp_pool = tf.reshape(spp_pool, [self.seq_len, self.batch_size, -1])
            spp_pool = tf.unstack(spp_pool)

        return spp_pool

    def atpl_layer(self, input1, input2):
        temp_mat = tf.get_variable(name="temp_mat",
                                   shape=[self.hidden_size, self.hidden_size],
                                   dtype=tf.float32,
                                   initializer=tf.random_normal_initializer)

        # in1_temp_mat = tf.scan(lambda a, x: tf.matmul(x, temp_mat), input1)
        in1_temp_mat = tf.matmul(tf.reshape(input1, [self.batch_size * self.seq_len, self.hidden_size]), temp_mat)
        in1_temp_mat = tf.reshape(in1_temp_mat, [self.batch_size, self.seq_len, self.hidden_size])
        self.in1_temp_mat_in2 = tf.matmul(in1_temp_mat, input2, transpose_b=True)

        self.atpl_mat = tf.tanh(self.in1_temp_mat_in2, name="atpl_mat")

        max_col = tf.reduce_max(self.atpl_mat, axis=1, name="max_col")
        max_row = tf.reduce_max(self.atpl_mat, axis=2, name="max_row")

        col_softmax = tf.nn.softmax(max_col)
        row_softmax = tf.nn.softmax(max_row)

        out1 = tf.squeeze(tf.matmul(input1, tf.expand_dims(col_softmax, -1), transpose_a=True))
        out2 = tf.squeeze(tf.matmul(input2, tf.expand_dims(row_softmax, -1), transpose_a=True))

        out1 = tf.reshape(out1, [self.batch_size, -1])
        out2 = tf.reshape(out2, [self.batch_size, -1])
        return out1, out2

    def rnn_layers(self, inputs, hidden_size):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
        outputs, _ = tf.nn.static_rnn(cell=cell,
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
        euclidian2 = tf.reduce_sum(euclidian2, len(euclidian2.get_shape()) - 1)
        return tf.sqrt(euclidian2 + 1e-6, name="distance")

    def contrastive_loss(self, out1, out2):
        c = tf.constant(self.margin, dtype=tf.float32)
        labels_true = self.similarity_labels
        labels_false = tf.subtract(1.0, self.similarity_labels, name="1-y")
        euclidian = self.euclidian_distance(out1, out2)
        pos = tf.multiply(labels_true, euclidian, name="y_x_distance")
        neg = tf.multiply(labels_false, tf.pow(tf.maximum(tf.subtract(c, euclidian), 0), 2),
                          name="ny_x_c-distance_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
