from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import os

import models.model_c64_c128_r512_d512_o128 as model
import dataprep.ilidsvid_vid as dataset
import numpy as np

model_path = 'model_rnn_c64_c128_r512_d512_o128'

sess = tf.InteractiveSession()

siamese = model.Siamese(training=True,
                        optical_flow=False,
                        batch_size=20,
                        seq_len=20,
                        hidden_size=512)
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

load = False
model_ckpt = './weights/' + model_path + '.ckpt.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True

if load:
    saver.restore(sess, './weights/' + model_path + '.ckpt')

losses_window = []
avg_loss = 0

for step in range(1000000):

    if siamese.batch_size > 1:
        batch_x1, batch_x2, batch_y, x1_label, x2_label = dataset.get_batch(training=siamese.training,
                                                                            optical_flow=siamese.optical_flow,
                                                                            augment=True,
                                                                            batch_size=siamese.batch_size,
                                                                            seq_len=siamese.seq_len)

    else:
        if step % 2 == 0:
            pair = dataset.get_positive_sequence_pair(training=siamese.training,
                                                      dense_optical_flow=siamese.optical_flow,
                                                      augment=True,
                                                      seq_len=siamese.seq_len)
        else:
            pair = dataset.get_negative_sequence_pair(training=siamese.training,
                                                      dense_optical_flow=siamese.optical_flow,
                                                      augment=True,
                                                      seq_len=siamese.seq_len)

        batch_x1 = pair.images1
        batch_x2 = pair.images2
        batch_y = pair.label

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
        siamese.input1: batch_x1,
        siamese.input2: batch_x2,
        siamese.labels: batch_y})

    losses_window.append(loss_v)

    if len(losses_window) > 1000:
        losses_window.remove(losses_window[0])

    if step % 10 == 0:
        avg_loss = np.asarray(losses_window)
        avg_loss = np.mean(avg_loss)
        print('step %d: avg_loss %.3f' % (step, avg_loss))

    if step % 100 == 0 and step > 0:
        saver.save(sess, './weights/' + model_path + '.ckpt')
