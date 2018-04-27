from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import os
import numpy as np
import pickle

import models.model_c128_c256_d4096_o256 as model
import dataprep.ilidsvid_seq as dataset

model_name = 'model_c128_c256_d4096_o256'

sess = tf.InteractiveSession()

siamese = model.Siamese(True)
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

losses_agg = []
steps_agg = []

log_dir = './results/' + model_name

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

load = False
model_ckpt = './weights/' + model_name + '.ckpt.meta'
model_losses_pickle = './results/' + model_name + '/losses.pickle'
model_steps_pickle = './results/' + model_name + '/steps.pickle'

if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True
        if os.path.isfile(model_losses_pickle) and os.path.isfile(model_steps_pickle):
            with open(model_losses_pickle, 'rb') as fp:
                losses_agg = pickle.load(fp)
            with open(model_steps_pickle, 'rb') as fp:
                steps_agg = pickle.load(fp)

if load:
    saver.restore(sess, './weights/' + model_name + '.ckpt')

losses_window = []
avg_loss = 0

for step in range(1000000):
    batch_x1, batch_x2, batch_y = dataset.get_augmented_batch(20, True)

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
        losses_agg.append(loss_v)
        if not steps_agg:
            steps_agg.append(0)
        else:
            steps_agg.append(steps_agg[-1] + 10)

        with open('./results/' + model_name + '/losses.pickle', 'w+b') as fp:
            pickle.dump(losses_agg, fp)
        with open('./results/' + model_name + '/steps.pickle', 'w+b') as fp:
            pickle.dump(steps_agg, fp)

    if step % 100 == 0 and step > 0:
        saver.save(sess, './weights/' + model_name + '.ckpt')
