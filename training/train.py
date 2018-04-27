from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import os

import models.model_c64_c128_d4096_o256 as model
import dataprep.ilidsvid_seq as dataset

sess = tf.InteractiveSession()

siamese = model.Siamese(True)
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

model_name = "model_c64_c128_d4096_o256"

load = False
model_ckpt = './weights/' + model_name + '.ckpt.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True


if load:
    saver.restore(sess, './weights/' + model_name + '.ckpt')

for step in range(1000000):
    batch_x1, batch_x2, batch_y = dataset.get_batch(80, True)

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
        siamese.input1: batch_x1,
        siamese.input2: batch_x2,
        siamese.labels: batch_y})

    if step % 10 == 0:
        print('step %d: loss %.3f' % (step, loss_v))

    if step % 100 == 0 and step > 0:
        saver.save(sess, './weights/' + model_name + '.ckpt')
