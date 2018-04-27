from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import matplotlib

matplotlib.use('MacOSX')
from matplotlib import pyplot as plt

import tensorflow as tf
import os
import numpy as np
import pickle

import models.model_f7_c64_c128_d4096_o256 as model
import dataprep.ilidsvid_seq as dataset

model_name = 'model_f7_c64_c128_d4096_o256'

sess = tf.InteractiveSession()

siamese = model.Siamese(False)
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

losses_agg = []
steps_agg = []

log_dir = './results/' + model_name
model_losses_pickle = './results/' + model_name + '/losses.pickle'
model_steps_pickle = './results/' + model_name + '/steps.pickle'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if os.path.isfile(model_losses_pickle) and os.path.isfile(model_steps_pickle):
    print("Loaded pickles")
    with open(model_losses_pickle, 'rb') as fp:
        losses_agg = pickle.load(fp)
    with open(model_steps_pickle, 'rb') as fp:
        steps_agg = pickle.load(fp)

print(losses_agg)

plt.plot(steps_agg, losses_agg, 'r')
plt.show()

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

batch_size = 2

batch_x1, batch_x2, batch_y = dataset.get_batch(batch_size, False)

shape, filters = sess.run([tf.shape(siamese.conv2), siamese.conv2], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.labels: batch_y})

var = [v for v in tf.trainable_variables() if v.name == 'siamese/conv1_weights:0'][0]
var_val, var_shape = sess.run([var, tf.shape(var)])

var_val = np.reshape(var_val, tuple(var_shape))
var_val = np.moveaxis(var_val, 3, 0)

i = 1
for filter_weights in var_val:
    minimum = np.min(np.reshape(filter_weights, (-1)), 0)
    maximum = np.max(np.reshape(filter_weights, (-1)), 0)
    normalized_filter = (filter_weights - minimum) / (maximum - minimum)

    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.imshow(normalized_filter)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

# filters = np.moveaxis(filters, 3, 1)
# i = 1
# plt.figure(figsize=(16, 8))
# for image in filters[0]:
#     plt.subplot(16, 8, i)
#     plt.axis('off')
#     plt.imshow(image)
#     i += 1
#
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()
