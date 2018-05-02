from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt

import tensorflow as tf
import os
import numpy as np

import models.model_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_of_ce as model
import dataprep.ilidsvid_vid as dataset

model_name = 'model_rnn_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_ce'

sess = tf.InteractiveSession()

siamese = model.Siamese(training=False,
                        optical_flow=True,
                        augment=False,
                        margin=5,
                        batch_size=2,
                        seq_len=20,
                        num_of_persons=len(dataset.get_persons()))
saver = tf.train.Saver()
tf.global_variables_initializer().run()

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

batch_x1, batch_x2, batch_y, x2_labels, x1_labels = dataset.get_batch(training=siamese.training,
                                                                      optical_flow=siamese.optical_flow,
                                                                      batch_size=siamese.batch_size,
                                                                      seq_len=siamese.seq_len,
                                                                      augment=siamese.augment)

shape, mat = sess.run([tf.shape(siamese.in1_temp_mat_in2), siamese.in1_temp_mat_in2], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.seq1_labels: x1_labels,
    siamese.seq2_labels: x2_labels,
    siamese.similarity_labels: batch_y})

i = 1
plt.figure(figsize=(2, 1))
for image in mat:
    plt.subplot(2, 1, i)
    plt.axis('off')
    plt.imshow(image)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/activations_atpl_mat_' + model_name + '.png')
plt.close()

shape, mat = sess.run([tf.shape(siamese.rnn), siamese.rnn], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.seq1_labels: x1_labels,
    siamese.seq2_labels: x2_labels,
    siamese.similarity_labels: batch_y})

i = 1
plt.figure(figsize=(2, 1))
for image in mat:
    plt.subplot(2, 1, i)
    plt.axis('off')
    plt.imshow(image)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/activations_rnn_' + model_name + '.png')
plt.close()


shape, mat = sess.run([tf.shape(siamese.conv1), siamese.conv1], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.seq1_labels: x1_labels,
    siamese.seq2_labels: x2_labels,
    siamese.similarity_labels: batch_y})


filters = np.moveaxis(mat, 3, 1)
i = 1
plt.figure(figsize=(8, 2))
for image in filters[0]:
    plt.subplot(8, 2, i)
    plt.axis('off')
    plt.imshow(image)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/activations_conv1_' + model_name + '.png')
plt.close()

shape, mat = sess.run([tf.shape(siamese.conv2), siamese.conv2], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.seq1_labels: x1_labels,
    siamese.seq2_labels: x2_labels,
    siamese.similarity_labels: batch_y})


filters = np.moveaxis(mat, 3, 1)
i = 1
plt.figure(figsize=(4, 8))
for image in filters[0]:
    plt.subplot(4, 8, i)
    plt.axis('off')
    plt.imshow(image)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/activations_conv2_' + model_name + '.png')
plt.close()

shape, mat = sess.run([tf.shape(siamese.conv3), siamese.conv3], feed_dict={
    siamese.input1: batch_x1,
    siamese.input2: batch_x2,
    siamese.seq1_labels: x1_labels,
    siamese.seq2_labels: x2_labels,
    siamese.similarity_labels: batch_y})


filters = np.moveaxis(mat, 3, 1)
i = 1
plt.figure(figsize=(8, 8))
for image in filters[0]:
    plt.subplot(8, 8, i)
    plt.axis('off')
    plt.imshow(image)
    i += 1

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/activations_conv3_' + model_name + '.png')
plt.close()



weight_name = 'temp_mat:0'
var = [v for v in tf.trainable_variables() if v.name == weight_name][0]
var_val, var_shape = sess.run([var, tf.shape(var)])

var_val = np.reshape(var_val, tuple(var_shape))

minimum = np.min(np.reshape(var_val, (-1)), 0)
maximum = np.max(np.reshape(var_val, (-1)), 0)
normalized = (var_val - minimum) / (maximum - minimum)

plt.axis('off')
plt.imshow(normalized)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('results/weights_' + model_name + '_' + weight_name + '.png')
plt.close()


#
# filters = np.moveaxis(filters, 3, 1)
# i = 1
# plt.figure(figsize=(8, 8))
# for image in filters[0]:
#     plt.subplot(8, 8, i)
#     plt.axis('off')
#     plt.imshow(image)
#     i += 1
#
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.savefig('results/activations_' + model_name + '.png')
# plt.close()
