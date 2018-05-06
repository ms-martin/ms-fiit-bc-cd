from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

# import system things
import tensorflow as tf
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import models.model_c16_c32_c32_spp_rBasicRnnNumPerson_atplScan_of_ce as model
import dataprep.ilidsvid_vid as dataset


def show_pair(_pair, _seq_len, _channels, _pair_id, _model_name, state):
    cam1_images = _pair.images1
    cam1_images_labels = _pair.images1_label

    cam2_images = _pair.images2
    cam2_images_labels = _pair.images2_label

    fig = plt.figure(figsize=(8, 8))

    col = 2
    row = _seq_len

    for i in range(0, _seq_len):
        sub1 = fig.add_subplot(col, row, i + 1)
        plt.imshow(np.reshape(cam1_images[i - 1], (128, 64, _channels))[:128, :64, 2:5])
        sub1.text(0.5, -0.1, cam1_images_labels, size=4, ha="center",
                  transform=sub1.transAxes)
        plt.axis('off')

        sub2 = fig.add_subplot(col, row, i + 1 + _seq_len)
        plt.imshow(np.reshape(cam2_images[i - 1], (128, 64, _channels))[:128, :64, 2:5])
        sub2.text(0.5, -0.1, cam2_images_labels, size=4, ha="center",
                  transform=sub2.transAxes)
        plt.axis('off')

    plt.savefig('results/' + _model_name + '/classification/' + state + '_match_' + str(_pair_id) + '.png')
    plt.close()


model_path = 'model_rnn_c16_c32_c32_spp_rBasicRnnNumPerson_atplScan_ce'

sess = tf.InteractiveSession()

siamese = model.Siamese(training=False,
                        optical_flow=True,
                        augment=False,
                        margin=5,
                        batch_size=1,
                        seq_len=10,
                        num_of_persons=len(dataset.get_persons()))
saver = tf.train.Saver()
tf.global_variables_initializer().run()

log_dir = './results/' + model_path + '/classification'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

threshold = 4.4279

pair_id = 0

for i in range(100):

    if i % 2:
        pair = dataset.get_positive_sequence_pair(training=siamese.training,
                                                  dense_optical_flow=siamese.optical_flow,
                                                  augment=siamese.augment,
                                                  seq_len=siamese.seq_len)
    else:
        pair = dataset.get_negative_sequence_pair(training=siamese.training,
                                                  dense_optical_flow=siamese.optical_flow,
                                                  augment=siamese.augment,
                                                  seq_len=siamese.seq_len)

    x1_test = pair.images1
    x2_test = pair.images2
    sim_labels = pair.label
    x1_label = pair.images1_label
    x2_label = pair.images2_label

    distance = sess.run([siamese.distance], feed_dict={
        siamese.input1: x1_test,
        siamese.input2: x2_test,
        siamese.similarity_labels: sim_labels,
        siamese.seq1_labels: x1_label,
        siamese.seq2_labels: x2_label})

    if int(sim_labels) == 0:
        if distance > i:
            show_pair(pair, siamese.seq_len, siamese.channels, pair_id, model_path, 'true_negative')
        else:
            show_pair(pair, siamese.seq_len, siamese.channels, pair_id, model_path, 'false_positive')

    elif int(sim_labels) == 1:
        if distance > i:
            show_pair(pair, siamese.seq_len, siamese.channels, pair_id, model_path, 'false_negative')
        else:
            show_pair(pair, siamese.seq_len, siamese.channels, pair_id, model_path, 'true_positive')
