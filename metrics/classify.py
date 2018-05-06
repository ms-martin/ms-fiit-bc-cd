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
import models.model_c64_c128_d4096_o128 as model
import dataprep.ilidsvid_seq as dataset


def show_pair(_pair, _pair_id, _model_name, state):
    plot_id = 1
    pair_image1 = _pair.image1
    pair_image2 = _pair.image2

    pair_image1 = np.reshape(pair_image1, (128, 64, 3))
    pair_image2 = np.reshape(pair_image2, (128, 64, 3))
    plt.figure(figsize=(1, 2))

    plt.subplot(1, 2, plot_id)
    plt.axis('off')
    plt.imshow(pair_image1)
    plt.text(0.5, -0.1, _pair.image1_label, size=4, ha="center")

    plot_id += 1

    plt.subplot(1, 2, plot_id)
    plt.axis('off')
    plt.imshow(pair_image2)
    plt.text(0.5, -0.1, _pair.image2_label, size=4, ha="center")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(
        'results/' + _model_name + '/classification/' + state + '_match_' + str(pair_id) + '_' + _model_name + '.png')
    plt.close()


model_path = 'model_c64_c128_d4096_o128'

sess = tf.InteractiveSession()

siamese = model.Siamese(training=False)
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

treshold = 4.4153

pair_id = 0

for i in range(100):

    if i % 2:
        pair = dataset.get_positive_pair(training=siamese.training)
    else:
        pair = dataset.get_negative_pair(training=siamese.training)

    x1_test = np.reshape(pair.image1, (1,24576)) / 255.0
    x2_test = np.reshape(pair.image2, (1,24576)) / 255.0
    sim_labels = np.reshape(pair.label, (1))
    x1_label = pair.image1_label
    x2_label = pair.image2_label

    distance = sess.run([siamese.distance], feed_dict={
        siamese.input1: x1_test,
        siamese.input2: x2_test,
        siamese.labels: sim_labels})

    distance = distance[0]
    print(distance)

    if sim_labels[0] == 0:
        if distance > treshold:
            show_pair(pair, pair_id, model_path, 'true_negative')
        else:
            show_pair(pair, pair_id, model_path, 'false_positive')

    elif sim_labels[0] == 1:
        if distance > treshold:
            show_pair(pair, pair_id, model_path, 'false_negative')
        else:
            show_pair(pair, pair_id, model_path, 'true_positive')

    pair_id += 1
