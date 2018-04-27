from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import os
import numpy as np

import models.model_c64_c128_spp_d4096_o128 as model
import dataprep.ilidsvid_rank as dataset

model_name = 'model_c64_c128_spp_d4096_o128'

sess = tf.InteractiveSession()

test_only = False

siamese = model.Siamese(False, 1)
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

persons = dataset.get_persons(test_only)

ranks_histogram = np.zeros(shape=[len(persons)], dtype=np.int32)
ranks = []
confused_pairs = []

for person in persons:
    pairs = dataset.get_person_pairs(person, test_only)

    for pair in pairs:
        image1 = np.reshape(pair.image1, (1, 24576))
        image2 = np.reshape(pair.image2, (1, 24576))
        distance = sess.run([siamese.distance], feed_dict={
            siamese.input1: image1,
            siamese.input2: image2
        })
        pair.set_distance(distance)
    pairs.sort(key=lambda x: x.distance)

    if not pairs[0].label:
        confused_pairs.append([pairs[0].image1_label, pairs[0].image2_label])

    rank = 0
    while not pairs[rank].label:
        rank += 1
    ranks_histogram[rank] += 1
    ranks.append(rank)
    print(person, rank)

for i in range(0, len(ranks_histogram) - 1):
    ranks_histogram[i + 1] += ranks_histogram[i]

ranks_percents = ranks_histogram / len(persons)
ranks = np.asarray(ranks, dtype=np.float32)
avg_rank = np.mean(ranks)

with open("results/ranking_" + model_name + "_" + str(test_only) + ".txt", "w") as file:
    file.write("top 1: {0}\n".format(ranks_percents[0]))
    file.write("top 2: {0}\n".format(ranks_percents[1]))
    file.write("top 3: {0}\n".format(ranks_percents[2]))
    file.write("top 4: {0}\n".format(ranks_percents[3]))
    file.write("top 5: {0}\n".format(ranks_percents[4]))
    file.write("top 10: {0}\n".format(ranks_percents[9]))
    file.write("top 20: {0}\n".format(ranks_percents[19]))
    file.write('\n')

with open("results/confused_pairs_" + model_name + "_" + str(test_only) + ".txt", "w") as file:
    for pair in confused_pairs:
        file.write("{0}, {1} \n".format(pair[0], pair[1]))
