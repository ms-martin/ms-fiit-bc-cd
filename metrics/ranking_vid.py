from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import os
import numpy as np

import models.model_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_of_ce as model
import dataprep.ilidsvid_rank_vid as dataset

model_name = 'model_rnn_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_of_ce'

sess = tf.InteractiveSession()

test_only = False

siamese = model.Siamese(training=True,
                        optical_flow=True,
                        augment=False,
                        margin=5,
                        batch_size=1,
                        seq_len=1,
                        num_of_persons=len(dataset.get_persons(False)))
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
confused_pairs = []
ranks = []

for person in persons:
    pairs = dataset.get_person_sequence_pairs(template=person,
                                              dense_optical_flow=siamese.optical_flow,
                                              seq_len=siamese.seq_len,
                                              test_only=test_only)

    for pair in pairs:
        images1 = np.reshape(pair.images1, (siamese.seq_len, 128 * 64 * siamese.channels))
        images2 = np.reshape(pair.images2, (siamese.seq_len, 128 * 64 * siamese.channels))
        distance = sess.run([siamese.distance], feed_dict={
            siamese.input1: images1,
            siamese.input2: images2
        })

        pair.set_distance(distance)

    pairs.sort(key=lambda x: x.distance)

    if not pairs[0].label:
        confused_pairs.append([pairs[0].images1_label, pairs[0].images2_label])

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
