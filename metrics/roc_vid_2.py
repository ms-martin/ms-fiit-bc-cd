from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os
import models.model_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_of_ce as model5
import models.model_c16_c32_c32_spp_rBasicRnnNumPerson_atplScan_of_ce as model6
import models.model_c16_c32_c32_spp_rNumPerson_atplScan_of_ce as model7
import models.model_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_of_ce as model8
import models.model_c16_c32_c128_spp_rBasicRnnNumPerson_atplScan_of_ce as model9
import models.model_c16_c64_c64_spp_rBasicRnnNumPerson_atplScan_of_ce as model10
import dataprep.ilidsvid_vid as dataset

model_paths = ['model_rnn_c16_c32_c128_spp_rBasicRnnNumPerson_atplScan_ce',
               'model_rnn_c16_c32_c32_spp_rBasicRnnNumPerson_atplScan_ce',
               'model_rnn_c16_c32_c32_spp_rNumPerson_atplScan_ce',
               'model_rnn_c16_c32_c64_spp_rBasicRnnNumPerson_atplScan_ce',
               'model_rnn_c16_c32_c128_spp_rBasicRnnNumPerson_atplScan_ce',
               'model_rnn_c16_c64_c64_spp_rBasicRnnNumPerson_atplScan_of_ce'
               ]

siamese_nets = [model5,
                model6,
                model7,
                model8,
                model9,
                model10]

models_tprs = []
models_fprs = []
models_f1 = []
models_auc = []
models_acc = []

for model in range(len(model_paths)):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    siamese = siamese_nets[model].Siamese(training=False,
                                          optical_flow=True,
                                          augment=False,
                                          margin=5,
                                          batch_size=1,
                                          seq_len=20,
                                          num_of_persons=len(dataset.get_persons()))
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    model_ckpt = './weights/' + model_paths[model] + '.ckpt.meta'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, './weights/' + model_paths[model] + '.ckpt')
    else:
        print("File not found.")

    distances = []
    test_labels = []

    for i in range(1000):
        if siamese.batch_size > 1:
            x1_test, x2_test, sim_labels, x1_label, x2_label = dataset.get_batch(training=siamese.training,
                                                                                 optical_flow=siamese.optical_flow,
                                                                                 augment=siamese.augment,
                                                                                 batch_size=siamese.batch_size,
                                                                                 seq_len=siamese.seq_len)

        else:
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

        distances.append(distance)
        test_labels.append(sim_labels)

    print(np.shape(distances))
    distances = np.asarray(distances)
    distances = np.reshape(distances, (-1))
    test_labels = np.asarray(test_labels)
    test_labels = np.reshape(test_labels, (-1))
    num_of_thresholds = 2000
    max_dist = max(distances)
    threshold_step = max_dist / num_of_thresholds

    thresholds = []
    for i in range(num_of_thresholds + 1):
        thresholds.append(i * threshold_step)

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    true_positive_rates = []
    false_positive_rates = []
    f1s = []
    accs = []

    for i in thresholds:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(distances)):
            if int(test_labels[j]) == 0:
                if distances[j] > i:
                    tn += 1
                else:
                    fp += 1

            elif int(test_labels[j]) == 1:
                if distances[j] > i:
                    fn += 1
                else:
                    tp += 1

        true_positives.append(tp)
        true_negatives.append(tn)
        false_positives.append(fp)
        false_negatives.append(fn)
        true_positive_rates.append(tp / (tp + fn))
        false_positive_rates.append(fp / (fp + tn))
        f1s.append(2 * tp / ((2 * tp) + fp + fn))
        accs.append((tp + tn) / (tp + tn + fp + fn))

    models_tprs.append(true_positive_rates)
    models_fprs.append(false_positive_rates)
    models_f1.append(max(f1s))
    models_acc.append(max(accs))

    true_positive_rates = np.asarray(true_positive_rates, dtype=np.float32)
    false_positive_rates = np.asarray(false_positive_rates, dtype=np.float32)
    auc = np.trapz(true_positive_rates, x=false_positive_rates)
    models_auc.append(auc)

    plt.plot(false_positive_rates, true_positive_rates, c=np.random.rand(3, ), label=model_paths[model])
    plt.legend()

plt.plot([0, 1], [0, 1], 'r')
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR (false positive rate)')
plt.ylabel('TPR (true positive rate)')
plt.savefig('results/roc_vid_aggregated.png')
plt.close()

with open("results/rnn_aggregate_metrics.txt", "w") as file:
    for i in range(len(models_f1)):
        file.write("F1: {0}: {1}\n".format(model_paths[i], models_f1[i]))

    file.write("\n")

    for i in range(len(models_auc)):
        file.write("AUC: {0}: {1}\n".format(model_paths[i], models_auc[i]))

    file.write("\n")

    for i in range(len(models_auc)):
        file.write("ACC: {0}: {1}\n".format(model_paths[i], models_acc[i]))
