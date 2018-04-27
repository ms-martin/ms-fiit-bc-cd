from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import os
import models.model_c64_c128_d2048_o128 as model1
import models.model_c64_c128_d4096_o128 as model2
import models.model_c64_c128_d4096_o256 as model3
import models.model_c64_c128_d4096_o512 as model4

import dataprep.ilidsvid_seq as dataset

model_paths = ['model_c64_c128_d2048_o128',
               'model_c64_c128_d4096_o128',
               'model_c64_c128_d4096_o256',
               'model_c64_c128_d4096_o512'
               ]

siamese_nets = [model1,
                model2,
                model3,
                model4,
                ]

models_tprs = []
models_fprs = []
models_f1s = []
models_accs = []
models_aucs = []

for model in range(len(model_paths)):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    siamese = siamese_nets[model].Siamese(False)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    model_ckpt = './weights/' + model_paths[model] + '.ckpt.meta'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, './weights/' + model_paths[model] + '.ckpt')

    distances = []
    test_labels = []

    for i in range(100):
        x1_test, x2_test, y_test = dataset.get_batch(80, False)

        distance = sess.run([siamese.distance], feed_dict={
            siamese.input1: x1_test,
            siamese.input2: x2_test,
            siamese.labels: y_test})

        distances.append(distance)
        test_labels.append(y_test)

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
        f1s.append((2 * tp) / (2 * tp + fp + fn))
        accs.append((tp + tn) / (tp + tn + fp + fn))

    models_tprs.append(true_positive_rates)
    models_fprs.append(false_positive_rates)
    models_f1s.append(max(f1s))
    models_accs.append(max(accs))
    models_aucs.append(np.trapz(true_positive_rates, x=false_positive_rates))

for model in range(len(model_paths)):
    plt.plot(models_fprs[model], models_tprs[model], c=np.random.rand(3, ), label=model_paths[model])

plt.legend()
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR (false positive rate)')
plt.ylabel('TPR (true positive rate)')
plt.savefig('results/roc_aggregate_c64_c128.png')
plt.close()

with open("results/metrics_aggregate_c64_c128.txt", "w") as file:
    for model in range(len(model_paths)):
        file.write("{0}\n".format(model_paths[model]))
        file.write("F1: {0}\n".format(models_f1s[model]))
        file.write("AUC: {0}\n".format(models_aucs[model]))
        file.write("ACC: {0}\n".format(models_accs[model]))
        file.write("\n")
