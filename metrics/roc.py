from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import models.model_c64_c128_spp_d4096_o128 as model
import dataprep.ilidsvid_seq as dataset

model_path = 'model_c64_c128_spp_d4096_o128'

sess = tf.InteractiveSession()

siamese = model.Siamese(False, 80)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

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
    f1s.append((2*tp) / (2*tp + fp + fn))
    accs.append((tp + tn) / (tp + tn + fp + fn))


for i in range(len(distances)):
    print('%d : Match %d: distance %.3f' % (i, test_labels[i], distances[i]))

for i in range(len(thresholds)):
    print('Threshold: ' + str(thresholds[i]))
    print('TP: ' + str(true_positives[i]))
    print('TN: ' + str(true_negatives[i]))
    print('FP: ' + str(false_positives[i]))
    print('FN: ' + str(false_negatives[i]))

acc = max(accs)
f1 = max(f1s)
auc = np.trapz(true_positive_rates, x=false_positive_rates)
plt.plot(false_positive_rates, true_positive_rates, 'g', label=model_path)
plt.legend()
plt.annotate('max F1', xy=(false_positive_rates[f1s.index(f1)], true_positive_rates[f1s.index(f1)]), 
            xytext=(false_positive_rates[f1s.index(f1)] - 0.2, true_positive_rates[f1s.index(f1)] - 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
plt.axis([0, 1, 0, 1])
plt.xlabel('FPR (false positive rate)')
plt.ylabel('TPR (true positive rate)')
plt.savefig('results/roc_'+model_path+'.png')
plt.close()

with open("results/metrics_" + model_path + ".txt", "w") as file:
    file.write("F1: {0} at threshold {1}\n".format(f1, thresholds[f1s.index(f1)]))
    file.write("AUC: {0}\n".format(auc))
    file.write("ACC: {0}\n".format(acc))
