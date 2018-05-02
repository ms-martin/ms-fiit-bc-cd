from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
import os
import numpy as np

import models.model_c16_c32_d512_o3 as model
import dataprep.ilidsvid_seq as dataset

model_name = 'model_c16_c32_d512_o3'

sess = tf.InteractiveSession()

siamese = model.Siamese(False)
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

image_labels = []
points = []

persons = dataset.get_persons()
classes = {}

for person in persons:
    classes[person] = []

for _ in range(10):
    batch_x1, batch_x2, batch_y, x1_labels, x2_labels = dataset.get_augmented_batch_image_labels(2, False)

    out1 = sess.run([siamese.out1], feed_dict={
        siamese.input1: batch_x1,
        siamese.input2: batch_x2,
        siamese.labels: batch_y})

    image_labels.append(x1_labels)
    points.append(out1)

points = np.reshape(points, newshape=(-1, 3))
image_labels = np.reshape(image_labels, newshape=(-1))


agg = []

for _ in range(len(points)):
    agg.append([image_labels[_],points[_]])

for entry in agg:
    classes[entry[0]].append(entry[1])

print(classes)

x = []
y = []
z = []

for point in points:
    x.append(point[0])
    y.append(point[1])
    z.append(point[2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for key in classes.keys():
    ax.scatter(x, y, z)


plt.savefig('results/embed_' + model_name + '.png')
plt.close()
