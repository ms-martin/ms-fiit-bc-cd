from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

sub_folders = [f.path for f in os.scandir('./results') if f.is_dir()]

for sub_folder in sub_folders:

    sub_folder = sub_folder[len('./results/'):]

    print(sub_folder)

    losses_agg = []
    steps_agg = []

    model_losses_pickle = './results/' + sub_folder + '/losses.pickle'
    model_steps_pickle = './results/' + sub_folder + '/steps.pickle'

    if os.path.isfile(model_losses_pickle) and os.path.isfile(model_steps_pickle):
        print("Loaded pickles")
        with open(model_losses_pickle, 'rb') as fp:
            losses_agg = pickle.load(fp)
        with open(model_steps_pickle, 'rb') as fp:
            steps_agg = pickle.load(fp)

    if len(losses_agg) == len(steps_agg):
        plt.plot(steps_agg, losses_agg, c=np.random.rand(3, ), label=sub_folder, linewidth=0.5)

plt.axis([0, 80000, 0, 600])
plt.legend()
plt.xlabel('Training steps [-]')
plt.ylabel('Loss [-]')
plt.savefig('results/losses_agg.png')
plt.close()
