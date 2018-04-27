from scipy import misc
import numpy as np
import random
import os
import cv2
import math
import matplotlib.pyplot as plt


class VideoPair:
    def __init__(self, images1, images1_label, images2, images2_label, label):
        self.images1 = images1
        self.images1_label = images1_label
        self.images2 = images2
        self.images2_label = images2_label
        self.label = label
        self.distance = math.inf

    def set_distance(self, distance):
        self.distance = distance


def get_persons(test_only):
    persons = os.listdir("./i-LIDS-VID/sequences/cam1")
    if test_only:
        return sorted(persons)[int(len(persons) * 0.8):len(persons)]
    else:
        return sorted(persons)


def get_person_sequence(per, cam):
    sequence = os.listdir("./i-LIDS-VID/sequences/cam" + str(cam) + "/" + per)
    return sorted(sequence)


def get_person_sequence_pairs(template, dense_optical_flow, seq_len, test_only):
    persons = get_persons(test_only)

    images1 = [misc.imread(name="./i-LIDS-VID/sequences/cam1/" + template + "/" + img, mode='RGB')
               for img in get_person_sequence(template, 1)[0:seq_len]]

    if dense_optical_flow:
        images1 = get_dense_optical_flow(images1)

    images1 = [np.reshape(image, (-1)) for image in images1]
    images1 = np.asarray(images1, dtype=np.float32) / 255.0

    pairs = []

    for person in persons:
        images2 = [misc.imread(name="./i-LIDS-VID/sequences/cam2/" + person + "/" + img, mode='RGB')
                   for img in get_person_sequence(person, 2)[0:seq_len]]

        if dense_optical_flow:
            images2 = get_dense_optical_flow(images2)

        images2 = [np.reshape(image, (-1)) for image in images2]
        images2 = np.asarray(images2, dtype=np.float32) / 255.0
        seq_label = np.asarray([1.0 if template == person else 0.0], dtype=np.float32)
        pairs.append(VideoPair(images1, persons.index(template), images2, persons.index(person), seq_label))

    return pairs


def get_dense_optical_flow(input_images):
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in input_images]
    flows = []

    for i in range(len(images)):
        flows.append(cv2.calcOpticalFlowFarneback(images[0 if i - 1 < 0 else i - 1],
                                                  images[i],
                                                  None, 0.5, 3, 15, 3, 5, 1.1, 0))

    mags = []
    angs = []

    for flow in flows:
        mags.append(cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX))
        angs.append(cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX))

    mags = np.asarray(mags, dtype=np.uint8)
    angs = np.asarray(angs, dtype=np.uint8)
    flows = np.stack([mags, angs], axis=3)
    return np.concatenate((flows, input_images), axis=3)
