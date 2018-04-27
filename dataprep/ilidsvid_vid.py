from scipy import misc
import numpy as np
import random
import os
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import matplotlib
#matplotlib.use('macosx')
from matplotlib import pyplot as plt


class VideoPair:
    def __init__(self, images1, images1_label, images2, images2_label, label):
        self.images1 = images1
        self.images1_label = images1_label
        self.images2 = images2
        self.images2_label = images2_label
        self.label = label


def get_persons():
    persons = os.listdir("./i-LIDS-VID/sequences/cam1")
    return sorted(persons)


def get_person_sequence(per, cam):
    sequence = os.listdir("./i-LIDS-VID/sequences/cam" + str(cam) + "/" + per)
    return sorted(sequence)


def get_positive_sequence_pair(training, dense_optical_flow, augment, seq_len):
    persons = get_persons()
    person_indices = []
    if training:
        person_indices = random.sample(range(0, int(len(persons) * 0.8)), 1)
    else:
        person_indices = random.sample(range(int(len(persons) * 0.8), len(persons)), 1)

    p1 = persons[person_indices[0]]

    si1 = random.randint(0, len(get_person_sequence(p1, 1)) - seq_len)
    si2 = random.randint(0, len(get_person_sequence(p1, 2)) - seq_len)

    images1 = [misc.imread(name="./i-LIDS-VID/sequences/cam1/" + p1 + "/" + img)
               for img in get_person_sequence(p1, 1)[si1:si1 + seq_len]]

    images2 = [misc.imread(name="./i-LIDS-VID/sequences/cam2/" + p1 + "/" + img)
               for img in get_person_sequence(p1, 2)[si2:si2 + seq_len]]

    if augment:
        if random.choice([True, False]):
            images1 = iaa.Fliplr(p=1.0).augment_images(images1)
            images2 = iaa.Fliplr(p=1.0).augment_images(images2)
        if random.choice([True, False]):
            images1 = iaa.Crop(px=4).augment_images(images1)
            images2 = iaa.Crop(px=4).augment_images(images2)

    if dense_optical_flow:
        images1 = get_dense_optical_flow(images1)
        images2 = get_dense_optical_flow(images2)

    images1 = [np.reshape(image, (-1)) for image in images1]
    images2 = [np.reshape(image, (-1)) for image in images2]

    images1 = np.asarray(images1, dtype=np.float32) / 255.0
    images2 = np.asarray(images2, dtype=np.float32) / 255.0

    seq_label = [1.0]

    return VideoPair(images1, np.asarray([person_indices[0]], dtype=np.int32),
                     images2, np.asarray([person_indices[0]], dtype=np.int32),
                     np.asarray(seq_label, dtype=np.float32))


def get_negative_sequence_pair(training, dense_optical_flow, augment, seq_len):
    persons = get_persons()
    person_indices = []
    if training:
        person_indices = random.sample(range(0, int(len(persons) * 0.8)), 2)
    else:
        person_indices = random.sample(range(int(len(persons) * 0.8), len(persons)), 2)

    p1 = persons[person_indices[0]]
    p2 = persons[person_indices[1]]

    si1 = random.randint(0, len(get_person_sequence(p1, 1)) - seq_len)
    si2 = random.randint(0, len(get_person_sequence(p2, 2)) - seq_len)

    images1 = [misc.imread(name="./i-LIDS-VID/sequences/cam1/" + p1 + "/" + img, mode='RGB')
               for img in get_person_sequence(p1, 1)[si1:si1 + seq_len]]

    images2 = [misc.imread(name="./i-LIDS-VID/sequences/cam2/" + p2 + "/" + img, mode='RGB')
               for img in get_person_sequence(p2, 2)[si2:si2 + seq_len]]

    if augment:
        if random.choice([True, False]):
            images1 = iaa.Fliplr(p=1.0).augment_images(images1)
            images2 = iaa.Fliplr(p=1.0).augment_images(images2)
        if random.choice([True, False]):
            images1 = iaa.Crop(px=4).augment_images(images1)
            images2 = iaa.Crop(px=4).augment_images(images2)

    if dense_optical_flow:
        images1 = get_dense_optical_flow(images1)
        images2 = get_dense_optical_flow(images2)

    images1 = [np.reshape(image, (-1)) for image in images1]
    images2 = [np.reshape(image, (-1)) for image in images2]

    images1 = np.asarray(images1, dtype=np.float32) / 255.0
    images2 = np.asarray(images2, dtype=np.float32) / 255.0

    seq_label = [0.0]

    return VideoPair(images1, np.asarray([person_indices[0]], dtype=np.int32),
                     images2, np.asarray([person_indices[1]], dtype=np.int32),
                     np.asarray(seq_label, dtype=np.float32))


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


def get_sparse_optical_flow(input_images):
    feature_params = dict(maxCorners=10,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in input_images]

    flows = []

    old_keypoints = cv2.goodFeaturesToTrack(images[0], mask=None, **feature_params)
    for i in range(1, len(images)):
        new_keypoints, state, err = cv2.calcOpticalFlowPyrLK(images[i - 1],
                                                             images[i],
                                                             old_keypoints,
                                                             None, **lk_params)

        good_new = new_keypoints[state == 1]
        good_old = old_keypoints[state == 1]
        if np.shape(good_new)[0] == 0 or i % 5 == 0:
            old_keypoints = cv2.goodFeaturesToTrack(images[i], mask=None, **feature_params)
        else:
            old_keypoints = good_new.reshape(-1, 1, 2)


def get_batch(training, optical_flow, augment, batch_size, seq_len):
    pairs = []

    for i in range(int(batch_size / 2)):
        pairs.append(get_positive_sequence_pair(training, optical_flow, augment, seq_len))

    for i in range(int(batch_size / 2)):
        pairs.append(get_negative_sequence_pair(training, optical_flow, augment, seq_len))

    random.shuffle(pairs)

    cam1_images = [pair.images1 for pair in pairs]
    cam1_images = np.asarray(cam1_images, dtype=np.float32)
    cam1_images = np.reshape(cam1_images, (batch_size * seq_len, -1))
    cam2_images = [pair.images2 for pair in pairs]
    cam2_images = np.asarray(cam2_images, dtype=np.float32)
    cam2_images = np.reshape(cam2_images, (batch_size * seq_len, -1))

    batch_labels = [pair.label for pair in pairs]
    batch_labels = np.asarray(batch_labels, dtype=np.float32)
    batch_labels = np.reshape(batch_labels, (-1))

    cam1_labels = [pair.images1_label for pair in pairs]
    cam1_labels = np.asarray(cam1_labels, dtype=np.int32)
    cam1_labels = np.reshape(cam1_labels, (-1))

    cam2_labels = [pair.images2_label for pair in pairs]
    cam2_labels = np.asarray(cam2_labels, dtype=np.int32)
    cam2_labels = np.reshape(cam2_labels, (-1))

    return cam1_images, cam2_images, batch_labels, cam1_labels, cam2_labels


def visualize_pair(pair):
    cam1_images = pair.images1
    cam1_images_labels = pair.images1_label

    cam2_images = pair.images2
    cam2_images_labels = pair.images2_label

    fig = plt.figure(figsize=(8, 8))

    col = 2
    row = 20

    for i in range(0, 20):
        sub1 = fig.add_subplot(col, row, i + 1)
        plt.imshow(np.reshape(cam1_images[i - 1], (128, 64, 3)))
        sub1.text(0.5, -0.1, cam1_images_labels, size=4, ha="center",
                  transform=sub1.transAxes)
        plt.axis('off')

        sub2 = fig.add_subplot(col, row, i + 1 + 20)
        plt.imshow(np.reshape(cam2_images[i - 1], (128, 64, 3)))
        sub2.text(0.5, -0.1, cam2_images_labels, size=4, ha="center",
                  transform=sub2.transAxes)
        plt.axis('off')

    plt.show()


#neg = get_negative_sequence_pair(True,False,True,20)
#visualize_pair(neg)
