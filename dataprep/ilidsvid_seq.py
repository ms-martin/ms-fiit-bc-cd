from scipy import misc
import numpy as np
import random
import os
from imgaug import augmenters as iaa
import imgaug as ia
from matplotlib import pyplot as plt

path = "./i-LIDS-VID/images/cam1"


class Pair:
    def __init__(self, image1, image1_label, image2, image2_label, label):
        self.image1 = image1
        self.image1_label = image1_label
        self.image2 = image2
        self.image2_label = image2_label
        self.label = label


def augmentor():
    seq = iaa.Sequential([
        iaa.Sometimes(0.4, iaa.CoarseDropout(p=0.2, size_percent=0.01))

    ])
    return iaa.OneOf([seq, iaa.Noop()])


def augment(images, height, width, channels, aug):
    images_aug = np.asarray(images, dtype=np.uint8)
    images_aug = np.reshape(images_aug, (-1, height, width, channels))

    images_aug = aug.augment_images(images_aug)
    images_aug = np.reshape(images_aug, (-1, height * width * channels))
    images_aug = np.asarray(images_aug, dtype=np.float32) / 255.0
    return images_aug


def get_persons():
    persons = os.listdir("./i-LIDS-VID/sequences/cam1")
    return sorted(persons)


def get_person_sequence(per, cam):
    sequence = os.listdir("./i-LIDS-VID/sequences/cam" + str(cam) + "/" + per)
    return sorted(sequence)


def get_positive_pair(training):
    persons = get_persons()
    person_indices = []
    if training:
        person_indices = random.sample(range(0, int(len(persons) * 0.8)), 1)
    else:
        person_indices = random.sample(range(int(len(persons) * 0.8), len(persons)), 1)

    p1 = persons[person_indices[0]]

    si1 = random.sample(range(len(get_person_sequence(p1, 1))), 1)
    si2 = random.sample(range(len(get_person_sequence(p1, 2))), 1)

    img1 = get_person_sequence(p1, 1)[si1[0]]
    img2 = get_person_sequence(p1, 2)[si2[0]]

    image1 = misc.imread(name="./i-LIDS-VID/sequences/cam1/" + p1 + "/" + img1)
    image1 = np.reshape(image1, (-1))
    image2 = misc.imread(name="./i-LIDS-VID/sequences/cam2/" + p1 + "/" + img2)
    image2 = np.reshape(image2, (-1))
    return Pair(image1, p1, image2, p1, 1.0)


def get_negative_pair(training):
    persons = get_persons()
    indices = []
    if training:
        indices = random.sample(range(0, int(len(persons) * 0.8)), 2)
    else:
        indices = random.sample(range(int(len(persons) * 0.8), len(persons)), 2)
    p1 = persons[indices[0]]
    p2 = persons[indices[1]]

    si1 = random.sample(range(len(get_person_sequence(p1, 1))), 1)
    si2 = random.sample(range(len(get_person_sequence(p2, 2))), 1)

    img1 = get_person_sequence(p1, 1)[si1[0]]
    img2 = get_person_sequence(p2, 2)[si2[0]]

    image1 = misc.imread(name="./i-LIDS-VID/sequences/cam1/" + p1 + "/" + img1)
    image1 = np.reshape(image1, (-1))
    image2 = misc.imread(name="./i-LIDS-VID/sequences/cam2/" + p2 + "/" + img2)
    image2 = np.reshape(image2, (-1))
    return Pair(image1, p1, image2, p2, 0.0)


def get_pairs(batch_size, training):
    pairs = []

    for i in range(int(batch_size / 2)):
        pairs.append(get_positive_pair(training))

    for i in range(int(batch_size / 2)):
        pairs.append(get_negative_pair(training))

    random.shuffle(pairs)
    return pairs


def get_batch(batch_size, training):
    pairs = get_pairs(batch_size, training)

    cam1_images = [pair.image1 for pair in pairs]
    cam1_images = np.asarray(cam1_images, dtype=np.float32) / 255.0
    cam2_images = [pair.image2 for pair in pairs]
    cam2_images = np.asarray(cam2_images, dtype=np.float32) / 255.0
    batch_labels = [pair.label for pair in pairs]
    batch_labels = np.asarray(batch_labels, dtype=np.float32)

    return cam1_images, cam2_images, batch_labels


def get_augmented_batch(batch_size, training):
    pairs = get_pairs(batch_size, training)

    cam1_images = [pair.image1 for pair in pairs]
    cam2_images = [pair.image2 for pair in pairs]
    det = augmentor().to_deterministic()
    cam1_images = augment(cam1_images, 128, 64, 3, det)
    cam2_images = augment(cam2_images, 128, 64, 3, det)
    batch_labels = [pair.label for pair in pairs]
    batch_labels = np.asarray(batch_labels, dtype=np.float32)

    cam1_images_labels = [pair.image1_label for pair in pairs]
    cam2_images_labels = [pair.image2_label for pair in pairs]

    # fig = plt.figure(figsize=(8, 8))
    #
    # col = 2
    # row = batch_size
    #
    # for i in range(0, batch_size):
    #     sub1 = fig.add_subplot(col, row, i + 1)
    #     plt.imshow(np.reshape(cam1_images[i - 1], (128, 64, 3)))
    #     sub1.text(0.5, -0.1, cam1_images_labels[i - 1], size=4, ha="center",
    #               transform=sub1.transAxes)
    #     plt.axis('off')
    #
    #     sub2 = fig.add_subplot(col, row, i + 1 + batch_size)
    #     plt.imshow(np.reshape(cam2_images[i - 1], (128, 64, 3)))
    #     sub2.text(0.5, -0.1, cam2_images_labels[i - 1], size=4, ha="center",
    #               transform=sub2.transAxes)
    #     plt.axis('off')
    #
    # plt.show()

    return cam1_images, cam2_images, batch_labels


def get_augmented_batch_image_labels(batch_size, training):
    pairs = get_pairs(batch_size, training)

    cam1_images = [pair.image1 for pair in pairs]
    cam2_images = [pair.image2 for pair in pairs]
    det = augmentor().to_deterministic()
    cam1_images = augment(cam1_images, 128, 64, 3, det)
    cam2_images = augment(cam2_images, 128, 64, 3, det)
    batch_labels = [pair.label for pair in pairs]
    batch_labels = np.asarray(batch_labels, dtype=np.float32)

    cam1_images_labels = [pair.image1_label for pair in pairs]
    cam2_images_labels = [pair.image2_label for pair in pairs]

    return cam1_images, cam2_images, batch_labels, cam1_images_labels, cam2_images_labels

# get_augmented_batch(10, True)
