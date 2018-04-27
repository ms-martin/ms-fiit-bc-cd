from scipy import misc
import numpy as np
import random
import os
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt

path = "./i-LIDS-VID/images/cam1"


class Pair:
    def __init__(self, image1, image1_label, image2, image2_label, label):
        self.image1 = image1
        self.image1_label = image1_label
        self.image2 = image2
        self.image2_label = image2_label
        self.label = label


def augment(images, height, width, channels):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.4, iaa.Crop(percent=(0, 0.2))),
        iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0.5, 3))),
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=(-30, 30),
            shear=(-10, 10),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        ))
    ])
    images_aug = np.asarray(images, dtype=np.uint8)
    images_aug = np.reshape(images_aug, (-1, height, width, channels))
    images_aug = seq.augment_images(images_aug)
    images_aug = np.reshape(images_aug, (-1, height * width * channels))
    images_aug = np.asarray(images_aug, dtype=np.float32) / 255.0
    return images_aug


def get_persons():
    return os.listdir("./i-LIDS-VID/images/cam1")


def get_positive_pair(training):
    persons = get_persons()
    indices = []
    if training:
        indices = random.sample(range(0, int(len(persons) * 0.8)), 1)
    else:
        indices = random.sample(range(int(len(persons) * 0.8), len(persons)), 1)

    p1 = persons[indices[0]]

    image1 = misc.imread(name="./i-LIDS-VID/images/cam1/" + p1 + "/cam1_" + p1 + ".png")
    image1 = np.reshape(image1, (-1))
    image2 = misc.imread(name="./i-LIDS-VID/images/cam2/" + p1 + "/cam2_" + p1 + ".png")
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

    image1 = misc.imread(name="./i-LIDS-VID/images/cam1/" + p1 + "/cam1_" + p1 + ".png")
    image1 = np.reshape(image1, (-1))
    image2 = misc.imread(name="./i-LIDS-VID/images/cam2/" + p2 + "/cam2_" + p2 + ".png")
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
    cam1_images = augment(cam1_images, 128, 64, 3)
    cam2_images = augment(cam2_images, 128, 64, 3)
    batch_labels = [pair.label for pair in pairs]
    batch_labels = np.asarray(batch_labels, dtype=np.float32)

    # cam1_images_labels = [pair.image1_label for pair in pairs]
    # cam2_images_labels = [pair.image2_label for pair in pairs]
    #
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


def get_test_images():
    persons = get_persons()
    images = []

    for i in range(100):
        p = persons[i]
        image = misc.imread(name="./i-LIDS-VID/images/cam1/" + p + "/cam1_" + p + ".png")
        images.append(np.reshape(image, (-1)))

    images = np.asarray(images, dtype=np.float32) / 255.0

    return images


def get_test_labels():
    persons = get_persons()
    test_labels = []

    for i in range(100):
        test_labels.append(int(persons[i][-3:]))

    return np.asarray(test_labels, dtype=np.float32)


get_augmented_batch(10, True)
