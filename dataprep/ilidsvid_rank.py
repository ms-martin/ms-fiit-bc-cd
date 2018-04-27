from scipy import misc
import numpy as np
import random
import os
import math

path = "./i-LIDS-VID/images/cam1"


class Pair:
    def __init__(self, image1, person1, image2, person2, label):
        self.image1 = image1
        self.image2 = image2
        self.image1_label = person1
        self.image2_label = person2
        self.label = label
        self.distance = math.inf

    def set_distance(self, dist):
        self.distance = dist


def get_persons(test_only):
    persons = os.listdir("./i-LIDS-VID/sequences/cam1")
    if test_only:
        return sorted(persons)[int(len(persons) * 0.8):len(persons)]
    else:
        return sorted(persons)


def get_person_pairs(input_person, test_only):
    persons = get_persons(test_only)

    image1 = misc.imread(name="./i-LIDS-VID/images/cam1/" + input_person + "/cam1_" + input_person + ".png")
    image1 = np.reshape(image1, (-1)) / 255.0

    pairs = []
    for person in persons:
        image2 = misc.imread(name="./i-LIDS-VID/images/cam2/" + person + "/cam2_" + person + ".png")
        image2 = np.reshape(image2, (-1)) / 255.0
        pairs.append(Pair(image1, input_person, image2, person, True if person == input_person else False))

    return pairs


def get_all_pairs(test_only):
    persons = get_persons(test_only)

    all_pairs = []

    for person1 in persons:
        person_pair = []
        for person2 in persons:
            image1 = misc.imread(name="./i-LIDS-VID/images/cam1/" + person1 + "/cam1_" + person1 + ".png")
            image1 = np.reshape(image1, (-1)) / 255.0
            image2 = misc.imread(name="./i-LIDS-VID/images/cam2/" + person2 + "/cam2_" + person2 + ".png")
            image2 = np.reshape(image2, (-1)) / 255.0
            person_pair.append(Pair(image1, person1, image2, person2, person1 == person2))
        all_pairs.append(person_pair)

    return all_pairs
