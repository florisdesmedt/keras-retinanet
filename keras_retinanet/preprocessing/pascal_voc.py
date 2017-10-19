from __future__ import division

import keras.applications.imagenet_utils
import keras.preprocessing.image
import keras.backend
from .anchors import anchors_for_image, anchor_targets

from .image import random_transform_batch, resize_image

import cv2
import xml.etree.ElementTree as ET

import os
import numpy as np
import time

voc_classes = {
    'aeroplane'      : 0,
    'bicycle'        : 1,
    'bird'           : 2,
    'boat'           : 3,
    'bottle'         : 4,
    'bus'            : 5,
    'car'            : 6,
    'cat'            : 7,
    'chair'          : 8,
    'cow'            : 9,
    'diningtable'    : 10,
    'dog'            : 11,
    'horse'          : 12,
    'motorbike'      : 13,
    'person'         : 14,
    'pottedplant'    : 15,
    'sheep'          : 16,
    'sofa'           : 17,
    'train'          : 18,
    'tvmonitor'      : 19
}


class PascalVocIterator(keras.preprocessing.image.Iterator):
    def __init__(
        self,
        data_dir,
        set_name,
        image_data_generator,
        classes=voc_classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        image_min_side=600,
        image_max_side=1024,
        batch_size=1,
        shuffle=True,
        seed=None,
    ):
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip() for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_data_generator = image_data_generator
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        if seed is None:
            seed = np.uint32(time.time() * 1000)

        assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        super(PascalVocIterator, self).__init__(len(self.image_names), batch_size, shuffle, seed)

    def parse_annotations(self, filename):
        boxes = np.zeros((0, 5), dtype=keras.backend.floatx())

        tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename + '.xml'))
        root = tree.getroot()

        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)

        for o in root.iter('object'):
            if int(o.find('truncated').text) and self.skip_truncated:
                continue

            if int(o.find('difficult').text) and self.skip_difficult:
                continue

            box = np.zeros((1, 5), dtype=keras.backend.floatx())

            class_name = o.find('name').text
            if class_name not in self.classes:
                raise Exception('Class name "{}" not found in classes "{}"'.format(class_name, self.classes))
            box[0, 4] = self.classes[class_name]

            bndbox = o.find('bndbox')
            box[0, 0] = float(bndbox.find('xmin').text) - 1
            box[0, 1] = float(bndbox.find('ymin').text) - 1
            box[0, 2] = float(bndbox.find('xmax').text) - 1
            box[0, 3] = float(bndbox.find('ymax').text) - 1

            boxes = np.append(boxes, box, axis=0)

        return boxes

    def next(self):
        # lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        # transformation of images is not under thread lock so it can be done in parallel
        boxes_batch = np.zeros((batch_size, 0, 5), dtype=keras.backend.floatx())

        for batch_index, image_index in enumerate(selection):
            path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image, image_scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

            # set ground truth boxes
            boxes = np.expand_dims(self.parse_annotations(self.image_names[image_index]), axis=0)
            boxes_batch = np.append(boxes_batch, boxes, axis=1)

            # scale the ground truth boxes to the selected image scale
            boxes_batch[batch_index, :, :4] *= image_scale

            # convert to batches (currently only batch_size = 1 is allowed)
            image_batch = np.expand_dims(image, axis=0).astype(keras.backend.floatx())

            # randomly transform images and boxes simultaneously
            image_batch, boxes_batch = random_transform_batch(image_batch, boxes_batch, self.image_data_generator)

            # generate the label and regression targets
            labels, regression_targets = anchor_targets(image, boxes_batch[0], len(self.classes))
            regression_targets         = np.append(regression_targets, labels, axis=1)

            # convert target to batch (currently only batch_size = 1 is allowed)
            regression_batch = np.expand_dims(regression_targets, axis=0)
            labels_batch     = np.expand_dims(labels, axis=0)

        # convert the image to zero-mean
        image_batch = keras.applications.imagenet_utils.preprocess_input(image_batch)
        image_batch = self.image_data_generator.standardize(image_batch)

        return image_batch, [regression_batch, labels_batch]
