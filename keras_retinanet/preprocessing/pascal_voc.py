"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
    '__background__' : 0,
    'aeroplane'      : 1,
    'bicycle'        : 2,
    'bird'           : 3,
    'boat'           : 4,
    'bottle'         : 5,
    'bus'            : 6,
    'car'            : 7,
    'cat'            : 8,
    'chair'          : 9,
    'cow'            : 10,
    'diningtable'    : 11,
    'dog'            : 12,
    'horse'          : 13,
    'motorbike'      : 14,
    'person'         : 15,
    'pottedplant'    : 16,
    'sheep'          : 17,
    'sofa'           : 18,
    'train'          : 19,
    'tvmonitor'      : 20
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

    def load_image(self, image_index):
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image, image_scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

        # set ground truth boxes
        boxes_batch = np.zeros((1, 0, 5), dtype=keras.backend.floatx())
        boxes       = np.expand_dims(self.parse_annotations(self.image_names[image_index]), axis=0)
        boxes_batch = np.append(boxes_batch, boxes, axis=1)

        # scale the ground truth boxes to the selected image scale
        boxes_batch[0, :, :4] *= image_scale

        # convert to batches (currently only batch_size = 1 is allowed)
        image_batch = np.expand_dims(image, axis=0).astype(keras.backend.floatx())

        # randomly transform images and boxes simultaneously
        image_batch, boxes_batch = random_transform_batch(image_batch, boxes_batch, self.image_data_generator)

        # generate the label and regression targets
        labels, regression_targets = anchor_targets(image, boxes_batch[0])
        regression_targets         = np.append(regression_targets, labels, axis=1)

        # convert target to batch (currently only batch_size = 1 is allowed)
        regression_batch = np.expand_dims(regression_targets, axis=0)
        labels_batch     = np.expand_dims(labels, axis=0)

        # convert the image to zero-mean
        image_batch = keras.applications.imagenet_utils.preprocess_input(image_batch)
        image_batch = self.image_data_generator.standardize(image_batch)

        return {
            'image'            : image,
            'image_scale'      : image_scale,
            'boxes_batch'      : boxes_batch,
            'image_batch'      : image_batch,
            'regression_batch' : regression_batch,
            'labels_batch'     : labels_batch,
        }

    def next(self):
        # lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        assert(batch_size == 1), "Currently only batch_size=1 is allowed."
        assert(len(selection) == 1), "Currently only batch_size=1 is allowed."

        # transformation of images is not under thread lock so it can be done in parallel
        image_data = self.load_image(selection[0])

        return image_data['image_batch'], [image_data['regression_batch'], image_data['labels_batch']]


class PascalVocIteratorBatch(keras.preprocessing.image.Iterator):
    def     __init__(
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

        #assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        super(PascalVocIteratorBatch, self).__init__(len(self.image_names), batch_size, shuffle, seed)

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

    def load_image(self, image_indeces):

        batch_size = len(image_indeces)

        # TODO fds: is this some value defined somewhere?
        max_annotations = 300
        # set ground truth boxes
        boxes_batch = np.zeros((batch_size, max_annotations, 5), dtype=keras.backend.floatx())

        max_width = 0
        max_height = 0

        temp_image_container = []
        valid_boxes = []
        for index in range(0,batch_size) :
            b = image_indeces[index]
            path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[b] + self.image_extension)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image, image_scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)
            # add the image to the temp list (required to find largest dimentions in batch for padding)
            temp_image_container.append(image)
            max_width = max(max_width, image.shape[1])
            max_height = max(max_height, image.shape[0])

            boxes = self.parse_annotations(self.image_names[b])
            # # scale the ground truth boxes to the selected image scale
            boxes[:, :4] *= image_scale
            valid_boxes.append(boxes.shape[0])
            boxes_batch[index,0:boxes.shape[0],:] = boxes

        image_batch = np.ndarray(shape=(batch_size,max_height,max_width,3),dtype=keras.backend.floatx())

        for index in range(0,batch_size):
            image = temp_image_container[index].astype(keras.backend.floatx())
            # pad image
            top_padding = 0
            left_padding = 0
            right_padding = max_width - image.shape[1]
            bottom_padding = max_height - image.shape[0]
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                       cv2.BORDER_CONSTANT, 0)

            # add image to batch
            image_batch[index] = image

        # randomly transform images and boxes simultaneously
        image_batch, boxes_batch = random_transform_batch(image_batch, boxes_batch, self.image_data_generator)

        for index in range(0,batch_size):
            image = image_batch[index]
            # generate the label and regression targets
            labels, regression_targets = anchor_targets(image, boxes_batch[index],valid_boxes=valid_boxes[index])
            regression_targets         = np.append(regression_targets, labels, axis=1)

            if index == 0:
                regression_batch = np.ndarray(
                    shape=(batch_size, regression_targets.shape[0], regression_targets.shape[1]),
                    dtype=keras.backend.floatx())
                labels_batch = np.ndarray(shape=(batch_size, labels.shape[0], labels.shape[1]),
                                          dtype=keras.backend.floatx())

            # convert target to batch (currently only batch_size = 1 is allowed)
            regression_batch[index] = regression_targets
            labels_batch[index]     = labels

        # convert the image to zero-mean
        image_batch = keras.applications.imagenet_utils.preprocess_input(image_batch)
        image_batch = self.image_data_generator.standardize(image_batch)

        return {
            'image'            : image,
            'image_scale'      : image_scale,
            'boxes_batch'      : boxes_batch,
            'image_batch'      : image_batch,
            'regression_batch' : regression_batch,
            'labels_batch'     : labels_batch,
        }

    def next(self):
        # lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        # transformation of images is not under thread lock so it can be done in parallel
        image_data = self.load_image(selection)

        return image_data['image_batch'], [image_data['regression_batch'], image_data['labels_batch']]
