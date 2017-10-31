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

import keras_retinanet

import cv2

import os
import numpy as np
import time

from pycocotools.coco import COCO

from .anchors import anchors_for_image, anchor_targets


class CocoIterator(keras.preprocessing.image.Iterator):
    def __init__(
        self,
        data_dir,
        set_name,
        image_data_generator,
        num_classes=90,
        image_min_side=600,
        image_max_side=1024,
        batch_size=1,
        shuffle=True,
        seed=None,
    ):
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.coco                 = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids            = self.coco.getImgIds()
        self.image_data_generator = image_data_generator
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side
        self.num_classes          = num_classes

        if seed is None:
            seed = np.uint32(time.time() * 1000)

        self.load_classes()

        #assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        super(CocoIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.classes = {}
        for c in categories:
            self.classes[c['name']] = c['id'] - 1 # start from 0

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def load_image(self, image_indeces):

        batch_size = len(image_indeces)

        temp_image_storage = []
        temp_boxes_storage = []

        max_image_width = 0
        max_image_height = 0

        max_boxes = 0
        valid_boxes = []
        for image_index in image_indeces:
            coco_image         = self.coco.loadImgs(self.image_ids[image_index])[0]
            path               = os.path.join(self.data_dir, self.set_name, coco_image['file_name'])

            image              = cv2.imread(path, cv2.IMREAD_COLOR)
            image, image_scale = keras_retinanet.preprocessing.image.resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

            # set ground truth boxes
            annotations_ids = self.coco.getAnnIds(imgIds=coco_image['id'], iscrowd=False)

            # some images appear to miss annotations (like image with id 257034)
            if len(annotations_ids) == 0:
                return None

            # parse annotations
            annotations = self.coco.loadAnns(annotations_ids)
            boxes       = np.zeros((0, 5), dtype=keras.backend.floatx())
            for idx, a in enumerate(annotations):
                box        = np.zeros((1, 5), dtype=keras.backend.floatx())
                box[0, :4] = a['bbox']
                box[0, 4]  = a['category_id'] - 1 # start from 0
                boxes      = np.append(boxes, box, axis=0)

            # transform from [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # scale the ground truth boxes to the selected image scale
            boxes[:, :4] *= image_scale
            temp_image_storage.append(image)
            temp_boxes_storage.append(boxes)
            max_image_width = max(max_image_width, image.shape[1])
            max_image_height = max(max_image_height, image.shape[0])
            max_boxes = max(max_boxes, boxes.shape[0])
            valid_boxes.append(boxes.shape[0])


        # convert to batches (currently only batch_size = 1 is allowed)
        #image_batch   = np.expand_dims(image.astype(keras.backend.floatx()), axis=0)
        image_batch   = np.zeros(shape=(batch_size,max_image_height,max_image_width,3),dtype=keras.backend.floatx())
        boxes_batch   = np.ones(shape=(batch_size,max_boxes,temp_boxes_storage[0].shape[1])) * -1


        # pad the images
        for i in range(0,batch_size):
            # pad image
            top_padding = 0
            left_padding = 0
            right_padding = max_image_width - temp_image_storage[i].shape[1]
            bottom_padding = max_image_height - temp_image_storage[i].shape[0]
            image_batch[i] = cv2.copyMakeBorder(temp_image_storage[i], top_padding, bottom_padding, left_padding, right_padding,
                                       cv2.BORDER_CONSTANT, 0)
            boxes_batch[i][:temp_boxes_storage[i].shape[0]] = temp_boxes_storage[i]


        # print("Boxes: {}".format(boxes_batch))

        # randomly transform images and boxes simultaneously
        image_batch, boxes_batch = keras_retinanet.preprocessing.image.random_transform_batch(image_batch, boxes_batch, self.image_data_generator)


        for i in range(0,batch_size):
            # generate the label and regression targets
            labels, regression_targets = anchor_targets(image_batch[i], boxes_batch[i], self.num_classes, valid_boxes=valid_boxes[i])
            regression_targets = np.append(regression_targets, labels, axis=1)

            if i == 0:
                regression_batch = np.ndarray(shape=(batch_size,regression_targets.shape[0],regression_targets.shape[1]), dtype=keras.backend.floatx())
                labels_batch = np.ndarray(shape=(batch_size,labels.shape[0], labels.shape[1]))

            # convert target to batch (currently only batch_size = 1 is allowed)
            regression_batch[i] = regression_targets
            labels_batch[i]     = labels

        # convert the image to zero-mean
        image_batch = keras_retinanet.preprocessing.image.preprocess_input(image_batch)
        image_batch = self.image_data_generator.standardize(image_batch)

        # for i in range(0,batch_size):
        #     print("Size of regression: {}".format(regression_batch[i].shape[0]))
        #     print("boxes {} {}".format(i,regression_batch[i]))
        #     cv2.imshow("Image " + str(i), image_batch[i]/255.0)
        #
        # cv2.waitKey(0)

        return {
       #     'image'            : image,
       #     'image_scale'      : image_scale,
       #     'coco_index'       : coco_image['id'],
            'boxes_batch'      : boxes_batch,
            'image_batch'      : image_batch,
            'regression_batch' : regression_batch,
            'labels_batch'     : labels_batch,
        }


    def next(self):
        # lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        #assert(batch_size == 1), "Currently only batch_size=1 is allowed."
        #assert(len(selection) == 1), "Currently only batch_size=1 is allowed."

        image_data = self.load_image(selection)

        if image_data is None:
            return self.next()

        return image_data['image_batch'], [image_data['regression_batch'], image_data['labels_batch']]
