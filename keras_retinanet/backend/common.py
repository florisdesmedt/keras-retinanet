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

import keras.backend
import keras_retinanet.backend

import numpy as np


def __bbox_transform_inv(boxes, deltas, batch_size = 1):
    shape_boxes = keras.backend.int_shape(boxes)


    if not shape_boxes[0]:
        return keras.backend.variable(np.zeros(shape=(batch_size,0,4)))

    max_boxes = 0
    P_boxes = []
    num_boxes = []

    for i in range(0, batch_size):

        _boxes = keras.backend.reshape(boxes[i], (-1, 4))
        _deltas = keras.backend.reshape(deltas[i], (-1, 4))

        widths = _boxes[:, 2] - _boxes[:, 0]
        heights = _boxes[:, 3] - _boxes[:, 1]
        ctr_x = _boxes[:, 0] + 0.5 * widths
        ctr_y = _boxes[:, 1] + 0.5 * heights

        dx = _deltas[:, 0]
        dy = _deltas[:, 1]
        dw = _deltas[:, 2]
        dh = _deltas[:, 3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = keras.backend.exp(dw) * widths
        pred_h = keras.backend.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)


        P_boxes.append(pred_boxes)
        max_boxes = max(max_boxes, keras.backend.int_shape(pred_boxes)[0])
        num_boxes.append(keras.backend.int_shape(pred_boxes)[0])


    p_batch = np.zeros(shape=(batch_size,max_boxes,4))

    for i in range(0, batch_size):
        p_batch[i][:num_boxes[i]] = keras.backend.eval(P_boxes[i])

    return keras.backend.variable(p_batch)


    # TODO fds: batch??
    # pred_boxes = keras.backend.expand_dims(pred_boxes, axis=0)

    return pred_boxes


def bbox_transform_inv(boxes, deltas):

    #boxes  = keras.backend.reshape(boxes, (-1, 4))
    #deltas = keras.backend.reshape(deltas, (-1, 4))

    widths = boxes[:,:, 2] - boxes[:,:, 0]
    heights = boxes[:,:, 3] - boxes[:,:, 1]
    ctr_x = boxes[:,:, 0] + 0.5 * widths
    ctr_y = boxes[:,:, 1] + 0.5 * heights

    dx = deltas[:,:, 0]
    dy = deltas[:,:, 1]
    dw = deltas[:,:, 2]
    dh = deltas[:,:, 3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = keras.backend.exp(dw) * widths
    pred_h = keras.backend.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)
    #pred_boxes = keras.backend.reshape(pred_boxes, b_shape)
    #    pred_boxes = keras.backend.expand_dims(pred_boxes, axis=0)

    return pred_boxes

def _bbox_transform_inv(boxes, deltas):
    b_shape = keras.backend.shape(boxes)

    boxes  = keras.backend.reshape(boxes, (-1, 4))
    deltas = keras.backend.reshape(deltas, (-1, 4))

    widths  = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x   = boxes[:, 0] + 0.5 * widths
    ctr_y   = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w     = keras.backend.exp(dw) * widths
    pred_h     = keras.backend.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h


    pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)
    pred_boxes = keras.backend.reshape(pred_boxes,b_shape)
#    pred_boxes = keras.backend.expand_dims(pred_boxes, axis=0)

    return pred_boxes


def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = keras_retinanet.backend.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
