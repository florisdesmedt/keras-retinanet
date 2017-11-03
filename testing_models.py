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

import argparse
import os

import keras
import keras.preprocessing.image

import tensorflow as tf

import sys
sys.path.append("/projects/keras-retinanet")

import cv2

from keras_retinanet.models.resnet import OnlyResNet, OnlyResNetPyramid, OnlyResNetSubmodels, OnlyResNetPyramidFeatures, OnlyResNetAnchors
#from keras_retinanet.preprocessing import CocoIterator
#import keras_retinanet
#from keras_retinanet import losses

import numpy as np


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(weights='imagenet', batch_size=1):
    image = keras.layers.Input((None, None, 3))
    return ResNet50RetinaNet(image, num_classes=90, weights=weights,batch_size=batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('coco_path', help='Path to COCO directory (ie. /tmp/COCO).')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--batchsize', help='Batch size', default=1)
    parser.add_argument('--epochscale', help='divide the epoch to increase amount of validation points', default=1)

    return parser.parse_args()

def pad_image(image,padded_shape):
    top_padding = 0
    left_padding = 0
    right_padding = padded_shape[1] - image.shape[1]
    bottom_padding = padded_shape[0] - image.shape[0]
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                        cv2.BORDER_CONSTANT, 0)

    return padded_image

def create_input():
    image = keras.layers.Input((None, None, 3))
    return image

def create_batch(image_path_list):
    images_list = []
    max_width = 0
    max_height = 0
    for f in image_path_list:
        I = cv2.imread(f)
        max_width = max(max_width, I.shape[1])
        max_height = max(max_height, I.shape[0])
        images_list.append(I)
    # pad images
    for index,I in enumerate(images_list):
        I = pad_image(I, (max_height, max_width))
        images_list[index] = np.expand_dims(I,axis=0)


    return np.concatenate(images_list,axis=0)

def create_mod_resnet():
    model = create_input()
    output = OnlyResNet(model)
    return output

def create_mod_pyramid():
    model = create_input()
    output = OnlyResNetPyramid(model)
    return output

def create_mod_anchors():
    model = create_input()
    output = OnlyResNetAnchors(model)
    return output


def create_mod_pyramidfeatures():
    model = create_input()
    output = OnlyResNetPyramidFeatures(model)
    return output

def create_temp_batch():
    images = ["Datasets/COCO/train2017/000000291597.jpg", "Datasets/COCO/val2017/000000212226.jpg"]
    #images = ["Datasets/COCO/train2017/000000291597.jpg"]
    return create_batch(images)

def test_input():
    print("test the input resnet calculation")
    output = create_mod_resnet()

    image_batch = create_temp_batch()

    G = output.predict(image_batch)
    print(G[0].shape)
    print(G[1].shape)
    print(G[2].shape)
    print(G[3].shape)


def test_with_pyramidfeatures():
    print("test the pyramid features")
    output = create_mod_pyramidfeatures()

    image_batch = create_temp_batch()

    G = output.predict(image_batch)
    for i in range(0,len(G)):
        print(G[i].shape)

def test_with_pyramid():
    print("test the pyramid")
    output = create_mod_pyramid()

    image_batch = create_temp_batch()

    G = output.predict(image_batch)
    for i in range(0,len(G)):
        print(G[i].shape)


def test_with_anchors():
    print("test the anchors")
    output = create_mod_anchors()

    image_batch = create_temp_batch()

    G = output.predict(image_batch)
    for i in range(0,len(G)):
        print(G[i].shape)



def test_submodels():
    print("test the submodels")

    G = OnlyResNetSubmodels()
    print(G)

if __name__ == '__main__':
    test_with_anchors()

    test_submodels()
    #test_input()
    test_with_pyramidfeatures()
    test_with_pyramid()
