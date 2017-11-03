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

from keras_retinanet.models.resnet import OnlyResNet
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

def test_input():
    model = create_input()
    I = cv2.imread("Datasets/COCO/train2017/000000291597.jpg")
    I2 = cv2.imread("Datasets/COCO/val2017/000000212226.jpg")

    I = pad_image(I,(max(I.shape[0],I2.shape[0]),max(I.shape[1],I2.shape[1])))
    I2 = pad_image(I,(max(I.shape[0],I2.shape[0]),max(I.shape[1],I2.shape[1])))

    I_ = np.expand_dims(I,axis=0)
    I2_ = np.expand_dims(I2,axis=0)
    I_ = np.concatenate([I_, I2_])
    output = OnlyResNet(model)

    G = output.predict(I_)
    print(G[0].shape)
    print(G[1].shape)
    print(G[2].shape)
    print(G[3].shape)
    #cv2.imshow("Image", I)
    #cv2.waitKey(0)

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    batch_size = int(args.batchsize)
    epoch_scaling = int(args.epochscale)

    test_batchsize = batch_size

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    test_input()
