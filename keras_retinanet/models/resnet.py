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

import keras
import keras_resnet.models
import keras_retinanet.models

from keras_retinanet.models.retinanet import __create_pyramid_features as create_pyramid_features
from keras_retinanet.models.retinanet import __build_pyramid
from keras_retinanet.models.retinanet import default_submodels, AnchorParameters, __build_anchors, retinanet
import numpy as np

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def ResNet50RetinaNet(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    model = keras_retinanet.models.retinanet_bbox(inputs=inputs, backbone=resnet,batch_size=batch_size, *args, **kwargs)
    model.load_weights(weights_path, by_name=True)

    return model


def OnlyResNet(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    return resnet


def OnlyResNetSubmodels():

    num_classes = 10
    anchor_parameters = AnchorParameters(
        sizes   = [32, 64, 128, 256, 512],
        strides = [8, 16, 32, 64, 128],
        ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    )

    submodels = default_submodels(num_classes, anchor_parameters)

    return submodels


def OnlyResNetPyramidFeatures(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    num_classes = 10
    anchor_parameters = AnchorParameters(
        sizes   = [32, 64, 128, 256, 512],
        strides = [8, 16, 32, 64, 128],
        ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    )

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)
    _, C3, C4, C5 = resnet.output  # we ignore C2
    pyramid_features = create_pyramid_features(C3, C4, C5)

    return keras.models.Model(inputs=inputs, outputs=pyramid_features)


def OnlyResNetAnchors(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    num_classes = 10
    anchor_parameters = AnchorParameters(
        sizes   = [32, 64, 128, 256, 512],
        strides = [8, 16, 32, 64, 128],
        ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    )

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)
    _, C3, C4, C5 = resnet.output  # we ignore C2
    pyramid_features = create_pyramid_features(C3, C4, C5)


    anchors = __build_anchors(anchor_parameters, pyramid_features)

    return keras.models.Model(inputs=inputs, outputs=anchors)

def OnlyResNetRetina(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    return retinanet(inputs=inputs, num_classes=10,batch_size=3,backbone=resnet, *args, **kwargs)

def OnlyResNetRegression(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)
    retina = retinanet(inputs=inputs, num_classes=10,batch_size=3,backbone=resnet, *args, **kwargs)

    predictions, anchors = retina.outputs
    regression = keras.layers.Lambda(lambda x: x[:, :, :4], name='regression')(predictions)


    return keras.models.Model(inputs=inputs, outputs=[regression])

def OnlyResNetPyramid(inputs, weights='imagenet',batch_size=1, *args, **kwargs):
    image = inputs

    num_classes = 10
    anchor_parameters = AnchorParameters(
        sizes   = [32, 64, 128, 256, 512],
        strides = [8, 16, 32, 64, 128],
        ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
    )

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    #model.load_weights(weights_path, by_name=True)

    _, C3, C4, C5 = resnet.output  # we ignore C2
    pyramid_features = create_pyramid_features(C3, C4, C5)

    submodels = default_submodels(num_classes, anchor_parameters)

    pyramid = __build_pyramid(submodels, pyramid_features)

    return keras.models.Model(inputs=inputs, outputs=pyramid)