import argparse
import os

import keras
import keras.preprocessing.image

from keras_retinanet.models import ResNet50RetinaNet
from keras_retinanet.preprocessing import PascalVocIterator
from keras_retinanet.preprocessing import PascalVocIteratorBatch
import keras_retinanet

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
keras.backend.tensorflow_backend.set_session(get_session())


def create_model():
    image = keras.layers.Input((None, None, 3))
    return ResNet50RetinaNet(image, num_classes=21, weights='imagenet')


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for Pascal VOC object detection.')
    parser.add_argument('voc_path', help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).')
    parser.add_argument('batch_size', help='the batch size used during training and testing')
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # TODO fds: make batch a program argument
    batch_size = int(args.batch_size)
    print("The batch size is {}".format(batch_size))

    seed = 1 # make the order of the generator predictable, set to None for real random

    # create the model
    print('Creating model, this may take a second...')
    model = create_model()

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(loss={'regression': keras_retinanet.losses.regression_loss, 'classification': keras_retinanet.losses.focal_loss()}, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    # print model summary
    print(model.summary())

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        #horizontal_flip=True,

        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #zoom_range=0.1,
    )
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    import numpy as np
    # create a generator for training data
    train_generator = PascalVocIteratorBatch(
        args.voc_path,
        'trainval',
        train_image_data_generator,
        batch_size=1,
        seed=np.uint32(1)
    )

    # create a generator for training data
    train_generator_batch = PascalVocIteratorBatch(
        args.voc_path,
        'trainval',
        train_image_data_generator,
        batch_size=batch_size,
        seed=np.uint32(1)
    )

    import cv2
    # import time
    for i in range(0,100000):
    #    start_time = time.time()
        batch = train_generator.next()
        print("size of batch is: {}".format(len(batch[0])))
        cv2.imshow("image", batch[0][0])
        batch = train_generator_batch.next()
        for j in range(0,len(batch[0])):
            cv2.imshow("image2_" + str(j), batch[0][j])
        cv2.waitKey(0)
    #    print("duration is {}".format(time.time() - start_time))

    # create a generator for testing data
    # test_generator = PascalVocIterator(
    #     args.voc_path,
    #     'test',
    #     test_image_data_generator
    # )
    # #
    # # start training
    # #
    # model.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=len(train_generator.image_names) // batch_size,
    #     epochs=50,
    #     verbose=1,
    #     validation_data=test_generator,
    #     validation_steps=500,  # len(test_generator.image_names) // batch_size,
    #     callbacks=[
    #         keras.callbacks.ModelCheckpoint('snapshots/resnet50_voc_best.h5', monitor='val_loss', verbose=1, save_best_only=True),
    #         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
    #     ],
    # )
    # #
    # # # store final result too
    # model.save('snapshots/resnet50_voc_final.h5')
