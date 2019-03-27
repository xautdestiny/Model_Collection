# -*- coding: utf-8 -*-

#import cv2
import numpy as np
import copy

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Lambda, GlobalAveragePooling2D, Add
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import Model
# from keras import initializations
from keras.engine import Layer, InputSpec
from keras import backend as K

import sys

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    # x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False, padding='same')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = Add(name='res' + str(stage) + block)([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    # x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False, padding='same')(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = Add(name='res' + str(stage) + block)([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet101_model(input_shape, weights_path=None):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=input_shape, name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    # x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding="same")(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    
    # x_fc = AveragePooling2D((7, 7), strides=1, name='avg_pool')(x)
    # x_fc = Flatten()(x_fc)
    x_fc = GlobalAveragePooling2D(name='avg_pool')(x)


    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)
    
    # load weights
    # if weights_path:
    #   model.load_weights(weights_path, by_name=True)

    return model

def create_resnet101(input_shape=(224, 224, 3)):
    model = resnet101_model(input_shape)
    return model

if __name__ == '__main__':
    model = resnet101_model((480, 640, 3))
    model.summary()
    model.save("resnet101_640w_480h_avgpool(stride=1).hdf5")


  # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)

  # # Remove train image mean
  # im[:,:,0] -= 103.939
  # im[:,:,1] -= 116.779
  # im[:,:,2] -= 123.68

  # # Transpose image dimensions (Theano uses the channels as the 1st dimension)
  # if K.image_dim_ordering() == 'th':
  #   im = im.transpose((2,0,1))

  #   # Use pre-trained weights for Theano backend
  #   weights_path = 'resnet101_weights_th.h5'
  # else:
  #   # Use pre-trained weights for Tensorflow backend
  #   weights_path = 'resnet101_weights_tf.h5'

  # # Insert a new dimension for the batch_size
  # im = np.expand_dims(im, axis=0)
  
  # # Test pretrained model
  # model = resnet101_model(weights_path)
  # sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
  # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  # out = model.predict(im)
  # print np.argmax(out)