import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.patches as patches
import glob
import functools
from PIL import Image
from operator import itemgetter
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import argparse
from tensorflow.python.framework import ops
import glob
import xml.etree.ElementTree as ET
import random
from scipy.ndimage.filters import gaussian_filter
import copy
from PIL import ImageEnhance
"""
Function:
build keras mmod model and load weights.

Input:
input shape for network

Output:
saved keras weights file as "mmod.h5"
saved model as "mmod.hdf5"
"""

# construct keras model
# inputsize = None
# input_shape = (inputsize, inputsize,  1)
def rconv3_block(x, num_filters, use_bias=True):
    x = keras.layers.Conv2D(num_filters, (3,3), strides=(1,1), padding='same', use_bias=use_bias,
                            kernel_regularizer=keras.regularizers.l2(0.001)
                           )(x)
    x = keras.layers.BatchNormalization(epsilon=0.0001)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def rconv5d_block(x, num_filters, use_bias=True):
    x = keras.layers.Conv2D(num_filters, (5,5), strides=(2,2), padding='same', use_bias=use_bias,
                            kernel_regularizer=keras.regularizers.l2(0.001)
                           )(x)
    x = keras.layers.BatchNormalization(epsilon=0.0001)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def create_FD(input_shape, use_bias=True):
    num_feature = 46
    inputs = keras.layers.Input(input_shape)
    x = rconv5d_block(inputs, num_feature, use_bias)
    x = rconv5d_block(x, num_feature, use_bias)
    x = rconv5d_block(x, num_feature, use_bias)

    x = rconv3_block(x, num_feature, use_bias)
    x = rconv3_block(x, num_feature, use_bias)
    x = rconv3_block(x, num_feature, use_bias)
    x = rconv3_block(x, num_feature, use_bias)
    x = rconv3_block(x, num_feature, use_bias)
    output = keras.layers.Conv2D(1, (1,1), strides=(1, 1))(x)

    model = keras.models.Model(inputs=inputs, outputs=output)

    # model.summary()
    return model