import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
from keras import backend as K

#### crelu activation layer
def CRelu(x):
    def crelu(x_):
        pos = K.relu(x_)
        neg = K.relu(-x_)
        return K.concatenate([pos, neg], axis=-1)
    x = keras.layers.Lambda(crelu, output_shape=None)(x)
    return x

#### relu activation layer
def Relu(x):
#     x = channel_wise_scale_bias(x)
    x = keras.layers.Activation('relu')(x)
    return x

#### a composition of convolution, batchnormalization, activation
def conv_bn_act(x, num_filters, kernel_size, strides, act='relu'):
    x = keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    
    if act == 'relu':
        x = Relu(x)
        return x
    else:
        x = CRelu(x)
        return x
    
#### a CRelu_block as defined in paper
def CRelu_block(x, num_filters_list, kernel_size, strides):
    x_proj = keras.layers.Conv2D(num_filters_list[2], (1,1), strides=strides, padding='same')(x)

    x = conv_bn_act(x, num_filters_list[0], (1,1), strides=(1,1), act='relu')
    x = conv_bn_act(x, num_filters_list[1], kernel_size, strides=strides, act='crelu')
    x = conv_bn_act(x, num_filters_list[2], (1,1), strides=(1,1), act='relu')
    
    ### sum up residual
    x = keras.layers.Add()([x_proj, x])
    x = keras.layers.BatchNormalization()(x)
    x = Relu(x)
    return x

#### a Inception_block as defined in paper
def Inception_block(x, num_filters_list, strides, max_pool=False):
    x_proj = keras.layers.Conv2D(num_filters_list[4][0], (1,1), strides=strides, padding='same')(x)
    
    x_1x1 = conv_bn_act(x, num_filters_list[0][0], (1,1), strides=strides, act='relu')
    
    x_3x3 = conv_bn_act(x, num_filters_list[1][0], (1,1), strides=strides, act='relu')
    x_3x3 = conv_bn_act(x_3x3, num_filters_list[1][1], (3,3), strides=(1,1), act='relu')
    
    x_5x5 = conv_bn_act(x, num_filters_list[2][0], (1,1), strides=strides, act='relu')
    x_5x5 = conv_bn_act(x_5x5, num_filters_list[2][1], (3,3), strides=(1,1), act='relu')
    x_5x5 = conv_bn_act(x_5x5, num_filters_list[2][2], (3,3), strides=(1,1), act='relu')
    
    if max_pool:
        assert(strides == (2,2))
        x_maxpool = keras.layers.MaxPool2D(pool_size=(3,3), strides=strides, padding='same')(x)
        x_maxpool = conv_bn_act(x_maxpool, num_filters_list[3][0], (1,1), strides=(1,1))
        x_cat = keras.layers.Concatenate(axis=-1)([x_1x1, x_3x3, x_5x5, x_maxpool])
    else:
        x_cat = keras.layers.Concatenate(axis=-1)([x_1x1, x_3x3, x_5x5])
    
    x_cat = conv_bn_act(x_cat, num_filters_list[4][0], (1,1), strides=(1,1), act='relu')
    
    ### sum up residual
    x = keras.layers.Add()([x_proj, x_cat])
    x = keras.layers.BatchNormalization()(x)
    x = Relu(x)
    return x
    

#### building model
def create_pvanet(input_shape=(480, 640, 3)):
# input_shape = (640, 480,  3)
    inputs = keras.layers.Input(input_shape)

    conv1_1 = conv_bn_act(inputs, 16, (7,7), strides=(2,2), act='crelu')
    pool1_1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1_1)

    conv2_1 = CRelu_block(pool1_1, [24, 24, 64], (3,3), (1,1))
    conv2_2 = CRelu_block(conv2_1, [24, 24, 64], (3,3), (1,1))
    conv2_3 = CRelu_block(conv2_2, [24, 24, 64], (3,3), (1,1))

    conv3_1 = CRelu_block(conv2_3, [48, 48, 128], (3,3), (2,2))
    conv3_2 = CRelu_block(conv3_1, [48, 48, 128], (3,3), (1,1))
    conv3_3 = CRelu_block(conv3_2, [48, 48, 128], (3,3), (1,1))
    conv3_4 = CRelu_block(conv3_3, [48, 48, 128], (3,3), (1,1))

    conv4_1 = Inception_block(conv3_4, [[64], [48, 128], [24, 48, 48], [128], [256]], strides=(2,2), max_pool=True)
    conv4_2 = Inception_block(conv4_1, [[64], [64, 128], [24, 48, 48], [], [256]], strides=(1,1))
    conv4_3 = Inception_block(conv4_2, [[64], [64, 128], [24, 48, 48], [], [256]], strides=(1,1))
    conv4_4 = Inception_block(conv4_3, [[64], [64, 128], [24, 48, 48], [], [256]], strides=(1,1))

    conv5_1 = Inception_block(conv4_4, [[64], [96, 192], [36, 64, 64], [128], [384]], strides=(2,2), max_pool=True)
    conv5_2 = Inception_block(conv5_1, [[64], [96, 192], [36, 64, 64], [], [384]], strides=(1,1))
    conv5_3 = Inception_block(conv5_2, [[64], [96, 192], [36, 64, 64], [], [384]], strides=(1,1))
    conv5_4 = Inception_block(conv5_3, [[64], [96, 192], [36, 64, 64], [], [384]], strides=(1,1))

    feature_by_downscale = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(conv3_4)
    feature_by_upscale = keras.layers.UpSampling2D(size=(2, 2))(conv5_4)
    feature_cat = keras.layers.Concatenate(axis=-1)([feature_by_downscale, conv4_4, feature_by_upscale])
    output = conv_bn_act(feature_cat, 512, (1,1), strides=(1,1), act='relu')


    model = keras.models.Model(inputs=inputs, outputs=output)
    # model.save("PVAnet.hdf5")
    return model
    # model.summary()
    # model.save('pvanet.hdf5')
