
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.layers import Lambda
import numpy as np

from keras.engine import Layer, InputSpec
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations
import keras.backend as K


def Act(act_type, name):
    if act_type=='prelu':
        body = layers.PReLU(name=name, shared_axes=[1, 2])
    elif act_type=='leaky':
        body = layers.LeakyReLU(0.001, name=name+'l')
    else:
        body = Activation(act_type, name=name)
    return body


def residual_unit_v3(data, num_filter, stride, dim_match, stage, block, **kwargs):
    use_se = kwargs.get('version_se', 0)
    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('act_type', 'prelu')
    name_base = 'stage%d_block%d'%(stage, block)
    # bn-conv-bn-relu-conv-bn
    bn1 = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name_base + '_bn1')(data)
    conv1 = Conv2D(filters=num_filter, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                   name=name_base + '_conv1')(bn1)
    bn2 = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name_base + '_bn2')(conv1)
    act1 = Act(act_type=act_type, name=name_base + '_relu1')(bn2)
    act1 = ZeroPadding2D(((1, 1), (1, 1)))(act1)
    conv2 = Conv2D(filters=num_filter, kernel_size=(3, 3), strides=stride, padding="valid", use_bias=False,
                   name=name_base + '_conv2')(act1)
    bn3 = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name_base + '_bn3')(conv2)

    if use_se:
        # Ques: kernel = (7,7)
        body = GlobalAveragePooling2D(name=name_base + '_se_pool1')(bn3)
        body = layers.Reshape((1, 1, num_filter))(body)
        body = Conv2D(filters=num_filter // 16, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      name=name_base + '_se_conv1')(body)
        body = Act(act_type=act_type, name=name_base + '_se_relu1')(body)
        body = Conv2D(filters=num_filter, kernel_size=(1, 1), strides=(1, 1), padding="same",
                      name=name_base + '_se_conv2')(body)
        body = Activation('sigmoid', name=name_base + '_se_sigmoid')(body)
        bn3 = layers.Multiply()([bn3, body])

    # short cut
    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv2D(filters=num_filter, kernel_size=(1, 1), strides=stride, padding="valid", use_bias=False,
                         name=name_base + '_conv1sc')(data)
        shortcut = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name_base + '_bnsc')(conv1sc)
    return Add()([bn3, shortcut])


def resnet(blocks, num_stages, filter_list, num_classes, residual_unit, **kwargs):
    """Return ResNet symbol of
    Parameters
    ----------
    blocks : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    """
    assert (len(blocks) == num_stages)

    bn_mom = kwargs.get('bn_mom', 0.9)
    act_type = kwargs.get('act_type', 'prelu')
    input_shape = kwargs.get('input_shape', (96, 96, 3))

    # building network
    input_img = Input(input_shape)
    body = input_img
    body = Conv2D(filters=filter_list[0], kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False,
                  name='conv0')(body)
    body = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn0')(body)
    body = Act(act_type=act_type, name='relu0')(body)

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (2, 2), False, stage=i+1, block=1, **kwargs)
        for j in range(blocks[i] - 1):
            body = residual_unit(body, filter_list[i+1], (1, 1), True, stage=i+1, block=j+2, **kwargs)

    # fc layer
    body = BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn1')(body)
#     body = GlobalAveragePooling2D()(body)
#     body = layers.Dropout(rate=0.1)(body)
    body = Flatten()(body)
    fc1 = Dense(units=num_classes, name='fc1')(body)
    fc1 = BatchNormalization(scale=False, epsilon=2e-5, momentum=bn_mom, name='bn2')(fc1)
    model = Model(input_img, fc1)
    return model


def base_model(input_shape, weights=None, bottleneck=256, **kwargs):
    filter_list = [32, 32, 64, 128, 256]
    num_stages = 4
    num_layers = kwargs.get('num_layers', 28)
    
    if num_layers == 18:
        blocks = [2, 2, 2, 2]
    elif num_layers == 28:
        blocks = [3, 4, 3, 3]
    elif num_layers == 34:
        blocks = [3, 4, 6, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    residual_unit = residual_unit_v3
    bn_mom = 0.9
    act_type = 'leaky'
    version_se = 0

    model = resnet(input_shape = input_shape,
                   blocks       = blocks,
                   num_stages  = num_stages,
                   filter_list = filter_list,
                   num_classes = bottleneck,
                   residual_unit=residual_unit,
                   bn_mom=bn_mom,
                   act_type=act_type,
                   version_se=version_se)
    if weights:
        model.load_weights(weights)
    return model


# def create_FR(input_shape):
#     model = base_model(input_shape)
#     model.save("asdf.hdf5")
#     return model

def create_FR(input_shape, name):

    ###### to Jiarong: change the path to correct directory
    if name == 'FR_leaky_mobilefacenet':
        return load_model("./FR_models/leaky_mobilefacenet/model_info/mobilefacenet_3_128_0.982_0.969_238.h5")
    elif name == 'FR_leaky_res18':
        return load_model("./FR_models/leaky_res18/model_info/res18_1.00_4_512_(112,112,3)_0.988_0.986_3.hdf5")
    elif name == 'FR_leaky_res28':
        return load_model("./FR_models/leaky_res28/model_info/res28_3_256_0.988_0.979_142.hdf5")
    elif name == 'FR_leaky_res34':
        return load_model("./FR_models/leaky_res34/model_info/res34_1.00_3_512_(112, 112, 3)_0.991_0.991_104.hdf5")
    elif name == 'FR_leaky_res50':
        return load_model("./FR_models/leaky_res50/model_info/res50_1.00_3_512_(112,112,3)_0.995_0.993_130.hdf5")
    elif name == 'FR_prelu_res34':
        return load_model("./FR_models/prelu_res34/model_info/res34_1.00_3_512_(112, 112, 3)_0.994_0.993_119.hdf5")
    elif name == 'FR_prelu_res100':
        return load_model("./FR_models/prelu_res100/model_info/res100_1.00_3_512_(112,112,3)_0.995_0.995_40.hdf5")
    else:
        raise ValueError("no model {}".format(name))
        return None