"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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
from keras.applications.mobilenetv2 import MobileNetV2
from keras_applications.mobilenet_v2 import BASE_WEIGHT_PATH
from keras.utils import get_file
from keras.layers import DepthwiseConv2D
import numpy as np
import keras
import tensorflow as tf

def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest' : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tf.image.ResizeMethod.BICUBIC,
        'area'    : tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)

class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]), method='bilinear')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


def default_classification_model(
    num_classes,
    num_anchors,
    input_shape_sub,
    inputs,
    pyramid_feature_size=32,
    prior_probability=0.01,
    classification_feature_size=32,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_regularizer' : keras.regularizers.l2(1e-3),
    }

    # inputs  = keras.layers.Input(shape=(input_shape_sub[0], input_shape_sub[1], pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            # name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer='zeros',
        # name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes))(outputs)
    outputs = keras.layers.Activation('sigmoid')(outputs)

    # return keras.models.Model(inputs=inputs, outputs=outputs, name=name+str(input_shape_sub[0])+str(np.random.random()))
    return outputs


def default_regression_model(num_anchors, input_shape_sub, inputs, pyramid_feature_size=32,
        regression_feature_size=32, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros',
        'kernel_regularizer' : keras.regularizers.l2(1e-3)
    }

    # inputs  = keras.layers.Input(shape=(input_shape_sub[0], input_shape_sub[1], pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            # name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)
    
    outputs = keras.layers.Conv2D(num_anchors * 4, **options)(outputs)
    outputs = keras.layers.Reshape((-1, 4))(outputs)
    
#     reg = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    
#     print "--------------------------------------REGREGREGREGREG"
#     print(reg.summary())
    
#     return reg
    
    # return keras.models.Model(inputs=inputs, outputs=outputs, name=name+str(input_shape_sub[0])+str(np.random.random()))
    return outputs

def __create_pyramid_features(C3, C4, C5, feature_size=32):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """

    options = {
        'kernel_regularizer' : keras.regularizers.l2(1e-3)
    }
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced', **options)(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5', **options)(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced', **options)(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', **options)(P4)

    # add P4 elementwise to C3
    P3           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced', **options)(C3)
    P3           = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
#     P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C2])
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', **options)(P3)
    
    # add P3 elementwise to C2
#     P2           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced', **options)(C2)
#     P2           = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
#     P2           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', **options)(P2)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6', **options)(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7', **options)(P7)

    return [P3, P4, P5, P6, P7]


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes   = [16, 32, 64, 128, 256],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, num_anchors, input_shape_sub, inputs):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        default_regression_model(num_anchors, input_shape_sub, inputs),
        default_classification_model(num_classes, num_anchors, input_shape_sub, inputs)
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors             = 9,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet',
    input_shape = None
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """


    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    # pyramids = __build_pyramid(submodels, features)

    # for i in range(5):
    #     regr_output
    # regr_pair = list(zip(regr_models, features))
    # clas_pair = list(zip(clas_models, features))

    # regr_output = [regr_model(feature) for regr_model, feature in regr_pair]
    # clas_output   = [clas_model(feature) for clas_model, feature in clas_pair]


    regr_output, clas_output = [], []
    if submodels is None:
        for i, input_shape_sub in enumerate([(np.ceil(input_shape[0]/16), np.ceil(input_shape[1]/16), 3),
                                (np.ceil(input_shape[0]/32), np.ceil(input_shape[1]/32), 3),
                                (np.ceil(input_shape[0]/32), np.ceil(input_shape[1]/32), 3),
                                (np.ceil(input_shape[0]/64), np.ceil(input_shape[1]/64), 3),
                                (np.ceil(input_shape[0]/128), np.ceil(input_shape[1]/128), 3)]):
            regr, clas = default_submodels(num_classes, num_anchors, input_shape_sub, features[i])
            regr_output.append(regr)
            clas_output.append(clas)

    regr_cat = keras.layers.Concatenate(axis=1, name='regression')(regr_output)
    clas_cat = keras.layers.Concatenate(axis=1, name='classification')(clas_output)

    pyramids = [regr_cat, clas_cat]

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
    model             = None,
    anchor_parameters = AnchorParameters.default,
    nms               = True,
    name              = 'retinanet-bbox',
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model             : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        name              : Name of the model.
        *kwargs           : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = retinanet(num_anchors=anchor_parameters.num_anchors(), **kwargs)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(anchor_parameters, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(nms=nms, name='filtered_detections')([boxes, classification] + other)

    outputs = detections
    
    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)


# class MobileNetBackbone(Backbone):
#     """ Describes backbone information and provides utility functions.
#     """

#     allowed_backbones = ['mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224']

#     def __init__(self, backbone):
#         super(MobileNetBackbone, self).__init__(backbone)

#         self.custom_objects.update({
#             'relu6': relu6,
#             'DepthwiseConv2D': DepthwiseConv2D
#         })

#     def retinanet(self, *args, **kwargs):
#         """ Returns a retinanet model using the correct backbone.
#         """
#         return mobilenet_retinanet(*args, backbone=self.backbone, **kwargs)

#     def download_imagenet(self):
#         """ Download pre-trained weights for the specified backbone name.
#         This name is in the format mobilenet{rows}_{alpha} where rows is the
#         imagenet shape dimension and 'alpha' controls the width of the network.
#         For more info check the explanation from the keras mobilenet script itself.
#         """

#         alpha = float(self.backbone.split('_')[1])
#         rows = int(self.backbone.split('_')[0].replace('mobilenet', ''))

#         # load weights
#         if keras.backend.image_data_format() == 'channels_first':
#             raise ValueError('Weights for "channels_last" format '
#                              'are not available.')
#         if alpha == 1.0:
#             alpha_text = '1.0'
#         elif alpha == 0.75:
#             alpha_text = '0.75'
#         elif alpha == 0.50:
#             alpha_text = '0.5'
#         else:
#             alpha_text = '0.35'

#         model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{}_{}.h5'.format(alpha_text,
#                 rows)
#         weights_url = BASE_WEIGHT_PATH + model_name
#         weights_path = get_file(model_name, weights_url, cache_subdir='models')

#         return weights_path

#     def validate(self):
#         """ Checks whether the backbone string is correct.
#         """
#         backbone = self.backbone.split('_')[0]

#         if backbone not in MobileNetBackbone.allowed_backbones:
#             raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackbone.allowed_backbones))


############## input shape = 320 320 3
def mobilenet_retinanet(num_classes, input_shape = (800, 1000, 3) ,backbone='mobilenet224_1.0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    alpha = float(backbone.split('_')[1])

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(input_shape)

    mobilenet = MobileNetV2(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights=None)
    # print(mobilenet.summary())
    # create the full model
    layer_names = ['block_6_project_BN', 'block_13_project_BN',
            'block_16_project_BN']
    layer_outputs = [mobilenet.get_layer(name).output for name in layer_names]
    mobilenet = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=mobilenet.name)

    # invoke modifier if given
    if modifier:
        mobilenet = modifier(mobilenet)

    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=mobilenet.outputs, input_shape=input_shape, **kwargs)

if __name__ == "__main__":
    model = mobilenet_retinanet(21)
    model.summary()