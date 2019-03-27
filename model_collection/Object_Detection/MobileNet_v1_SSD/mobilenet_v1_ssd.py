from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
import keras.backend as K

# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_L2Normalization import L2Normalization
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import add
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
# from keras.applications.imagenet_utils import _obtain_input_shape
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf

def nothing(x):
	return x

def l2normalization(x):
    para_trained = 2
    x = K.l2_normalize(x, axis=3) * para_trained
    return x


class L2Normalization(Layer):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

    Arguments:
        gamma_init (int): The initial scaling parameter. Defaults to 20 following the
            SSD paper.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Returns:
        The scaled tensor. Same shape as the input tensor.

    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''

    def __init__(self, gamma_init=20, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.tf.nn.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def relu6(x):
    return K.relu(x, max_value=6)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), bn_epsilon=2e-5,
                bn_momentum=0.95, block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = filters * alpha
    filters = _make_divisible(filters)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon,
                           name='conv%d_bn' % block_id)(x)
    return Activation(relu6, name='conv%d_relu' % block_id)(x)


# taken from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/conv_blocks.py
def _make_divisible(v, divisor=8, min_value=8):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1),
                          bn_epsilon=2e-5, block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        bn_epsilon: Epsilon value for BatchNormalization
        block_id: Integer, a unique identification designating the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = _make_divisible(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, epsilon=bn_epsilon, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, epsilon=bn_epsilon, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def ssd_mn_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False,
            alpha = 1.0,
           depth_multiplier=1,
            expansion_factor=6,
           ):
    '''
    Build a Keras model with SSD300 architecture, see references.
    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.
    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.
    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.
    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).
    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        mode (str, optional): One of 'training', 'inference' and 'inference_fast'. In 'training' mode,
            the model outputs the raw prediction tensor, while in 'inference' and 'inference_fast' modes,
            the raw predictions are decoded into absolute coordinates and filtered via confidence thresholding,
            non-maximum suppression, and top-k filtering. The difference between latter two modes is that
            'inference' follows the exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.
        l2_regularization (float, optional): The L2-regularization rate. Applies to all convolutional layers.
            Set to zero to deactivate L2-regularization.
        min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
            of the shorter side of the input images. All scaling factors between the smallest and the
            largest will be linearly interpolated. Note that the second to last of the linearly interpolated
            scaling factors will actually be the scaling factor for the last predictor layer, while the last
            scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
            if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
            This list must be one element longer than the number of predictor layers. The first `k` elements are the
            scaling factors for the `k` predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
            last scaling factor must be passed either way, even if it is not being used. If a list is passed,
            this argument overrides `min_scale` and `max_scale`. All scaling factors must be greater than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
            generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer individually, which is the case for the
            original SSD300 implementation. If a list is passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer how many
            pixels apart the anchor box center points should be vertically and horizontally along the spatial grid over
            the image. If the list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent `(step_height, step_width)`.
            If no steps are provided, then they will be computed such that the anchor box center points will form an
            equidistant grid within the image dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements can be
            either floats or tuples of two floats. These numbers represent for each predictor layer how many
            pixels from the top and left boarders of the image the top-most and left-most anchor box center points should be
            as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel values, but fractions
            of the step size specified in the `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
            its respective variance value.
        coords (str, optional): The box coordinate format to be used internally by the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height), 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model is supposed to use relative instead of absolute coordinates,
            i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
            of any shape that is broadcast-compatible with the image shape. The elements of this array will be
            subtracted from the image pixel intensity values. For example, pass a list of three integers
            to perform per-channel mean normalization for color images.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
            floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
            intensity values will be divided by the elements of this array. For example, pass a list
            of three integers to perform per-channel standard deviation normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the desired order in which the input
            image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the model, but also
            a list containing the spatial dimensions of the predictor layers. This isn't strictly necessary since
            you can always get their sizes easily via the Keras API, but it's convenient and less error-prone
            to get them this way. They are only relevant for training anyway (SSDBoxEncoder needs to know the
            spatial dimensions of the predictor layers), for inference you don't need them.
             depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.
    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    inputs = Input(shape=(img_height, img_width, img_channels))

    # # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    # x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(inputs)
    # if not (subtract_mean is None):
    #     x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    # if not (divide_by_stddev is None):
    #     x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    # if swap_channels:
    #     x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    # 300
    x = _conv_block(inputs, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    # 150
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    # 75
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    # 38
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    conv11 = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    # 19
    x = _depthwise_conv_block(conv11, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    conv13 = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    # 10
    conv14_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_1')(conv13)
    conv14_1 = BatchNormalization(epsilon=2e-5, name='bn14_1')(conv14_1)
    conv14_1 = Activation('relu')(conv14_1)
    conv14_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv14_padding_1')(conv14_1)
    conv14_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_2')(conv14_1)
    conv14_2 = BatchNormalization(epsilon=2e-5, name='bn14_2')(conv14_2)
    conv14_2 = Activation('relu')(conv14_2)
    # 5
    conv15_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_1')(conv14_2)
    conv15_1 = BatchNormalization(epsilon=2e-5, name='bn15_1')(conv15_1)
    conv15_1 = Activation('relu')(conv15_1)
    conv15_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv15_padding')(conv15_1)
    conv15_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_2')(conv15_1)
    conv15_2 = BatchNormalization(epsilon=2e-5, name='bn15_2')(conv15_2)
    conv15_2 = Activation('relu')(conv15_2)
    # 3
    conv16_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_1')(conv15_2)
    conv16_1 = BatchNormalization(epsilon=2e-5, name='bn16_1')(conv16_1)
    conv16_1 = Activation('relu')(conv16_1)
    conv16_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv16_padding')(conv16_1)
    conv16_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_2')(conv16_1)
    conv16_2 = BatchNormalization(epsilon=2e-5, name='bn16_2')(conv16_2)
    conv16_2 = Activation('relu')(conv16_2)
    # 2
    conv17_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_1')(conv16_2)
    conv17_1 = BatchNormalization(epsilon=2e-5, name='bn17_1')(conv17_1)
    conv17_1 = Activation('relu')(conv17_1)
    conv17_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv17_padding')(conv17_1)
    conv17_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_2')(conv17_1)
    conv17_2 = BatchNormalization(epsilon=2e-5, name='bn17_2')(conv17_2)
    conv17_2 = Activation('relu')(conv17_2)
    # Feed conv4_3 into the L2 normalization layer

    ## L2Norm to be cut off
    # conv11_norm = L2Normalization(gamma_init=20, name='conv11_norm')(conv11)
    # conv11_norm = Lambda(nothing, name='conv11_norm')(conv11)
    conv11_norm = Lambda(l2normalization, name='conv11_norm')(conv11)

    ### Build the convolutional predictor layers on top of the base network

    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    conv11_norm_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_norm_conf')(conv11_norm)
    conv13_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv13_mbox_conf')(conv13)
    conv14_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_conf')(conv14_2)
    conv15_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_conf')(conv15_2)
    conv16_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_conf')(conv16_2)
    conv17_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_conf')(conv17_2)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv11_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_norm_mbox_loc')(conv11_norm)
    conv13_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv13_mbox_loc')(conv13)
    conv14_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_loc')(conv14_2)
    conv15_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_loc')(conv15_2)
    conv16_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_loc')(conv16_2)
    conv17_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_loc')(conv17_2)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    # conv11_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
    #                                          two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
    #                                          variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv11_norm_mbox_priorbox')(conv11_norm_mbox_loc)
    # conv13_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
    #                                 two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
    #                                 variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv13_mbox_priorbox')(conv13_mbox_loc)
    # conv14_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
    #                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
    #                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv14_2_mbox_priorbox')(conv14_2_mbox_loc)
    # conv15_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
    #                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
    #                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv15_2_mbox_priorbox')(conv15_2_mbox_loc)
    # conv16_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
    #                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
    #                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv16_2_mbox_priorbox')(conv16_2_mbox_loc)
    # conv17_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
    #                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
    #                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv17_2_mbox_priorbox')(conv17_2_mbox_loc)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv11_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv11_norm_mbox_conf_reshape')(conv11_norm_conf)
    conv13_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_mbox_conf_reshape')(conv13_mbox_conf)
    conv14_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv14_2_mbox_conf_reshape')(conv14_2_mbox_conf)
    conv15_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv15_2_mbox_conf_reshape')(conv15_2_mbox_conf)
    conv16_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv16_2_mbox_conf_reshape')(conv16_2_mbox_conf)
    conv17_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv17_2_mbox_conf_reshape')(conv17_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv11_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv11_norm_mbox_loc_reshape')(conv11_norm_mbox_loc)
    conv13_mbox_loc_reshape = Reshape((-1, 4), name='conv13_mbox_loc_reshape')(conv13_mbox_loc)
    conv14_2_mbox_loc_reshape = Reshape((-1, 4), name='conv14_2_mbox_loc_reshape')(conv14_2_mbox_loc)
    conv15_2_mbox_loc_reshape = Reshape((-1, 4), name='conv15_2_mbox_loc_reshape')(conv15_2_mbox_loc)
    conv16_2_mbox_loc_reshape = Reshape((-1, 4), name='conv16_2_mbox_loc_reshape')(conv16_2_mbox_loc)
    conv17_2_mbox_loc_reshape = Reshape((-1, 4), name='conv17_2_mbox_loc_reshape')(conv17_2_mbox_loc)

    # # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    # conv11_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv11_norm_mbox_priorbox_reshape')(conv11_norm_mbox_priorbox)
    # conv13_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_mbox_priorbox_reshape')(conv13_mbox_priorbox)
    # conv14_mbox_priorbox_reshape = Reshape((-1, 8), name='conv14_mbox_priorbox_reshape')(conv14_2_mbox_priorbox)
    # conv15_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv15_2_mbox_priorbox_reshape')(conv15_2_mbox_priorbox)
    # conv16_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv16_2_mbox_priorbox_reshape')(conv16_2_mbox_priorbox)
    # conv17_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv17_2_mbox_priorbox_reshape')(conv17_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv11_norm_mbox_conf_reshape,
                                                       conv13_mbox_conf_reshape,
                                                       conv14_2_mbox_conf_reshape,
                                                       conv15_2_mbox_conf_reshape,
                                                       conv16_2_mbox_conf_reshape,
                                                       conv17_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv11_norm_mbox_loc_reshape,
                                                     conv13_mbox_loc_reshape,
                                                     conv14_2_mbox_loc_reshape,
                                                     conv15_2_mbox_loc_reshape,
                                                     conv16_2_mbox_loc_reshape,
                                                     conv17_2_mbox_loc_reshape])

    # # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    # mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv11_norm_mbox_priorbox_reshape,
    #                                                            conv13_mbox_priorbox_reshape,
    #                                                            conv14_mbox_priorbox_reshape,
    #                                                            conv15_2_mbox_priorbox_reshape,
    #                                                            conv16_2_mbox_priorbox_reshape,
    #                                                            conv17_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc])#, mbox_priorbox])

    # if mode == 'training':
    #     model = Model(inputs=inputs, outputs=predictions)
    # elif mode == 'inference':
    #     decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
    #                                            iou_threshold=iou_threshold,
    #                                            top_k=top_k,
    #                                            nms_max_output_size=nms_max_output_size,
    #                                            coords=coords,
    #                                            normalize_coords=normalize_coords,
    #                                            img_height=img_height,
    #                                            img_width=img_width,
    #                                            name='decoded_predictions')(predictions)
    #     model = Model(inputs=inputs, outputs=decoded_predictions)
    # elif mode == 'inference_fast':
    #     decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
    #                                                iou_threshold=iou_threshold,
    #                                                top_k=top_k,
    #                                                nms_max_output_size=nms_max_output_size,
    #                                                coords=coords,
    #                                                normalize_coords=normalize_coords,
    #                                                img_height=img_height,
    #                                                img_width=img_width,
    #                                                name='decoded_predictions')(predictions)
    #     model = Model(inputs=inputs, outputs=decoded_predictions)
    # else:
    #     raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    # if return_predictor_sizes:
    #     predictor_sizes = np.array([conv11_norm_conf._keras_shape[1:3],
    #                                  conv13_mbox_conf._keras_shape[1:3],
    #                                  conv14_2_mbox_conf._keras_shape[1:3],
    #                                  conv15_2_mbox_conf._keras_shape[1:3],
    #                                  conv16_2_mbox_conf._keras_shape[1:3],
    #                                  conv17_2_mbox_conf._keras_shape[1:3]])
    #     return model, predictor_sizes
    # else:
    #     return model

    model = Model(inputs=inputs, outputs=predictions)
    return model



def create_mobilenetv1_ssd(input_shape):
    img_height = input_shape[0] # Height of the input images
    img_width = input_shape[1] # Width of the input images
    img_channels = input_shape[2] # Number of color channels of the input images
    intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    n_classes = 5 # Number of positive classes
    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
    # aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
    aspect_ratios = [1.0] # The list of aspect ratios for the anchor boxes
    two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
    steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
    offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
    clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
    normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
    model = ssd_mn_300(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)
    # model.save("mobv1_ssd.hdf5")
    return model

if __name__ == "__main__":
    # model = create_mobilenetv1_ssd((480, 640, 3))
    # model.summary()
    # model.save("mobv1_ssd_640w_480h_lambda(l2).hdf5")
    model = create_mobilenetv1_ssd((224, 224, 3))
    model.summary()
    model.save("mobilenet_v1_ssd_224_224.hdf5")