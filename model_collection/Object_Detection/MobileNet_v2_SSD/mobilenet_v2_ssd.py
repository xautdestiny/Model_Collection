import keras.backend as K
from keras.layers import *
from keras import backend
from keras.models import *
from keras.regularizers import l2
import os
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(input_shape,
                n_classes,
                mode='training',
                weights=None,
                input_tensor=None,
                min_scale=None,
                max_scale=None,
                scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
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
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False,
                alpha = 1.0,
                **kwargs
                ):

    l2_reg = kwargs.get('l2_reg', 0.00005)
    alpha = kwargs.get('alpha', 1.)

    if weights and os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match

    if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
        raise ValueError('If imagenet weights are being loaded, '
                         'alpha can be one of `0.35`, `0.50`, `0.75`, '
                         '`1.0`, `1.3` or `1.4` only.')

    # prepare config
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    img_height, img_width = input_shape[0], input_shape[1]

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
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    img_input = Input(shape=input_shape)
    # 300
    x = _inverted_res_block(img_input, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)
    # 150
    x = _inverted_res_block(x, filters=24, stride=2, expansion=6, block_id=1, **kwargs)
    x = _inverted_res_block(x, filters=24, stride=1, expansion=6, block_id=2, **kwargs)
    # 75
    x = _inverted_res_block(x, filters=32, stride=2, expansion=6, block_id=3, **kwargs)
    x = _inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=4, **kwargs)
    x = _inverted_res_block(x, filters=32, stride=1, expansion=6, block_id=5, **kwargs)
    # 38
    x = _inverted_res_block(x, filters=64, stride=2, expansion=6, block_id=6, **kwargs)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=7, **kwargs)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=8, **kwargs)
    x = _inverted_res_block(x, filters=64, stride=1, expansion=6, block_id=9, **kwargs)
    x = _inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=10, **kwargs)
    x = _inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=11, **kwargs)
    x = _inverted_res_block(x, filters=96, stride=1, expansion=6, block_id=12, **kwargs)




    # 19
    conv0_pw = _inverted_res_block(x, filters=160, stride=2, expansion=6, block_id=13, **kwargs)    ###########
    x = _inverted_res_block(conv0_pw, filters=160, stride=1, expansion=6, block_id=14, **kwargs)
    x = _inverted_res_block(x, filters=160, stride=1, expansion=6, block_id=15, **kwargs)
    x = _inverted_res_block(x, filters=320, stride=1, expansion=6, block_id=16, **kwargs)

    # 10
    # SSD lite
    x = Conv2D(1280, kernel_size=1, strides=(2, 2), use_bias=False,
               kernel_regularizer=l2(l2_reg), name='conv_ft_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_ft_1')(x)    
    conv1_pw = ReLU(6., name='relu_ft_1')(x)         ############

    # 5
    conv2_pw = SSDConv(conv1_pw, 2, 512,  **kwargs)         #########
    # 3
    conv3_pw = SSDConv(conv2_pw, 3, 256,  **kwargs)         ##########
    # 2
    conv4_pw = SSDConv(conv3_pw, 4, 128,  **kwargs)         ##########
    # 1
    conv5_pw = SSDConv(conv4_pw, 5, 128,  **kwargs)         ###########

    conv0_mbox_conf = pred_cls(conv0_pw, n_boxes[0] * n_classes, name='conv0_mbox_conf',  **kwargs)
    conv1_mbox_conf = pred_cls(conv1_pw, n_boxes[1] * n_classes, name='conv1_mbox_conf',  **kwargs)
    conv2_mbox_conf = pred_cls(conv2_pw, n_boxes[2] * n_classes, name='conv2_mbox_conf',  **kwargs)
    conv3_mbox_conf = pred_cls(conv3_pw, n_boxes[3] * n_classes, name='conv3_mbox_conf',  **kwargs)
    conv4_mbox_conf = pred_cls(conv4_pw, n_boxes[4] * n_classes, name='conv4_mbox_conf',  **kwargs)
    conv5_mbox_conf = pred_cls(conv5_pw, n_boxes[5] * n_classes, name='conv5_mbox_conf',  **kwargs)
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv0_mbox_loc = pred_cls(conv0_pw, n_boxes[0] * 4, name='conv0_mbox_loc',  **kwargs)
    conv1_mbox_loc = pred_cls(conv1_pw, n_boxes[1] * 4, name='conv1_mbox_loc',  **kwargs)
    conv2_mbox_loc = pred_cls(conv2_pw, n_boxes[2] * 4, name='conv2_mbox_loc',  **kwargs)
    conv3_mbox_loc = pred_cls(conv3_pw, n_boxes[3] * 4, name='conv3_mbox_loc',  **kwargs)
    conv4_mbox_loc = pred_cls(conv4_pw, n_boxes[4] * 4, name='conv4_mbox_loc',  **kwargs)
    conv5_mbox_loc = pred_cls(conv5_pw, n_boxes[5] * 4, name='conv5_mbox_loc',  **kwargs)


    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv0_mbox_conf_reshape = Reshape((-1, n_classes), name='conv0_mbox_conf_reshape')(conv0_mbox_conf)
    conv1_mbox_conf_reshape = Reshape((-1, n_classes), name='conv1_mbox_conf_reshape')(conv1_mbox_conf)
    conv2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv2_mbox_conf_reshape')(conv2_mbox_conf)
    conv3_mbox_conf_reshape = Reshape((-1, n_classes), name='conv3_mbox_conf_reshape')(conv3_mbox_conf)
    conv4_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_mbox_conf_reshape')(conv4_mbox_conf)
    conv5_mbox_conf_reshape = Reshape((-1, n_classes), name='conv5_mbox_conf_reshape')(conv5_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv0_mbox_loc_reshape = Reshape((-1, 4), name='conv0_mbox_loc_reshape')(conv0_mbox_loc)
    conv1_mbox_loc_reshape = Reshape((-1, 4), name='conv1_mbox_loc_reshape')(conv1_mbox_loc)
    conv2_mbox_loc_reshape = Reshape((-1, 4), name='conv2_mbox_loc_reshape')(conv2_mbox_loc)
    conv3_mbox_loc_reshape = Reshape((-1, 4), name='conv3_mbox_loc_reshape')(conv3_mbox_loc)
    conv4_mbox_loc_reshape = Reshape((-1, 4), name='conv4_mbox_loc_reshape')(conv4_mbox_loc)
    conv5_mbox_loc_reshape = Reshape((-1, 4), name='conv5_mbox_loc_reshape')(conv5_mbox_loc)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv0_mbox_conf_reshape,
                                                       conv1_mbox_conf_reshape,
                                                       conv2_mbox_conf_reshape,
                                                       conv3_mbox_conf_reshape,
                                                       conv4_mbox_conf_reshape,
                                                       conv5_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv0_mbox_loc_reshape,
                                                     conv1_mbox_loc_reshape,
                                                     conv2_mbox_loc_reshape,
                                                     conv3_mbox_loc_reshape,
                                                     conv4_mbox_loc_reshape,
                                                     conv5_mbox_loc_reshape])



    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc])#, mbox_priorbox])


    model = Model(inputs=img_input, outputs=predictions)
    return model



def pred_cls(x, filter_num, name='', **kwargs):
    l2_reg = kwargs.get('l2_reg', 0.00005)
    return Conv2D(filter_num, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=name)(x)


def SSDConv(x, i, filter_num,  **kwargs):
    l2_reg = kwargs.get('l2_reg', 0.00005)
    x = Conv2D(filter_num//2, (1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg),   name='conv_ft_%s_1'%(str(i)))(x)
    x = BatchNormalization(epsilon=2e-5, name='bn_ft_%s_1'%(str(i)))(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv_ft_%s_padding_1'%(str(i)))(x)

    x = Conv2D(filter_num, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv_ft_%s_2'%(str(i)))(x)
    x = BatchNormalization(epsilon=2e-5, name='bn_ft_%s_2' %(str(i))  )(x)
    x = Activation('relu')(x) 
    return x


def _inverted_res_block(inputs, expansion, stride, filters, block_id, **kwargs):
    l2_reg = kwargs.get('l2_reg', 0.00005)
    alpha = kwargs.get('alpha', 1.)
    use_bias = kwargs.get('use_bias', False)
    epsilon = kwargs.get('epsilon', 1e-3)

    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=use_bias,
                      kernel_regularizer=l2(l2_reg), name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=epsilon, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters, kernel_size=3, strides=2, padding='same', use_bias=use_bias,
                      kernel_regularizer=l2(l2_reg), name='Conv1')(x)
        x = BatchNormalization(epsilon=epsilon, momentum=0.999, name='bn_Conv1')(x)
        x = ReLU(6., name='Conv1_relu')(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=use_bias, padding='same',
                        depthwise_regularizer= l2(l2_reg), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.999,name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=use_bias,
               kernel_regularizer=l2(l2_reg), name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def create_mobilenetv2_ssd(input_shape):
    model = MobileNetV2(input_shape, 21)
    return model

# if __name__ == "__main__":
#   model = MobileNetV2((300, 300, 3), 21)
#   model.summary()
#   model.save("mobilenet_v2_ssd_224_224.hdf5")