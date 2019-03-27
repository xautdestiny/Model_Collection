import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Convolution2D, Input, TimeDistributed, Flatten, Dense, Activation, BatchNormalization
from keras.engine.topology import Layer
import numpy as np

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
def create_pvanet(input_shape=(480, 640, 3), inside_fcnn=False):
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

    if inside_fcnn:
        return inputs, output

    model = keras.models.Model(inputs=inputs, outputs=output)
    # model.save("PVAnet.hdf5")
    return model
    # model.summary()
    # model.save('pvanet.hdf5')



class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):

        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            num_pool_regions = self.pool_size

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size,self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return final_output


def rpn_create(input_shape=(480, 640, 3),num_anchors=42):
    inputs, base_feature = create_pvanet(input_shape, inside_fcnn=True)

    x = Convolution2D(384, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_feature)

    x_class = Convolution2D(num_anchors * 2, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)


    return keras.models.Model(inputs, [base_feature, x_class, x_regr])




def rcnn_create(feature_shape, num_rois=64, pooling_regions = 7, nb_classes = 21, trainable=False):
    feature_input = Input(shape=feature_shape)
    rois_input = Input(shape=(num_rois, 4))
    

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([feature_input, rois_input])

    out = TimeDistributed(Flatten())(out_roi_pool)


    out = TimeDistributed(Dense(4096))(out)
    out = TimeDistributed(BatchNormalization())(out)
    out = TimeDistributed(Activation('relu'))(out)

    out = TimeDistributed(Dense(4096))(out)
    out = TimeDistributed(BatchNormalization())(out)
    out = TimeDistributed(Activation('relu'))(out)


    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * nb_classes, activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    rcnn = keras.models.Model([feature_input, rois_input], [out_class, out_regr])
    return rcnn


def apply_regr(x, y, w, h, tx, ty, tw, th):
    cx = x + w/2.
    cy = y + h/2.
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = np.exp(tw) * w
    h1 = np.exp(th) * h
    x1 = cx1 - w1/2.
    y1 = cy1 - h1/2.
    x1 = np.round(x1)
    y1 = np.round(y1)
    w1 = np.round(w1)
    h1 = np.round(h1)

    return x1, y1, w1, h1

def rpn_to_roi(
    anchor_cls, 
    anchor_regr, 
    rpn_stride=10,
    anchor_box=[
                [128, 128],
                [128,128*2],
                [128*2,128],
                [256,256],
                [256,256*2],
                [256*2, 256],
                [384,384],
                [384,384*2],
                [384*2,384]
               ]
    ):

    rows, cols = anchor_cls.shape[1:3]

    anchor_idx = 0

    all_roi = []
    all_score = []
    for anchor_x, anchor_y in anchor_box:
        anchor_x /= rpn_stride
        anchor_y /= rpn_stride
        
        score = anchor_cls[0 ,..., anchor_idx]
        offset = anchor_regr[0,..., 4*anchor_idx: 4*anchor_idx+4]

        offset = np.transpose(offset, (2,0,1))

        cy, cx = np.where(score > 0.)
        x, y = cx - anchor_x/2, cy - anchor_y/2
        tx, ty, tw, th = offset[:, cy, cx]

        x, y, w, h = apply_regr(x, y, anchor_x, anchor_y, tx, ty, tw, th)

        
        x = np.maximum(0, x)
        y = np.maximum(0, y)
        w = np.maximum(1, w)
        h = np.maximum(1, h)
        
        roi = list(zip(x, y, w, h))

        all_roi.extend(roi)
        anchor_idx += 1

    ### suppose to do NMS here
    ### all_roi = NMS(all_roi)
    return np.expand_dims(np.array(all_roi), axis=0)


if __name__ == "__main__":    
    num_rois = 10
    rpn = rpn_create()
    rpn.summary()
    rpn.save('faterRCNN(pvanet).hdf5')

    rcnn = rcnn_create((30, 40, 512), num_rois)
    rcnn.summary()
    rcnn.save('faterRCNN(rcnn).hdf5')

    ### compelete run of Faster RCNN

    image = np.random.random((1, 480, 640, 3))

    [base_feature, anchor_class, anchor_regr] = rpn.predict(image)
    rois = rpn_to_roi(anchor_class, anchor_regr, rpn_stride=16)
    rois = rois[:,:num_rois,...]

    [x_class, anchor_regr] = rcnn.predict([base_feature, rois])