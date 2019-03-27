from keras import applications
from keras.layers import Layer
from keras.layers import *
import keras
from keras import backend as K
from .resnet101 import *
from .frcnn_head import frcnn_pred

def create_feature_map(input_shape):
    #### changed maxpooling padding to same
    # model = applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape, classes=1000)
    model = create_resnet101(input_shape)
    model.summary()
    return model.input, model.get_layer("res2c_relu").output, model.get_layer("res3b3_relu").output, model.get_layer("res4b22_relu").output, model.get_layer("res5c_relu").output
    # return model.input, model.get_layer("activation_10").output, model.get_layer("activation_22").output, model.get_layer("activation_40").output, model.get_layer("activation_49").output


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


# def rpn_create(input_shape=(480, 640, 3),num_anchors=15):
#     # inputs, base_feature = create_pvanet(input_shape, inside_fcnn=True)
#     inputs, base_feature = create_resnet50_v1_fRCNN(input_shape)

#     x = Convolution2D(1024, (3, 3), padding='same', kernel_initializer='normal', name='rpn_conv1')(base_feature)
#     x = ReLU()(x)

#     x_logit = Convolution2D(num_anchors, (1, 1), kernel_initializer='uniform', name='rpn_out_class')(x)
#     x_pred = Activation('sigmoid')(x_logit)

#     x_regr = Convolution2D(num_anchors * 4, (1, 1), kernel_initializer='zero', name='rpn_out_regress')(x)

#     return keras.models.Model(inputs, [base_feature, x_pred, x_regr])


def create_rpn(input_shape=(480, 640, 3),num_anchors=15):
    # inputs, base_feature = create_pvanet(input_shape, inside_fcnn=True)
    C1, C2, C3, C4, C5 = create_feature_map(input_shape)

    M5 = Conv2D(256, (1,1), padding='same')(C5)

    P6 = MaxPooling2D((1,1), 2)(M5)
    P5 = Conv2D(256, (3,3), padding='same')(M5)


    M5_upsample = UpSampling2D((2,2))(M5)
    inner_C4 = Conv2D(256, (1,1), padding='same')(C4)
    M4 = Add()([M5_upsample, inner_C4])
    P4 = Conv2D(256, (3,3), padding='same')(M4)


    M4_upsample = UpSampling2D((2,2))(M4)
    inner_C3 = Conv2D(256, (1,1), padding='same')(C3)
    M3 = Add()([M4_upsample, inner_C3])
    P3 = Conv2D(256, (3,3), padding='same')(M3)

    M3_upsample = UpSampling2D((2,2))(M3)
    inner_C2 = Conv2D(256, (1,1), padding='same')(C2)
    M2 = Add()([M3_upsample, inner_C2])
    P2 = Conv2D(256, (3,3), padding='same')(M2)




    P6 = Conv2D(256, (3,3), padding='same')(P6)
    P6 = ReLU()(P6)
    regr6 = Conv2D(num_anchors * 4, (1,1))(P6)
    logit6 = Conv2D(num_anchors, (1,1))(P6)
    pred6 = Activation('sigmoid')(logit6)

    P5 = Conv2D(256, (3,3), padding='same')(P5)
    P5 = ReLU()(P5)
    regr5 = Conv2D(num_anchors * 4, (1,1))(P5)
    logit5 = Conv2D(num_anchors, (1,1))(P5)
    pred5 = Activation('sigmoid')(logit5)

    P4 = Conv2D(256, (3,3), padding='same')(P4)
    P4 = ReLU()(P4)
    regr4 = Conv2D(num_anchors * 4, (1,1))(P4) #box
    logit4 = Conv2D(num_anchors, (1,1))(P4)
    pred4 = Activation('sigmoid')(logit4) 

    P3 = Conv2D(256, (3,3), padding='same')(P3)
    P3 = ReLU()(P3)
    regr3 = Conv2D(num_anchors * 4, (1,1))(P3)
    logit3 = Conv2D(num_anchors, (1,1))(P3)
    pred3 = Activation('sigmoid')(logit3)

    P2 = Conv2D(256, (3,3), padding='same')(P2)
    P2 = ReLU()(P2)
    regr2 = Conv2D(num_anchors * 4, (1,1))(P2)
    logit2 = Conv2D(num_anchors, (1,1))(P2)
    pred2 = Activation('sigmoid')(logit2)


    # rpn_conv = Conv2D(256, (3,3), padding='same')
    # rpn_relu = ReLU()
    # rpn_regr_conv = Conv2D(num_anchors * 4, (1,1))
    # rpn_logit_conv = Conv2D(num_anchors, (1,1))
    # rpn_sigmoid = Activation('sigmoid')
    
    # P5 = rpn_conv(P5)
    # P5 = rpn_relu(P5)
    # regr5 = rpn_regr_conv(P5)
    # logit5 = rpn_logit_conv(P5)
    # pred5 = rpn_sigmoid(logit5)

    # P4 = rpn_conv(P4)
    # P4 = rpn_relu(P4)
    # regr4 = rpn_regr_conv(P4)
    # logit4 = rpn_logit_conv(P4)
    # pred4 = rpn_sigmoid(logit4)

    # P3 = rpn_conv(P3)
    # P3 = rpn_relu(P3)
    # regr3 = rpn_regr_conv(P3)
    # logit3 = rpn_logit_conv(P3)
    # pred3 = rpn_sigmoid(logit3)

    # P2 = rpn_conv(P2)
    # P2 = rpn_relu(P2)
    # regr2 = rpn_regr_conv(P2)
    # logit2 = rpn_logit_conv(P2)
    # pred2 = rpn_sigmoid(logit2)

    return keras.models.Model(C1, [regr5, pred5, regr4, pred4, regr3, pred3, regr2, pred2, P5, P4, P3, P2])
    # return keras.models.Model(inputs, [base_feature, x_pred, x_regr])





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


def create_detectron_fpn_frcnn_resnet101v1(shape):
    model1 = create_rpn(shape)
    model2 = frcnn_pred(nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256)
    return [model1, model2]


if __name__ == "__main__":
    # res = create_resnet50_v1((224, 224, 3))
    K.clear_session()
    rpn_model = rpn_create((800, 640, 3))
    rpn_model.save("R-101_FPN_800x640.hdf5")
    
    K.clear_session()
    rpn_model = rpn_create((640, 480, 3))
    rpn_model.save("R-101_FPN_640x480.hdf5")

    K.clear_session()
    rpn_model = rpn_create((480, 320, 3))
    rpn_model.save("R-101_FPN_480x320.hdf5")



