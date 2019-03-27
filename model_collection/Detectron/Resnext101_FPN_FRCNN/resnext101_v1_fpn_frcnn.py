from keras import applications
from keras.layers import Layer
from keras.layers import *
from keras.models import load_model
import keras
from .resnext101 import *
from keras import backend as K
from .frcnn_head import frcnn_pred

def create_feature_map(input_shape, cardinality, width):
    # #### changed maxpooling padding to same
    # model = applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape, classes=1000)
    # model = load_model("resnext101_.hdf5")
    print(input_shape, cardinality, width)
    model =  ResNextImageNet(input_shape=input_shape, cardinality=cardinality, width=width)
    model.summary()

    # For resnext101
    # feature1 = (cardinality + 2) * 3 + 1
    # feature2 = (cardinality + 2) * 7 + 1
    # feature3 = (cardinality + 2) * 30 + 1
    # feature4 = (cardinality + 2) * 33 + 1

    if cardinality == 64 and width == 4:
        return model.input, model.get_layer("activation_199").output, model.get_layer("activation_463").output, model.get_layer("activation_1981").output, model.get_layer("activation_2179").output
    elif cardinality == 32 and width == 8:
        return model.input, model.get_layer("activation_103").output, model.get_layer("activation_239").output, model.get_layer("activation_1021").output, model.get_layer("activation_1123").output
    else:
        raise NotImplemented
def create_rpn(input_shape=(480, 640, 3),num_anchors=15, cardinality= 64, width= 4):
    # inputs, base_feature = create_pvanet(input_shape, inside_fcnn=True)
    C1, C2, C3, C4, C5 = create_feature_map(input_shape, cardinality, width)

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

def create_detectron_fpn_frcnn_resnxet101v1_64x4(shape):
    model1 = create_rpn(num_anchors=15, input_shape=shape, cardinality= 64, width= 4)
    model2 = frcnn_pred(nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256)
    return [model1, model2]

def create_detectron_fpn_frcnn_resnxet101v1_32x8(shape):
    model1 = create_rpn(num_anchors=15, input_shape=shape, cardinality= 32, width= 8)
    model2 = frcnn_pred(nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256)
    return [model1, model2]



if __name__ == "__main__":
 
    K.clear_session()
    #rpn_model = rpn_create()
    rpn_model1 = rpn_create(num_anchors=15, input_shape=(800, 640, 3), cardinality= 64, width= 4)
    rpn_model1.save("RXNT-101_FPN_800x640_64x4d.hdf5")
    print("RXNT-101_FPN_800x640_64x4d fininshed")

    K.clear_session()
    rpn_model2 = rpn_create(num_anchors=15, input_shape=(640, 480, 3), cardinality= 64, width= 4)
    rpn_model2.save("RXNT-101_FPN_640x480_64x4d.hdf5")
    print("RXNT-101_FPN_640x480_64x4d fininshed")

    K.clear_session()
    rpn_model3 = rpn_create(num_anchors=15, input_shape=(480, 320, 3), cardinality= 64, width= 4)
    rpn_model3.save("RXNT-101_FPN_480x320_64x4d.hdf5")
    print("RXNT-101_FPN_480x320_64x4d fininshed")

    K.clear_session()
    rpn_model4 = rpn_create(num_anchors=15, input_shape=(800, 640, 3), cardinality= 32, width= 8)
    rpn_model4.save("RXNT-101_FPN_800x640_32x8d.hdf5")
    print("RXNT-101_FPN_800x640_32x8d fininshed")

    K.clear_session()
    rpn_model5 = rpn_create(num_anchors=15, input_shape=(640, 480, 3), cardinality= 32, width= 8)
    rpn_model5.save("RXNT-101_FPN_640x480_32x8d.hdf5")
    print("RXNT-101_FPN_640x480_32x8d fininshed")

    K.clear_session()
    rpn_model6 = rpn_create(num_anchors=15, input_shape=(480, 320, 3), cardinality= 32, width= 8)
    rpn_model6.save("RXNT-101_FPN_480x320_32x8d.hdf5")
    print("RXNT-101_FPN_480x320_32x8d fininshed")
