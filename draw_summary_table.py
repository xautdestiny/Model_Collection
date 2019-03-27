import sys
import os
import argparse
from xlwt import *
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
from model_collection import *
# from network import Onet, Pnet, Rnet, densenet161, GoogLeNet, inception_v3, inception_v4, mobilenet, nasnet, resnet_common, resnext, shufflenet, xception
# from network import yolov2, PVAnet, mobilenetv1_ssd, FR, FD, Onet48, Onet56, MobileNetV2, PVAnet_fasterRCNN, mobilenetv1_ssd_reduced, mobilenetv2_retinanet
# from network import vgg, Deeplabv3, Deeplabv3plus, mobilenet_noAct, MobileNetV2_noAct, inception_v3, mobilenet, MobileNetV2
# from network import resnet, resnext, tiny_yolo

def create_keras_model(model_name, input_shape):
    model = None
    if model_name == "ONet":
        model = Onet.create_Onet(input_shape)
    elif model_name == "PNet":
        model = Pnet.create_Pnet(input_shape)
    elif model_name == "RNet":
        model = Rnet.create_Rnet(input_shape)
    elif model_name == "ResNet50":
        model = resnet_common.ResNet50(input_shape=input_shape)
    elif model_name == "ResNet101":
        model = resnet_common.ResNet101(input_shape=input_shape)
    elif model_name == "ResNet152":
        model = resnet_common.ResNet152(input_shape=input_shape)
    elif model_name == "ResNet50v2":
        model = resnet_common.ResNet50V2(input_shape=input_shape)
    elif model_name == "ResNet101v2":
        model = resnet_common.ResNet101V2(input_shape=input_shape)
    elif model_name == "ResNet152v2":
        model = resnet_common.ResNet152V2(input_shape=input_shape)
    elif model_name == "Inceptionv1":
        model = GoogLeNet.create_GoogLeNet(input_shape=input_shape)
    elif model_name == "Inceptionv3":
        model = inception_v3.create_InceptionV3(input_shape=input_shape)
    elif model_name == "Xception":
        model = xception.create_Xception(input_shape)
    elif model_name == "ShuffleNet-2":
        model = shufflenet.ShuffleNet(input_shape=input_shape, groups=2)
    elif model_name == "ShuffleNet-4":
        model = shufflenet.ShuffleNet(input_shape=input_shape, groups=4)
    elif model_name == "ShuffleNet-8":
        model = shufflenet.ShuffleNet(input_shape=input_shape, groups=8)
    elif model_name == "MobileNet":
        model = mobilenet_v1.create_MobileNet(input_shape)
    elif model_name == "MobileNet_noAct":
        model = mobilenet_noAct.create_mobilenet_noAct(input_shape)
    elif model_name == "NasNet":
        # Use nasnet mobile
        model = nasnet.NASNetMobile(input_shape=input_shape)
    elif model_name == "DenseNet":
        model = densenet161. create_densenet161(input_shape=input_shape)
    elif model_name == "Inceptionv4":
        model = inception_v4.create_inceptionV4(input_shape=input_shape)
    elif model_name == "MobileNetv2":
        model = MobileNetV2.MobileNetV2(input_shape=input_shape)
    elif model_name == "mobilenetv2_ssd":
        model = mobilenet_v2_ssd.create_mobilenetv2_ssd(input_shape=input_shape)
    elif model_name == "mobilenetv2_ssdlite":
        model = mobilenet_v2_ssdlite.create_mobilenetv2_ssdlite(input_shape = input_shape)
    elif model_name == "mobilenetv1_ssd":
        model = mobilenet_v1_ssd.create_mobilenetv1_ssd(input_shape=input_shape)
    elif model_name == "mobilenetv1_ssd_reduced":
        model = mobilenet_v1_ssd_reduced.create_mobilenetv1_ssd_reduced(input_shape=input_shape)
    elif model_name == "mobilenetv2_retinanet":
        model = mobilenet_v2_retinanet.mobilenet_retinanet(21, input_shape=input_shape)
    elif model_name == "mobilenetv2_retinanet_sharedSubmodel":
        # default num_classes=21
        model = mobilenet_v2_retinanet_sharedSubmodel.mobilenet_retinanet(input_shape=input_shape)
    elif model_name == "MobileNetV2_noAct":
        model = MobileNetV2_noAct.MobileNetV2(input_shape=input_shape)
    # elif model_name == "ResNext":
    #     # default depth = 29, cardinality = 8
    #     model = resnext.ResNext(input_shape=input_shape)
    elif model_name == "ResNext50":
        model = resnet_common.ResNeXt50(input_shape=input_shape)
    elif model_name == "ResNext101":
        model = resnet_common.ResNeXt101(input_shape=input_shape)

    elif model_name == "ssd512":
        model = ssd512.create_ssd512(input_shape=input_shape)
    elif model_name == "ssd300":
        model = ssd300.create_ssd300(input_shape=input_shape)

    elif model_name == "ResNext50_detectron":
        model = resnext.ResNextImageNet(input_shape=input_shape, depth=[3, 4, 6, 3])
    elif model_name == "ResNext101_detectron":
        model = resnext.ResNextImageNet(input_shape=input_shape, depth=[3, 4, 23, 3])

    elif model_name == "PVAnet":
        model = PVAnet.create_pvanet(input_shape=input_shape)
    elif model_name == "yolov2":
        model = yolo_v2.create_yolov2(input_shape=input_shape)
    elif model_name == "tiny_yolo":
        model = tiny_yolo_v3.create_tiny_yolo(input_shape)
    elif "FR" in model_name: ### FR-leaky-res18
        model = FR.create_FR(input_shape, model_name)
    elif model_name == "FD":
        model = FD.create_FD(input_shape=input_shape)
    elif model_name == "Onet48":
        model = Onet48.create_onet(input_shape=input_shape)
    elif model_name == "Onet56":
        model = Onet56.create_onet(input_shape=input_shape)
    elif model_name == "PVAnet_fasterRCNN_rpn":
        model = PVAnet_fasterRCNN.rpn_create(input_shape=(480, 640, 3))
    elif model_name == "PVAnet_fasterRCNN_rcnn":
        model = PVAnet_fasterRCNN.rcnn_create((30, 40, 512), num_rois=20)
    elif model_name == "vgg16":
        model = vgg.create_vgg16(input_shape)
    elif model_name == "vgg19":
        model = vgg.create_vgg19(input_shape)
    elif model_name == "Deeplabv3":
        model = Deeplabv3.create_deeplabv3(input_shape) 
    elif model_name == "Deeplabv3plus":
        model = Deeplabv3plus.create_deeplabv3plus(input_shape) ### has to be (512, 512, 3)
    elif model_name == "Deeplabv3plus_mobv2":
        model = Deeplabv3plus.create_deeplabv3plus_mobv2(input_shape)### has to be (512, 512, 3)
    elif model_name == "SRCNN":
        model = SRCNN.create_srcnn(input_shape)
    elif model_name == "SRGAN":
        model = SRGAN.create_srgan(input_shape)
    elif model_name == "VDSR_vgg19":
        model = VDSR_vgg19.create_vgg19(input_shape)
    elif model_name == "Resnet_12":
        model = Resnet_12.create_resnet12(input_shape)
    elif model_name == "Inception_resnet_v1":
        model = Inception_resnet_v1.create_inception_resnet_v1(input_shape)
    elif model_name == "ICNet":
        model = ICNet.ICNetModelFactory.create_icnet(input_shape)
    elif model_name == "img_edsr":
        model = img_edsr.create_edsr(input_shape)
    elif model_name == "img_ocr":
        model = img_ocr.create_img_ocr(input_shape)
    elif model_name == "img_pspnet":
        model = img_pspnet.create_pspnet(input_shape)
    elif model_name == "lstm_attention":
        model = lstm_attention.create_lstm_attention()
    elif model_name == "openpose":
        model = openpose.create_openpose()

    elif model_name == "detectron_frcnn_resnet50v1c4":
        model = resnet50_v1_c4_frcnn.create_detectron_frcnn_resnet50v1c4(input_shape)
    elif model_name == "detectron_fpn_frcnn_resnet50v1":
        model = resnet50_v1_fpn_frcnn.create_detectron_fpn_frcnn_resnet50v1(input_shape)
    elif model_name == "detectron_fpn_frcnn_resnet101v1":
        model = resnet101_v1_fpn_frcnn.create_detectron_fpn_frcnn_resnet101v1(input_shape)
    elif model_name == "detectron_fpn_frcnn_resnext101v1_32":
        model = resnext101_v1_fpn_frcnn.create_detectron_fpn_frcnn_resnxet101v1_64x4(input_shape)
    elif model_name == "detectron_fpn_frcnn_resnext101v1_64":
        model = resnext101_v1_fpn_frcnn.create_detectron_fpn_frcnn_resnxet101v1_32x8(input_shape)


    elif model_name == "detectron_mfrcnn_resnet50v1c4":
        model = resnet50_v1_c4_mfrcnn.create_detectron_mfrcnn_resnet50v1c4(input_shape)
    elif model_name == "detectron_fpn_mfrcnn_resnet50v1":
        model = resnet50_v1_fpn_mfrcnn.create_detectron_fpn_mfrcnn_resnet50v1(input_shape)
    elif model_name == "detectron_fpn_mfrcnn_resnet101v1":
        model = resnet101_v1_fpn_mfrcnn.create_detectron_fpn_mfrcnn_resnet101v1(input_shape)
    elif model_name == "detectron_fpn_mfrcnn_resnext101v1_32":
        model = resnext101_v1_fpn_mfrcnn.create_detectron_fpn_mfrcnn_resnxet101v1_64x4(input_shape)
    elif model_name == "detectron_fpn_mfrcnn_resnext101v1_64":
        model = resnext101_v1_fpn_mfrcnn.create_detectron_fpn_mfrcnn_resnxet101v1_32x8(input_shape)

    else:
        raise ValueError('No such model.')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically generate model summary mem_tables",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model',type=str, required=True)
    parser.add_argument('-d', '--destination',type=str, required=True)
    parser.add_argument('-i', '--inputshape',type=int, nargs="+")

    args = parser.parse_args()

    model_name = args.model
    table_path = args.destination

    print(model_name)

    if args.inputshape:
        input_shape = tuple(args.inputshape)
    else:
        input_shape = (224, 224, 3)

    if 'detectron' in model_name:
        [model1, model2] = create_keras_model(model_name, input_shape)
        model1.save("./"+table_path+"/%s_%dw_%dh.h5" % (model_name, input_shape[1], input_shape[0]))
        model2.save("./"+table_path+"/%s_%dw_%dh.h5" % (model_name + '_head', input_shape[1], input_shape[0]))

    else:
        model = create_keras_model(model_name, input_shape)
        model.summary()
        # model.save('./tmp.hdf5')
        model.save("./"+table_path+"/%s_%dw_%dh.h5" % (model_name, input_shape[1], input_shape[0]))
    #model.save(os.path.join(os.path.split(table_path)[0], "%s_%dw_%dh.h5" % (model_name, input_shape[1], input_shape[0])))

    #if input_shape[2] == 1 or input_shape[2] == 3:
    #    generate_testdata_and_output(model_name, input_shape, model, os.path.split(table_path)[0])