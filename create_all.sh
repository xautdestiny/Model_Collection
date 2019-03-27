set -e


# python draw_summary_table.py -m ResNet101 -d "./model_collection/Feature_Extraction/ResNet" -i 224 224 3
# python draw_summary_table.py -m ResNet152 -d "./model_collection/Feature_Extraction/ResNet"
# python draw_summary_table.py -m vgg16 -d "./model_collection/Feature_Extraction/VGG16"
# python draw_summary_table.py -m Inceptionv1 -d "./model_collection/Feature_Extraction/GoogleNet"
# python draw_summary_table.py -m Inceptionv3 -d "./model_collection/Feature_Extraction/inception_v3"
# python draw_summary_table.py -m Inceptionv4 -d "./model_collection/Feature_Extraction/inception_v4" -i 299 299 3
# python draw_summary_table.py -m Xception -d "./model_collection/Feature_Extraction/xception"
# python draw_summary_table.py -m ShuffleNet-2 -d "./model_collection/Feature_Extraction/shufflenet"
# python draw_summary_table.py -m ShuffleNet-4 -d "./model_collection/Feature_Extraction/shufflenet"
# python draw_summary_table.py -m ShuffleNet-8 -d "./model_collection/Feature_Extraction/shufflenet"
# python draw_summary_table.py -m MobileNet -d "./model_collection/Feature_Extraction/MobileNet_v1"

# python draw_summary_table.py -m NasNet -d "./model_collection/Feature_Extraction/nasnet"
# python draw_summary_table.py -m MobileNetv2 -d "./model_collection/Feature_Extraction/MobileNet_v2"
# python draw_summary_table.py -m MobileNetV2_noAct -d "./model_collection/Feature_Extraction/MobileNetV2_noAct"

# python draw_summary_table.py -m ResNext50_detectron -d "./model_collection/Feature_Extraction/ResNext"
# python draw_summary_table.py -m ResNext101_detectron -d "./model_collection/Feature_Extraction/ResNext"

# python draw_summary_table.py -m ResNext50 -d "./model_collection/Feature_Extraction/ResNet"
# python draw_summary_table.py -m ResNext101 -d "./model_collection/Feature_Extraction/ResNet"




# python draw_summary_table.py -m DenseNet -d "./model_collection/Feature_Extraction/densenet"
# python draw_summary_table.py -m ONet -d "./model_collection/Object_Detection/Onet"
# python draw_summary_table.py -m PNet -d "./model_collection/Object_Detection/Pnet"
# python draw_summary_table.py -m RNet -d "./model_collection/Object_Detection/Rnet"
# python draw_summary_table.py -m mobilenetv1_ssd -d "./model_collection/Object_Detection/MobileNet_v1_SSD" -i 480 640 3
# python draw_summary_table.py -m mobilenetv1_ssd -d "./model_collection/Object_Detection/MobileNet_v1_SSD" -i 720 960 3
# python draw_summary_table.py -m mobilenetv1_ssd -d "./model_collection/Object_Detection/MobileNet_v1_SSD" -i 224 224 3
# python draw_summary_table.py -m mobilenetv1_ssd_reduced -d "./model_collection/Object_Detection/MobileNet_v1_SSD_Reduced" -i 480 640 3
# python draw_summary_table.py -m mobilenetv1_ssd_reduced -d "./model_collection/Object_Detection/MobileNet_v1_SSD_Reduced" -i 720 960 3
# python draw_summary_table.py -m mobilenetv2_ssd -d "./model_collection/Object_Detection/MobileNet_v2_SSD" -i 480 640 3
# python draw_summary_table.py -m mobilenetv2_ssd -d "./model_collection/Object_Detection/MobileNet_v2_SSD" -i 720 960 3
# python draw_summary_table.py -m mobilenetv2_ssdlite -d "./model_collection/Object_Detection/MobileNet_v2_SSDlite" -i 480 640 3
# python draw_summary_table.py -m mobilenetv2_ssdlite -d "./model_collection/Object_Detection/MobileNet_v2_SSDlite" -i 720 960 3
# python draw_summary_table.py -m FD -d "./model_collection/Other_Models/FD" -i 160 160 3
# python draw_summary_table.py -m FD_nobias -d "./model_collection/Other_Models/FD" -i 160 160 3                                      
# python draw_summary_table.py -m FR_leaky_mobilefacenet -d  -i 112 112 3                                                                ######### command not defined######### 
# python draw_summary_table.py -m FR_leaky_res18 -d "./model_collection/Other_Models/FR" -i 112 112 3                                         ######### command not defined#########                    
# python draw_summary_table.py -m FR_leaky_res28 -d "./model_collection/Other_Models/FR" -i 112 112 3                                         ######### command not defined#########                     
# python draw_summary_table.py -m FR_leaky_res34 -d "./model_collection/Other_Models/FR" -i 112 112 3                                         ######### command not defined#########                       
# python draw_summary_table.py -m FR_leaky_res50 -d "./model_collection/Other_Models/FR" -i 112 112 3                                         ######### command not defined#########                       
# python draw_summary_table.py -m FR_prelu_res34 -d "./model_collection/Other_Models/FR" -i 112 112 3                                         ######### command not defined#########                         
# python draw_summary_table.py -m FR_prelu_res100 -d "./model_collection/Other_Models/FR" -i 112 112 3                                        ######### command not defined#########                                                                     
# python draw_summary_table.py -m PVAnet_fasterRCNN_rpn -d "./model_collection/Feature_Extraction/PVAnet" -i 480 640 3
# python draw_summary_table.py -m PVAnet_fasterRCNN_rcnn -d "./model_collection/Object_Detection/PVAnet_fasterRCNN" -i 30 40 512
# python draw_summary_table.py -m mobilenetv2_retinanet -d "./model_collection/Object_Detection/MobileNet_v2_retinanet" -i 800 1000 3
# python draw_summary_table.py -m yolov2 -d "./model_collection/Object_Detection/yolo_v2" -i 480 640 3
# python draw_summary_table.py -m yolov2 -d "./model_collection/Object_Detection/yolo_v2" -i 736 960 3
# python draw_summary_table.py -m yolov2 -d "./model_collection/Object_Detection/yolo_v2" -i 608 608 3
# python draw_summary_table.py -m tiny_yolo -d "./model_collection/Object_Detection/tiny_yolo_v3" -i 416 416 3
# python draw_summary_table.py -m ssd512 -d "./model_collection/Object_Detection/SSD" -i 512 512 3                                                 
# python draw_summary_table.py -m ssd300 -d "./model_collection/Object_Detection/SSD" -i 300 300 3                                                 
# python draw_summary_table.py -m Deeplabv3 -d "./model_collection/Segmentation/Deeplabv3" -i 512 512 3
# python draw_summary_table.py -m Deeplabv3plus -d "./model_collection/Segmentation/Deeplabv3plus" -i 512 512 3                               
# python draw_summary_table.py -m Deeplabv3plus_mobv2 -d "./model_collection/Segmentation/Deeplabv3plus" -i 512 512 3

# python draw_summary_table.py -m SRCNN -d "./model_collection/AI_Benchmark/SRCNN" -i 300 300 1
# python draw_summary_table.py -m SRGAN -d "./model_collection/AI_Benchmark/SRGAN" -i 512 512 3
# python draw_summary_table.py -m VDSR_vgg19 -d "./model_collection/AI_Benchmark/VDSR" -i 192 192 3
# python draw_summary_table.py -m Resnet_12 -d "./model_collection/AI_Benchmark/Resnet_12" -i 128 192 3
# python draw_summary_table.py -m Inception_resnet_v1 -d "./model_collection/AI_Benchmark/Inception_resnet_v1" -i 299 299 3
# python draw_summary_table.py -m ICNet -d "./model_collection/AI_Benchmark/ICNet" -i 384 384 3

# python draw_summary_table.py -m img_edsr -d "./model_collection/Super_Resolution/edsr" -i 48 48 3           
# python draw_summary_table.py -m img_ocr -d "./model_collection/Other_Models/OCR" -i 64 256 1       
# python draw_summary_table.py -m img_pspnet -d "./model_collection/Segmentation/pspnet" -i 473 473 3  
# python draw_summary_table.py -m lstm_attention -d "./model_collection/Other_Models/LSTM_Attention"
# python draw_summary_table.py -m ResNet50 -d "./model_collection/Feature_Extraction/ResNet" -i 720 960 3
# python draw_summary_table.py -m ResNet50 -d "./model_collection/Feature_Extraction/ResNet" -i 1080 1920 3
# python draw_summary_table.py -m Inceptionv3 -d "./model_collection/Feature_Extraction/inception_v3" -i 720 960 3
# python draw_summary_table.py -m Inceptionv3 -d "./model_collection/Feature_Extraction/inception_v3" -i 1080 1920 3
# python draw_summary_table.py -m MobileNet -d "./model_collection/Feature_Extraction/MobileNet_v1" -i 720 960 3
# python draw_summary_table.py -m MobileNet -d "./model_collection/Feature_Extraction/MobileNet_v1" -i 1080 1920 3
# python draw_summary_table.py -m MobileNetv2 -d "./model_collection/Feature_Extraction/MobileNet_v2" -i 720 960 3
# python draw_summary_table.py -m MobileNetv2 -d "./model_collection/Feature_Extraction/MobileNet_v2" -i 1080 1920 3
# python draw_summary_table.py -m Deeplabv3plus_mobv2 -d "./model_collection/Segmentation/Deeplabv3plus" -i 720 960 3
# python draw_summary_table.py -m Deeplabv3plus_mobv2 -d "./model_collection/Segmentation/Deeplabv3plus" -i 1080 1920 3
# python draw_summary_table.py -m openpose -d "./model_collection/Other_Models/openpose" -i 256 256 3                              

# python draw_summary_table.py -m detectron_frcnn_resnet50v1c4 -d "./model_collection/Detectron/Resnet50_C4_FRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_frcnn_resnet50v1 -d "./model_collection/Detectron/Resnet50_FPN_FRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_frcnn_resnet101v1 -d "./model_collection/Detectron/Resnet101_FPN_FRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_frcnn_resnext101v1_64 -d "./model_collection/Detectron/Resnext101_FPN_FRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_frcnn_resnext101v1_32 -d "./model_collection/Detectron/Resnext101_FPN_FRCNN" -i 800 640 3 # 640 480 3 # 480 320 3




# python draw_summary_table.py -m detectron_mfrcnn_resnet50v1c4 -d "./model_collection/Detectron/Resnet50_C4_MFRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_mfrcnn_resnet50v1 -d "./model_collection/Detectron/Resnet50_FPN_MFRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_mfrcnn_resnet101v1 -d "./model_collection/Detectron/Resnet101_FPN_MFRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_mfrcnn_resnext101v1_64 -d "./model_collection/Detectron/Resnext101_FPN_MFRCNN" -i 800 640 3 # 640 480 3 # 480 320 3
# python draw_summary_table.py -m detectron_fpn_mfrcnn_resnext101v1_32 -d "./model_collection/Detectron/Resnext101_FPN_MFRCNN" -i 800 640 3 # 640 480 3 # 480 320 3

