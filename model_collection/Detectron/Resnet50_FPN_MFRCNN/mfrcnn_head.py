from keras import applications
from keras.layers import Layer
from keras.layers import *
import keras
from keras import backend as K
from keras.models import Model



# def frcnn_pred(num_rois=4, nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256):

#     out_roi_pool = Input(shape=(num_rois, pooling_regions, pooling_regions, dim_in))

#     #for x in input:
#     out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
#     out = TimeDistributed(Dense(hidden_dim, activation='relu', name='fc1'))(out)

#     out = TimeDistributed(Dense(hidden_dim, activation='relu', name='fc2'))(out)


#     # There are two output layer
#     # out_class: softmax acivation function for classify the class name of the object
#     # out_regr: linear activation function for bboxes coordinates regression
#     out_class = TimeDistributed(Dense(nb_classes-1, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
#     # note: no regression target for bg class
#     out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
#     return Model(out_roi_pool, [out_class, out_regr])

# # model = frcnn_pred (num_rois=4, nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256)
# # model.summary()


# def mfrcnn_pred(num_rois=4, nb_classes=81, pooling_regions=14, hidden_dim= 1024, dim_in=256):
#     out_roi_pool = Input(shape=(num_rois, pooling_regions, pooling_regions, dim_in))
#     # Flatten the convlutional layer and connected to 2 FC and 2 dropout
#     out_mask = TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu'))(out_roi_pool)
#     out_mask = TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu'))(out_mask)
#     out_mask = TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu'))(out_mask)
#     out_mask = TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu'))(out_mask)


#     out_mask = TimeDistributed(Conv2DTranspose(256, (2,2), strides=2), name = 'mask_28')(out_mask)
#     out_mask = TimeDistributed(Activation('relu'))(out_mask)
#     out_mask = TimeDistributed(Conv2D(nb_classes-1, (1, 1), activation='sigmoid'))(out_mask)
#     return Model(out_roi_pool, [out_mask])



# model = mfrcnn_pred (num_rois=4, nb_classes=81, pooling_regions=14, hidden_dim = 1024, dim_in=256)
# model.summary()
def mfrcnn_pred( nb_classes=81, pooling_regions=14, hidden_dim= 1024, dim_in=256):
    out_roi_pool = Input(shape=( pooling_regions, pooling_regions, dim_in))
    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_roi_pool)
    out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)
    out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)
    out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)


    out_mask = Conv2DTranspose(256, (2,2), strides=2, name = 'mask_28')(out_mask)
    out_mask = Activation('relu')(out_mask)
    out_mask = Conv2D(nb_classes-1, (1, 1), activation='sigmoid')(out_mask)
    return Model(out_roi_pool, out_mask)


def mask_frcnn_pred (nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256):

    out_roi_pool = Input(shape=(pooling_regions, pooling_regions, dim_in))
    #for x in input:
    out = Flatten(name='flatten')(out_roi_pool)
    out = Dense(hidden_dim, activation='relu', name='fc1')(out)
    out = Dense(hidden_dim, activation='relu', name='fc2')(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = Dense(nb_classes-1, activation='softmax', kernel_initializer='zero', name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero', name='dense_regress_{}'.format(nb_classes))(out)
    
    mask_model = mfrcnn_pred()
    return Model([out_roi_pool, mask_model.input], [out_class, out_regr, mask_model.output])

# model = mask_frcnn_pred (nb_classes=81, pooling_regions=7, hidden_dim= 1024, dim_in=256)
# model.summary()
# model.save('Resnet50_FPN_faster_head.hdf5')


# def mfrcnn_pred( nb_classes=81, pooling_regions=14, hidden_dim= 1024, dim_in=256):
#     out_roi_pool = Input(shape=( pooling_regions, pooling_regions, dim_in))
#     # Flatten the convlutional layer and connected to 2 FC and 2 dropout
#     out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_roi_pool)
#     out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)
#     out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)
#     out_mask = Conv2D(256, (3,3), padding='same', activation='relu')(out_mask)


#     out_mask = Conv2DTranspose(256, (2,2), strides=2, name = 'mask_28')(out_mask)
#     out_mask = Activation('relu')(out_mask)
#     out_mask = Conv2D(nb_classes-1, (1, 1), activation='sigmoid')(out_mask)
#     return Model(out_roi_pool, [out_mask])

# # model =  mfrcnn_pred (nb_classes=81, pooling_regions=14, hidden_dim= 1024, dim_in=256)
# # model.summary()
# # model.save('Resnet50_FPN_mask_head.hdf5')
