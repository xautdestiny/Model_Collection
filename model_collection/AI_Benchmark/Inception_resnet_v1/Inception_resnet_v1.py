from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation

from keras.layers.normalization import BatchNormalization

from keras.layers import MaxPooling2D, Conv2D, AveragePooling2D

from keras.models import Model


from keras import backend as K



import warnings

warnings.filterwarnings('ignore')



"""

Implementation of Inception-Residual Network v1 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.



Some additional details:

[1] Each of the A, B and C blocks have a 'scale_residual' parameter.

    The scale residual parameter is according to the paper. It is however turned OFF by default.



    Simply setting 'scale=True' in the create_inception_resnet_v1() method will add scaling.

"""





def inception_resnet_stem(input):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1


    #inp = Input(shape=input)
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)

    c = Conv2D(32, (3, 3), activation='relu', strides=2)(input)

    c = Conv2D(32, (3, 3), activation='relu', )(c)

    c = Conv2D(64, (3, 3), activation='relu', )(c)

    c = MaxPooling2D((3, 3), strides=(2, 2))(c)

    c = Conv2D(80, (1, 1), activation='relu', border_mode='same')(c)

    c = Conv2D(192, (3, 3), activation='relu')(c)

    c = Conv2D(256, (3, 3), activation='relu', strides=2, border_mode='same')(c)

    b = BatchNormalization(axis=channel_axis)(c)

    b = Activation('relu')(b)

    return b



def inception_resnet_A(input, scale_residual=True):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1



    # Input is relu activation

    init = input



    ir1 = Conv2D(32, (1, 1),  activation='relu', border_mode='same')(input)



    ir2 = Conv2D(32, (1, 1),  activation='relu', border_mode='same')(input)

    ir2 = Conv2D(32, (3, 3),  activation='relu', border_mode='same')(ir2)



    ir3 = Conv2D(32, (1, 1),  activation='relu', border_mode='same')(input)

    ir3 = Conv2D(32, (3, 3),  activation='relu', border_mode='same')(ir3)

    ir3 = Conv2D(32, (3, 3),  activation='relu', border_mode='same')(ir3)


    ir_merge = merge.concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = Conv2D(256, (1, 1),  activation='linear', border_mode='same')(ir_merge)

    #if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)



    out = merge.add([init, ir_conv])

    out = BatchNormalization(axis=channel_axis)(out)

    out = Activation("relu")(out)

    return out



def inception_resnet_B(input, scale_residual=True):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1



    # Input is relu activation

    init = input



    ir1 = Conv2D(128, (1, 1),  activation='relu', border_mode='same')(input)



    ir2 = Conv2D(128, (1, 1), activation='relu', border_mode='same')(input)

    ir2 = Conv2D(128, (1, 7), activation='relu', border_mode='same')(ir2)

    ir2 = Conv2D(128, (7, 1), activation='relu', border_mode='same')(ir2)



    ir_merge = merge.concatenate([ir1, ir2], axis=channel_axis)


    ir_conv = Conv2D(896, (1, 1), activation='linear', border_mode='same')(ir_merge)

    #if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)



    out = merge.add([init, ir_conv])

    out = BatchNormalization(axis=channel_axis)(out)

    out = Activation("relu")(out)

    return out



def inception_resnet_C(input, scale_residual=True):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1



    # Input is relu activation

    init = input



    ir1 = Conv2D(128, (1, 1), activation='relu', border_mode='same')(input)



    ir2 = Conv2D(192, (1, 1), activation='relu', border_mode='same')(input)

    ir2 = Conv2D(192, (1, 3), activation='relu', border_mode='same')(ir2)

    ir2 = Conv2D(192, (3, 1), activation='relu', border_mode='same')(ir2)



    ir_merge = merge.concatenate([ir1, ir2], axis=channel_axis)


    ir_conv = Conv2D(1792, (1, 1), activation='linear', border_mode='same')(ir_merge)

    #if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)



    out = merge.add([init, ir_conv])

    out = BatchNormalization(axis=channel_axis)(out)

    out = Activation("relu")(out)

    return out



def reduction_A(input, k=192, l=224, m=256, n=384):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1



    r1 = MaxPooling2D((3,3), strides=(2,2))(input)



    r2 = Conv2D(n, (3, 3), activation='relu', subsample=(2,2))(input)



    r3 = Conv2D(k, (1, 1), activation='relu', border_mode='same')(input)

    r3 = Conv2D(l, (3, 3), activation='relu', border_mode='same')(r3)

    r3 = Conv2D(m, (3, 3), activation='relu', strides=2)(r3)



    m = merge.concatenate([r1, r2, r3], axis=channel_axis)

    m = BatchNormalization(axis=channel_axis)(m)

    m = Activation('relu')(m)

    return m





def reduction_resnet_B(input):

    if K.image_dim_ordering() == "th":

        channel_axis = 1

    else:

        channel_axis = -1



    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)



    r2 = Conv2D(256,(1, 1), activation='relu', border_mode='same')(input)

    r2 = Conv2D(384, (3, 3), activation='relu', strides=2)(r2)



    r3 = Conv2D(256, (1, 1), activation='relu', border_mode='same')(input)

    r3 = Conv2D(256, (3, 3), activation='relu', strides=2)(r3)



    r4 = Conv2D(256, (1, 1), activation='relu', border_mode='same')(input)

    r4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(r4)

    r4 = Conv2D(256, (3, 3), activation='relu', strides=2)(r4)



    m = merge.concatenate([r1, r2, r3, r4], axis=channel_axis)

    m = BatchNormalization(axis=channel_axis)(m)

    m = Activation('relu')(m)

    return m



def create_inception_resnet_v1(image_input=Input((299, 299 ,3)), nb_classes=1001, scale=True):

    '''

    Creates a inception resnet v1 network



    :param nb_classes: number of classes.txt

    :param scale: flag to add scaling of activations

    :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)

    '''



    # if K.image_dim_ordering() == 'th':

    #     init = Input((3, 299, 299))

    # else:

    #     init = Input((299, 299, 3))


    init = Input(shape=image_input)
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)

    x = inception_resnet_stem(init)



    # 5 x Inception Resnet A

    for i in range(5):

        x = inception_resnet_A(x, scale_residual=scale)



    # Reduction A - From Inception v4

    x = reduction_A(x, k=192, l=192, m=256, n=384)



    # 10 x Inception Resnet B

    for i in range(10):

        x = inception_resnet_B(x, scale_residual=scale)



    # Auxiliary tower

    aux_out = AveragePooling2D((5, 5), strides=(3, 3))(x)

    aux_out = Conv2D(128, (1, 1), border_mode='same', activation='relu')(aux_out)

    aux_out = Conv2D(768, (5, 5), activation='relu')(aux_out)

    aux_out = Flatten()(aux_out)

    aux_out = Dense(nb_classes, activation='softmax')(aux_out)



    # Reduction Resnet B

    x = reduction_resnet_B(x)



    # 5 x Inception Resnet C

    for i in range(5):

        x = inception_resnet_C(x, scale_residual=scale)



    # Average Pooling

    x = AveragePooling2D((8,8))(x)



    # Dropout

    x = Dropout(0.8)(x)

    x = Flatten()(x)



    # Output

    out = Dense(output_dim=nb_classes, activation='softmax')(x)



    model = Model(init, output=[out, aux_out], name='Inception-Resnet-v1')



    return model



# if __name__ == "__main__":


#     inception_resnet_v1 = create_inception_resnet_v1()

#     inception_resnet_v1.summary()

#     inception_resnet_v1.save('inception_resnet_v1.hdf5')



