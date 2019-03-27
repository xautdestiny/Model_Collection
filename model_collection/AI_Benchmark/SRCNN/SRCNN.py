from keras.models import Sequential

from keras.layers import Conv2D, Input, BatchNormalization

# from keras.layers.advanced_activations import LeakyReLU



def create_srcnn(input_shape):

    # lrelu = LeakyReLU(alpha=0.1)

    SRCNN = Sequential()

    SRCNN.add(Conv2D(nb_filter=64, nb_row=9, nb_col=9, init='glorot_uniform',

                     activation='relu', border_mode='valid', bias=True, input_shape=input_shape))

    SRCNN.add(Conv2D(nb_filter=32, nb_row=1, nb_col=1, init='glorot_uniform',

                     activation='relu', border_mode='same', bias=True))


    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',

                     activation='linear', border_mode='valid', bias=True))

    return SRCNN

# if __name__ == "__main__":
#     model = create_srcnn(input_shape=(300, 300, 1))
#     model.summary()
    #model.save('SRCNN.hdf5')