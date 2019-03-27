from keras.layers import Conv2D, MaxPooling2D, Input, Reshape
from keras import backend as K
from keras import Model
from keras import regularizers, initializers
from keras.layers import Conv2D, BatchNormalization, LeakyReLU

import numpy as np




def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                        kernel_regularizer=regularizers.l2(0.0005),
                        kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                        use_bias=False
                    )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)


class TinyYOLOv2:
    def __init__(self, image_size, num_anchors, n_classes, is_learning_phase=False):
        K.set_learning_phase(int(is_learning_phase))
        K.reset_uids()

        self.image_size = image_size
        self.n_cells = self.image_size[0] // 32
        self.B = num_anchors
        self.n_classes = n_classes

    def buildModel(self):
        model_in = Input(self.image_size)
        
        model = model_in
        for i in range(0, 5):
            
            model = conv_batch_lrelu(model, 16 * 2**i, 3)
            model = MaxPooling2D(2, padding='valid')(model)

        model = conv_batch_lrelu(model, 512, 3)
        model = MaxPooling2D(2, 1, padding='same')(model)

        model = conv_batch_lrelu(model, 1024, 3)
        model = conv_batch_lrelu(model, 1024, 3)
        
        model = Conv2D(125, (1, 1), padding='same', activation='linear')(model)
        print(model)
        model_out = Reshape(
            [self.n_cells, self.n_cells, self.B, 4 + 1 + self.n_classes]
            )(model)

        return Model(inputs=model_in, outputs=model_out)


if __name__ == '__main__':
    net =  TinyYOLOv2(image_size = (416, 416, 3), num_anchors= 5, n_classes= 20, is_learning_phase=False)
    model = net.buildModel()
    print(model.summary())
    model.save("tiny_yolo_v2_416_416.hdf5")

