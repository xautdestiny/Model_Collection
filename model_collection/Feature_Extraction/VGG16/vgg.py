from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

def create_vgg16(input_shape):
    model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=1000)
    return model

def create_vgg19(input_shape):
    model = VGG19(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=1000)
    return model


if __name__ == '__main__':
    model = create_vgg16(input_shape=(128, 128, 3))
    print(model.summary())
    model.save("vgg16_128_128.hdf5")

    model = create_vgg16(input_shape=(224, 224, 3))
    print(model.summary())
    model.save("vgg16_224_224.hdf5")