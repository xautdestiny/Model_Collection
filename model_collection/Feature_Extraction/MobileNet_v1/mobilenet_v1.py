from keras.applications.mobilenet import MobileNet
def create_MobileNet(input_shape = (224, 224, 3)):
    # instantiate pre-trained MobileNet model
    # the default input shape is (299, 299, 3)
    model = MobileNet(include_top=True, weights=None, input_shape=input_shape)
    return model


if __name__ == "__main__":
	model = create_MobileNet()
	model.summary()
	model.save("mobilenet_v1_224_224.hdf5")