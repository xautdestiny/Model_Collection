from keras.applications.xception import Xception, preprocess_input
def create_Xception(input_shape = (299, 299, 3)):
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    model = Xception(include_top=True, weights=None, input_shape=input_shape)
    return model