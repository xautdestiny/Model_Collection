
# coding: utf-8

# In[1]:


from keras.layers import *
from keras.models import *


# In[6]:

def create_img_ocr(in_shape=(64,256,1)):

	img_h = in_shape[0]
	img_w = in_shape[1]
	words_per_epoch = 16000
	val_split = 0.2
	val_words = int(words_per_epoch * (val_split))
	act = 'relu'

	# Network parameters
	conv_filters = 16
	kernel_size = (3, 3)
	pool_size = 2
	time_dense_size = 32
	rnn_size = 512
	minibatch_size = 32

	if K.image_data_format() == 'channels_first':
	    input_shape = (1, img_w, img_h)
	else:
	    input_shape = (img_w, img_h, 1)
	input_data = Input(name='the_input', shape=input_shape, dtype='float32')
	inner = Conv2D(conv_filters, kernel_size, padding='same',
	               activation=act, kernel_initializer='he_normal',
	               name='conv1')(input_data)
	inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
	inner = Conv2D(conv_filters, kernel_size, padding='same',
	               activation=act, kernel_initializer='he_normal',
	               name='conv2')(inner)
	inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

	conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
	inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

	# cuts down input size going into RNN:
	inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

	# Two layers of bidirectional GRUs
	# GRU seems to work as well, if not better than LSTM:
	gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
	gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
	gru1_merged = add([gru_1, gru_1b])
	gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
	gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

	# transforms RNN output to character activations:
	inner = Dense(36, kernel_initializer='he_normal',
	              name='dense2')(concatenate([gru_2, gru_2b]))
	y_pred = Activation('softmax', name='softmax')(inner)
	model_ocr = Model(inputs=input_data, outputs=y_pred)

	return model_ocr
# In[7]:


# model = create_img_ocr()

# model.summary()
#model_ocr.save('./img_ocr.hdf5')